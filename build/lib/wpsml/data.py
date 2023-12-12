# Python core
from typing import Optional, Callable, TypedDict, Union, Iterable, NamedTuple
from dataclasses import dataclass

# Scientific python
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import glob

# PyTorch
import torch
import zarr
from zarr.storage import KVStore


def get_forward_data(filename) -> xr.DataArray:
    """Lazily opens the Zarr store on gladefilesystem.
    """
    dataset = xr.open_zarr(filename, consolidated=True)
    return dataset


Array = Union[np.ndarray, xr.DataArray]
IMAGE_ATTR_NAMES = ('historical_ERA5_images', 'target_ERA5_images')


class Sample(TypedDict):
    """Simple class for structuring data for the ML model.
    
    Using typing.TypedDict gives us several advantages:
      1. Single 'source of truth' for the type and documentation of each example.
      2. A static type checker can check the types are correct.

    Instead of TypedDict, we could use typing.NamedTuple,
    which would provide runtime checks, but the deal-breaker with Tuples is that they're immutable
    so we cannot change the values in the transforms.
    """
    # IMAGES
    # Shape: batch_size, seq_length, lat, lon, lev
    historical_ERA5_images: Array
    target_ERA5_images: Array
        
    # METADATA
    datetime_index: Array


@dataclass
class Reshape_Data():
    size: int = 128  #: Size of the cropped image.

    def __call__(self, sample: Sample) -> Sample:
        for attr_name in IMAGE_ATTR_NAMES:
            image = sample[attr_name]
            # TODO: Random crop!
            cropped_image = image[..., :self.size, :self.size]
            sample[attr_name] = cropped_image
        return sample


class CheckForBadData():
    def __call__(self, sample: Sample) -> Sample:
        for attr_name in IMAGE_ATTR_NAMES:
            image = sample[attr_name]
            if np.any(image < 0):
                raise ValueError(f'\n{attr_name} has negative values at {image.time.values}')
        return sample

class Normalize():
    def __init__(self,mean_file,std_file):
        self.mean_ds = xr.open_dataset(mean_file)
        self.std_ds = xr.open_dataset(std_file)

    def __call__(self, sample:Sample)->Sample:
        normalized_sample = {}
        for key, value in sample.items():
            if isinstance(value, xr.Dataset):
                #key_change = key
                #value_change = (value - self.mean_ds)/self.std_ds
                #sample[key]=value_change
                normalized_sample[key] = (value - self.mean_ds) / self.std_ds
        return normalized_sample


class ToTensor():
    def __init__(self):
        
        self.allvarsdo = ['U','V','T','Q','SP','t2m','V500','U500','T500','Z500','Q500']
        self.surfvars = ['SP', 't2m','V500','U500','T500','Z500','Q500']
        
    def __call__(self, sample: Sample) -> Sample:
    
        return_dict = {}
        
        for key, value in sample.items():
            
            if isinstance(value, xr.DataArray):
                value_var = value.values
                
            elif isinstance(value, xr.Dataset):
                surface_vars = []
                concatenated_vars = []
                for vv in self.allvarsdo: 
                    value_var = value[vv].values
                    if vv in self.surfvars:
                        surface_vars_temp = value_var
                        surface_vars.append(surface_vars_temp)
                    else:
                        concatenated_vars.append(value_var)
                    
            else: 
                value_var = value        
                    
            if key == 'historical_ERA5_images':
                return_dict['x_surf'] = torch.as_tensor(surface_vars).squeeze()
                return_dict['x'] = torch.as_tensor(np.vstack(concatenated_vars))
            elif key == 'target_ERA5_images':
                y_surf = torch.as_tensor(surface_vars)
                #print(y_surf)
                y = torch.as_tensor(np.hstack([np.expand_dims(x, axis=1) for x in concatenated_vars]))
                return_dict['y1_surf'] = y_surf[:,0]
                return_dict['y2_surf'] = y_surf[:,1]
                return_dict['y1'] = y[0]
                return_dict['y2'] = y[1]
                
        return return_dict


class Segment(NamedTuple):
    """Represents the start and end indicies of a segment of contiguous samples."""
    start: int
    end: int
    
    
    
def get_contiguous_segments(dt_index: pd.DatetimeIndex, min_timesteps: int, max_gap: pd.Timedelta) -> Iterable[Segment]:
    """Chunk datetime index into contiguous segments, each at least min_timesteps long.
    
    max_gap defines the threshold for what constitutes a 'gap' between contiguous segments.
    
    Throw away any timesteps in a sequence shorter than min_timesteps long.
    """
    gap_mask = np.diff(dt_index) > max_gap
    gap_indices = np.argwhere(gap_mask)[:, 0]

    # gap_indicies are the indices into dt_index for the timestep immediately before the gap.
    # e.g. if the datetimes at 12:00, 12:05, 18:00, 18:05 then gap_indicies will be [1].
    segment_boundaries = gap_indices + 1

    # Capture the last segment of dt_index.
    segment_boundaries = np.concatenate((segment_boundaries, [len(dt_index)]))

    segments = []
    start_i = 0
    for end_i in segment_boundaries:
        n_timesteps = end_i - start_i
        if n_timesteps >= min_timesteps:
            segment = Segment(start=start_i, end=end_i)
            segments.append(segment)
        start_i = end_i
        
    return segments


def get_zarr_chunk_sequences(
    n_chunks_per_disk_load: int, 
    zarr_chunk_boundaries: Iterable[int], 
    contiguous_segments: Iterable[Segment]) -> Iterable[Segment]:
    """
    
    Args:
      n_chunks_per_disk_load: Maximum number of Zarr chunks to load from disk in one go.
      zarr_chunk_boundaries: The indicies into the Zarr store's time dimension which define the Zarr chunk boundaries.
        Must be sorted.
      contiguous_segments: Indicies into the Zarr store's time dimension that define contiguous timeseries.
        That is, timeseries with no gaps.
    
    Returns zarr_chunk_sequences: a list of Segments representing the start and end indicies of contiguous sequences of multiple Zarr chunks,
    all exactly n_chunks_per_disk_load long (for contiguous segments at least as long as n_chunks_per_disk_load zarr chunks),
    and at least one side of the boundary will lie on a 'natural' Zarr chunk boundary.
    
    For example, say that n_chunks_per_disk_load = 3, and the Zarr chunks sizes are all 5:
    
    
                  0    5   10   15   20   25   30   35 
                  |....|....|....|....|....|....|....|

    INPUTS:
                     |------CONTIGUOUS SEGMENT----|
                     
    zarr_chunk_boundaries:
                  |----|----|----|----|----|----|----|
    
    OUTPUT:
    zarr_chunk_sequences:
           3 to 15:  |-|----|----|
           5 to 20:    |----|----|----|
          10 to 25:         |----|----|----|
          15 to 30:              |----|----|----|
          20 to 32:                   |----|----|-|
    
    """
    assert n_chunks_per_disk_load > 0
    
    zarr_chunk_sequences = []

    for contig_segment in contiguous_segments:
        # searchsorted() returns the index into zarr_chunk_boundaries at which contig_segment.start
        # should be inserted into zarr_chunk_boundaries to maintain a sorted list.
        # i_of_first_zarr_chunk is the index to the element in zarr_chunk_boundaries which defines
        # the start of the current contig chunk.
        i_of_first_zarr_chunk = np.searchsorted(zarr_chunk_boundaries, contig_segment.start)
        
        # i_of_first_zarr_chunk will be too large by 1 unless contig_segment.start lies
        # exactly on a Zarr chunk boundary.  Hence we must subtract 1, or else we'll
        # end up with the first contig_chunk being 1 + n_chunks_per_disk_load chunks long.
        if zarr_chunk_boundaries[i_of_first_zarr_chunk] > contig_segment.start:
            i_of_first_zarr_chunk -= 1
            
        # Prepare for looping to create multiple Zarr chunk sequences for the current contig_segment.
        zarr_chunk_seq_start_i = contig_segment.start
        zarr_chunk_seq_end_i = None  # Just a convenience to allow us to break the while loop by checking if zarr_chunk_seq_end_i != contig_segment.end.
        while zarr_chunk_seq_end_i != contig_segment.end:
            zarr_chunk_seq_end_i = zarr_chunk_boundaries[i_of_first_zarr_chunk + n_chunks_per_disk_load]
            zarr_chunk_seq_end_i = min(zarr_chunk_seq_end_i, contig_segment.end)
            zarr_chunk_sequences.append(Segment(start=zarr_chunk_seq_start_i, end=zarr_chunk_seq_end_i))
            i_of_first_zarr_chunk += 1
            zarr_chunk_seq_start_i = zarr_chunk_boundaries[i_of_first_zarr_chunk]
            
    return zarr_chunk_sequences


Array = Union[np.ndarray, xr.DataArray]
IMAGE_ATTR_NAMES = ('historical_ERA5_images', 'target_ERA5_images')

class Sample(TypedDict):
    """Simple class for structuring data for the ML model.
    
    Using typing.TypedDict gives us several advantages:
      1. Single 'source of truth' for the type and documentation of each example.
      2. A static type checker can check the types are correct.

    Instead of TypedDict, we could use typing.NamedTuple,
    which would provide runtime checks, but the deal-breaker with Tuples is that they're immutable
    so we cannot change the values in the transforms.
    """
    # IMAGES
    # Shape: batch_size, seq_length, lat, lon, lev
    historical_ERA5_images: Array
    target_ERA5_images: Array
        
    # METADATA
    datetime_index: Array



def flatten_list(list_of_lists):
    """
    Flatten a list of lists.

    Parameters:
    - list_of_lists (list): A list containing sublists.

    Returns:
    - flattened_list (list): A flattened list containing all elements from sublists.
    """
    return [item for sublist in list_of_lists for item in sublist]

def generate_integer_list_around(number, spacing=10):
    """
    Generate a list of integers on either side of a given number with a specified spacing.

    Parameters:
    - number (int): The central number around which the list is generated.
    - spacing (int): The spacing between consecutive integers in the list. Default is 10.

    Returns:
    - integer_list (list): List of integers on either side of the given number.
    """
    lower_limit = number - spacing
    upper_limit = number + spacing + 1  # Adding 1 to include the upper limit
    integer_list = list(range(lower_limit, upper_limit))

    return integer_list

def find_key_for_number(input_number, data_dict):
    """
    Find the key in the dictionary based on the given number.

    Parameters:
    - input_number (int): The number to search for in the dictionary.
    - data_dict (dict): The dictionary with keys and corresponding value lists.

    Returns:
    - key_found (str): The key in the dictionary where the input number falls within the specified range.
    """
    for key, value_list in data_dict.items():
        if value_list[1] <= input_number <= value_list[2]:
            return key

    # Return None if the number is not within any range
    return None

class ERA5Dataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        filenames: list = ['/glade/derecho/scratch/wchapman/STAGING/TOTAL_2012-01-01_2012-12-31_staged.zarr','/glade/derecho/scratch/wchapman/STAGING/TOTAL_2013-01-01_2013-12-31_staged.zarr'],
        history_len: int = 1,
        forecast_len: int = 2,
        transform: Optional[Callable] = None,
        SEED=42,
    ):
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.transform = transform
        self.total_seq_len = self.history_len + self.forecast_len
        all_fils = []
        filenames = sorted(filenames)
        for fn in filenames:
            all_fils.append(get_forward_data(filename=fn))
        self.all_fils = all_fils
        self.rng = np.random.default_rng(seed=SEED)
        
        #set data places: 
        indo = 0
        self.meta_data_dict = {}
        for ee,bb in enumerate(self.all_fils):
            self.meta_data_dict[str(ee)]=[len(bb['time']),indo,indo+len(bb['time'])]
            indo += len(bb['time'])+1
            
        #set out of bounds indexes... 
        OOB = []
        for kk in self.meta_data_dict.keys():
            OOB.append(generate_integer_list_around(self.meta_data_dict[kk][2]))
        self.OOB = flatten_list(OOB)
                
    def __post_init__(self):
        #: Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self):
        tlen = 0
        for bb in self.all_fils:
            tlen+=len(bb['time']) - self.total_seq_len + 1
        return tlen

    def __getitem__(self, index):
        
        #find the result key:
        result_key = find_key_for_number(index, self.meta_data_dict)
        #get the data selection:
        true_ind = index-self.meta_data_dict[result_key][1]
        
        if true_ind > (len(self.all_fils[int(result_key)]['time'])-(self.forecast_len+1)):
            true_ind = len(self.all_fils[int(result_key)]['time'])-(self.forecast_len+1)
        
        datasel = self.all_fils[int(result_key)].isel(time=slice(true_ind, true_ind+self.forecast_len+1)).load()
        
        sample = Sample(
            historical_ERA5_images=datasel.isel(time=slice(0, self.history_len)),
            target_ERA5_images=datasel.isel(time=slice(self.history_len, len(datasel['time']))),
            datetime_index=datasel.time.values.astype('datetime64[s]').astype(int)
        )
    
        if self.transform:
            sample = self.transform(sample)
        return sample
