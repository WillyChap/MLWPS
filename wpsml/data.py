# Python core
from typing import Optional, Callable, TypedDict, Union, Iterable, NamedTuple
from dataclasses import dataclass

# Scientific python
import numpy as np
import pandas as pd
import xarray as xr

# PyTorch
import torch


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
        for key, value in sample.items():
            if isinstance(value, xr.Dataset):
                key_change = key
                value_change = (value - self.mean_ds)/self.std_ds
                sample[key]=value_change
        return sample

class ToTensor():
    def __call__(self, sample: Sample) -> Sample:
        
        return_dict = {}
        
        for key, value in sample.items():
            
            if isinstance(value, xr.DataArray):
                value_var = value.values
                
            elif isinstance(value, xr.Dataset):
                surface_vars = 0
                concatenated_vars = []
                varsdo = ['U','V','T','Q','SP']
                for vv in varsdo: 
                    value_var = value[vv].values
                    if vv == 'SP':
                        surface_vars = np.expand_dims(value_var,axis=1)
                    else:
                        concatenated_vars.append(value_var)
                    
            else: 
                value_var = value        
                    
            if key == 'historical_ERA5_images':
                return_dict['x_surf'] = torch.from_numpy(surface_vars).squeeze(1)
                return_dict['x'] = torch.from_numpy(np.vstack(concatenated_vars))
            elif key == 'target_ERA5_images':
                y_surf = torch.from_numpy(surface_vars)
                y = torch.from_numpy(np.hstack([np.expand_dims(x, axis=1) for x in concatenated_vars]))
                return_dict['y1_surf'] = y_surf[0]
                return_dict['y2_surf'] = y_surf[1]
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


class ERA5Dataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        filename: str = '/glade/derecho/scratch/wchapman/STAGING/All_2010_staged.zarr',
        history_len: int = 1,
        forecast_len: int = 2,
        transform: Optional[Callable] = None,
    ):
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.transform = transform
        self.total_seq_len = self.history_len + self.forecast_len
        self.data_array = get_forward_data(filename=filename)
        self.rng = np.random.default_rng(seed=torch.initial_seed())
    
    def __post_init__(self):
        #: Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def __len__(self):
        return len(self.data_array['time'])-2

    def __getitem__(self, index):
        
        datasel = self.data_array.isel(time=slice(index, index+self.forecast_len+1)).load()
        
        sample = Sample(
            historical_ERA5_images=datasel.isel(time=slice(0, self.history_len)),
            target_ERA5_images=datasel.isel(time=slice(self.history_len, len(datasel['time']))),
            datetime_index=datasel.time.values.astype('datetime64[s]').astype(int)
        )
    
        if self.transform:
            sample = self.transform(sample)
        return sample
