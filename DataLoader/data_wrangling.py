# Python core
from typing import Optional, Callable, TypedDict, Union, Iterable, Tuple, NamedTuple, List
from dataclasses import dataclass
import datetime
from itertools import product
from concurrent import futures

# Scientific python
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# PyTorch
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl



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
        crop_params = None
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
                raise BadData(f'\n{attr_name} has negative values at {image.time.values}')
        return sample

class ToTensor():
    def __call__(self, sample: Sample) -> Sample:
        for key, value in sample.items():
            #print('the key is: ',key)
            
            if isinstance(value, xr.DataArray):
                #print('its an array')
                value_var = value.values
                
            elif isinstance(value, xr.Dataset):
                #print('its a dataset')
                concatenated_vars = []
                varsdo = ['U','V','T','Q','SP'] #hardcoded and bad !!!FIX THIS TO MAKE MORE GENERAL 
                for vv in varsdo: 
                    value_var = value[vv].values
                    if vv == 'SP':
                        value_var = np.expand_dims(value_var,axis=1)
                        concatenated_vars.append(value_var) 
                        #print('valvar.shape:',value_var.shape)
                    else:
                        #print('valvar.shape:',value_var.shape)
                        concatenated_vars.append(value_var)
                    
                value_var = np.concatenate(concatenated_vars, axis=1)
            else: 
                #print('its something else')
                value_var = value
                                    
            sample[key] = torch.from_numpy(value_var)
        return sample


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






@dataclass
class ERA5Dataset(torch.utils.data.IterableDataset):
    zarr_chunk_sequences: Iterable[Segment]  #: Defines multiple Zarr chunks to be loaded from disk at once.
    history_len: int = 1  #: The number of timesteps of 'history' to load.
    forecast_len: int = 2  #: The number of timesteps of 'forecast' to load.
    transform: Optional[Callable] = None
    n_disk_loads_per_epoch: int = 10_000  #: Number of disk loads per worker process per epoch.
    min_n_samples_per_disk_load: int = 1_000  #: Number of samples each worker will load for each disk load.
    max_n_samples_per_disk_load: int = 2_000  #: Max number of disk loader.  Actual number is chosen randomly between min & max.
    n_zarr_chunk_sequences_to_load_at_once: int = 8  #: Number of chunk seqs to load at once.  These are sampled at random.
    
    def __post_init__(self):
        #: Total sequence length of each sample.
        self.total_seq_len = self.history_len + self.forecast_len

    def per_worker_init(self, worker_id: int) -> None:
        """Called by worker_init_fn on each copy of SatelliteDataset after the worker process has been spawned."""
        self.worker_id = worker_id
        self.data_array = get_forward_data(filename=ZARR)
        # Each worker must have a different seed for its random number generator.
        # Otherwise all the workers will output exactly the same data!
        seed = torch.initial_seed()
        self.rng = np.random.default_rng(seed=seed)
    
    def __iter__(self):
        """
        Asynchronously loads next data from disk while sampling from data_in_mem.
        """
        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_data = executor.submit(self._load_data_from_disk)
            for _ in range(self.n_disk_loads_per_epoch):
                data_in_mem = future_data.result()
                future_data = executor.submit(self._load_data_from_disk)
                n_samples = self.rng.integers(self.min_n_samples_per_disk_load, self.max_n_samples_per_disk_load)
                for _ in range(n_samples):
                    sample = self._get_sample(data_in_mem)
                    if self.transform:
                        sample = self.transform(sample)
                        # try:
                        #     sample = self.transform(sample)
                        # except BadData as e:
                        #     print(e)
                        #     continue
                    yield sample

    def _load_data_from_disk(self) -> List[xr.DataArray]:
        """Loads data from contiguous Zarr chunks from disk into memory."""
        ERA5_images_list = []
        for _ in range(self.n_zarr_chunk_sequences_to_load_at_once):
            zarr_chunk_sequence = self.rng.choice(self.zarr_chunk_sequences)
            ERA5_images = self.data_array.isel(time=slice(*zarr_chunk_sequence))

            # Sanity checks
            n_timesteps_available = len(ERA5_images)
            if n_timesteps_available < self.total_seq_len:
                raise RuntimeError(f'Not enough timesteps in loaded data!  Need at least {self.total_seq_len}.  Got {n_timesteps_available}!')

            ERA5_images_list.append(ERA5_images.load())
        return ERA5_images_list

    def _get_sample(self, data_in_mem_list: List[xr.DataArray]) -> Sample:
        i = self.rng.integers(0, len(data_in_mem_list))
        data_in_mem = data_in_mem_list[i]
        n_timesteps_available = len(data_in_mem)
        max_start_idx = n_timesteps_available - self.total_seq_len
        start_idx = self.rng.integers(low=0, high=max_start_idx, dtype=np.uint32)
        end_idx = start_idx + self.total_seq_len
        #print('s e idx', start_idx, end_idx)
        #print('data_in_mem: ',data_in_mem)
        ERA5_images = data_in_mem.isel(time=slice(start_idx, end_idx))
        return Sample(
            historical_ERA5_images=ERA5_images.isel(time = slice(0,self.history_len)),
            target_ERA5_images=ERA5_images.isel(time = slice(self.history_len,len(ERA5_images['time']))),
            datetime_index=ERA5_images.time.values.astype('datetime64[s]').astype(int)
        )
    

def worker_init_fn(worker_id):
    """Configures each dataset worker process.
    
    Just has one job!  To call SatelliteDataset.per_worker_init().
    """
    # get_worker_info() returns information specific to each worker process.
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        print('worker_info is None!')
    else:
        dataset_obj = worker_info.dataset  # The Dataset copy in this worker process.
        dataset_obj.per_worker_init(worker_id=worker_info.id)


# torch.manual_seed(42)    
# dataset = ERA5Dataset(
#     zarr_chunk_sequences=zarr_chunk_sequences,
#     transform=transforms.Compose([
#         Reshape_Data(),
#         CheckForBadData(),
#         ToTensor(),
#     ]),
# )