import xarray as xr
import glob
import os
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
import time
from dask import delayed
from dask import delayed, persist
import dask
import dask.array as da
import xarray as xr
from dask import delayed
from dask.diagnostics import ProgressBar
import re

#### settings !!! MODIFY THIS BLOCK
start_date = '2010-01-01'
end_date = '2011-01-02' #make sure this date is after the start date... 
interval_hours = 1 #what hour interval would you like to get? [i.e: 1 = 24 files/day, 6 = 4 files/day]
FPout = '/glade/derecho/scratch/wchapman/STAGING/' #where do you want the files stored?
prefix_out = 'ERA5_e5.oper.ml.v3' #what prefix do you want the files stored with?
project_num = 'NAML0001'
print('try this')
#### settings !!! MODIFY THIS BLOCK

if 'client' in locals():
    client.shutdown()
    print('...shutdown client...')
else:
    print('client does not exist yet')

###dask NCAR client: 
from distributed import Client
from dask_jobqueue import PBSCluster

cluster = PBSCluster(project='NAML0001',walltime='06:00:00',cores=1, memory='70GB',shared_temp_directory='/glade/scratch/wchapman/tmp',queue='casper')
cluster.scale(jobs=40)
client = Client(cluster)
#client

def find_staged_files(start_date,end_date):
    date_range_daily=pd.date_range(start_date,end_date)
    files_=[]
    for dtdt in date_range_daily:
        d_file = FPout+prefix_out+str(dtdt)[:10].replace('-','')+'.nc'
        files_.append(d_file)

        if not os.path.exists(d_file):
            raise FileNotFoundError(f"File not found: {d_file}")   
    return files_


if __name__ == '__main__':
    
    
    files_ = find_staged_files(start_date,end_date)
    
    DS = xr.open_mfdataset(files_,parallel=True)
    print('opened')
    DS = DS.chunk({'time':10})
    print('chunked')
    print('send to zarr')
    yrz = start_date[:4]
    DS.to_zarr(FPout+'All_'+yrz+'_staged.zarr')
    print('finished')
    
    if 'client' in locals():
        client.shutdown()
        print('...shutdown client...')
    else:
        print('client does not exist yet')     
