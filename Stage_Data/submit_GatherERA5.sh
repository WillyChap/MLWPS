#!/bin/bash -l
# PBS -N ERA5_stage
# PBS -A NAML0001
# PBS -l walltime=12:00:00
# PBS -o Nudge4.out
# PBS -e Nudge4.out
# PBS -q casper
# PBS -l select=1:ncpus=20:mem=200GB
# PBS -m a
# PBS -M wchapman@ucar.edu

# ######



qsub -I -q casper -A P54048000 -l walltime=12:00:00 -l select=3:ncpus=3:mem=30GB
conda activate MLWPS
python ./GatherERA5_DistributedDask.py > Gather.run.out

python GatherERA5_DistributedDask_toZarr_dev.py --start_date 2010-01-01 --end_date 2011-01-01
python GatherERA5_DistributedDask_toZarr_dev.py --start_date 2013-01-01 --end_date 2014-01-01
python GatherERA5_DistributedDask_toZarr_dev.py --start_date 2014-01-01 --end_date 2015-01-01




python GatherERA5_DistributedDask_toZarr.py --start_date 2012-01-01 --end_date 2013-01-01
python GatherERA5_DistributedDask_toZarr.py --start_date 2013-01-01 --end_date 2014-01-01
python GatherERA5_DistributedDask_toZarr.py --start_date 2014-01-01 --end_date 2015-01-01
python GatherERA5_DistributedDask_toZarr.py --start_date 2015-01-01 --end_date 2016-01-01
python GatherERA5_DistributedDask_toZarr.py --start_date 2016-01-01 --end_date 2017-01-01
python GatherERA5_DistributedDask_toZarr.py --start_date 2017-01-01 --end_date 2018-01-01
python GatherERA5_DistributedDask_toZarr.py --start_date 2018-01-01 --end_date 2019-01-01
python GatherERA5_DistributedDask_toZarr.py --start_date 2019-01-01 --end_date 2020-01-01
