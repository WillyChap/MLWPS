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

qsub -I -q main -A P54048000 -l walltime=12:00:00 -l select=3:ncpus=3:mem=30GB


qsub -I -q casper -A P54048000 -l walltime=12:00:00 -l select=3:ncpus=3:mem=30GB
conda activate MLWPS
python ./GatherERA5_DistributedDask.py > Gather.run.out

python GatherERA5_DistributedDask_toZarr_dev.py --start_date 2011-01-01 --end_date 2012-01-01
python GatherERA5_DistributedDask_toZarr_dev.py --start_date 2014-01-01 --end_date 2015-01-01
python GatherERA5_DistributedDask_toZarr_dev.py --start_date 2015-01-01 --end_date 2016-01-01

conda activate MLWPS
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2004-01-01 --end_date 2005-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2009-01-01 --end_date 2010-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2017-01-01 --end_date 2018-01-01

qsub -I -q main -A P54048000 -l walltime=12:00:00 -l select=3:ncpus=3:mem=30GB
conda activate MLWPS
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1996-01-01 --end_date 1997-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1995-01-01 --end_date 1996-01-01

python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1994-01-01 --end_date 1995-01-01

python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1993-01-01 --end_date 1994-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1992-01-01 --end_date 1993-01-01

qsub -I -q main -A P54048000 -l walltime=12:00:00 -l select=3:ncpus=3:mem=30GB
conda activate MLWPS
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1991-01-01 --end_date 1992-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1994-01-01 --end_date 1995-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1986-01-01 --end_date 1987-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1985-01-01 --end_date 1986-01-01


qsub -I -q main -A P54048000 -l walltime=12:00:00 -l select=3:ncpus=3:mem=30GB
conda activate MLWPS
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1979-01-01 --end_date 1980-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1980-01-01 --end_date 1981-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1981-01-01 --end_date 1982-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1982-01-01 --end_date 1983-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1983-01-01 --end_date 1984-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1984-01-01 --end_date 1985-01-01


python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1990-01-01 --end_date 1991-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1989-01-01 --end_date 1990-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1988-01-01 --end_date 1989-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 1987-01-01 --end_date 1988-01-01




python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2000-01-01 --end_date 2001-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2001-01-01 --end_date 2002-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2002-01-01 --end_date 2003-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2003-01-01 --end_date 2004-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2004-01-01 --end_date 2005-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2005-01-01 --end_date 2006-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2006-01-01 --end_date 2007-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2007-01-01 --end_date 2008-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2008-01-01 --end_date 2009-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2009-01-01 --end_date 2010-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2010-01-01 --end_date 2011-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2011-01-01 --end_date 2012-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2012-01-01 --end_date 2013-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2013-01-01 --end_date 2014-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2014-01-01 --end_date 2015-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2015-01-01 --end_date 2016-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2016-01-01 --end_date 2017-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2017-01-01 --end_date 2018-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2018-01-01 --end_date 2019-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2019-01-01 --end_date 2020-01-01

python GatherERA5_DistributedDask_toZarr_dev.py --start_date 2017-01-01 --end_date 2018-01-01
python GatherERA5_DistributedDask_toZarr_dev.py --start_date 2018-01-01 --end_date 2019-01-01
python GatherERA5_DistributedDask_toZarr_dev.py --start_date 2019-01-01 --end_date 2020-01-01

python GatherERA5_DistributedDask_toZarr_dev.py --start_date 2009-01-01 --end_date 2010-01-01

python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2003-01-01 --end_date 2004-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2005-01-01 --end_date 2006-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2007-01-01 --end_date 2008-01-01

python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2004-01-01 --end_date 2005-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2006-01-01 --end_date 2007-01-01
python GatherERA5_DistributedDask_toZarr_derecho_dev.py --start_date 2008-01-01 --end_date 2009-01-01


python GatherERA5_DistributedDask_toZarr.py --start_date 2012-01-01 --end_date 2013-01-01
python GatherERA5_DistributedDask_toZarr.py --start_date 2013-01-01 --end_date 2014-01-01
python GatherERA5_DistributedDask_toZarr.py --start_date 2014-01-01 --end_date 2015-01-01
python GatherERA5_DistributedDask_toZarr.py --start_date 2015-01-01 --end_date 2016-01-01
python GatherERA5_DistributedDask_toZarr.py --start_date 2016-01-01 --end_date 2017-01-01
python GatherERA5_DistributedDask_toZarr.py --start_date 2017-01-01 --end_date 2018-01-01
python GatherERA5_DistributedDask_toZarr.py --start_date 2018-01-01 --end_date 2019-01-01
python GatherERA5_DistributedDask_toZarr.py --start_date 2019-01-01 --end_date 2020-01-01
