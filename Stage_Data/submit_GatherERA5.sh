xsx#!/bin/bash -l
# PBS -N ERA5_stage
# PBS -A NAML0001
# PBS -l walltime=12:00:00
# PBS -o Nudge4.out
# PBS -e Nudge4.out
# PBS -q casper
# PBS -l select=1:ncpus=20:mem=200GB
# PBS -m a
# PBS -M wchapman@ucar.edu

# ######qsub -I -q casper -A P54048000 -l walltime=12:00:00 -l select=3:ncpus=3:mem=30GB ./submit_GatherERA5.sh

module purge                               
module load conda 
conda activate /glade/u/apps/opt/conda/envs/npl-2023b
python ./GatherERA5_DistributedDask.py > Gather.run.out
