#!/bin/bash
#SBATCH --job-name=VirtualArgos
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH --time=8:00:00
#SBATCH --partition=base

export OMP_NUM_THREADS=16

module load gcc12-env
module load miniconda3

conda activate amoc_recons



dask_url='tcp://10.0.4.100:8786'

python data_scripts/extract_argos.py --deep_argo True --dask_url ${dask_url} --cycle_number 1 --smoothing_days 90;

python data_scripts/extract_argos.py --deep_argo True --dask_url ${dask_url} --cycle_number 2 --smoothing_days 90;

python data_scripts/extract_argos.py --deep_argo True --dask_url ${dask_url} --cycle_number 3 --smoothing_days 90;

python data_scripts/extract_argos.py --deep_argo True --dask_url ${dask_url} --cycle_number 4 --smoothing_days 90;

python data_scripts/extract_argos.py --deep_argo True --dask_url ${dask_url} --cycle_number 5 --smoothing_days 90;

python data_scripts/extract_argos.py --deep_argo True --dask_url ${dask_url} --cycle_number 6 --smoothing_days 90;