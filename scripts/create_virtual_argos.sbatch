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
smoothing_days=90

for cycle_number in {1..6} 
do
    python data_scripts/extract_argos.py --deep_argo True --dask_url ${dask_url} --cycle_number ${cycle_number} --smoothing_days ${smoothing_days};
done