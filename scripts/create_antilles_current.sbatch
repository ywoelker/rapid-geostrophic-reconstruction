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

for cycle_number in {3..6} 
do
    python data_scripts/extract_antilles_current.py --dask_url ${dask_url} --cycle_number ${cycle_number};
done