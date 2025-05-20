#!/bin/bash
#SBATCH --job-name=AMOCRecons
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH --time=1:00:00
#SBATCH --partition=base

export OMP_NUM_THREADS=16

module load gcc12-env
module load miniconda3

conda activate amoc_recons

python $WORK/geomar/rapid-geostrophic-reconstruction/train.py --n_compartments 29 --n_embedding 48 --epochs 400 