
module load gcc12-env
module load miniconda3

conda activate amoc_recons

echo $HOSTNAME


dask scheduler --scheduler-file scheduler.json