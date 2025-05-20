
module load gcc12-env
module load miniconda3

echo "Module loaded"
conda activate amoc_recons

echo "Conda activated"

dask worker --scheduler-file scheduler.json --nworkers=1 --nthreads=1 --memory-limit=12G --local-directory ${TMPDIR}

echo "Worker started"