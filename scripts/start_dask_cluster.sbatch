#!/bin/bash
#SBATCH --job-name=dask_job
#SBATCH -n 401
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=03:00:00
#SBATCH --partition=base

module load gcc12-env
module load miniconda3
 

# Set up dask cluster
srun --ntasks=1 -N1 --exclusive sh scripts/start_scheduler.sh &
echo "Waiting for scheduler to come up"
sleep 10  # allow for the scheduler to come up
echo "Starting workers"
# srun --ntasks=100 --exclusive sh scripts/start_worker.sh &
# iterate over the number of workers
for i in {1..400}
do
    srun --ntasks=1 --exclusive sh scripts/start_worker.sh &
done
echo "Waiting for workers to come up"

wait

# print resource infos
jobinfo 