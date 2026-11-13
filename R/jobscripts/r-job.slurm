#!/bin/bash
#SBATCH --job-name=[JOBNAME]
#SBATCH --time=[TIME] --mem=[MEM]
#SBATCH --nodes=[NODES] --ntasks=[TASKS] --cpus-per-task=[CPUS]
#SBATCH --output=out.%j --error=error.%j
[PARTITION]
[EXTRA]
[RSBATCH]

module purge
module load WebProxy
module load [LOADR]
[RLIBS]

# parallel workers selected in Drona; size your cluster with as.integer(Sys.getenv("DRONA_R_WORKERS"))
export DRONA_R_WORKERS=[R_WORKERS]
export DRONA_R_PARALLEL=[R_PARALLEL]
# default core count for mclapply/mcmapply (parallel package): the workers, or 1 without workers
export MC_CORES=$(( DRONA_R_WORKERS > 0 ? DRONA_R_WORKERS : 1 ))
# threads per R process for the math library (BLAS) and OpenMP; more threads than cores slows R down
export DRONA_R_THREADS=[R_THREADS]
export OPENBLAS_NUM_THREADS=$DRONA_R_THREADS OMP_NUM_THREADS=$DRONA_R_THREADS MKL_NUM_THREADS=$DRONA_R_THREADS

[CPU_MONITOR_START]
# R worker monitor for the R Run Check on the Manage page. DO NOT REMOVE.
srun --jobid="$SLURM_JOB_ID" --overlap --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --gpus-per-task=0 \
     python3 .drona/drona_r_monitor.py 15 & DRONA_R_MONITOR_PID=$!

[RRUN]
DRONA_EXIT=$?

[CPU_MONITOR_STOP]
kill "$DRONA_R_MONITOR_PID" 2>/dev/null  # Stop R worker monitor. DO NOT REMOVE.
exit $DRONA_EXIT
