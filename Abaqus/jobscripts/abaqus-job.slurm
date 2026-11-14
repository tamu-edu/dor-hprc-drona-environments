#!/bin/bash
#SBATCH --job-name=[JOBNAME]
#SBATCH --time=[TIME] --mem=[MEM]
#SBATCH --nodes=[NODES] --ntasks=[TASKS] --cpus-per-task=[CPUS]
#SBATCH --output=out.%j --error=error.%j
[PARTITION]
[EXTRA]

module purge
module load WebProxy
module load [LOADABAQUS]
[ABQ_COMPILER]

[GPU_MONITOR_START]
[CPU_MONITOR_START]
[ABQ_HOSTS]
[ABQ_INPUT_LINKS]
[ABQ_LINKS]
# Abaqus memory: what Slurm gave the job, less 10% for Abaqus itself. Abaqus' default is a share of
# the node's physical memory, which can be more than the job may use, and then Slurm kills the job.
ABQ_MEM_MB=$(( ${SLURM_MEM_PER_NODE:-4096} * 9 / 10 ))

abaqus job=[ABQJOB] input=[INPUT] cpus=[ABQ_CPUS] [ABQ_MPMODE][ABQ_GPUS]memory="${ABQ_MEM_MB} mb" scratch="${TMPDIR:-$PWD}" [ABQ_EXTRA][ABQ_OPTIONS]interactive
DRONA_EXIT=$?

[GPU_MONITOR_STOP]
[CPU_MONITOR_STOP]
exit $DRONA_EXIT
