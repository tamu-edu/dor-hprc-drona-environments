#!/bin/bash
#SBATCH --job-name=[JOBNAME]
#SBATCH --time=[TIME] --mem=[MEM]
#SBATCH --ntasks=[TASKS] --nodes=[NODES] --ntasks-per-node=[TASKS_PER_NODE] --cpus-per-task=[CPUS]
#SBATCH --output=out.%j --error=error.%j
[PARTITION]
[EXTRA]

module --force purge
module load WebProxy

export PYTHONNOUSERSITE=1

[setupEnv]

cd [flocation]
[GPU_MONITOR_START]
[CPU_MONITOR_START]
[tbLaunch]

# one training process per task (per GPU); Lightning reads the ranks from Slurm
srun --cpus-per-task=$SLURM_CPUS_PER_TASK python3 train.py

[GPU_MONITOR_STOP]
[CPU_MONITOR_STOP]
