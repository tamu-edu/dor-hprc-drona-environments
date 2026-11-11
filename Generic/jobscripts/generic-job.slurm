#!/bin/bash
#SBATCH --job-name=[JOBNAME]
#SBATCH --time=[TIME] --mem=[MEM]
#SBATCH --ntasks=[TASKS] --nodes=[NODES] --cpus-per-task=[CPUS]
#SBATCH --output=out.%j --error=error.%j
[PARTITION]
[EXTRA]

module purge
module load WebProxy [MODULES]
[GPU_MONITOR_START]
[CPU_MONITOR_START]
# ADD YOUR COMMANDS BELOW

[IMPORTSCRIPT]

[GPU_MONITOR_STOP]
[CPU_MONITOR_STOP]
