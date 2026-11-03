#!/bin/bash
#SBATCH --job-name=[JOBNAME]
#SBATCH --time=[TIME] --mem=[MEM]
#SBATCH --nodes=[NODES] --ntasks=[TASKS] --ntasks-per-node=1 --cpus-per-task=[CPUS]
#SBATCH --output=out.%j --error=error.%j
[PARTITION]
[EXTRA]

module purge
[MODULES]
cd [flocation]
[CPU_MONITOR_START]

# tamulauncher starts one launcher per node; each runs [CPN] commands at a time
# with [CPUPT] thread(s) per command. Progress is kept in .tamulauncher-log/ so a
# rerun picks up where the previous run stopped (unless --norestart is set).
export OMP_NUM_THREADS=[CPUPT]

tamulauncher --commands-pernode [CPN] [NORESTART] [COMMANDFILE]
DRONA_EXIT=$?

[CPU_MONITOR_STOP]
exit $DRONA_EXIT
