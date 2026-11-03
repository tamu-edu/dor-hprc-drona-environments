#!/bin/bash
#SBATCH --job-name=[JOBNAME]
#SBATCH --time=[TIME] --mem=[MEM]
#SBATCH --nodes=1 --ntasks=[TASKS] --cpus-per-task=[CPUS]
#SBATCH --output=out.%j --error=error.%j
[PARTITION]
[EXTRA]

module purge
module load Matlab/[fversion]
cd [flocation]
[matlabpath]

# usage log shared with matlabsubmit (same format)
echo "$(date)   $USER   [fmainscript] workers=[MATLAB_WORKERS],threads=[MATLAB_THREADS],numnodes=1,mempnode=[MEM],walltime=[TIME]" >> "$MATLABROOT/../log/matlabsubmit.log" 2>/dev/null

[GPU_MONITOR_START]
[CPU_MONITOR_START]

# drona_matlab_wrapper.m sets the threads, starts a local pool when workers > 0, then runs the main script
export DRONA_MATLAB_MAIN=[fmainscript]
export DRONA_MATLAB_WORKERS=[MATLAB_WORKERS]
export DRONA_MATLAB_THREADS=[MATLAB_THREADS]
matlab -batch "drona_matlab_wrapper"
DRONA_EXIT=$?

[GPU_MONITOR_STOP]
[CPU_MONITOR_STOP]
exit $DRONA_EXIT
