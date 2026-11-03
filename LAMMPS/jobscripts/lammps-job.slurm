#!/bin/bash
#SBATCH --job-name=[JOBNAME]
#SBATCH --time=[TIME] --mem=[MEM]
#SBATCH --ntasks=[TASKS] --nodes=[NODES] --cpus-per-task=[CPUS]
#SBATCH --output=out.%j --error=error.%j
[PARTITION]
[EXTRA]

# The Manage thermo plots read out.<jobid> in this (the submit) directory
DRONA_SUBMIT_DIR=$PWD

[GPU_MONITOR_START]
[CPU_MONITOR_START]

# LAMMPS input file
INFILE=[inputfile]
# Choose a directory for LAMMPS read/dump
cd [workdir]

# LAMMPS software
module purge
[setup]

# [summary]
[lmp_cmd]
DRONA_EXIT=$?

cd "$DRONA_SUBMIT_DIR"
[GPU_MONITOR_STOP]
[CPU_MONITOR_STOP]
exit $DRONA_EXIT
