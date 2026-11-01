#!/bin/bash
source /etc/profile

cd [flocation]

$DRONA_RUNTIME_DIR/driver_scripts/drona_wf_driver_sbatch lightning-job.slurm
