#!/bin/bash
source /etc/profile

cd [CREATE_DIR]


$DRONA_RUNTIME_DIR/driver_scripts/drona_wf_driver_sbatch generic-job.slurm

