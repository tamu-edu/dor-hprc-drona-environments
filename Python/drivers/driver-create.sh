#!/bin/bash
source /etc/profile

cd [CREATE_DIR]


$DRONA_RUNTIME_DIR/schedulers/slurm/drona_wf_driver_sbatch python-job.slurm
