#!/bin/bash
source /etc/profile

[PRECHECK]

# Submit the pipeline with the shared driver: each job file waits for the previous one
# (afterok), and the job IDs are recorded in the workflow's database record.
$DRONA_RUNTIME_DIR/schedulers/slurm/drona_wf_driver_sbatch [JOBFILES]
