#!/bin/bash
# Run report card (setup, performance, warnings & errors) from out.$JOBID / error.$JOBID.
python3 lammps_run_report.py "$JOBDIR" "$JOBID"
