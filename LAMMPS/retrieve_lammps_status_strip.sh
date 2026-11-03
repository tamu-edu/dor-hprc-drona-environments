#!/bin/bash
# One-line status strip (state, elapsed/limit, step, steps/s, last thermo values, warnings/errors).
python3 lammps_run_report.py "$JOBDIR" "$JOBID" --status-line
