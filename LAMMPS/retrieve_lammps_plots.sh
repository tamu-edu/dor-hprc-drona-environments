#!/bin/bash

if [ -n "$FILTEROPTS" ]; then
    python3 lammps_monitor.py --generate-plots "$JOBDIR" "$JOBID" --filter "$FILTEROPTS"
else
    python3 lammps_monitor.py --generate-plots "$JOBDIR" "$JOBID"
fi