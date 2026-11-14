#!/bin/bash
# Abaqus Progress card on the Manage page: progress from <job>.sta, errors and warnings (see abaqus_report.py).
python3 "$(dirname "$0")/abaqus_report.py" "$JOBDIR" "$JOBID" "$STATUS"
