#!/bin/bash
# R Run Check card on the Manage page: worker check and R problems in error.<jobid> (see r_report.py).
python3 "$(dirname "$0")/r_report.py" "$JOBDIR" "$JOBID" "$STATUS"
