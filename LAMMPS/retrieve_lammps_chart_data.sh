#!/bin/bash
# Thermo rows from out.$JOBID as a JSON array for the "chart" element.
# MAX_POINTS: downsample to at most N rows (default 400; zoom after DONE uses 20000; 0 = all rows).
# FILTEROPTS: JSON list of keywords to emit; empty = all non-constant keywords.

ARGS=(--chart-data "$JOBDIR" "$JOBID" --max-points "${MAX_POINTS:-400}")
if [ -n "$FILTEROPTS" ] && [ "$FILTEROPTS" != "[]" ]; then
    ARGS+=(--filter "$FILTEROPTS")
fi
python3 lammps_monitor.py "${ARGS[@]}"
