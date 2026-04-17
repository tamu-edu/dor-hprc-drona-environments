#!/bin/bash
# Given $LOCATION (job output directory), reads slurm_jobids.txt and queries
# SLURM to determine the overall status: PENDING, RUNNING, DONE, or FAILED.

JOBIDS_FILE="${LOCATION}/slurm_jobids.txt"

if [ ! -f "$JOBIDS_FILE" ]; then
    echo "PENDING"
    exit 0
fi

mapfile -t JOBIDS < <(grep -v '^\s*$' "$JOBIDS_FILE")

if [ ${#JOBIDS[@]} -eq 0 ]; then
    echo "PENDING"
    exit 0
fi

has_running=false
has_failed=false

for JID in "${JOBIDS[@]}"; do
    # Try squeue first (catches PENDING and RUNNING)
    STATUS=$(squeue -j "$JID" -h -o "%T" 2>/dev/null | head -1)

    if [ -z "$STATUS" ]; then
        # Job no longer in squeue — query sacct
        STATUS=$(sacct -j "$JID" --noheader --format=State --parsable2 2>/dev/null | head -1 | cut -d'|' -f1 | xargs)
    fi

    case "$STATUS" in
        PENDING)                                  echo "PENDING"; exit 0 ;;
        RUNNING)                                  has_running=true ;;
        FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
                                                  has_failed=true ;;
        COMPLETED)                                : ;;   # continue checking others
    esac
done

if $has_failed; then
    echo "FAILED"
elif $has_running; then
    echo "RUNNING"
else
    echo "DONE"
fi
