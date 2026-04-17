#!/bin/bash
# Given $LOCATION, emits an HTML table summarising the SLURM jobs for this run.

JOBIDS_FILE="${LOCATION}/slurm_jobids.txt"

if [ ! -f "$JOBIDS_FILE" ]; then
    echo "<div style='font-size:0.83em;color:#6c757d;padding:4px 2px'>Waiting for SLURM job IDs — driver script may still be starting.</div>"
    exit 0
fi

mapfile -t JOBIDS < <(grep -v '^\s*$' "$JOBIDS_FILE")

if [ ${#JOBIDS[@]} -eq 0 ]; then
    echo "<div style='font-size:0.83em;color:#6c757d;padding:4px 2px'>No SLURM job IDs recorded yet.</div>"
    exit 0
fi

rows=""
has_running=false
has_failed=false
has_pending=false
all_done=true

for JID in "${JOBIDS[@]}"; do
    # Try squeue (active jobs)
    ROW=$(squeue -j "$JID" -h -o "%i|%j|%T|%M|%N" 2>/dev/null | head -1)

    if [ -n "$ROW" ]; then
        STATUS=$(echo "$ROW" | cut -d'|' -f3)
        all_done=false
        case "$STATUS" in
            RUNNING) ROW_CLASS="table-success"; has_running=true ;;
            PENDING) ROW_CLASS="table-warning"; has_pending=true ;;
            *)       ROW_CLASS="" ;;
        esac
    else
        # Fall back to sacct (completed jobs)
        ROW=$(sacct -j "$JID" --noheader --format=JobID,JobName,State,Elapsed,NodeList \
              --parsable2 2>/dev/null | head -1)
        STATUS=$(echo "$ROW" | cut -d'|' -f3 | xargs)
        case "$STATUS" in
            COMPLETED)   ROW_CLASS="table-success" ;;
            FAILED|CANCELLED*|TIMEOUT|NODE_FAIL)
                         ROW_CLASS="table-danger"; has_failed=true ;;
            *)           ROW_CLASS=""; all_done=false ;;
        esac
    fi

    C1=$(echo "$ROW" | cut -d'|' -f1)
    C2=$(echo "$ROW" | cut -d'|' -f2)
    C3=$(echo "$ROW" | cut -d'|' -f3)
    C4=$(echo "$ROW" | cut -d'|' -f4)
    C5=$(echo "$ROW" | cut -d'|' -f5)

    rows+="<tr class='$ROW_CLASS'><td>${C1:-$JID}</td><td>${C2:-—}</td><td>${C3:-unknown}</td><td>${C4:-—}</td><td>${C5:-—}</td></tr>"
done

# Compact overall status badge
if $has_failed; then
    dot_color="#dc3545"
    status_label="Failed &mdash; one or more jobs did not complete. Check logs below."
elif $has_running; then
    dot_color="#0d6efd"
    status_label="Running &mdash; logs update every 30 seconds."
elif $has_pending; then
    dot_color="#ffc107"
    status_label="Pending &mdash; queued, waiting to start."
else
    dot_color="#198754"
    status_label="Completed &mdash; ranked structures shown below."
fi

echo "<div style='display:flex;align-items:center;gap:7px;margin-bottom:8px;font-size:0.83em;color:#495057'>"
echo "  <span style='width:8px;height:8px;border-radius:50%;background:${dot_color};flex-shrink:0;display:inline-block'></span>"
echo "  ${status_label}"
echo "</div>"

echo "<table class='table table-sm table-bordered mb-0' style='font-size:0.85em'>"
echo "<thead class='table-light'><tr><th>Job ID</th><th>Name</th><th>Status</th><th>Elapsed</th><th>Node(s)</th></tr></thead>"
echo "<tbody>${rows}</tbody></table>"
