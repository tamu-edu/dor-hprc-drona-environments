#!/bin/bash
# Training progress (epoch, speed, percent) for the selected PyTorch-Lightning job.
# Split out of retrieve_pt_monitor_dashboard.sh so it can be its own element.


export LOCATION="${LOCATION:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

html_escape() {
    sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g'
}

emit_epoch_progress() {
    if [ -z "$LOCATION" ] || [ ! -d "$LOCATION" ]; then
        return 0
    fi

    local JOBIDS_FILE="${LOCATION}/slurm_jobids.txt"
    local -a JOBIDS=()

    if [ -f "$JOBIDS_FILE" ]; then
        mapfile -t JOBIDS < <(grep -v '^\s*$' "$JOBIDS_FILE")
    fi

    if [ ${#JOBIDS[@]} -eq 0 ]; then
        shopt -s nullglob
        local f base
        for f in "$LOCATION"/out.*; do
            base="$(basename "$f")"
            if [[ "$base" =~ ^out\.([0-9]+)$ ]]; then
                JOBIDS+=("${BASH_REMATCH[1]}")
            fi
        done
        shopt -u nullglob
    fi

    if [ ${#JOBIDS[@]} -eq 0 ]; then
        return 0
    fi

    local LATEST_JID="${JOBIDS[-1]}"
    LATEST_JID="${LATEST_JID//$'\r'/}"
    LATEST_JID="$(echo "$LATEST_JID" | xargs)"
    
    local STATUS="unknown"
    local ROW
    ROW=$(squeue -j "$LATEST_JID" -h -o "%T" 2>/dev/null | head -1 | xargs)
    if [ -n "$ROW" ]; then
        STATUS="$ROW"
    else
        local RAW_SACCT
        RAW_SACCT=$(sacct -j "$LATEST_JID" --noheader --format=State 2>/dev/null | head -1 | xargs)
        if [ -n "$RAW_SACCT" ]; then
            STATUS="$RAW_SACCT"
        fi
    fi

    if [ "$STATUS" = "PENDING" ]; then
        echo "<div class='pt-epoch-section' style='margin-top:14px'>"
        echo "  <div style='font-size:0.9em;font-weight:600;color:#495057;margin-bottom:8px'>Training Progress</div>"
        echo "  <div class='card' style='border: 1px solid #dee2e6; border-radius: 8px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.05); overflow: hidden;'>"
        echo "    <div class='card-body' style='padding: 15px; display: flex; flex-direction: column; gap: 8px;'>"
        echo "      <div style='display:flex; justify-content:space-between; align-items:center; font-size:0.9em; font-weight:600;'>"
        echo "        <span>Status: <span style='color:#ea580c;'>Pending for running</span></span>"
        echo "        <span style='color:#868e96;'>0%</span>"
        echo "      </div>"
        echo "      <div style='background: #e9ecef; border-radius: 4px; height: 10px; overflow: hidden;'>"
        echo "        <div style='background: #cbd5e1; width: 0%; height: 100%;'></div>"
        echo "      </div>"
        echo "    </div>"
        echo "  </div>"
        echo "</div>"
        return 0
    fi

    local LATEST_LOG=""
    if [ -f "$LOCATION/out.$LATEST_JID" ]; then
        LATEST_LOG="$LOCATION/out.$LATEST_JID"
    fi
    if [ -z "$LATEST_LOG" ]; then
        LATEST_LOG=$(ls -t "$LOCATION"/out.* 2>/dev/null | head -n 1)
    fi

    if [ -z "$LATEST_LOG" ] || [ ! -f "$LATEST_LOG" ]; then
        return 0
    fi

    local max_epochs="10"
    if [ -f "$LOCATION/train.py" ]; then
        local parsed_max
        parsed_max=$(grep -E '^\s*MAX_EPOCHS\s*=' "$LOCATION/train.py" | grep -o -E '[0-9]+' | head -n 1)
        if [ -n "$parsed_max" ]; then
            max_epochs="$parsed_max"
        fi
    fi

    local PARSED
    PARSED=$(python3 - <<EOF "$LATEST_LOG" "$max_epochs" "$STATUS"
import sys
import os
import re

log_path = sys.argv[1]
max_epochs = int(sys.argv[2])
status = sys.argv[3].upper()

if status in ("COMPLETED", "COMPLETE"):
    print(f"{max_epochs}|{max_epochs}|100|100|—")
    sys.exit(0)

try:
    with open(log_path, "rb") as f:
        try:
            f.seek(-102400, os.SEEK_END)
        except OSError:
            f.seek(0)
        content = f.read().decode("utf-8", errors="ignore")
except Exception:
    content = ""

matches = re.findall(r'Epoch\s+(\d+)(?:/\d+)?:?\s*(\d+)%', content)

curr_ep = 0
ep_percent = 0

if matches:
    curr_ep_str, ep_percent_str = matches[-1]
    try:
        curr_ep = int(curr_ep_str)
        ep_percent = int(ep_percent_str)
    except ValueError:
        pass
else:
    simple_epochs = re.findall(r'Epoch\s+(\d+)', content)
    if simple_epochs:
        try:
            curr_ep = int(simple_epochs[-1])
        except ValueError:
            pass

display_epoch = curr_ep + 1
if display_epoch > max_epochs:
    display_epoch = max_epochs

if status in ("COMPLETED", "COMPLETE") or display_epoch == max_epochs and ep_percent == 100:
    overall_percent = 100
    display_epoch = max_epochs
    ep_percent = 100
else:
    overall_percent = int(((curr_ep + (ep_percent / 100.0)) / max_epochs) * 100)

speed_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(it/s|s/it)', content)
if speed_matches:
    val, unit = speed_matches[-1]
    speed = f"{val} {unit}"
else:
    speed = "—"

print(f"{display_epoch}|{max_epochs}|{ep_percent}|{overall_percent}|{speed}")
EOF
)

    local display_epoch max_ep ep_percent overall_percent speed
    display_epoch=$(echo "$PARSED" | cut -d'|' -f1)
    max_ep=$(echo "$PARSED" | cut -d'|' -f2)
    ep_percent=$(echo "$PARSED" | cut -d'|' -f3)
    overall_percent=$(echo "$PARSED" | cut -d'|' -f4)
    speed=$(echo "$PARSED" | cut -d'|' -f5)

    if [ -z "$display_epoch" ] || [ -z "$max_ep" ]; then
        return 0
    fi

    echo "<div class='pt-epoch-section' style='margin-top:14px'>"
    echo "  <div style='font-size:0.9em;font-weight:600;color:#495057;margin-bottom:8px'>Training Progress</div>"
    echo "  <div class='card' style='border: 1px solid #dee2e6; border-radius: 8px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.05); overflow: hidden;'>"
    echo "    <div class='card-body' style='padding: 15px; display: flex; flex-direction: column; gap: 8px;'>"
    
    echo "      <div style='display:flex; justify-content:space-between; align-items:center; font-size:0.9em; font-weight:600;'>"
    echo "        <span>Epoch: <span style='color:#0d6efd;'>${display_epoch}</span> / ${max_ep}</span>"
    if [ -n "$speed" ] && [ "$speed" != "—" ]; then
        echo "        <span style='font-size:0.85em; font-weight:normal; color:#6c757d;'>Speed: <span style='font-weight:600; color:#495057;'>${speed}</span></span>"
    fi
    echo "        <span style='color:#198754;'>${ep_percent}%</span>"
    echo "      </div>"

    echo "      <div style='background: #e9ecef; border-radius: 4px; height: 10px; overflow: hidden;'>"
    echo "        <div style='background: #198754; width: ${ep_percent}%; height: 100%; transition: width 0.4s ease;'></div>"
    echo "      </div>"

    echo "    </div>"
    echo "  </div>"
    echo "</div>"
}

emit_epoch_progress
