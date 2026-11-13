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
    local ELAPSED=""
    local ROW
    ROW=$(squeue -j "$LATEST_JID" -h -o "%T|%M" 2>/dev/null | head -1)
    ROW="${ROW//$'\r'/}"
    ROW="$(echo "$ROW" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    if [ -n "$ROW" ]; then
        STATUS=$(echo "$ROW" | cut -d'|' -f1)
        ELAPSED=$(echo "$ROW" | cut -d'|' -f2)
    else
        local RAW_SACCT
        RAW_SACCT=$(sacct -j "$LATEST_JID" --noheader -X -P --format=State,Elapsed 2>/dev/null | head -1)
        RAW_SACCT="${RAW_SACCT//$'\r'/}"
        RAW_SACCT="$(echo "$RAW_SACCT" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        if [ -n "$RAW_SACCT" ]; then
            STATUS=$(echo "$RAW_SACCT" | cut -d'|' -f1)
            ELAPSED=$(echo "$RAW_SACCT" | cut -d'|' -f2)
        fi
    fi

    if [ "$STATUS" = "PENDING" ]; then
        echo "<div class='pt-epoch-section' style='margin-top:14px'>"
        echo "  <div style='font-size:0.9em;font-weight:600;color:#495057;margin-bottom:8px'>Training Progress</div>"
        echo "  <div class='card' style='border: 1px solid #dee2e6; border-radius: 8px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.05); overflow: hidden;'>"
        echo "    <div class='card-body' style='padding: 15px; display: flex; flex-direction: column; gap: 8px;'>"
        echo "      <div style='display:flex; justify-content:space-between; align-items:center; font-size:0.9em; font-weight:600;'>"
        echo "        <span>Status: <span style='color:#ea580c;'>Pending for running</span></span>"
        echo "        <span style='font-size:0.85em; font-weight:normal; color:#6c757d;'>Time left: <span style='font-weight:600; color:#495057;'>—</span></span>"
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
    local parsed_max=""
    if [ -f "$LOCATION/config.yaml" ]; then
        # settings file (newer workflows): "  max_epochs: 3"
        parsed_max=$(grep -E '^\s+max_epochs:\s*[0-9]+' "$LOCATION/config.yaml" | grep -o -E '[0-9]+' | head -n 1)
    fi
    if [ -z "$parsed_max" ] && [ -f "$LOCATION/train.py" ]; then
        # older workflows: "MAX_EPOCHS = 3" in train.py
        parsed_max=$(grep -E '^\s*MAX_EPOCHS\s*=\s*[0-9]' "$LOCATION/train.py" | grep -o -E '[0-9]+' | head -n 1)
    fi
    if [ -n "$parsed_max" ]; then
        max_epochs="$parsed_max"
    fi

    local PARSED
    PARSED=$(python3 - <<'PY' "$LATEST_LOG" "$max_epochs" "$STATUS" "$ELAPSED"
import sys
import os
import re

log_path = sys.argv[1]
max_epochs = int(sys.argv[2])
status = sys.argv[3].upper()
elapsed_str = sys.argv[4] if len(sys.argv) > 4 else ""

def parse_elapsed(s):
    s = (s or "").strip()
    if not s or s in ("—", "-", "Unknown", "INVALID"):
        return None
    days = 0
    if "-" in s:
        day_part, s = s.split("-", 1)
        try:
            days = int(day_part)
        except ValueError:
            return None
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = 0, parts[0], parts[1]
    elif len(parts) == 1:
        h, m, sec = 0, 0, parts[0]
    else:
        return None
    return days * 86400 + h * 3600 + m * 60 + sec

def format_elapsed(total):
    if total < 0:
        total = 0
    days, rem = divmod(int(total), 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    if days > 0:
        return f"{days}-{h:02d}:{m:02d}:{sec:02d}"
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"

def estimate_eta(frac):
    terminal_other = status.startswith("CANCELLED") or status in (
        "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL",
    )
    if status in ("COMPLETED", "COMPLETE") or (frac is not None and frac >= 0.999):
        return "0:00"
    if status in ("PENDING", "CONFIGURING") or terminal_other or not status:
        return "—"
    elapsed_s = parse_elapsed(elapsed_str)
    if frac is None or elapsed_s is None or elapsed_s <= 0:
        return "—"
    if frac < 0.01:
        return "—"
    remaining_s = int(round(elapsed_s * (1.0 - frac) / frac))
    return format_elapsed(remaining_s)

if status in ("COMPLETED", "COMPLETE"):
    print(f"{max_epochs}|{max_epochs}|100|100|—|0:00")
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

frac = (curr_ep + (ep_percent / 100.0)) / float(max_epochs)
if status in ("COMPLETED", "COMPLETE") or (display_epoch == max_epochs and ep_percent == 100):
    frac = 1.0
eta = estimate_eta(frac)

print(f"{display_epoch}|{max_epochs}|{ep_percent}|{overall_percent}|{speed}|{eta}")
PY
)

    local display_epoch max_ep ep_percent overall_percent speed eta
    display_epoch=$(echo "$PARSED" | cut -d'|' -f1)
    max_ep=$(echo "$PARSED" | cut -d'|' -f2)
    ep_percent=$(echo "$PARSED" | cut -d'|' -f3)
    overall_percent=$(echo "$PARSED" | cut -d'|' -f4)
    speed=$(echo "$PARSED" | cut -d'|' -f5)
    eta=$(echo "$PARSED" | cut -d'|' -f6)
    if [ -z "$eta" ]; then
        eta="—"
    fi

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
    echo "        <span style='font-size:0.85em; font-weight:normal; color:#6c757d;' title='Estimated time remaining based on training progress'>Time left: <span style='font-weight:600; color:#495057;'>${eta}</span></span>"
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
