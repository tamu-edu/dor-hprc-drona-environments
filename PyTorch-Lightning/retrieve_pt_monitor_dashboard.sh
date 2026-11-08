#!/bin/bash
# Combined monitor dashboard: Slurm summary + output/error logs + shared UI scripts.

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

LOCATION_ESC=$(echo "$LOCATION" | html_escape | sed 's/"/\&quot;/g')
echo "<div id='pt-monitor-dashboard' data-pt-location=\"$LOCATION_ESC\">"


# Slurm job listing now comes from the runtime_support slurmJobsTable element;
# keep an empty anchor because the refresh JS inserts the epoch section after it.
echo "<div class='pt-slurm-section' style='display:none'></div>"

emit_epoch_progress

# GPU utilization and output/error logs are provided by runtime_support elements.

echo "</div>"

cat << 'SCRIPT'
<script>
(function () {
  var COOLDOWN_MS = 5000;

  function updateDashboard(existing, newDash) {
    // 1. Update the slurm section
    var existingSlurm = existing.querySelector('.pt-slurm-section');
    var newSlurm = newDash.querySelector('.pt-slurm-section');
    if (existingSlurm && newSlurm) {
      existingSlurm.innerHTML = newSlurm.innerHTML;
    }

    // 1.2. Update the epoch progress section
    var existingEpoch = existing.querySelector('.pt-epoch-section');
    var newEpoch = newDash.querySelector('.pt-epoch-section');
    if (existingEpoch && newEpoch) {
      existingEpoch.innerHTML = newEpoch.innerHTML;
    } else if (newEpoch) {
      var slurmSec = existing.querySelector('.pt-slurm-section');
      if (slurmSec) {
        var div = document.createElement('div');
        div.className = 'pt-epoch-section';
        div.style.marginTop = '14px';
        div.innerHTML = newEpoch.innerHTML;
        slurmSec.parentNode.insertBefore(div, slurmSec.nextSibling);
      }
    } else if (existingEpoch) {
      existingEpoch.parentNode.removeChild(existingEpoch);
    }

    // 1.5. Update the gpu section
    // Skip GPU update for 30s after any TensorBoard interaction so the bars
    // don't reset to 0% while the TB-triggered refresh re-runs the srun query.
    var tbGuardActive = window.__tb_interaction_ts &&
      (Date.now() - window.__tb_interaction_ts < 30000);
    var existingGpu = existing.querySelector('.pt-gpu-section');
    var newGpu = newDash.querySelector('.pt-gpu-section');
    if (!tbGuardActive) {
      if (existingGpu && newGpu) {
        existingGpu.innerHTML = newGpu.innerHTML;
      } else if (newGpu) {
        var logsSec = existing.querySelector('.pt-logs-section');
        if (logsSec) {
          var div = document.createElement('div');
          div.className = 'pt-gpu-section';
          div.style.marginTop = '14px';
          div.innerHTML = newGpu.innerHTML;
          logsSec.parentNode.insertBefore(div, logsSec);
        } else {
          var div = document.createElement('div');
          div.className = 'pt-gpu-section';
          div.style.marginTop = '14px';
          div.innerHTML = newGpu.innerHTML;
          existing.appendChild(div);
        }
      } else if (existingGpu) {
        existingGpu.parentNode.removeChild(existingGpu);
      }
    }

    // 2. Update the logs section
    var existingLogsRoot = existing.querySelector('.pt-logs-root');
    var newLogsRoot = newDash.querySelector('.pt-logs-root');
    if (existingLogsRoot && newLogsRoot) {
      var newDetails = newLogsRoot.querySelectorAll('.pt-log-container');
      var existingDetails = existingLogsRoot.querySelectorAll('.pt-log-container');

      var existingMap = {};
      existingDetails.forEach(function (d) {
        var logfile = d.getAttribute('data-pt-logfile');
        if (logfile) {
          existingMap[logfile] = d;
        }
      });

      var presentKeys = {};
      newDetails.forEach(function (newD) {
        var logfile = newD.getAttribute('data-pt-logfile');
        if (!logfile) return;
        presentKeys[logfile] = true;

        var existingD = existingMap[logfile];
        if (existingD) {
          // Update header text
          var existingHeader = existingD.querySelector('.pt-log-header');
          var newHeader = newD.querySelector('.pt-log-header');
          if (existingHeader && newHeader) {
            if (existingHeader.innerHTML !== newHeader.innerHTML) {
              existingHeader.innerHTML = newHeader.innerHTML;
            }
          }
        } else {
          // New file stream, append it!
          var importedD = document.importNode(newD, true);
          existingLogsRoot.appendChild(importedD);
        }
      });

      // Remove removed file streams
      existingDetails.forEach(function (d) {
        var logfile = d.getAttribute('data-pt-logfile');
        if (logfile && !presentKeys[logfile]) {
          d.parentNode.removeChild(d);
        }
      });
    }

    // 3. Update cooldown hint
    var existingHint = existing.querySelector('#pt-refresh-cooldown-hint');
    var newHint = newDash.querySelector('#pt-refresh-cooldown-hint');
    if (existingHint && newHint) {
      existingHint.innerHTML = newHint.innerHTML;
    }
  }

  // Monkeypatch Element.prototype.innerHTML setter once to intercept dashboard updates
  if (!window.__pt_inner_html_patched) {
    window.__pt_inner_html_patched = true;
    var descriptor = Object.getOwnPropertyDescriptor(Element.prototype, 'innerHTML');
    if (descriptor && descriptor.set) {
      var originalSet = descriptor.set;
      Object.defineProperty(Element.prototype, 'innerHTML', {
        set: function (html) {
          if (typeof html === 'string' && html.indexOf('pt-monitor-dashboard') !== -1) {
            var existingDashboard = document.getElementById('pt-monitor-dashboard');
            if (existingDashboard) {
              try {
                var parser = new DOMParser();
                var newDoc = parser.parseFromString(html, 'text/html');
                var newDashboard = newDoc.getElementById('pt-monitor-dashboard');
                if (newDashboard) {
                  var oldLoc = existingDashboard.getAttribute('data-pt-location');
                  var newLoc = newDashboard.getAttribute('data-pt-location');
                  if (oldLoc === newLoc) {
                    updateDashboard(existingDashboard, newDashboard);
                    return; // Skip replacing innerHTML of the container
                  }
                }
              } catch (err) {
                console.error('Error doing smart dashboard update:', err);
              }
            }
          }
          originalSet.call(this, html);
        },
        configurable: true,
        enumerable: true
      });
    }
  }

  function findRefreshButton(root) {
    var el = root;
    while (el && el !== document.body) {
      var parent = el.parentElement;
      if (parent) {
        var buttons = parent.querySelectorAll('button');
        for (var i = 0; i < buttons.length; i++) {
          var b = buttons[i];
          if (b.dataset.ptRefreshBtn) return b;
          var label = ((b.textContent || '') + ' ' + (b.title || '') + ' ' + (b.getAttribute('aria-label') || '')).toLowerCase();
          if (label.indexOf('refresh') !== -1) {
            b.dataset.ptRefreshBtn = '1';
            return b;
          }
        }
      }
      el = parent;
    }
    return null;
  }

  // Stamp window.__tb_interaction_ts whenever a TensorBoard trigger input
  // changes value, so updateDashboard() can skip the GPU section update
  // for the next 30 seconds and avoid a false 0% flash.
  function bindTbInteractionGuard() {
    if (window.__ptTbGuardBound) return;
    window.__ptTbGuardBound = true;

    function markInteraction(e) {
      if (e.target && (e.target.name === 'tbStartTrigger' || e.target.name === 'tbStopTrigger')) {
        if (e.target.value !== '') {
          window.__tb_interaction_ts = Date.now();
        }
      }
    }
    document.addEventListener('input', markInteraction, true);
    document.addEventListener('change', markInteraction, true);
  }

  function bindRefreshCooldown() {
    var root = document.getElementById('pt-monitor-dashboard');
    if (!root) return;

    var btn = findRefreshButton(root);
    if (!btn || btn.dataset.ptCooldownBound) return;
    btn.dataset.ptCooldownBound = '1';

    var hint = document.getElementById('pt-refresh-cooldown-hint');
    var last = 0;
    var timer = null;

    function setHint(msg) {
      if (hint) hint.textContent = msg || '';
    }

    function startCooldown(remaining) {
      btn.disabled = true;
      btn.style.opacity = '0.65';
      btn.style.cursor = 'not-allowed';
      if (timer) clearInterval(timer);
      var left = Math.ceil(remaining / 1000);
      setHint('Refresh available in ' + left + 's');
      timer = setInterval(function () {
        left -= 1;
        if (left <= 0) {
          clearInterval(timer);
          timer = null;
          btn.disabled = false;
          btn.style.opacity = '';
          btn.style.cursor = '';
          setHint('');
          return;
        }
        setHint('Refresh available in ' + left + 's');
      }, 1000);
    }

    btn.addEventListener('click', function (e) {
      var now = Date.now();
      if (last && now - last < COOLDOWN_MS) {
        e.preventDefault();
        e.stopPropagation();
        startCooldown(COOLDOWN_MS - (now - last));
        return false;
      }
      last = now;
      startCooldown(COOLDOWN_MS);
    }, true);
  }

  function init() {
    var root = document.getElementById('pt-monitor-dashboard');
    if (!root) return;
    bindRefreshCooldown();
    bindTbInteractionGuard();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  setTimeout(init, 200);
  setTimeout(init, 800);
 })();
</script>
SCRIPT
