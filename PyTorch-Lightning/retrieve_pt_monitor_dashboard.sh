#!/bin/bash
# Combined monitor dashboard: Slurm summary + output/error logs + shared UI scripts.

export LOCATION="${LOCATION:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

html_escape() {
    sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g'
}

emit_slurm_summary() {
    if [ -z "$LOCATION" ] || [ ! -d "$LOCATION" ]; then
        echo "<div style='font-size:0.83em;color:#6c757d;padding:4px 2px'>Workflow directory not available yet.</div>"
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
        echo "<div style='font-size:0.83em;color:#6c757d;padding:4px 2px'>Waiting for SLURM job IDs — training job may still be starting.</div>"
        return 0
    fi

    local rows=""
    local has_running=false
    local has_failed=false
    local has_pending=false
    local JID ROW STATUS ROW_CLASS
    local C1 C2 C3 C4 C5

    for JID in "${JOBIDS[@]}"; do
        JID="${JID//$'\r'/}"
        JID="$(echo "$JID" | xargs)"
        [ -z "$JID" ] && continue

        ROW=$(squeue -j "$JID" -h -o "%i|%j|%T|%M|%N" 2>/dev/null | head -1)

        if [ -n "$ROW" ]; then
            STATUS=$(echo "$ROW" | cut -d'|' -f3)
            case "$STATUS" in
                RUNNING) ROW_CLASS="table-success"; has_running=true ;;
                PENDING) ROW_CLASS="table-warning"; has_pending=true ;;
                *)       ROW_CLASS="" ;;
            esac
        else
            RAW_SACCT=$(sacct -j "$JID" --noheader --format=JobID,JobName,State,Elapsed,NodeList,Submit --parsable2 2>/dev/null)
            ROW=$(python3 -c "
import sys, os, datetime
raw_sacct = sys.argv[1]
location = sys.argv[2]
target_jid = sys.argv[3]
ref_path = os.path.join(location, 'slurm_jobids.txt')
if not os.path.exists(ref_path):
    ref_path = location
ref_time = os.path.getmtime(ref_path) if os.path.exists(ref_path) else None
best_row = ''
min_diff = float('inf')
for line in raw_sacct.strip().split('\n'):
    if not line.strip():
        continue
    parts = line.split('|')
    if len(parts) < 6:
        continue
    jid, name, state, elapsed, nodelist, submit_str = parts[:6]
    if jid != target_jid:
        continue
    try:
        dt = datetime.datetime.strptime(submit_str.strip(), '%Y-%m-%dT%H:%M:%S')
        submit_time = dt.timestamp()
    except Exception:
        submit_time = None
    if ref_time is not None and submit_time is not None:
        diff = abs(ref_time - submit_time)
        if diff < min_diff:
            min_diff = diff
            best_row = f'{jid}|{name}|{state}|{elapsed}|{nodelist}'
    elif not best_row:
        best_row = f'{jid}|{name}|{state}|{elapsed}|{nodelist}'
if best_row:
    print(best_row)
" "$RAW_SACCT" "$LOCATION" "$JID")
            STATUS=$(echo "$ROW" | cut -d'|' -f3 | xargs)
            case "$STATUS" in
                COMPLETED)   ROW_CLASS="table-success" ;;
                FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
                             ROW_CLASS="table-danger"; has_failed=true ;;
                *)           ROW_CLASS="" ;;
            esac
        fi

        C1=$(echo "$ROW" | cut -d'|' -f1 | html_escape)
        C2=$(echo "$ROW" | cut -d'|' -f2 | html_escape)
        C3=$(echo "$ROW" | cut -d'|' -f3 | html_escape)
        C4=$(echo "$ROW" | cut -d'|' -f4 | html_escape)
        C5=$(echo "$ROW" | cut -d'|' -f5 | html_escape)
        JID_ESC=$(echo "$JID" | html_escape)

        rows+="<tr class='${ROW_CLASS}'><td>${C1:-$JID_ESC}</td><td>${C2:-—}</td><td>${C3:-unknown}</td><td>${C4:-—}</td><td>${C5:-—}</td></tr>"
    done

    local dot_color status_label
    if $has_failed; then
        dot_color="#dc3545"
        status_label="Failed — check output/error logs below."
    elif $has_running; then
        dot_color="#0d6efd"
        status_label="Running — dashboard auto-refreshes every 15 seconds."
    elif $has_pending; then
        dot_color="#ffc107"
        status_label="Pending — queued, waiting to start."
    else
        dot_color="#198754"
        status_label="Completed — see logs below."
    fi

    echo "<div style='display:flex;align-items:center;gap:7px;margin-bottom:8px;font-size:0.83em;color:#495057'>"
    echo "  <span style='width:8px;height:8px;border-radius:50%;background:${dot_color};flex-shrink:0;display:inline-block'></span>"
    echo "  ${status_label}"
    echo "</div>"

    echo "<table class='table table-sm table-bordered mb-0' style='font-size:0.85em'>"
    echo "<thead class='table-light'><tr><th>Job ID</th><th>Name</th><th>Status</th><th>Elapsed</th><th>Node(s)</th></tr></thead>"
    echo "<tbody>${rows}</tbody></table>"
}

emit_logs() {
  if [ -f "$SCRIPT_DIR/retrieve_pt_logs.sh" ]; then
      bash "$SCRIPT_DIR/retrieve_pt_logs.sh"
  else
      echo "<div style='font-size:0.85em;color:#6c757d'>Log viewer unavailable.</div>"
  fi
}

echo "<div id='pt-monitor-dashboard'>"

echo "<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap'>"
echo "  <span style='font-size:0.82em;color:#868e96'>Auto-refreshes every 15 seconds.</span>"
echo "  <span id='pt-refresh-cooldown-hint' style='font-size:0.82em;color:#e67700'></span>"
echo "</div>"

echo "<div class='pt-slurm-section'>"
echo "<div style='font-size:0.9em;font-weight:600;color:#495057;margin-bottom:8px'>Slurm Jobs</div>"
emit_slurm_summary
echo "</div>"

echo "<div class='pt-logs-section' style='margin-top:14px'>"
echo "<div style='font-size:0.9em;font-weight:600;color:#495057;margin-bottom:8px'>Output/Error logs</div>"
emit_logs
echo "</div>"

echo "</div>"

cat << 'SCRIPT'
<script>
(function () {
  var COOLDOWN_MS = 5000;

  function bindLogArrows(root) {
    root.querySelectorAll('.pt-logs-root details').forEach(function (d) {
      if (d.dataset.ptArrowBound) return;
      d.dataset.ptArrowBound = '1';
      d.addEventListener('toggle', function () {
        var arrow = d.querySelector('.pt-log-arrow');
        if (arrow) arrow.textContent = d.open ? '\u25BC' : '\u25B6';
        if (d.open) {
          var vp = d.querySelector('.pt-log-viewport');
          if (vp) vp.scrollTop = vp.scrollHeight;
        }
      });
    });
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

  function scrollToBottom(root) {
    root.querySelectorAll('.pt-logs-root details[open] .pt-log-viewport').forEach(function (vp) {
      if (vp.dataset.ptScrolledOnce) return;
      vp.dataset.ptScrolledOnce = '1';
      vp.scrollTop = vp.scrollHeight;
    });
  }

  function init() {
    var root = document.getElementById('pt-monitor-dashboard');
    if (!root) return;
    bindLogArrows(root);
    bindRefreshCooldown();
    scrollToBottom(root);
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
