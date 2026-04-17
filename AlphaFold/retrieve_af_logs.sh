#!/bin/bash
# Given $LOCATION, emits collapsible log viewers for all AlphaFold log files.

mapfile -t LOG_FILES < <(
    find "$LOCATION" -maxdepth 2 \
        \( -name "out-alphafold.*" -o -name "out-parafold.*" -o -name "error.alphafold.*" -o -name "slurm-*.out" \) \
        -type f 2>/dev/null | sort
)

if [ ${#LOG_FILES[@]} -eq 0 ]; then
    echo "<div class='alert alert-secondary mb-0' style='font-size:0.85em'>No log files found yet in <code>$(echo "$LOCATION" | sed 's/&/\&amp;/g')</code>.<br>Logs will appear once the job starts writing output.</div>"
    exit 0
fi

echo "<div style='font-size:0.82em'>"

for LOG_FILE in "${LOG_FILES[@]}"; do
    BASENAME=$(basename "$LOG_FILE")
    LINE_COUNT=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
    LINES=$(tail -n 300 "$LOG_FILE" 2>/dev/null | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g')

    echo "<details style='margin-bottom:10px'>"
    echo "  <summary style='cursor:pointer;user-select:none;padding:5px 8px;"
    echo "     background:#f1f3f5;border-radius:4px;font-weight:600;color:#495057;"
    echo "     list-style:none;display:flex;align-items:center;gap:8px'>"
    echo "    <span>&#9654;</span>"
    echo "    <span>&#128196; $BASENAME</span>"
    echo "    <span style='font-weight:400;color:#868e96;font-size:0.9em'>($LINE_COUNT lines)</span>"
    echo "  </summary>"
    echo "  <pre style='background:#1e1e1e;color:#d4d4d4;padding:10px;border-radius:0 0 4px 4px;"
    echo "       overflow-x:auto;max-height:400px;overflow-y:auto;margin:0;white-space:pre-wrap;"
    echo "       font-size:0.9em'>$LINES</pre>"
    echo "</details>"
done

echo "</div>"

# Toggle triangle via inline script
cat << 'SCRIPT'
<script>
document.querySelectorAll('details').forEach(d => {
  d.addEventListener('toggle', () => {
    const arrow = d.querySelector('summary span:first-child');
    if (arrow) arrow.textContent = d.open ? '▼' : '▶';
  });
});
</script>
SCRIPT
