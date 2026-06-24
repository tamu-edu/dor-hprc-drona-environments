#!/bin/bash
# Given $LOCATION, emits collapsible log viewers for all out.* and error.* files.
# Uses a CSS scaleY(-1) flip so the viewport opens at the end of each file without JS.

LOCATION="${LOCATION:-}"

if [ -z "$LOCATION" ] || [ ! -d "$LOCATION" ]; then
    echo "<div class='alert alert-secondary mb-0' style='font-size:0.85em'>Workflow directory not available yet.</div>"
    exit 0
fi

mapfile -t LOG_FILES < <(
    find "$LOCATION" -maxdepth 1 \
        \( -name "out.*" -o -name "error.*" \) \
        -type f 2>/dev/null | sort
)

if [ ${#LOG_FILES[@]} -eq 0 ]; then
    echo "<div class='alert alert-secondary mb-0' style='font-size:0.85em'>No log files found yet in <code>$(echo "$LOCATION" | sed 's/&/\&amp;/g')</code>.<br>Logs will appear once the job starts writing output.</div>"
    exit 0
fi

escape_html() {
    sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g'
}

echo "<div class='pt-logs-root' style='font-size:0.82em'>"

for LOG_FILE in "${LOG_FILES[@]}"; do
    BASENAME=$(basename "$LOG_FILE")
    LINE_COUNT=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
    CONTENT=$(cat "$LOG_FILE" 2>/dev/null | escape_html)
    [ -z "$CONTENT" ] && CONTENT="(empty)"

    echo "<details style='margin-bottom:10px' open>"
    echo "  <summary style='cursor:pointer;user-select:none;padding:5px 8px;"
    echo "     background:#f1f3f5;border-radius:4px;font-weight:600;color:#495057;"
    echo "     list-style:none;display:flex;align-items:center;gap:8px'>"
    echo "    <span class='pt-log-arrow'>&#9660;</span>"
    echo "    <span>$BASENAME</span>"
    echo "    <span style='font-weight:400;color:#868e96;font-size:0.9em'>($LINE_COUNT lines, newest at bottom)</span>"
    echo "  </summary>"
    echo "  <div class='pt-log-viewport' style='max-height:400px;overflow:auto;"
    echo "       transform:scaleY(-1);background:#1e1e1e;border-radius:0 0 4px 4px'>"
    echo "    <pre class='pt-log-pre' style='margin:0;padding:10px;transform:scaleY(-1);"
    echo "         white-space:pre-wrap;word-break:break-word;color:#d4d4d4;"
    echo "         font-size:0.9em'>${CONTENT}</pre>"
    echo "  </div>"
    echo "</details>"
done

echo "</div>"
