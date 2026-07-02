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
    FILE_SIZE=$(du -sh "$LOG_FILE" 2>/dev/null | awk '{print $1}')
    if [[ "$FILE_SIZE" =~ K$ ]]; then
        FILE_SIZE="${FILE_SIZE%K} KB"
    elif [[ "$FILE_SIZE" =~ M$ ]]; then
        FILE_SIZE="${FILE_SIZE%M} MB"
    elif [[ "$FILE_SIZE" =~ G$ ]]; then
        FILE_SIZE="${FILE_SIZE%G} GB"
    elif [[ "$FILE_SIZE" =~ T$ ]]; then
        FILE_SIZE="${FILE_SIZE%T} TB"
    elif [ -n "$FILE_SIZE" ]; then
        if [[ "$FILE_SIZE" =~ ^[0-9]+$ ]]; then
            FILE_SIZE="${FILE_SIZE} B"
        fi
    fi
    CONTENT=$(tail -n 1000 "$LOG_FILE" 2>/dev/null | escape_html)
    [ -z "$CONTENT" ] && CONTENT="(empty)"

    if [ -n "$FILE_SIZE" ]; then
        SIZE_TEXT=", $FILE_SIZE"
    else
        SIZE_TEXT=""
    fi

    echo "<div class='pt-log-container' data-pt-logfile=\"$BASENAME\" style='margin-bottom:10px; border:1px solid #dee2e6; border-radius:4px; overflow:hidden'>"
    echo "  <div class='pt-log-header' style='padding:5px 8px;"
    echo "     background:#f1f3f5;font-weight:600;color:#495057;"
    echo "     display:flex;align-items:center;gap:8px;border-bottom:1px solid #dee2e6'>"
    echo "    <span><a href=\"/pun/sys/files/fs${LOG_FILE}\" target=\"_blank\" style=\"color:#0d6efd;text-decoration:none;cursor:pointer\" onmouseover=\"this.style.textDecoration='underline'\" onmouseout=\"this.style.textDecoration='none'\">$BASENAME</a></span>"
    echo "    <span style='font-weight:400;color:#868e96;font-size:0.9em'>($LINE_COUNT lines${SIZE_TEXT})</span>"
    echo "  </div>"
    if [ "$LINE_COUNT" -gt 1000 ]; then
        SHOWN_LINES=1000
    else
        SHOWN_LINES=$LINE_COUNT
    fi

    echo "  <div class='pt-log-sub-header' style='font-size:0.85em;color:#adb5bd;background:#2d2d2d;padding:4px 10px;border-bottom:1px solid #3c3c3c;font-family:var(--bs-font-sans-serif, sans-serif)'>"
    echo "    Showing the latest $SHOWN_LINES lines"
    echo "  </div>"
    echo "  <div class='pt-log-viewport' style='max-height:250px;overflow:auto;"
    echo "       background:#1e1e1e'>"
    echo "    <pre class='pt-log-pre' style='margin:0;padding:10px;"
    echo "         white-space:pre-wrap;word-break:break-word;color:#d4d4d4;"
    echo "         font-size:0.9em'>${CONTENT}</pre>"
    echo "  </div>"
    echo "</div>"
done

echo "</div>"
