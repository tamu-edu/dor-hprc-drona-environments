#!/bin/bash
# Given $LOCATION, emits links to open all out.* and error.* files.

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

echo "<style>"
echo "  .pt-log-link-container {"
echo "    display: flex;"
echo "    align-items: center;"
echo "    gap: 8px;"
echo "    padding: 8px 12px;"
echo "    background: #f8f9fa;"
echo "    border: 1px solid #dee2e6;"
echo "    border-radius: 6px;"
echo "    transition: background-color 0.15s ease, border-color 0.15s ease;"
echo "    text-decoration: none !important;"
echo "  }"
echo "  .pt-log-link-container:hover {"
echo "    background: #e9ecef;"
echo "    border-color: #ced4da;"
echo "  }"
echo "  .pt-log-link-title {"
echo "    color: #0d6efd;"
echo "    font-weight: 500;"
echo "    font-size: 0.95em;"
echo "  }"
echo "  .pt-log-link-container:hover .pt-log-link-title {"
echo "    text-decoration: underline;"
echo "  }"
echo "</style>"

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

    if [ -n "$FILE_SIZE" ]; then
        SIZE_TEXT=", $FILE_SIZE"
    else
        SIZE_TEXT=""
    fi

    echo "<div class='pt-log-container' data-pt-logfile=\"$BASENAME\" style='margin-bottom:8px'>"
    echo "  <div class='pt-log-header'>"
    echo "    <a class='pt-log-link-container' href=\"/pun/sys/files/fs${LOG_FILE}\" target=\"_blank\">"
    echo "      <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" fill=\"currentColor\" class=\"bi bi-file-earmark-text\" viewBox=\"0 0 16 16\" style=\"color:#6c757d;flex-shrink:0\">"
    echo "        <path d=\"M5.5 7a.5.5 0 0 0 0 1h5a.5.5 0 0 0 0-1zM5 9.5a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5m0 2a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5\"/>"
    echo "        <path d=\"M14 4.5V14a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V2a2 2 0 0 1 2-2h5.5zm-3 0A1.5 1.5 0 0 1 9.5 3V1H4a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V4.5z\"/>"
    echo "      </svg>"
    echo "      <span class='pt-log-link-title'>$BASENAME</span>"
    echo "      <span class='pt-log-metadata' style='font-weight:400;color:#6c757d;font-size:0.85em;margin-left:auto'>($LINE_COUNT lines${SIZE_TEXT})</span>"
    echo "    </a>"
    echo "  </div>"
    echo "</div>"
done

echo "</div>"
