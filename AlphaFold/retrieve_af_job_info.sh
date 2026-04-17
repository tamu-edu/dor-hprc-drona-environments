#!/bin/bash
# Given $DRONA_ID, emits the job's output location as plain text.
# Used by a hidden field so downstream retrievers can reference $af_location.

JSON_DATA=$($DRONA_RUNTIME_DIR/db_access/drona_db_retriever.py -i "$DRONA_ID" 2>/dev/null)
export JSON_DATA

python3 <<'EOF'
import json, os, sys

raw = os.environ.get("JSON_DATA", "").strip()
try:
    record = json.loads(raw) if raw else {}
except Exception:
    record = {}

sys.stdout.write(record.get("location", ""))
EOF
