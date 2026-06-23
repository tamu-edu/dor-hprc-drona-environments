#!/bin/bash
# Returns PyTorch-Lightning runs from the job history DB for dynamicSelect.

JSON_DATA=$($DRONA_RUNTIME_DIR/db_access/drona_db_retriever.py -e PyTorch-Lightning 2>/dev/null)
export JSON_DATA

python3 <<'EOF'
import json
import os

raw = os.environ.get("JSON_DATA", "[]").strip()
try:
    records = json.loads(raw) if raw else []
except Exception:
    records = []

options = []
for r in records:
    drona_id = r.get("drona_id", "")
    if not drona_id:
        continue
    name = r.get("name") or "unnamed"
    date = (r.get("start_time") or "")[:10]
    status = r.get("status") or "submitted"
    label = f"{name}  ({date})  [{status}]"
    options.append({"value": drona_id, "label": label})

print(json.dumps(options))
EOF
