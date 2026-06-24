#!/bin/bash
# Returns PyTorch-Lightning runs from the Drona job history DB for dynamicSelect.
# Only includes workflows whose staging directory still exists on disk.

ENV_NAME="${DRONA_ENV_NAME:-PyTorch-Lightning}"

JSON_DATA=$("$DRONA_RUNTIME_DIR/db_access/drona_db_retriever.py" -e "$ENV_NAME" 2>/dev/null)
export JSON_DATA
export ENV_NAME

python3 <<'EOF'
import json
import os

raw = os.environ.get("JSON_DATA", "[]").strip()
env_name = os.environ.get("ENV_NAME", "PyTorch-Lightning")

try:
    records = json.loads(raw) if raw else []
except Exception:
    records = []

if not isinstance(records, list):
    records = []

def format_submitted_at(value):
    from datetime import datetime
    if not value:
        return "unknown"
    s = str(value).strip().replace("Z", "").split("+")[0].split(".")[0]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    if len(s) >= 19:
        return s[:19].replace("T", " ")
    return s[:10] if s else "unknown"

candidates = []
for r in records:
    if not isinstance(r, dict):
        continue

    drona_id = r.get("drona_id", "")
    if not drona_id:
        continue

    record_env = r.get("env") or r.get("environment") or r.get("env_name") or ""
    if record_env and record_env != env_name:
        continue

    location = (r.get("location") or "").strip()
    if not location:
        continue
    location = os.path.expandvars(os.path.expanduser(location))
    if not os.path.isdir(location):
        continue

    name = r.get("name") or "unnamed"
    submitted_at = format_submitted_at(r.get("start_time") or "")
    status = r.get("status") or "submitted"
    label = f"{name}  ({submitted_at})  [{status}]"
    start_time = r.get("start_time") or ""
    candidates.append((start_time, drona_id, label))

candidates.sort(key=lambda item: item[0], reverse=True)
options = [{"value": drona_id, "label": label} for _, drona_id, label in candidates]

print(json.dumps(options))
EOF
