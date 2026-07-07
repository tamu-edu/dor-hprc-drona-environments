#!/bin/bash
# Resolve TensorBoard log directory for a PyTorch-Lightning workflow run.

JOB_DIR="${JOB_DIR:-}"
WORKFLOW_ID="${WORKFLOW_ID:-}"

if [ -z "$JOB_DIR" ] && [ -n "$WORKFLOW_ID" ]; then
  JOB_DIR=$($DRONA_RUNTIME_DIR/db_access/drona_db_retriever.py -i "$WORKFLOW_ID" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('location',''))" 2>/dev/null)
fi

python3 <<EOF
import os
import re

job_dir = """${JOB_DIR}""".strip()
log_dir_name = "lightning_logs"

if job_dir:
    train_py = os.path.join(job_dir, "train.py")
    if os.path.isfile(train_py):
        try:
            with open(train_py, "r", encoding="utf-8") as f:
                content = f.read()
            match = re.search(r'LOG_DIR\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                val = match.group(1)
                if val.startswith("./"):
                    val = val[2:]
                log_dir_name = val
        except Exception:
            pass

candidates = [
    os.path.join(job_dir, log_dir_name),
    os.path.join(job_dir, "lightning_logs"),
]

for path in candidates:
    if path and os.path.isdir(path):
        print(path)
        break
else:
    if job_dir:
        print(os.path.join(job_dir, log_dir_name))
    else:
        print("")
EOF
