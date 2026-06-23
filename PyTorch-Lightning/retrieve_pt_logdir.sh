#!/bin/bash
# Resolve TensorBoard log directory for a PyTorch-Lightning workflow run.

JOB_DIR="${JOB_DIR:-}"
WORKFLOW_ID="${WORKFLOW_ID:-}"

if [ -z "$JOB_DIR" ] && [ -n "$WORKFLOW_ID" ]; then
  JOB_DIR=$($DRONA_RUNTIME_DIR/db_access/drona_db_retriever.py -i "$WORKFLOW_ID" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('location',''))" 2>/dev/null)
fi

python3 <<EOF
import os

job_dir = """${JOB_DIR}""".strip()
candidates = []
if job_dir:
    candidates.append(os.path.join(job_dir, "lightning_logs"))
    candidates.append(os.path.join(job_dir, "./lightning_logs"))

for path in candidates:
    if path and os.path.isdir(path):
        print(path)
        break
else:
    if job_dir:
        print(os.path.join(job_dir, "lightning_logs"))
    else:
        print("")
EOF
