#!/bin/bash
# Resolve TensorBoard log directory for a PyTorch-Lightning workflow run.

JOB_DIR="${JOB_DIR:-}"

python3 <<EOF
import os
import re

job_dir = """${JOB_DIR}""".strip()
log_dir_name = "lightning_logs"

# Settings file (newer workflows: "  log_dir: ./x" in config.yaml), else LOG_DIR = "./x" in train.py
SOURCES = [
    ("config.yaml", r'^\s+log_dir:\s*["\']?([^"\'#\n]+?)["\']?\s*(?:#.*)?$'),
    ("train.py", r'^LOG_DIR\s*=\s*["\']([^"\']+)["\']'),
]
for fname, pattern in (SOURCES if job_dir else []):
    path = os.path.join(job_dir, fname)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                val = match.group(1)
                if val.startswith("./"):
                    val = val[2:]
                log_dir_name = val
                break
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
