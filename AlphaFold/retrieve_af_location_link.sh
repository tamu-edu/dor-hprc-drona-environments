#!/bin/bash
# Emits an HTML link to the OOD file browser for the job's output directory.

python3 << 'PYEOF'
import os

LOCATION = os.environ.get('LOCATION', '').strip()
if not LOCATION:
    raise SystemExit(0)

# For AF3 runs the actual output is in a subdirectory written by the CPU job
output_dir = None
try:
    with open(os.path.join(LOCATION, 'af3_output_dir.txt')) as f:
        candidate = f.read().strip()
    if candidate and os.path.isdir(candidate):
        output_dir = candidate
except Exception:
    pass

show_dir = output_dir or LOCATION

# OOD Files app URL: relative path works inside the portal
ood_url = '/pun/sys/files/fs/' + show_dir.lstrip('/')

parts = [
    "<div>",
    f"  Output: <a href='{ood_url}' target='_blank'>{show_dir}</a>",
    "</div>",
]
print('\n'.join(parts))
PYEOF
