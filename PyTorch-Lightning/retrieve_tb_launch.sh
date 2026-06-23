#!/bin/bash
# Emit TensorBoard launch instructions for a training run log directory.

RESOLVED_LOGDIR="${RESOLVED_LOGDIR:-}"

python3 <<EOF
import html
import os

logdir = """${RESOLVED_LOGDIR}""".strip()
port = "6006"

if not logdir:
    print("<p><em>Select a training run above to resolve the log directory.</em></p>")
    raise SystemExit(0)

if not os.path.isdir(logdir):
    note = f"<p style='color:#b8860b;'>Directory not found yet: <code>{html.escape(logdir)}</code>. It may appear after training starts writing logs.</p>"
else:
    note = f"<p>Log directory: <code>{html.escape(logdir)}</code></p>"

cmd = "module load WebProxy && module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch-Lightning/2.2.1-CUDA-12.1.1 && tensorboard --logdir={logdir} --host=0.0.0.0 --port={port}"

print(note)
print("<p><strong>Interactive / login node command:</strong></p>")
print(f"<pre style='background:#f5f5f5;padding:10px;border-radius:4px;'>{html.escape(cmd)}</pre>")
print("<p style='font-size:0.9em;color:#555;'>Use an OOD desktop session or port forwarding to open TensorBoard in your browser.</p>")
EOF
