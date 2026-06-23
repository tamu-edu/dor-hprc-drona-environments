#!/bin/bash
# Emit TensorBoard launch instructions for a training run log directory.

LOGDIR="${LOGDIR:-}"
RESOLVED_LOGDIR="${RESOLVED_LOGDIR:-}"
TB_LAUNCH_MODE="${TB_LAUNCH_MODE:-command_only}"
TB_PORT="${TB_PORT:-6006}"

python3 <<EOF
import html
import os

logdir = """${LOGDIR}""".strip() or """${RESOLVED_LOGDIR}""".strip()
mode = """${TB_LAUNCH_MODE}""".strip() or "command_only"
port = """${TB_PORT}""".strip() or "6006"

if not logdir:
    print("<p><em>Select a training run above or enter a log directory path.</em></p>")
    raise SystemExit(0)

if not os.path.isdir(logdir):
    note = f"<p style='color:#b8860b;'>Directory not found yet: <code>{html.escape(logdir)}</code>. It may appear after training starts writing logs.</p>"
else:
    note = f"<p>Log directory: <code>{html.escape(logdir)}</code></p>"

cmd = f"module load WebProxy && tensorboard --logdir={logdir} --host=0.0.0.0 --port={port}"
helper = f"""#!/bin/bash
#SBATCH --job-name=tensorboard
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=tb-out.%j --error=tb-error.%j
module purge
module load WebProxy
{cmd}
"""

print(note)
print(f"<p><strong>Interactive / login node command:</strong></p>")
print(f"<pre style='background:#f5f5f5;padding:10px;border-radius:4px;'>{html.escape(cmd)}</pre>")

if mode == "submit_helper":
    print("<p><strong>Helper Slurm script (save as <code>tensorboard.job</code> and submit with <code>sbatch tensorboard.job</code>):</strong></p>")
    print(f"<pre style='background:#f5f5f5;padding:10px;border-radius:4px;white-space:pre-wrap;'>{html.escape(helper)}</pre>")

print("<p style='font-size:0.9em;color:#555;'>After starting TensorBoard, use Monitor → Connect to Server with the node hostname and port.</p>")
EOF
