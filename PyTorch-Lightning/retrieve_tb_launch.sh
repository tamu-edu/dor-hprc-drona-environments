#!/bin/bash
# Emit TensorBoard log directory and Checkpoint file location link.

RESOLVED_LOGDIR="${RESOLVED_LOGDIR:-}"

python3 <<EOF
import html
import os
import glob

logdir = """${RESOLVED_LOGDIR}""".strip()

if not logdir:
    print("<p><em>Select a training run above to resolve the log directory.</em></p>")
    raise SystemExit(0)

if not os.path.isdir(logdir):
    note = f"<p style='color:#b8860b;'>Directory not found yet: <code>{html.escape(logdir)}</code>. It may appear after training starts writing logs.</p>"
    ckpt_dir = logdir
else:
    note = f"<p>Log directory: <code>{html.escape(logdir)}</code></p>"
    
    # Try to find a checkpoints directory or any saved .ckpt files
    ckpt_dirs = glob.glob(os.path.join(logdir, "**", "checkpoints"), recursive=True)
    if ckpt_dirs:
        # Get the most recently modified one if there are multiple
        ckpt_dir = sorted(ckpt_dirs, key=os.path.getmtime, reverse=True)[0]
    else:
        # Check for any .ckpt files
        ckpt_files = glob.glob(os.path.join(logdir, "**", "*.ckpt"), recursive=True)
        if ckpt_files:
            ckpt_dir = os.path.dirname(ckpt_files[0])
        else:
            ckpt_dir = logdir

print(note)

# Generate Open OnDemand Files app link
style = """
<style>
  .ckpt-box {
    display: inline-block;
    background-color: #f5f5f5;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 6px 12px;
    font-family: monospace;
    margin-top: 5px;
  }
  .ckpt-box a {
    color: #003c71;
    text-decoration: none;
    font-weight: bold;
  }
  .ckpt-box a:hover {
    text-decoration: underline;
  }
</style>
"""

escaped_dir = html.escape(ckpt_dir)
link = f'<a href="/pun/sys/dashboard/files/fs{escaped_dir}" target="_blank">Open Checkpoint Directory ({escaped_dir})</a>'

print(style)
print("<p><strong>Model Checkpoints:</strong></p>")
print(f'<div class="ckpt-box">{link}</div>')
EOF
