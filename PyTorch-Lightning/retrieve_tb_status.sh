#!/bin/bash
# Check reachability of a TensorBoard server and return HTML status.

TB_HOST="${TB_HOST:-localhost}"
TB_PORT="${TB_PORT:-6006}"

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 "http://${TB_HOST}:${TB_PORT}/" 2>/dev/null || echo "000")

python3 <<EOF
host = """${TB_HOST}""".strip() or "localhost"
port = """${TB_PORT}""".strip() or "6006"
code = """${HTTP_CODE}""".strip()
url = f"http://{host}:{port}/"

if code in ("200", "301", "302"):
    status = "<span style='color:green;font-weight:bold;'>Reachable</span>"
elif code == "000":
    status = "<span style='color:#b8860b;font-weight:bold;'>Unreachable</span> (connection failed or timed out)"
else:
    status = f"<span style='color:#b8860b;font-weight:bold;'>HTTP {code}</span>"

print(f"""<div style='padding:8px;'>
  <p>TensorBoard at <code>{url}</code>: {status}</p>
  <p><a href="{url}" target="_blank" rel="noopener noreferrer">Open TensorBoard</a></p>
  <p style='font-size:0.9em;color:#555;'>If TensorBoard runs on a compute node, use OOD desktop port forwarding or an SSH tunnel to reach it from your browser.</p>
</div>""")
EOF
