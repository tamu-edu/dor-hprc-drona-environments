#!/bin/bash
# Reworked TensorBoard Setup Procedure and Access Link Generator.

export RESOLVED_LOGDIR="${RESOLVED_LOGDIR:-}"
export CUSTOM_LOGDIR="${CUSTOM_LOGDIR:-}"
export TB_VERSION="${TB_VERSION:-tensorboard/2.18.0}"
export TB_HOST="${TB_HOST:-localhost}"
export TB_PORT="${TB_PORT:-6006}"
export TB_START_TRIGGER="${TB_START_TRIGGER:-}"
export TB_STOP_TRIGGER="${TB_STOP_TRIGGER:-}"

python3 <<'EOF'
import html
import os
import glob
import subprocess
import socket
import sys

def clean_env_val(val):
    if not val:
        return ""
    val = val.strip()
    # Strip surrounding quotes
    if val.startswith('"') and val.endswith('"'):
        val = val[1:-1].strip()
    if val.startswith("'") and val.endswith("'"):
        val = val[1:-1].strip()
    # Strip literal '\n' text or newlines
    while val.endswith('\\n') or val.endswith('\n'):
        if val.endswith('\\n'):
            val = val[:-2].strip()
        else:
            val = val[:-1].strip()
    # Strip quotes again in case they were inside the newlines
    if val.startswith('"') and val.endswith('"'):
        val = val[1:-1].strip()
    return val

# Retrieve environment variables
resolved_logdir = clean_env_val(os.environ.get("RESOLVED_LOGDIR", ""))
custom_logdir = clean_env_val(os.environ.get("CUSTOM_LOGDIR", ""))
logdir = custom_logdir if custom_logdir else resolved_logdir

tb_version = clean_env_val(os.environ.get("TB_VERSION", "tensorboard/2.18.0"))
tb_host = clean_env_val(os.environ.get("TB_HOST", "localhost"))
tb_port_str = clean_env_val(os.environ.get("TB_PORT", "6006"))
tb_start_trigger = clean_env_val(os.environ.get("TB_START_TRIGGER", ""))
tb_stop_trigger = clean_env_val(os.environ.get("TB_STOP_TRIGGER", ""))

try:
    port = int(tb_port_str)
except ValueError:
    port = 6006

if not logdir:
    print("<p><em>Select a training run above to resolve the log directory.</em></p>")
    sys.exit(0)

# Resolve checkpoints directory
ckpt_dir = logdir
if os.path.isdir(logdir):
    ckpt_dirs = glob.glob(os.path.join(logdir, "**", "checkpoints"), recursive=True)
    if ckpt_dirs:
        ckpt_dir = sorted(ckpt_dirs, key=os.path.getmtime, reverse=True)[0]
    else:
        ckpt_files = glob.glob(os.path.join(logdir, "**", "*.ckpt"), recursive=True)
        if ckpt_files:
            ckpt_dir = os.path.dirname(ckpt_files[0])

launched_port = None
last_trigger = ""
job_dir = os.path.dirname(logdir)
tb_port_file = os.path.join(job_dir, "tb_port.txt")
if os.path.isfile(tb_port_file):
    try:
        with open(tb_port_file, "r") as f:
            port_val = f.read().strip()
            if port_val:
                launched_port = int(port_val)
    except Exception:
        pass

tb_last_trigger_file = os.path.join(job_dir, "tb_last_trigger.txt")
if os.path.isfile(tb_last_trigger_file):
    try:
        with open(tb_last_trigger_file, "r") as f:
            last_trigger = f.read().strip()
    except Exception:
        pass



# Determine GCC toolchain based on selected TensorBoard version
gcc_module = "GCC/12.3.0"
if "2.18.0" in tb_version:
    gcc_module = "GCC/13.2.0"

# Auto-detect compute node if host is set to localhost
def get_first_node(nodelist_str):
    if not nodelist_str:
        return ""
    import re
    first = nodelist_str.split(",")[0].strip()
    if "[" in first:
        m = re.match(r"^([^\[]+)\[([^\]]+)\]", first)
        if m:
            prefix = m.group(1)
            num_range = m.group(2)
            first_num = num_range.split("-")[0].split(",")[0].strip()
            return f"{prefix}{first_num}"
    return first

node_name = tb_host
if not node_name or node_name == "localhost":
    job_id = None
    job_dir = os.path.dirname(logdir)
    jobids_file = os.path.join(job_dir, "slurm_jobids.txt")
    if os.path.isfile(jobids_file):
        try:
            with open(jobids_file, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
                if lines:
                    job_id = lines[0]
        except Exception:
            pass
    # Fallback to out.* files if jobids_file not found or empty
    if not job_id:
        try:
            import re
            out_files = glob.glob(os.path.join(job_dir, "out.*"))
            candidates = []
            for f in out_files:
                base = os.path.basename(f)
                m = re.match(r"^out\.([0-9]+)$", base)
                if m:
                    candidates.append((os.path.getmtime(f), m.group(1)))
            if candidates:
                candidates.sort(reverse=True)
                job_id = candidates[0][1]
        except Exception:
            pass

    if job_id:
        try:
            res = subprocess.run(["squeue", "-j", job_id, "-h", "-o", "%N"], capture_output=True, text=True, check=True)
            node = get_first_node(res.stdout.strip())
            if node and not node.startswith("("):
                node_name = node
        except Exception:
            try:
                res = subprocess.run(["sacct", "-j", job_id, "--noheader", "--format=JobID,NodeList,Submit", "--parsable2"], capture_output=True, text=True, check=True)
                ref_path = jobids_file if os.path.isfile(jobids_file) else job_dir
                ref_time = os.path.getmtime(ref_path) if os.path.exists(ref_path) else None
                best_node = None
                min_diff = float('inf')
                for line in res.stdout.strip().split('\n'):
                    if not line.strip():
                        continue
                    parts = line.split('|')
                    if len(parts) < 3:
                        continue
                    jid, raw_node, submit_str = parts[:3]
                    if jid != job_id:
                        continue
                    if raw_node == "None" or not raw_node:
                        continue
                    try:
                        from datetime import datetime
                        dt = datetime.strptime(submit_str.strip(), "%Y-%m-%dT%H:%M:%S")
                        submit_time = dt.timestamp()
                    except Exception:
                        submit_time = None
                    if ref_time is not None and submit_time is not None:
                        diff = abs(ref_time - submit_time)
                        if diff < min_diff:
                            min_diff = diff
                            best_node = raw_node
                    elif best_node is None:
                        best_node = raw_node
                if best_node:
                    node = get_first_node(best_node)
                    if node:
                        node_name = node
            except Exception:
                pass

if not node_name:
    node_name = "localhost"

# Process stop trigger if set
if tb_stop_trigger:
    last_stop_trigger = ""
    tb_last_stop_trigger_file = os.path.join(job_dir, "tb_last_stop_trigger.txt")
    if os.path.isfile(tb_last_stop_trigger_file):
        try:
            with open(tb_last_stop_trigger_file, "r") as f:
                last_stop_trigger = f.read().strip()
        except Exception:
            pass

    if last_stop_trigger != tb_stop_trigger:
        try:
            with open(tb_last_stop_trigger_file, "w") as f:
                f.write(tb_stop_trigger)
        except Exception:
            pass

        if node_name == "localhost" or node_name == "127.0.0.1":
            stop_cmd = "pkill -u $USER -f tensorboard"
        else:
            import shlex
            stop_cmd = f"ssh -o StrictHostKeyChecking=no {node_name} 'pkill -u $USER -f tensorboard'"
        try:
            subprocess.run(stop_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

        for fn in [tb_port_file, tb_last_trigger_file]:
            try:
                if os.path.exists(fn):
                    os.remove(fn)
            except Exception:
                pass
        
        launched_port = None

# If TensorBoard is already launched, default to that port unless user typed another one
if launched_port is not None and tb_port_str == "6006":
    port = launched_port

# Step 1: Check if logdir exists
logdir_exists = os.path.isdir(logdir)
if logdir_exists:
    step1_status = f'<span style="color: #16a34a; font-weight: 600;">Log directory found</span>: <code>{html.escape(logdir)}</code>'
else:
    step1_status = f'<span style="color: #ea580c; font-weight: 600;">Waiting for log directory to be generated</span>: <code>{html.escape(logdir)}</code><div class="tb-tip">Make sure your PyTorch-Lightning training job has started and has begun writing logs.</div>'

# Check if TensorBoard is already running on the target node and port
def check_port(host, port_val):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        res = s.connect_ex((host, port_val))
        s.close()
        return res == 0
    except Exception:
        return False

socket_host = "127.0.0.1" if node_name == "localhost" else node_name

is_active = check_port(socket_host, port)
port_taken = False
port_invalid = not (1024 <= port <= 65535)
still_launching = False

if port_invalid:
    is_running = False
elif is_active:
    if launched_port is not None and port == launched_port:
        is_running = True
    else:
        is_running = False
        port_taken = True
else:
    is_running = False
    if tb_start_trigger and launched_port is not None and port == launched_port:
        still_launching = True



# Step 2: Build commands
tb_cmd = f'module load {gcc_module} {tb_version} && timeout 1h tensorboard --logdir="{logdir}" --port={port} --bind_all'
nohup_cmd = f'module load {gcc_module} {tb_version} && nohup timeout 1h tensorboard --logdir="{logdir}" --port={port} --bind_all >/dev/null 2>&1 &'

## Automate launching TensorBoard if trigger is set, not already running, not taken, and not invalid
if tb_start_trigger and not is_running and not port_taken and not port_invalid:
    if last_trigger != tb_start_trigger:
        # Write trigger to last_trigger file first to prevent duplicate launches
        try:
            with open(tb_last_trigger_file, "w") as f:
                f.write(tb_start_trigger)
        except Exception:
            pass

        # Write to tb_port.txt immediately so we register this port as our target TensorBoard port
        try:
            with open(tb_port_file, "w") as f:
                f.write(str(port))
            launched_port = port
        except Exception:
            pass

        import shlex
        import time
        if node_name == "localhost" or node_name == "127.0.0.1":
            cmd = nohup_cmd
        else:
            cmd = f"ssh -f -o StrictHostKeyChecking=no {node_name} {shlex.quote(nohup_cmd)}"
        try:
            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Sleep to allow the TensorBoard process to start and bind to the port
            time.sleep(1.5)
            is_running = check_port(socket_host, port)
            if is_running:
                port_taken = False
            else:
                still_launching = True
        except Exception:
            pass

# Step 3: Relative proxy URL
proxy_url = f'/rnode/{node_name}/{port}/'

esc_node = html.escape(node_name)
esc_logdir = html.escape(logdir)
escaped_dir = html.escape(ckpt_dir)
esc_job_dir = html.escape(job_dir)
link = f'<a href="/pun/sys/dashboard/files/fs{escaped_dir}" target="_blank">Open Checkpoint Directory</a>'

# Open TensorBoard Web UI button styling based on run state
if is_running:
    tb_button_html = f'<a href="{html.escape(proxy_url)}" target="_blank" class="tb-btn-primary">Open TensorBoard Web UI</a>'
else:
    tb_button_html = f'<a href="#" onclick="return false;" class="tb-btn-primary tb-btn-disabled">Open TensorBoard Web UI</a>'

# Render Step 2 Content dynamically based on status
if is_running:
    stop_button_html = f"""<button type="button" class="tb-btn-start" style="background-color: #dc3545;" onclick="this.disabled = true; this.innerHTML = '<span class=\\'tb-spinner\\'></span> Stopping TensorBoard...'; this.style.opacity = '0.7'; this.style.cursor = 'not-allowed'; var input = document.querySelector('input[name=\\'tbStopTrigger\\']'); if (input) {{ var val = Date.now().toString(); var lastVal = input.value; var setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; if (setter) {{ setter.call(input, val); }} else {{ input.value = val; }} var tracker = input._valueTracker; if (tracker) {{ tracker.setValue(lastVal); }} input.dispatchEvent(new Event('input', {{ bubbles: true }})); input.dispatchEvent(new Event('change', {{ bubbles: true }})); input.dispatchEvent(new Event('blur', {{ bubbles: true }})); }}">Stop TensorBoard</button>"""
    step2_content = f"""
    <div style="color: #16a34a; font-weight: 600; margin-bottom: 12px;">TensorBoard server is active and running on node <code>{esc_node}</code> port <code>{port}</code>.</div>
    <div style="margin-bottom: 16px;">
        {stop_button_html}
    </div>
    <div class="tb-tip" style="color: #475569;">No manual setup is required. Proceed to Step 3.</div>
    """
else:
    if still_launching:
        status_msg = f'<span class="tb-status-msg" style="color: #64748b; font-size: 0.88em; margin-left: 5px; vertical-align: middle;">It may take up to 2 minutes to launch Tensorboard</span>'
        button_html = f'<button type="button" class="tb-btn-start tb-btn-disabled" style="background-color: #cbd5e1 !important; color: #64748b !important; cursor: not-allowed !important; pointer-events: none !important;" disabled><span class="tb-spinner"></span> Starting TensorBoard...</button>'
        port_disabled_attr = "disabled"
        port_style = "width: 80px; padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.88em; opacity: 0.7; cursor: not-allowed;"
    elif port_invalid:
        status_msg = f'<span class="tb-status-msg" style="color: #dc3545; font-weight: 600; font-size: 0.88em; margin-left: 5px; vertical-align: middle;">Port must be between 1024 and 65535.</span>'
        button_html = f'<button type="button" class="tb-btn-start tb-btn-disabled" style="background-color: #cbd5e1 !important; color: #64748b !important; cursor: not-allowed !important; pointer-events: none !important;" disabled>Start TensorBoard on Node</button>'
        port_disabled_attr = ""
        port_style = "width: 80px; padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.88em;"
    elif port_taken:
        status_msg = f'<span class="tb-status-msg" style="color: #dc3545; font-weight: 600; font-size: 0.88em; margin-left: 5px; vertical-align: middle;">Port {port} is taken. Please choose another port.</span>'
        button_html = f'<button type="button" class="tb-btn-start tb-btn-disabled" style="background-color: #cbd5e1 !important; color: #64748b !important; cursor: not-allowed !important; pointer-events: none !important;" disabled>Start TensorBoard on Node</button>'
        port_disabled_attr = ""
        port_style = "width: 80px; padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.88em;"
    else:
        status_msg = "It may take up to 2 minutes to launch Tensorboard" if tb_start_trigger else ""
        if status_msg:
            status_msg = f'<span class="tb-status-msg" style="font-size: 0.88em; color: #64748b; margin-left: 5px; vertical-align: middle;">{status_msg}</span>'
        else:
            status_msg = '<span class="tb-status-msg" style="font-size: 0.88em; color: #64748b; margin-left: 5px; vertical-align: middle;"></span>'
        button_html = f"""<button type="button" class="tb-btn-start" onclick="this.disabled = true; this.innerHTML = '<span class=\\'tb-spinner\\'></span> Starting TensorBoard...'; this.style.opacity = '0.7'; this.style.cursor = 'not-allowed'; var portInput = this.parentElement.querySelector('.tb-port-input'); if (portInput) {{ portInput.disabled = true; portInput.style.opacity = '0.7'; portInput.style.cursor = 'not-allowed'; window.__tb_launching_port = portInput.value; }} var textSpan = this.parentElement.querySelector('.tb-status-msg'); if (textSpan) {{ textSpan.textContent = 'It may take up to 2 minutes to launch Tensorboard'; }} var input = document.querySelector('input[name=\\'tbStartTrigger\\']'); console.log('TensorBoard button clicked. Input element:', input); if (input) {{ var val = Date.now().toString(); var lastVal = input.value; var setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; if (setter) {{ setter.call(input, val); }} else {{ input.value = val; }} var tracker = input._valueTracker; if (tracker) {{ tracker.setValue(lastVal); }} input.dispatchEvent(new Event('input', {{ bubbles: true }})); input.dispatchEvent(new Event('change', {{ bubbles: true }})); input.dispatchEvent(new Event('blur', {{ bubbles: true }})); console.log('Dispatched input/change/blur events with value:', val); }}">Start TensorBoard on Node</button>"""
        port_disabled_attr = ""
        port_style = "width: 80px; padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.88em;"

    step2_content = f"""
    <div style="margin-bottom: 12px; color: #ea580c; font-weight: 600;">TensorBoard server is not running on node <code>{esc_node}</code> port <code class="tb-port-display">{port}</code>.</div>
    
    <div style="margin-bottom: 16px; display: inline-flex; align-items: center; gap: 10px; flex-wrap: wrap;">
        {button_html}
        <label style="font-weight: 600; color: #475569; font-size: 0.88em; margin: 0; margin-left: 5px;">Port:</label>
        <input type="number" class="tb-port-input" value="{port}" min="1024" max="65535" style="{port_style}" {port_disabled_attr} onchange="var hiddenPort = document.querySelector('input[name=\\'tbPort\\']'); if (hiddenPort) {{ var val = this.value; var lastVal = hiddenPort.value; var setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; if (setter) {{ setter.call(hiddenPort, val); }} else {{ hiddenPort.value = val; }} var tracker = hiddenPort._valueTracker; if (tracker) {{ tracker.setValue(lastVal); }} hiddenPort.dispatchEvent(new Event('input', {{ bubbles: true }})); hiddenPort.dispatchEvent(new Event('change', {{ bubbles: true }})); hiddenPort.dispatchEvent(new Event('blur', {{ bubbles: true }})); }}" />
        {status_msg}
    </div>
    
    <details class="tb-details">
        <summary class="tb-details-summary">Manual Start Method</summary>
        <div class="tb-details-content">
            <div style="margin-bottom: 8px;">To start the TensorBoard server manually, open a terminal on the cluster and follow these steps:</div>
            <ol style="margin: 0; padding-left: 20px; line-height: 1.6;">
                <li>SSH into the target compute node:
                    <div class="tb-code-box">ssh {esc_node}</div>
                </li>
                <li style="margin-top: 8px;">Run the command to launch TensorBoard:
                    <div class="tb-code-box">{html.escape(tb_cmd)}</div>
                </li>
            </ol>
        </div>
    </details>
    """

print(f"""
<style>
  .tb-container {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px;
    margin-top: 15px;
  }}
  .tb-header {{
    font-size: 1.05em;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 16px;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 8px;
  }}
  .tb-step-item {{
    margin-bottom: 18px;
  }}
  .tb-step-item:last-child {{
    margin-bottom: 0;
  }}
  .tb-step-num {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    background-color: #003c71;
    color: #ffffff;
    border-radius: 50%;
    font-size: 0.8em;
    font-weight: 700;
    margin-right: 8px;
  }}
  .tb-step-title {{
    font-size: 0.9em;
    font-weight: 600;
    color: #1e293b;
    display: inline-block;
    vertical-align: middle;
  }}
  .tb-step-content {{
    margin-top: 6px;
    margin-left: 30px;
    font-size: 0.88em;
  }}
  .tb-code-box {{
    background-color: #0f172a;
    color: #f1f5f9;
    border-radius: 6px;
    padding: 10px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.85em;
    line-height: 1.45;
    word-break: break-all;
    white-space: pre-wrap;
    margin-top: 6px;
  }}
  .tb-btn-primary {{
    display: inline-block;
    background-color: #10b981;
    color: #ffffff !important;
    font-weight: 600;
    text-decoration: none !important;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 0.88em;
    transition: background-color 0.2s;
    border: none;
  }}
  .tb-btn-primary:hover {{
    background-color: #059669;
  }}
  .tb-btn-disabled {{
    background-color: #cbd5e1 !important;
    color: #64748b !important;
    cursor: not-allowed !important;
    pointer-events: none !important;
  }}
  .tb-btn-secondary {{
    display: inline-block;
    background-color: #64748b;
    color: #ffffff !important;
    font-weight: 600;
    text-decoration: none !important;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 0.88em;
    transition: background-color 0.2s, transform 0.1s;
    border: none;
    cursor: pointer;
  }}
  .tb-btn-secondary:hover {{
    background-color: #475569;
  }}
  .tb-btn-secondary:active {{
    transform: scale(0.98);
  }}
  /* Hide spinner arrows for number input */
  .tb-port-input::-webkit-outer-spin-button,
  .tb-port-input::-webkit-inner-spin-button {{
    -webkit-appearance: none;
    margin: 0;
  }}
  .tb-port-input {{
    -moz-appearance: textfield;
  }}
  .tb-btn-start {{
    display: inline-block;
    background-color: #003c71;
    color: #ffffff !important;
    font-weight: 600;
    text-decoration: none !important;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 0.88em;
    transition: background-color 0.2s, transform 0.1s;
    border: none;
    cursor: pointer;
  }}
  .tb-btn-start:hover {{
    background-color: #002850;
  }}
  .tb-btn-start:active {{
    transform: scale(0.98);
  }}
  .tb-spinner {{
    display: inline-block;
    width: 12px;
    height: 12px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    border-top-color: #fff;
    animation: tb-spin 1s ease-in-out infinite;
    margin-right: 6px;
    vertical-align: middle;
  }}
  @keyframes tb-spin {{
    to {{ transform: rotate(360deg); }}
  }}
  input[name="tbStartTrigger"] {{
    display: none !important;
  }}
  input[name="customLogdir"] {{
    display: none !important;
  }}
  input[name="tbPort"] {{
    display: none !important;
  }}
  input[name="tbStopTrigger"] {{
    display: none !important;
  }}
  .tb-tip {{
    font-size: 0.82em;
    color: #64748b;
    margin-top: 4px;
    line-height: 1.4;
  }}
  .tb-details {{
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    background-color: #f8fafc;
    margin-top: 14px;
  }}
  .tb-details-summary {{
    padding: 10px 12px;
    font-weight: 600;
    color: #475569;
    cursor: pointer;
    outline: none;
    user-select: none;
    font-size: 0.9em;
  }}
  .tb-details-summary:hover {{
    color: #0f172a;
    background-color: #f1f5f9;
    border-radius: 5px;
  }}
  .tb-details[open] .tb-details-summary {{
    border-bottom: 1px solid #e2e8f0;
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
  }}
  .tb-details-content {{
    padding: 12px;
    color: #334155;
    font-size: 0.88em;
  }}
  .ckpt-box {{
    display: inline-block;
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 12px;
    font-family: monospace;
    margin-top: 5px;
    margin-bottom: 10px;
  }}
  .ckpt-box a {{
    color: #003c71;
    text-decoration: none;
    font-weight: 600;
  }}
  .ckpt-box a:hover {{
    text-decoration: underline;
  }}
</style>

<div class="ckpt-box">{link}</div>

<div class="tb-container">
  <div class="tb-header">TensorBoard Setup Procedure</div>
  
  <!-- Step 1 -->
  <div class="tb-step-item">
    <div>
      <span class="tb-step-num">1</span>
      <span class="tb-step-title">Wait for the lightning logs files to be generated</span>
    </div>
    <div class="tb-step-content">
      {step1_status}
      <div style="margin-top: 8px;">
        <button type="button" class="tb-btn-secondary" onclick="var input = document.querySelector('input[name=\\'customLogdir\\']'); if (input) {{ var container = input.closest('.input-group') || input.closest('.form-group') || input.parentElement; var btn = container ? container.querySelector('button') : null; if (!btn) {{ btn = input.nextElementSibling; while (btn && btn.tagName !== 'BUTTON') {{ btn = btn.nextElementSibling; }} }} if (btn) {{ btn.click(); }} else {{ console.error('Picker button not found'); }} }}">Change Log directory</button>
      </div>
    </div>
  </div>
  
  <!-- Step 2 -->
  <div class="tb-step-item">
    <div>
      <span class="tb-step-num">2</span>
      <span class="tb-step-title">Run command to start TensorBoard server on cluster node <code>{esc_node}</code></span>
    </div>
    <div class="tb-step-content">
      {step2_content}
    </div>
  </div>
  
  <!-- Step 3 -->
  <div class="tb-step-item">
    <div>
      <span class="tb-step-num">3</span>
      <span class="tb-step-title">Access the TensorBoard Web UI</span>
    </div>
    <div class="tb-step-content">
      <div style="margin-bottom: 8px;">Once the server is running, click the button below to open the TensorBoard Web UI directly in your browser:</div>
      {tb_button_html}
    </div>
  </div>
</div>

<img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" onload="var input = document.querySelector('input[name=\\'tbStartTrigger\\']'); console.log('Auto-clearing check. Input element:', input, 'Value:', input ? input.value : null); if (input && input.value !== '') {{ var val = ''; var lastVal = input.value; var setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; if (setter) {{ setter.call(input, val); }} else {{ input.value = val; }} var tracker = input._valueTracker; if (tracker) {{ tracker.setValue(lastVal); }} input.dispatchEvent(new Event('input', {{ bubbles: true }})); input.dispatchEvent(new Event('change', {{ bubbles: true }})); input.dispatchEvent(new Event('blur', {{ bubbles: true }})); console.log('Auto-cleared tbStartTrigger.'); }} var stopInput = document.querySelector('input[name=\\'tbStopTrigger\\']'); if (stopInput && stopInput.value !== '') {{ var val = ''; var lastVal = stopInput.value; var setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; if (setter) {{ setter.call(stopInput, val); }} else {{ stopInput.value = val; }} var tracker = stopInput._valueTracker; if (tracker) {{ tracker.setValue(lastVal); }} stopInput.dispatchEvent(new Event('input', {{ bubbles: true }})); stopInput.dispatchEvent(new Event('change', {{ bubbles: true }})); stopInput.dispatchEvent(new Event('blur', {{ bubbles: true }})); console.log('Auto-cleared tbStopTrigger.'); }} var customInput = document.querySelector('input[name=\\'customLogdir\\']'); if (customInput) {{ var container = customInput.closest('.form-group') || customInput.closest('.form-row') || customInput.closest('[class*=\\'form-group\\']') || customInput.parentElement; if (container) {{ container.style.setProperty('display', 'none', 'important'); }} }} var portInput = document.querySelector('input[name=\\'tbPort\\']'); if (portInput) {{ var container = portInput.closest('.form-group') || portInput.closest('.form-row') || portInput.closest('[class*=\\'form-group\\']') || portInput.parentElement; if (container) {{ container.style.setProperty('display', 'none', 'important'); }} }} var headers = document.querySelectorAll('h1, h2, h3, h4, h5, h6, .card-header, .section-title, label, legend, .form-section-title'); for (var i = 0; i < headers.length; i++) {{ if (headers[i].textContent.trim().indexOf('TensorBoard — Settings') !== -1) {{ var section = headers[i].closest('.card') || headers[i].closest('.section') || headers[i].closest('.form-section') || headers[i].closest('.rowContainer') || headers[i].closest('[class*=\\'rowContainer\\']') || headers[i].parentElement; if (section) {{ section.style.setProperty('display', 'none', 'important'); }} }} }} var tbContainer = this.previousElementSibling; if (tbContainer && tbContainer.classList.contains('tb-container')) {{ var pInput = tbContainer.querySelector('.tb-port-input'); var startBtn = tbContainer.querySelector('.tb-btn-start'); var statusMsg = tbContainer.querySelector('.tb-status-msg'); var portDisplay = tbContainer.querySelector('.tb-port-display'); if (startBtn && pInput) {{ var isStartBtn = startBtn.textContent.trim().indexOf('Start') !== -1 || startBtn.textContent.trim().indexOf('Starting') !== -1; if (isStartBtn) {{ var currentPort = pInput.value; if (window.__tb_launching_port === currentPort) {{ startBtn.disabled = true; startBtn.innerHTML = '<span class=\\'tb-spinner\\'></span> Starting TensorBoard...'; startBtn.style.opacity = '0.7'; startBtn.style.cursor = 'not-allowed'; startBtn.classList.add('tb-btn-disabled'); pInput.disabled = true; pInput.style.opacity = '0.7'; pInput.style.cursor = 'not-allowed'; if (statusMsg) {{ statusMsg.innerHTML = '<span class=\\'tb-status-msg\\' style=\\'font-size: 0.88em; color: #64748b; margin-left: 5px; vertical-align: middle;\\'>It may take up to 2 minutes to launch Tensorboard</span>'; }} }} }} else {{ window.__tb_launching_port = null; }} }} if (pInput && startBtn) {{ var updateUI = function () {{ var val = parseInt(pInput.value, 10); if (portDisplay) {{ portDisplay.textContent = pInput.value; }} if (isNaN(val) || val < 1024 || val > 65535) {{ startBtn.disabled = true; startBtn.style.opacity = '0.7'; startBtn.style.cursor = 'not-allowed'; startBtn.classList.add('tb-btn-disabled'); if (statusMsg) {{ statusMsg.innerHTML = '<span class=\\'tb-status-msg\\' style=\\'color: #dc3545; font-weight: 600; font-size: 0.88em; margin-left: 5px; vertical-align: middle;\\'>Port must be between 1024 and 65535.</span>'; }} }} else {{ if (window.__tb_launching_port !== pInput.value) {{ startBtn.disabled = false; startBtn.style.opacity = ''; startBtn.style.cursor = ''; startBtn.classList.remove('tb-btn-disabled'); if (statusMsg) {{ statusMsg.textContent = ''; }} }} }} }}; pInput.addEventListener('input', updateUI); pInput.addEventListener('change', updateUI); }} }}" style="display: none;" />
""")
EOF
