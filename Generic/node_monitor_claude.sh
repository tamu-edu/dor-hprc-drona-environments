#!/bin/bash
# Usage: ./node_monitor.sh <nodename> <jobid> [partition]


SRUN_ARGS="--nodelist=${NODE} --nodes=1 --ntasks=1 --oversubscribe -t 0:30"

# Collect all data in a single srun call to minimize overhead.
# JOBID is passed as $1 to the remote bash script to avoid heredoc quoting issues.
RAW=$(srun $SRUN_ARGS bash -s "$JOBID" <<'REMOTE'
JOBID="$1"

# CPU usage from /proc/stat (two samples, 200ms apart)
read -ra cpu1 < /proc/stat
sleep 0.2
read -ra cpu2 < /proc/stat

user=$(( cpu2[1] - cpu1[1] ))
nice=$(( cpu2[2] - cpu1[2] ))
sys=$(( cpu2[3] - cpu1[3] ))
idle=$(( cpu2[4] - cpu1[4] ))
iowait=$(( cpu2[5] - cpu1[5] ))
total=$(( user + nice + sys + idle + iowait ))
cpu_pct=$(( total > 0 ? (total - idle) * 100 / total : 0 ))

# Memory from /proc/meminfo
mem_total=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
mem_free=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
mem_used=$(( mem_total - mem_free ))
mem_pct=$(( mem_total > 0 ? mem_used * 100 / mem_total : 0 ))

# Load average
read load1 load5 load15 _ _ < /proc/loadavg

# CPU count
ncpus=$(nproc)

# --- Collect PIDs belonging to this SLURM job via cgroups ---
# Try cgroup v2 first, then fall back to cgroup v1.
job_pids=""

# cgroup v2: unified hierarchy
cg2=$(find /sys/fs/cgroup -maxdepth 6 -name "cgroup.procs" -path "*/job_${JOBID}/*" 2>/dev/null)
if [[ -n "$cg2" ]]; then
    job_pids=$(cat $cg2 2>/dev/null | sort -u | tr '\n' ',')
fi

# cgroup v1: memory or cpu subsystem
if [[ -z "$job_pids" ]]; then
    cg1=$(find /sys/fs/cgroup/{memory,cpu} -maxdepth 6 \( -name "cgroup.procs" -o -name "tasks" \) \
          -path "*/job_${JOBID}/*" 2>/dev/null | head -1)
    [[ -n "$cg1" ]] && job_pids=$(cat "$cg1" 2>/dev/null | sort -u | tr '\n' ',')
fi

job_pids="${job_pids%,}"  # strip trailing comma

# Build ps output filtered to job PIDs only
if [[ -n "$job_pids" ]]; then
    procs=$(ps -p "$job_pids" -o user,pid,pcpu,pmem,stat,comm --no-headers --sort=-pcpu 2>/dev/null | \
      awk '{printf "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n",
            $1,$2,$3,$4,$5,$6}')
else
    procs='<tr><td colspan="6" style="color:#888;text-align:center">No cgroup found for job '"$JOBID"'</td></tr>'
fi

echo "CPU_PCT=${cpu_pct}"
echo "MEM_TOTAL_KB=${mem_total}"
echo "MEM_USED_KB=${mem_used}"
echo "MEM_PCT=${mem_pct}"
echo "LOAD=${load1} ${load5} ${load15}"
echo "NCPUS=${ncpus}"
echo "PROCS_START"
echo "${procs}"
echo "PROCS_END"
REMOTE
)

# Parse collected data
CPU_PCT=$(echo "$RAW"   | awk -F= '/^CPU_PCT=/{print $2}')
MEM_TOTAL=$(echo "$RAW" | awk -F= '/^MEM_TOTAL_KB=/{printf "%.1f", $2/1024/1024}')
MEM_USED=$(echo "$RAW"  | awk -F= '/^MEM_USED_KB=/{printf "%.1f", $2/1024/1024}')
MEM_PCT=$(echo "$RAW"   | awk -F= '/^MEM_PCT=/{print $2}')
LOAD=$(echo "$RAW"      | awk -F= '/^LOAD=/{print $2}')
NCPUS=$(echo "$RAW"     | awk -F= '/^NCPUS=/{print $2}')
PROCS=$(echo "$RAW"     | sed -n '/^PROCS_START/,/^PROCS_END/{/^PROCS_START/d;/^PROCS_END/d;p}')
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Color thresholds helper (returns a CSS class name)
cpu_class="ok";  [[ $CPU_PCT -ge 70 ]] && cpu_class="warn"; [[ $CPU_PCT -ge 90 ]] && cpu_class="crit"
mem_class="ok";  [[ $MEM_PCT -ge 70 ]] && mem_class="warn"; [[ $MEM_PCT -ge 90 ]] && mem_class="crit"

OUTFILE = tmp.$JOB

cat > "$OUTFILE" <<HTML
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Node Report: ${NODE} — Job ${JOBID}</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f9; color: #222; padding: 2rem; }
    h1   { font-size: 1.6rem; margin-bottom: 0.25rem; }
    .meta { color: #666; font-size: 0.85rem; margin-bottom: 2rem; }

    .cards { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 2rem; }
    .card  { background: #fff; border-radius: 8px; padding: 1.25rem 1.5rem;
              box-shadow: 0 1px 4px rgba(0,0,0,.1); flex: 1; min-width: 180px; }
    .card h2 { font-size: 0.75rem; text-transform: uppercase; letter-spacing: .05em;
                color: #888; margin-bottom: .5rem; }
    .card .val { font-size: 2rem; font-weight: 700; line-height: 1; }
    .card .sub { font-size: 0.8rem; color: #666; margin-top: .35rem; }

    .bar-wrap { background: #e8eaed; border-radius: 4px; height: 8px; margin-top: .6rem; overflow: hidden; }
    .bar      { height: 100%; border-radius: 4px; transition: width .3s; }
    .ok   .bar { background: #34a853; }
    .warn .bar { background: #fbbc04; }
    .crit .bar { background: #ea4335; }
    .ok   .val { color: #34a853; }
    .warn .val { color: #f59300; }
    .crit .val { color: #ea4335; }

    table  { width: 100%; border-collapse: collapse; background: #fff;
              border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.1); }
    thead  { background: #3b4a6b; color: #fff; }
    th, td { padding: .6rem 1rem; text-align: left; font-size: .85rem; }
    tbody tr:nth-child(even) { background: #f8f9fb; }
    tbody tr:hover { background: #eef2ff; }
  </style>
</head>
<body>
  <h1>Node: ${NODE} &nbsp;<span style="font-weight:400;color:#666">Job ${JOBID}</span></h1>
  <p class="meta">Generated: ${TIMESTAMP} &nbsp;|&nbsp; CPUs: ${NCPUS} &nbsp;|&nbsp; Load avg: ${LOAD}</p>

  <div class="cards">
    <div class="card ${cpu_class}">
      <h2>CPU Utilization</h2>
      <div class="val">${CPU_PCT}%</div>
      <div class="bar-wrap"><div class="bar" style="width:${CPU_PCT}%"></div></div>
      <div class="sub">${NCPUS} logical CPUs</div>
    </div>

    <div class="card ${mem_class}">
      <h2>Memory Used</h2>
      <div class="val">${MEM_PCT}%</div>
      <div class="bar-wrap"><div class="bar" style="width:${MEM_PCT}%"></div></div>
      <div class="sub">${MEM_USED} GB / ${MEM_TOTAL} GB</div>
    </div>

    <div class="card ok">
      <h2>Load Average</h2>
      <div class="val" style="color:#3b4a6b">$(echo $LOAD | awk '{print $1}')</div>
      <div class="sub">1 min &nbsp;|&nbsp; 5 min: $(echo $LOAD | awk '{print $2}') &nbsp;|&nbsp; 15 min: $(echo $LOAD | awk '{print $3}')</div>
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th>User</th><th>PID</th><th>%CPU</th><th>%MEM</th><th>State</th><th>Command</th>
      </tr>
    </thead>
    <tbody>
${PROCS}
    </tbody>
  </table>
</body>
</html>
HTML
 
cat $OUTFILE
rm $OUTFILE

