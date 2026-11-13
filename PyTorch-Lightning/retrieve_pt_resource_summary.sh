#!/bin/bash
# One-line summary of the training processes the job will start, from the form values.
# Mirrors retrieve_pt_slurm_params: one process (task) per GPU on each node (one per node
# without a GPU), DataLoader workers + 1 CPUs per process. The cluster module may still cap
# the GPUs per node at submit time.
# Params: NODES, WORKERS, GPU, NUMGPU, BATCH

int_or() { [[ "$1" =~ ^[0-9]+$ ]] && echo "$1" || echo "$2"; }

nodes=$(int_or "$NODES" 1);   [ "$nodes" -lt 1 ] && nodes=1
workers=$(int_or "$WORKERS" 0)
numgpu=$(int_or "$NUMGPU" 1); [ "$numgpu" -lt 1 ] && numgpu=1
batch=$(int_or "$BATCH" "")
gpu=$(echo "${GPU:-}" | tr -d '"[] ' | tr '[:lower:]' '[:upper:]')

STYLE_INFO="color:#0c5460;background:#d1ecf1;border:1px solid #bee5eb;border-radius:4px;padding:4px 8px"
cpus=$((workers + 1))
s() { [ "$1" -eq 1 ] && echo "$2" || echo "${3:-$2s}"; }

if [ -z "$gpu" ] || [ "$gpu" = "NONE" ]; then
    procs=$nodes
    line="$procs training $(s $procs process processes) ($nodes $(s $nodes node) &times; 1, no GPU) &middot; $cpus $(s $cpus CPU) each"
else
    procs=$((nodes * numgpu))
    line="$procs training $(s $procs process processes) ($nodes $(s $nodes node) &times; $numgpu $gpu, 1 per GPU) &middot; $cpus $(s $cpus CPU) each"
fi
[ -n "$batch" ] && [ "$procs" -gt 1 ] && line="$line &middot; effective batch $((batch * procs))"

echo "<div style=\"$STYLE_INFO;font-size:0.9em\">&#9432; $line</div>"
if [ -n "$gpu" ] && [ "$gpu" != "NONE" ] && [ "$workers" -eq 0 ]; then
    echo "<div style=\"font-size:0.85em;color:#6c757d;margin-top:2px\">With 0 DataLoader workers data loads in the main process; 2&ndash;8 workers usually keep a GPU busier.</div>"
fi
