#!/bin/bash
# Whether the GPU stage can be re-run (Manage "RE-RUN PREDICTION" card):
#   failed : CPU job completed, GPU job ended without success (failed, timeout, cancelled, out of memory)
#   done   : CPU and GPU job completed (re-run with other settings)
#   (empty): one-job workflow, CPU stage not completed, GPU job still pending/running, or the intermediate
#            files the GPU stage needs were removed (.drona/af_cleaned)
# Param: DRONA_ID (workflow record).
python3 - <<'PYEOF'
import json, os, subprocess
wf = os.environ.get("DRONA_ID", "").strip()
db = os.path.join(os.environ.get("DRONA_RUNTIME_DIR", ""), "db_access", "drona_db_retriever.py")
try:
    rec = json.loads(subprocess.check_output(["python3", db, "-i", wf], universal_newlines=True,
                                             stderr=subprocess.DEVNULL))
    ids = [str(j["id"]) for j in (rec.get("runtime_meta") or {}).get("jobinfo", [])]
    if rec.get("location") and os.path.exists(os.path.join(rec["location"], ".drona", "af_cleaned")):
        ids = []
except Exception:
    ids = []
if len(ids) < 2:
    raise SystemExit(0)
try:
    out = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", ids[0] + "," + ids[-1], "--format=JobID,State"],
                                  universal_newlines=True, stderr=subprocess.DEVNULL)
except Exception:
    raise SystemExit(0)
state = {}
for line in out.splitlines():
    p = line.split("|")
    if len(p) >= 2:
        state[p[0]] = p[1].split()[0]
cpu, gpu = state.get(ids[0], ""), state.get(ids[-1], "")
if cpu != "COMPLETED":
    raise SystemExit(0)
if gpu == "COMPLETED":
    print("done")
elif gpu in ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"):
    print("failed")
PYEOF
