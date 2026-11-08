#!/usr/bin/env python3
"""Re-run the GPU stage of a two-stage AlphaFold workflow (AF3, AF2 full_dbs) on the finished CPU output.

Called from the Manage driver (see utils.rerun_gpu_stage). Steps:
  1. back up the GPU job file to <file>.<old job id>
  2. apply the overrides (walltime, partition/GPU line, AF3 recycles + model directory)
  3. sbatch the job file (no dependency: the CPU stage has finished)
  4. workflow record: the GPU entry of runtime_meta.jobinfo is replaced by the new job id,
     the old id is kept in runtime_meta.replaced_jobs

--register JOBID skips 1-3 and only records an already submitted job (e.g. one submitted by hand).
Python 3.6 compatible, stdlib only.
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys

DB = os.path.join(os.environ.get("DRONA_RUNTIME_DIR", ""), "db_access", "drona_db_retriever.py")
SBATCH = "/sw/local/bin/sbatch" if os.path.exists("/sw/local/bin/sbatch") else "sbatch"


def fail(msg):
    print("ERROR: " + msg)
    sys.exit(1)


def read_record(wf_id):
    out = subprocess.check_output(["python3", DB, "-i", wf_id], universal_newlines=True)
    rec = json.loads(out)
    meta = rec.get("runtime_meta") or {}
    if isinstance(meta, str):
        meta = json.loads(meta) if meta.strip() else {}
    return meta


def write_record(wf_id, meta):
    subprocess.check_call(["python3", DB, "--edit", "-i", wf_id, "--runtime-meta", json.dumps(meta)],
                          stdout=subprocess.DEVNULL)


def edit_job_file(path, time, partition, recycles, modeldir, seeds=""):
    with open(path) as f:
        text = f.read()
    changes = []

    def sub(pattern, repl, what):
        nonlocal text
        new, n = re.subn(pattern, lambda m: repl, text, flags=re.M)
        if n == 0:
            print("WARNING: could not set %s (line not found in %s)" % (what, os.path.basename(path)))
        else:
            text = new
            changes.append(what)

    if time:
        sub(r"^#SBATCH --time=\S+.*$", "#SBATCH --time=" + time, "walltime " + time)
    if partition:
        sub(r"^#SBATCH --partition=.*$", partition, partition.replace("#SBATCH ", ""))
    if recycles:
        sub(r"--num_recycles=\d+", "--num_recycles=" + recycles, "recycles " + recycles)
    if seeds:
        if re.search(r"--num_seeds=\d+", text):
            sub(r"--num_seeds=\d+", "--num_seeds=" + seeds, "seeds " + seeds)
        else:
            sub(r"--num_recycles=\d+", "--num_recycles=" + (re.search(r"--num_recycles=(\d+)", text) or [0, "10"])[1]
                + " --num_seeds=" + seeds, "seeds " + seeds)
    if modeldir:
        sub(r"^AF3_MODEL_PARAMETERS_DIR=.*$", "AF3_MODEL_PARAMETERS_DIR=" + modeldir, "model directory " + modeldir)
    with open(path, "w") as f:
        f.write(text)
    return changes


def job_file_label(path):
    """Stage label from a '#DRONA label: ...' line (as read by drona_wf_driver_sbatch)."""
    try:
        with open(path) as f:
            for line in f:
                m = re.match(r"#DRONA\s+label:\s*(.*)", line)
                if m:
                    return m.group(1).strip()
    except OSError:
        pass
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", required=True, help="drona_id of the workflow")
    ap.add_argument("--jobfile", required=True, help="GPU job file in the current directory")
    ap.add_argument("--time", default="")
    ap.add_argument("--partition", default="", help="full '#SBATCH --partition=... --gres=...' line")
    ap.add_argument("--recycles", default="")
    ap.add_argument("--modeldir", default="")
    ap.add_argument("--seeds", default="", help="AF3 --num_seeds (input must list a single seed)")
    ap.add_argument("--register", default="", help="record this already submitted job instead of submitting")
    a = ap.parse_args()

    meta = read_record(a.workflow)
    jobs = meta.get("jobinfo") or []
    if len(jobs) < 2:
        fail("workflow %s has %d job(s); re-running the GPU stage needs a CPU job and a GPU job." % (a.workflow, len(jobs)))
    old_id = str(jobs[-1]["id"])

    if a.register:
        new_id = a.register
        print("Registering job %s as the GPU stage (replaces %s)." % (new_id, old_id))
    else:
        if not os.path.isfile(a.jobfile):
            fail("job file %s not found in %s" % (a.jobfile, os.getcwd()))
        backup = "%s.%s" % (a.jobfile, old_id)
        shutil.copy2(a.jobfile, backup)
        print("Backed up %s -> %s" % (a.jobfile, backup))
        changes = edit_job_file(a.jobfile, a.time, a.partition, a.recycles, a.modeldir, a.seeds)
        print("Changed: " + (", ".join(changes) if changes else "nothing (same settings as before)"))
        out = subprocess.check_output([SBATCH, a.jobfile], universal_newlines=True, stderr=subprocess.STDOUT)
        print(out.strip())
        m = re.search(r"Submitted batch job (\d+)", out)
        if not m:
            fail("could not read the job id from the sbatch output")
        new_id = m.group(1)

    label = jobs[-1].get("label") or job_file_label(a.jobfile)
    jobs[-1] = {"id": int(new_id), "label": label} if label else {"id": int(new_id)}
    meta["jobinfo"] = jobs
    meta.setdefault("replaced_jobs", []).append({
        "id": int(old_id), "replaced_by": int(new_id),
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    write_record(a.workflow, meta)
    print("GPU stage job %s recorded in workflow %s (previous attempt %s kept in its out/error files)."
          % (new_id, a.workflow, old_id))


if __name__ == "__main__":
    main()
