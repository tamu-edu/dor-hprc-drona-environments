#!/usr/bin/env python3
"""Drona R worker monitor: one copy per node (started with srun from the job script).

Every INTERVAL seconds it finds this job's R processes on the node (all R processes are
named "R": the main Rscript, SOCK/MPI workers and forked mclapply children), measures
their CPU use since the previous sample and their thread counts, and replaces
drona_monitoring/r_workers_<node>.json with the latest sample plus running peaks.
The R Run Check card on the Manage page reads these files. Only reads /proc; it never
touches the R processes.
"""
import json
import os
import signal
import socket
import sys
import time

INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 15
BUSY_CPU = 50.0          # a process using at least this % of a core counts as busy
LOG_DIR = "drona_monitoring"
JOB = os.environ.get("SLURM_JOB_ID", "")
NODE = os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0]
UID = os.getuid()
TICK = os.sysconf("SC_CLK_TCK")
PAGE_MB = os.sysconf("SC_PAGE_SIZE") / 1048576.0


def env_int(name, default=0):
    value = os.environ.get(name, "")
    return int(value) if value.isdigit() else default


def in_job(pid):
    if not JOB:
        return True
    try:
        with open("/proc/%s/cgroup" % pid) as f:
            text = f.read()
    except OSError:
        return False
    return ("/job_%s/" % JOB) in text or text.rstrip().endswith("/job_%s" % JOB)


def read_proc(pid):
    """(ppid, cpu ticks, threads, rss MB, is_main) for an R process of this user and job, else None."""
    try:
        if os.stat("/proc/%s" % pid).st_uid != UID:
            return None
        with open("/proc/%s/comm" % pid) as f:
            if f.read().strip() != "R":
                return None
        with open("/proc/%s/stat" % pid) as f:
            stat = f.read()
        with open("/proc/%s/cmdline" % pid, "rb") as f:
            cmdline = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return None
    if not in_job(pid):
        return None
    fields = stat[stat.rindex(")") + 2:].split()   # fields after "pid (comm) "
    ppid = int(fields[1])
    ticks = int(fields[11]) + int(fields[12])        # utime + stime
    threads = int(fields[17])
    rss_mb = int(fields[21]) * PAGE_MB
    # SOCK workers run "R ... -e ...workRSOCK", Rmpi workers are started by mpirun; the main
    # script runs "R ... --file=<script>". Forked children copy the main's command line.
    is_main = "--file=" in cmdline and "workRSOCK" not in cmdline
    return ppid, ticks, threads, rss_mb, is_main


def sample(prev, dt):
    procs = {}
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            info = read_proc(pid)
            if info:
                procs[int(pid)] = info
    out = []
    for pid, (ppid, ticks, threads, rss_mb, is_main) in procs.items():
        main = is_main and ppid not in procs        # a forked child's parent is also R
        cpu = None
        if pid in prev and dt > 0:
            cpu = max(0.0, (ticks - prev[pid]) * 100.0 / TICK / dt)
        out.append({"pid": pid, "role": "main" if main else "worker", "cpu": None if cpu is None else round(cpu, 1),
                    "threads": threads, "rss_mb": round(rss_mb, 1)})
    out.sort(key=lambda p: (p["role"] != "main", p["pid"]))
    return out, {pid: info[1] for pid, info in procs.items()}


def write(state):
    path = os.path.join(LOG_DIR, "r_workers_%s.json" % NODE)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)


def main():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except OSError:
        return
    state = {
        "node": NODE, "job": JOB, "cpus": env_int("SLURM_CPUS_ON_NODE", os.cpu_count() or 1),
        "nnodes": env_int("SLURM_NNODES", 1), "workers": env_int("DRONA_R_WORKERS"), "threads": env_int("DRONA_R_THREADS", 1),
        "parallel": os.environ.get("DRONA_R_PARALLEL", ""), "interval": INTERVAL,
        "started": int(time.time()), "time": 0, "samples": 0, "procs": [],
        "peak_procs": 0, "peak_workers": 0, "peak_busy_workers": 0, "peak_threads": 0,
        "busy_worker_samples": 0, "overload_samples": 0, "final": False,
    }
    stop = {"now": False}

    def on_term(signum, frame):
        stop["now"] = True
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    prev, last = {}, time.time()
    while True:
        now = time.time()
        procs, prev = sample(prev, now - last)
        last = now
        measured = [p for p in procs if p["cpu"] is not None]
        workers = [p for p in procs if p["role"] == "worker"]
        busy_workers = [p for p in workers if (p["cpu"] or 0) >= BUSY_CPU]
        threads = sum(p["threads"] for p in procs)
        state["time"] = int(now)
        state["procs"] = procs
        if measured:
            state["samples"] += 1
            state["peak_procs"] = max(state["peak_procs"], len(procs))
            state["peak_workers"] = max(state["peak_workers"], len(workers))
            state["peak_busy_workers"] = max(state["peak_busy_workers"], len(busy_workers))
            state["peak_threads"] = max(state["peak_threads"], threads)
            state["busy_worker_samples"] += 1 if busy_workers else 0
            # more runnable threads than cores: count only threads of busy processes
            busy_threads = sum(p["threads"] for p in measured if p["cpu"] >= BUSY_CPU)
            state["overload_samples"] += 1 if busy_threads > state["cpus"] else 0
        state["final"] = stop["now"]
        try:
            write(state)
        except OSError:
            pass
        if stop["now"]:
            return
        for _ in range(INTERVAL * 2):
            if stop["now"]:
                break
            time.sleep(0.5)


if __name__ == "__main__":
    main()
