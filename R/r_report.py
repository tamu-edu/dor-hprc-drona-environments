#!/usr/bin/env python3
"""R Run Check card for the Manage page (HTML on stdout).

  r_report.py <workflow dir> <job id> [job status]

1. Workers: from drona_monitoring/r_workers_<node>.json (written in the job by
   .drona/drona_r_monitor.py): R processes found vs. workers requested, whether the
   workers did any work, and whether busy threads outnumber the cores.
2. Problems: the common R failures found in error.<job id> / out.<job id>, explained.
3. Next run: after the job ends, cores/memory/time requested vs. used (one sacct call, cached in
   drona_monitoring/r_usage_<job id>.json, so the web server does no repeated work) and what to request next time.
"""
import glob
import html
import json
import os
import math
import re
import subprocess
import sys
import time

e = html.escape

STYLE = """
<style>
#dc-r-report { --t:#1e293b; --m:#64748b; --b:#e2e8f0; --d:#f1f5f9; color:var(--t);
  font-family:'Inter',system-ui,sans-serif; font-size:13px; width:100%; box-sizing:border-box; }
#dc-r-report h4 { font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--m);
  margin:14px 0 6px; font-weight:600; }
#dc-r-report h4:first-child { margin-top:0; }
#dc-r-report table { width:100%; border-collapse:collapse; }
#dc-r-report th { text-align:left; font-size:11px; color:var(--m); font-weight:600; padding:4px 6px; }
#dc-r-report td { padding:4px 6px; border-top:1px solid var(--d); vertical-align:top; }
#dc-r-report .mono { font-family:'JetBrains Mono',monospace; font-size:12px; }
#dc-r-report pre { font-family:'JetBrains Mono',monospace; font-size:12px; background:var(--d); padding:6px 8px;
  border-radius:6px; margin:4px 0 0; white-space:pre-wrap; word-break:break-word; }
#dc-r-report .hint { padding:6px 9px; border-left:3px solid; margin:5px 0; border-radius:0 6px 6px 0; }
#dc-r-report .hint.warn { border-color:#f59e0b; background:#fffbeb; }
#dc-r-report .hint.bad { border-color:#ef4444; background:#fef2f2; }
#dc-r-report .hint.info { border-color:#94a3b8; background:#f8fafc; }
#dc-r-report .hint b { font-weight:600; }
#dc-r-report .ok { color:#047857; }
#dc-r-report .muted { color:var(--m); font-size:11px; }
</style>
"""

DONE_STATES = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "DONE", "PREEMPTED")


def hint(level, title, text="", code=""):
    body = "<b>%s</b>" % e(title)
    if text:
        body += "<div>%s</div>" % text
    if code:
        body += "<pre>%s</pre>" % e(code)
    return "<div class='hint %s'>%s</div>" % (level, body)


# ---------------------------------------------------------------- workers

def load_nodes(jobdir, jobid):
    nodes = []
    for path in sorted(glob.glob(os.path.join(jobdir, "drona_monitoring", "r_workers_*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if str(data.get("job", "")) == str(jobid):
            nodes.append(data)
    return nodes


def workers_section(nodes, status):
    out = ["<h4>Workers</h4>"]
    if not nodes:
        if status and status.upper() in DONE_STATES:
            out.append("<div class='muted'>No worker data for this job (it ended before the first sample, "
                       "or it was submitted before the R Run Check existed).</div>")
        else:
            out.append("<div class='muted'>Waiting for the first sample; it appears about 30 seconds "
                       "after the job starts.</div>")
        return out

    first = nodes[0]
    parallel = first.get("parallel") or "none"
    requested = int(first.get("workers") or 0)
    running = not (status and status.upper() in DONE_STATES) and not all(n.get("final") for n in nodes)
    samples = max(n.get("samples", 0) for n in nodes)

    out.append("<table><tr><th>Node</th><th>Cores</th><th>R processes now</th><th>Busy now</th>"
               "<th>Threads now</th><th>Peak workers (busy)</th></tr>")
    for n in nodes:
        procs = n.get("procs") or []
        busy = [p for p in procs if (p.get("cpu") or 0) >= 50]
        age = int(time.time()) - int(n.get("time") or 0)
        node = e(n.get("node", "?"))
        if running and age > 3 * int(n.get("interval") or 15) + 30:
            node += " <span class='muted'>(last sample %d min ago)</span>" % (age // 60)
        out.append("<tr><td class='mono'>%s</td><td class='mono'>%s</td><td class='mono'>%d</td><td class='mono'>%d</td>"
                   "<td class='mono'>%d</td><td class='mono'>%d (%d)</td></tr>"
                   % (node, n.get("cpus", "?"), len(procs), len(busy), sum(p.get("threads", 0) for p in procs),
                      n.get("peak_workers", 0), n.get("peak_busy_workers", 0)))
    out.append("</table>")

    peak_workers = sum(n.get("peak_workers", 0) for n in nodes)
    peak_busy = sum(n.get("peak_busy_workers", 0) for n in nodes)
    cores = sum(int(n.get("cpus") or 0) for n in nodes)
    hints = []

    if parallel == "none":
        if peak_workers:
            hints.append(hint("info", "Your script started %d extra R process%s, but the form says NONE."
                              % (peak_workers, "es" if peak_workers != 1 else ""),
                              "If your code runs in parallel, set <i>How your script runs in parallel</i> and #workers on the "
                              "form, so the job gets one core per worker."))
        else:
            out.append("<div class='ok' style='margin-top:6px'>Single R process, as requested.</div>")
    elif samples >= 4:   # about a minute of data before judging the workers
        mpi = parallel == "mpi"
        if peak_workers == 0:
            hints.append(hint("bad", "No R workers were started (%d requested)." % requested,
                              "The script ran on the main R process only, so the other cores sat idle. "
                              "Check that your code creates the cluster "
                              + ("(e.g. <span class='mono'>snow::makeMPIcluster()</span> or Rmpi)" if mpi else
                                 "(e.g. <span class='mono'>cl &lt;- makeCluster(n)</span>)")
                              + " and that the parallel part is actually reached."))
        elif peak_busy == 0:
            hints.append(hint("bad", "%d workers were started but never did any work." % peak_workers,
                              "This usually means the loop runs serially: with foreach, register the cluster "
                              "(<span class='mono'>registerDoParallel(cl)</span>) and use "
                              "<span class='mono'>%dopar%</span>, not <span class='mono'>%do%</span>; or use "
                              "<span class='mono'>parLapply(cl, ...)</span> instead of "
                              "<span class='mono'>lapply</span>."))
        if peak_workers > requested and requested:
            hints.append(hint("warn", "%d R workers for %d requested." % (peak_workers, requested),
                              "More workers than cores slows the job down. The cluster size probably comes from "
                              "<span class='mono'>detectCores()</span>, which counts every core on the node, not "
                              "the ones the job got. Use <span class='mono'>as.integer(Sys.getenv("
                              "\"DRONA_R_WORKERS\"))</span> instead."))
        elif 0 < peak_workers < requested:
            hints.append(hint("warn", "Only %d of %d requested workers were started." % (peak_workers, requested),
                              "The cluster size in your code is smaller than #workers on the form, so some "
                              "requested cores stay idle. Use <span class='mono'>as.integer(Sys.getenv("
                              "\"DRONA_R_WORKERS\"))</span> for the cluster size, or lower #workers."))
        elif peak_busy and peak_busy < peak_workers and not running:
            hints.append(hint("info", "At most %d of %d workers were busy at the same time." % (peak_busy, peak_workers),
                              "Fine for short or uneven tasks; if it stays low, there may be fewer tasks than workers."))
        if not hints:
            out.append("<div class='ok' style='margin-top:6px'>%d of %d workers started and busy.</div>"
                       % (peak_busy, requested))
    else:
        out.append("<div class='muted' style='margin-top:6px'>Checking the workers after about a minute of "
                   "samples.</div>")

    overload = sum(n.get("overload_samples", 0) for n in nodes)
    if overload and samples and overload * 4 >= samples:   # in at least a quarter of the samples
        hints.append(hint("warn", "More busy threads than cores (%d cores)." % cores,
                          "Threads compete for the cores and the job slows down. Common causes: a package that "
                          "starts its own threads in every worker (e.g. data.table, ranger, xgboost; set their "
                          "thread option to 1), or <i>Threads per R process</i> above 1 together with workers, so every "
                          "worker starts several math-library (BLAS) threads."))
    out.extend(hints)
    return out


# ---------------------------------------------------------------- problems

QUOTE = "[‘’'\"]"
CHECKS = [
    # (regex, level, title, explanation); groups go into the title with %s
    (r"there is no package called %s([^‘’'\"]+)%s" % (QUOTE, QUOTE), "bad", "Missing R package: %s",
     "Install it for this R version (start R with the same R_tamu module and run install.packages(\"%s\")), "
     "or add the directory that has it under <i>Additional R_LIBS_USER paths</i>."),
    (r"unable to load shared object %s([^‘’'\"]+)%s" % (QUOTE, QUOTE), "bad", "A package could not be loaded: %s",
     "It was most likely installed with a different R version or toolchain. Reinstall it with the R version "
     "selected for this job, or remove that directory from the R_LIBS_USER paths."),
    (r"package %s([^‘’'\"]+)%s was built under R version (\S+)" % (QUOTE, QUOTE), "info",
     "Package %s was built for R %s",
     "Usually harmless, but if it fails, reinstall it with the R version selected for this job."),
    (r"Fatal error: cannot open file %s([^‘’'\"]+)%s" % (QUOTE, QUOTE), "bad", "R could not open the main script: %s",
     "Check the main R script selected on the form; a relative path is relative to the workflow directory."),
    (r"cannot open file %s([^‘’'\"]+)%s: No such file or directory" % (QUOTE, QUOTE), "bad", "File not found: %s",
     "The job runs in the workflow directory, so relative paths start there. Upload the file with the job "
     "or use its full path."),
    (r"cannot allocate vector of size ([\d.]+ \w+)", "bad", "R ran out of memory (could not allocate %s)",
     "Request more memory on the form. With SOCK workers, every worker holds its own copy of the data."),
    (r"(oom[-_ ]kill|Out Of Memory|Exceeded job memory limit)", "bad", "The job was killed for using too much memory",
     "Request more memory on the form. With SOCK workers, every worker holds its own copy of the data."),
    (r"DUE TO TIME LIMIT", "bad", "The job reached its time limit",
     "Request more time on the form, or save intermediate results so a rerun can continue."),
    (r"CANCELLED AT (?!.*DUE TO TIME LIMIT)", "info", "The job was cancelled", ""),
    (r"(MPI_ABORT|mpirun noticed that process|mpirun has exited|PRTE has lost communication)", "bad",
     "The MPI run failed (%s)", "Check the R error above it; with MPI the first failing worker stops the run."),
]


def read_text(path, limit=2 * 1024 * 1024):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - limit))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return None


def first_r_error(text):
    """The first "Error in ..." / "Error:" message with its continuation lines."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"\s*(Error in |Error: |Error\b.*:\s*$)", line):
            msg = [line.rstrip()]
            for nxt in lines[i + 1:i + 6]:
                if not nxt.strip() or re.match(r"\s*(Calls:|Execution halted|In addition:|Warning)", nxt):
                    break
                msg.append(nxt.rstrip())
            return "\n".join(msg)
    return ""


def match_checks(text, limit=8, prefix=""):
    """Hints for the known R problems in text, and the matched snippets."""
    found, seen, matched = [], set(), []
    for pattern, level, title, explanation in CHECKS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            groups = m.groups()
            key = (title, groups)
            if key in seen:
                continue
            seen.add(key)
            matched.append(m.group(0))
            arg = groups[0] if groups else ""
            found.append(hint(level, prefix + (title % groups if "%s" in title else title),
                              e(explanation).replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
                              .replace("%s", e(arg))))
            if len(found) >= limit:
                return found, matched
    return found, matched


def problems_section(jobdir, jobid, status):
    out = ["<h4>Problems found</h4>"]
    err_name, out_name = "error.%s" % jobid, "out.%s" % jobid
    err = read_text(os.path.join(jobdir, err_name))
    log = read_text(os.path.join(jobdir, out_name)) or ""
    if err is None:
        out.append("<div class='muted'>No %s yet.</div>" % e(err_name))
        return out
    text = err + "\n" + log

    found, matched = match_checks(text)

    error = first_r_error(err)
    if error and not any(text_ in error for text_ in matched):   # already explained above
        found.append(hint("bad", "R stopped with this error:", code=error))
    elif re.search(r"^Execution halted", err, re.MULTILINE) and not found:
        found.append(hint("bad", "R stopped with an error", "See %s for the message." % e(err_name)))

    if found:
        out.extend(found)
    elif status and status.upper() in DONE_STATES:
        out.append("<div class='ok'>No known R problems in %s.</div>" % e(err_name))
    else:
        out.append("<div class='ok'>None so far in %s.</div>" % e(err_name))
    return out


# ---------------------------------------------------------------- many runs (tamulauncher)

RUNS_FILE = "r-runs.commands"
RUN_LINE = re.compile(r"^(\d+) : exit code (-?\d+): elapsed time = ([\d.]+)s")


def runs_logdir(jobdir):
    """The .tamulauncher-log/<md5>/ directory of r-runs.commands (newest if there are several)."""
    dirs = [d for d in glob.glob(os.path.join(jobdir, ".tamulauncher-log", "*"))
            if os.path.isfile(os.path.join(d, RUNS_FILE))]
    return max(dirs, key=os.path.getmtime) if dirs else None


def count_lines(path):
    try:
        with open(path, "rb") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def runs_section(jobdir, jobid, status, nodes):
    out = ["<h4>Runs</h4>"]
    done_job = bool(status and status.upper() in DONE_STATES)
    total = count_lines(os.path.join(jobdir, RUNS_FILE))
    logdir = runs_logdir(jobdir)
    results = {}                       # run number -> (exit code, seconds); the last result counts
    if logdir:
        for path in [os.path.join(logdir, "done")] + sorted(glob.glob(os.path.join(logdir, "log.*"))):
            try:
                with open(path, errors="replace") as f:
                    for line in f:
                        m = RUN_LINE.match(line)
                        if m:
                            results[int(m.group(1))] = (int(m.group(2)), float(m.group(3)))
            except OSError:
                pass
    if not logdir and not done_job:
        out.append("<div class='muted'>Waiting for tamulauncher to start the runs.</div>")
        return out

    ok = sorted(n for n, (code, _) in results.items() if code == 0)
    failed = sorted(n for n, (code, _) in results.items() if code != 0)
    finished = len(results)
    active = sum(len([p for p in (n.get("procs") or []) if p.get("role") == "main"]) for n in nodes) \
        if not done_job else 0
    left = max(total - finished, 0)
    pct = 100.0 * finished / total if total else 0.0
    colour = "#047857" if total and not left and not failed else "#b45309" if failed or (done_job and left) else "#2563eb"
    out.append("<div style='display:flex;align-items:center;gap:10px'>"
               "<div style='flex:1;height:8px;border-radius:4px;background:var(--b);overflow:hidden'>"
               "<div style='height:100%%;width:%.1f%%;background:%s'></div></div>"
               "<span class='mono' style='white-space:nowrap'>%d / %d finished (%.0f%%)</span></div>"
               % (pct, colour, finished, total, pct))
    parts = ["<span class='ok'>%d succeeded</span>" % len(ok)]
    if failed:
        parts.append("<span style='color:#b91c1c'>%d failed</span>" % len(failed))
    if active:
        parts.append("%d running now" % active)
    if left:
        parts.append("%d not finished" % left if done_job else "%d to go" % left)
    times = [sec for code, sec in results.values() if code == 0]
    if times:
        parts.append("a successful run takes %s on average" % hm_short(sum(times) / len(times)))
    out.append("<div style='margin-top:4px'>%s</div>" % " &middot; ".join(parts))

    if done_job and left:
        out.append(hint("warn", "%d run%s did not finish." % (left, "s" if left != 1 else ""),
                        "Submit the job again from the workflow directory (<span class='mono'>sbatch r-job.slurm</span>) to run "
                        "them: tamulauncher skips the runs that already finished. If the job hit its time limit, request more time or more nodes."))
    if failed:
        shown = ", ".join("%d" % n for n in failed[:15]) + (", ..." if len(failed) > 15 else "")
        out.append(hint("bad", "Failed runs: %s" % shown,
                        "Each run's output is in <span class='mono'>r-runs/run-&lt;n&gt;.log</span>. A resubmit does "
                        "<i>not</i> rerun failed runs (they count as finished); fix the cause and submit a new "
                        "workflow for them."))
        for n in failed[:3]:
            text = read_text(os.path.join(jobdir, "r-runs", "run-%d.log" % n), limit=256 * 1024)
            if not text:
                continue
            hints, matched = match_checks(text, limit=3, prefix="Run %d: " % n)
            error = first_r_error(text)
            if error and not any(t in error for t in matched):
                hints.append(hint("bad", "Run %d stopped with this error:" % n, code=error))
            out.extend(hints)
    return out


def hm_short(sec):
    sec = int(sec)
    if sec < 90:
        return "%d s" % sec
    return "%d min" % round(sec / 60.0) if sec < 5400 else "%.1f h" % (sec / 3600.0)


# ---------------------------------------------------------------- next run

FINAL_SACCT = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED")


def seconds(text):
    """Slurm [D-][HH:]MM:SS[.mmm] -> seconds."""
    text = (text or "").strip()
    if not text:
        return 0.0
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d) if d.isdigit() else 0
    total = 0.0
    for part in text.split(":"):
        try:
            total = total * 60 + float(part)
        except ValueError:
            return 0.0
    return days * 86400 + total


def mem_bytes(text):
    """Slurm memory like 409728K, 60G, 4000Mc -> bytes (0 when unknown)."""
    m = re.match(r"^([\d.]+)([KMGTP]?)", (text or "").strip())
    if not m:
        return 0
    return int(float(m.group(1)) * 1024 ** " KMGTP".index(m.group(2) or " "))


def gb(n):
    return "%.1f GB" % (n / 1024.0 ** 3) if n < 10 * 1024 ** 3 else "%d GB" % round(n / 1024.0 ** 3)


def hm(sec):
    sec = int(sec)
    return "%d:%02d h" % (sec // 3600, sec % 3600 // 60) if sec >= 3600 else "%d min" % max(1, round(sec / 60.0))


def job_usage(jobdir, jobid):
    """Requested and used resources of a finished job, from one sacct call; cached in the workflow directory."""
    cache = os.path.join(jobdir, "drona_monitoring", "r_usage_%s.json" % jobid)
    try:
        with open(cache) as f:
            return json.load(f)
    except (OSError, ValueError):
        pass
    try:
        out = subprocess.run(["sacct", "-j", str(jobid), "-n", "-P", "-o",
                              "JobID,State,ElapsedRaw,TimelimitRaw,AllocCPUS,TotalCPU,MaxRSS,ReqMem"],
                             capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    usage = None
    maxrss = 0
    for line in out.splitlines():
        f = line.split("|")
        if len(f) < 8:
            continue
        if f[0] == str(jobid):
            limit_min = int(f[3]) if f[3].isdigit() else 0
            usage = {"state": f[1].split()[0] if f[1] else "", "elapsed": int(f[2] or 0), "limit": limit_min * 60,
                     "cpus": int(f[4] or 0), "cpu_time": seconds(f[5]), "reqmem": mem_bytes(f[7])}
        maxrss = max(maxrss, mem_bytes(f[6]))
    if not usage:
        return None
    usage["maxrss"] = maxrss
    if usage["state"] in FINAL_SACCT:
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "w") as f:
                json.dump(usage, f)
        except OSError:
            pass
    return usage


def round_mem_gb(n):
    """Suggested memory in whole GB: 1.5x the peak, at least 2 GB."""
    return max(2, int(math.ceil(n * 1.5 / 1024.0 ** 3)))


def round_time(sec):
    """Suggested time: 1.5x the run time, rounded up to 15 minutes, at least 30 minutes."""
    return max(1800, int(math.ceil(sec * 1.5 / 900.0)) * 900)


def next_run_section(jobdir, jobid, status, nodes):
    if not (status and status.upper() in DONE_STATES):
        return []
    out = ["<h4>Next run</h4>"]
    u = job_usage(jobdir, jobid)
    if not u or u["state"] not in FINAL_SACCT:
        out.append("<div class='muted'>The job's usage is not available from Slurm yet.</div>")
        return out
    if u["elapsed"] < 60:
        out.append("<div class='muted'>The job ran for less than a minute, too short to judge its resources.</div>")
        return out

    first = nodes[0] if nodes else {}
    workers = int(first.get("workers") or 0)
    threads = int(first.get("threads") or 1)
    parallel = first.get("parallel") or ""
    rows, tips = [], []

    # cores: average cores busy = CPU time / wall time
    cpus, used = u["cpus"], u["cpu_time"] / u["elapsed"]
    cpu_cell = "%.1f on average (%d%%)" % (used, round(100 * used / cpus)) if cpus else "?"
    cpu_tip = ""
    if cpus >= 2 and used < 0.5 * cpus:
        want = max(1, int(math.ceil(used * 1.1 - 0.2)))   # 1.0 -> 1, 2.1 -> 3: a little headroom
        if parallel == "many":
            cpu_tip = "about %d cores: fewer runs at the same time per node" % want
        elif workers and threads == 1:
            cpu_tip = "about %d cores: %d worker%s" % (want, max(1, want - 1), "s" if want - 1 != 1 else "")
        elif not workers:
            cpu_tip = "threads per R process: %d" % want
        else:
            cpu_tip = "about %d cores: fewer workers or threads per R process" % want
    elif cpus == 1 and used > 0.9 and not workers and u["elapsed"] >= 300:
        tips.append(hint("info", "The single core was busy the whole run.",
                         "If your script does matrix math (or uses data.table, ranger, xgboost), raising "
                         "<i>Threads per R process</i> may make it faster."))
    rows.append(("Cores", str(cpus), cpu_cell, cpu_tip))

    # memory: peak resident memory of all job steps
    req, peak = u["reqmem"], u["maxrss"]
    mem_tip = ""
    if u["state"] == "OUT_OF_MEMORY":
        mem_tip = "more than %s (the job ran out)" % gb(req)
    elif req and peak and peak < 0.5 * req and req > 4 * 1024 ** 3:
        mem_tip = "%d GB" % round_mem_gb(peak)
    rows.append(("Memory", gb(req) if req else "?", gb(peak) if peak else "?", mem_tip))

    # time: shorter requests start sooner
    limit, elapsed = u["limit"], u["elapsed"]
    time_tip = ""
    if u["state"] == "TIMEOUT":
        time_tip = "more than %s (the job hit the limit)" % hm(limit)
    elif u["state"] == "COMPLETED" and limit >= 3600 and elapsed < 0.25 * limit:
        time_tip = hm(round_time(elapsed))
    rows.append(("Time", hm(limit) if limit else "?", hm(elapsed), time_tip))

    out.append("<table><tr><th></th><th>Requested</th><th>Used</th><th>Suggested next time</th></tr>")
    for name, req_s, used_s, tip in rows:
        out.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                   % (name, e(req_s), e(used_s), e(tip) if tip else "<span class='muted'>keep</span>"))
    out.append("</table>")
    if any(r[3] for r in rows) and u["state"] not in ("TIMEOUT", "OUT_OF_MEMORY"):
        out.append("<div class='muted' style='margin-top:4px'>Smaller requests usually start sooner and use less "
                   "of your allocation.%s</div>"
                   % (" Based on a run that did not complete, so check the numbers." if u["state"] != "COMPLETED" else ""))
    out.extend(tips)
    return out


def main():
    if len(sys.argv) < 3 or not sys.argv[1] or not sys.argv[2]:
        print("<div class='muted'>Select a job to see its R Run Check.</div>")
        return
    jobdir, jobid = sys.argv[1], sys.argv[2]
    status = sys.argv[3] if len(sys.argv) > 3 else ""
    out = ["<div id='dc-r-report'>", STYLE]
    nodes = load_nodes(jobdir, jobid)
    many = (nodes and nodes[0].get("parallel") == "many") or os.path.isfile(os.path.join(jobdir, RUNS_FILE))
    out.extend(runs_section(jobdir, jobid, status, nodes) if many else workers_section(nodes, status))
    out.extend(problems_section(jobdir, jobid, status))
    out.extend(next_run_section(jobdir, jobid, status, nodes))
    out.append("</div>")
    print("\n".join(out))


if __name__ == "__main__":
    main()
