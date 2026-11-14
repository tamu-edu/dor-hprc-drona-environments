#!/usr/bin/env python3
"""Abaqus Progress card for the Manage page (HTML on stdout).

  abaqus_report.py <workflow dir> <job id> [job status]

Cheap enough for the web server: it reads only the job script (for job= and input=), the
*STEP/time-period lines of the input file, and the last part of Abaqus' text output:
  <job>.sta   progress: step, increment, step/total time, cutbacks (Standard) or stable increment (Explicit)
  <job>.msg   ***WARNING / ***ERROR (Standard)
  <job>.dat   ***ERROR from the pre-processor
  out.<id>, error.<id>, <job>.log   license, lock-file, memory and user-subroutine problems, "COMPLETED"
"""
import html
import os
import re
import sys
import time

e = html.escape
TAIL = 512 * 1024            # bytes read from the end of a file
INPUT_LIMIT = 20 * 1024 * 1024
DONE_STATES = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "DONE", "PREEMPTED")

STYLE = """
<style>
#dc-abq { --t:#1e293b; --m:#64748b; --b:#e2e8f0; --d:#f1f5f9; color:var(--t);
  font-family:'Inter',system-ui,sans-serif; font-size:13px; width:100%; box-sizing:border-box; }
#dc-abq h4 { font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--m); margin:14px 0 6px; font-weight:600; }
#dc-abq h4:first-child { margin-top:0; }
#dc-abq table { width:100%; border-collapse:collapse; }
#dc-abq th { text-align:left; font-size:11px; color:var(--m); font-weight:600; padding:4px 6px; }
#dc-abq td { padding:4px 6px; border-top:1px solid var(--d); vertical-align:top; }
#dc-abq .mono { font-family:'JetBrains Mono',monospace; font-size:12px; }
#dc-abq pre { font-family:'JetBrains Mono',monospace; font-size:12px; background:var(--d); padding:6px 8px;
  border-radius:6px; margin:4px 0 0; white-space:pre-wrap; word-break:break-word; }
#dc-abq .hint { padding:6px 9px; border-left:3px solid; margin:5px 0; border-radius:0 6px 6px 0; }
#dc-abq .hint.warn { border-color:#f59e0b; background:#fffbeb; }
#dc-abq .hint.bad { border-color:#ef4444; background:#fef2f2; }
#dc-abq .hint.info { border-color:#94a3b8; background:#f8fafc; }
#dc-abq .ok { color:#047857; }
#dc-abq .muted { color:var(--m); font-size:11px; }
</style>
"""


def hint(level, title, text="", code=""):
    body = "<b>%s</b>" % e(title)
    if text:
        body += "<div>%s</div>" % text
    if code:
        body += "<pre>%s</pre>" % e(code)
    return "<div class='hint %s'>%s</div>" % (level, body)


def tail(path, limit=TAIL):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - limit))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return None


def job_and_input(jobdir):
    """job= and input= of the abaqus line in the workflow's job script."""
    text = tail(os.path.join(jobdir, "abaqus-job.slurm"), 64 * 1024) or ""
    for line in text.splitlines():
        if line.lstrip().startswith("abaqus "):
            job = re.search(r"\bjob=(\S+)", line)
            inp = re.search(r"\binput=('[^']*'|\S+)", line)
            return (job.group(1) if job else ""), (inp.group(1).strip("'") if inp else "")
    return "", ""


def step_periods(path):
    """Time period of every *STEP in the input file (None for steps without one, e.g. *FREQUENCY)."""
    periods, in_step, want_data = [], False, False
    try:
        with open(path, errors="replace") as f:
            text = f.read(INPUT_LIMIT)
    except OSError:
        return []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("**"):
            continue
        if s.startswith("*"):
            key = s.replace(" ", "").lower()
            want_data = False
            if key.startswith("*step"):
                periods.append(None)
                in_step = True
            elif key.startswith("*endstep"):
                in_step = False
            elif in_step and periods and periods[-1] is None and re.match(
                    r"\*(static|dynamic|visco|heattransfer|coupledtemperature|soils|coupledthermal|"
                    r"geostatic|massdiffusion|directcyclic)", key):
                want_data = True
            continue
        if want_data:
            fields = [x.strip() for x in s.split(",")]
            try:
                periods[-1] = float(fields[1]) if len(fields) > 1 and fields[1] else 1.0
            except ValueError:
                periods[-1] = 1.0
            want_data = False
    return periods


STD_LINE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)(U?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)")
EXP_STEP = re.compile(r"^\s*STEP\s+(\d+)\s+ORIGIN\s+(\S+)")
EXP_LINE = re.compile(r"^\s*(\d+)\s+(\S+E[+-]\d+)\s+(\S+E[+-]\d+)\s+(\d+:\d+:\d+)\s+(\S+E[+-]\d+)\s+(\S+)")


def num(text):
    try:
        return float(text)
    except ValueError:
        return None


def read_sta(path):
    text = tail(path)
    if text is None:
        return None
    sta = {"kind": "Explicit" if "Abaqus/Explicit" in text[:4096] or EXP_STEP.search(text) else "Standard",
           "done": "COMPLETED SUCCESSFULLY" in text, "not_done": "HAS NOT BEEN COMPLETED" in text,
           "step": 0, "inc": 0, "step_time": None, "total_time": None, "cutbacks": 0, "recent_cutbacks": 0,
           "stable": [], "mtime": os.path.getmtime(path)}
    rows = []
    for line in text.splitlines():
        if sta["kind"] == "Standard":
            m = STD_LINE.match(line)
            if m:
                rows.append(m)
                if m.group(4) == "U":
                    sta["cutbacks"] += 1
        else:
            m = EXP_STEP.match(line)
            if m:
                sta["step"] = int(m.group(1))
                continue
            m = EXP_LINE.match(line)
            if m:
                sta["inc"] = int(m.group(1))
                sta["step_time"], sta["total_time"] = num(m.group(2)), num(m.group(3))
                sta["cpu"] = m.group(4)
                sta["stable"].append(num(m.group(5)))
    if rows:
        last = rows[-1]
        sta["step"], sta["inc"] = int(last.group(1)), int(last.group(2))
        sta["total_time"], sta["step_time"] = num(last.group(8)), num(last.group(9))
        sta["recent_cutbacks"] = sum(1 for m in rows[-20:] if m.group(4) == "U")
        sta["attempts_last"] = int(last.group(3))
    return sta


CHECKS = [
    (r"license|token", r"(not available|unavailable|denied|exceeded|no such feature|cannot connect|"
                       r"failed to check ?out|queued)", "bad", "License problem",
     "Abaqus could not get its license tokens. If all tokens are in use, try again later or with fewer cores "
     "(tokens grow with the cores: about 5 x cores^0.422)."),
    (r"\.lck|lock file", r"", "bad", "A lock file from an earlier run is in the way",
     "Delete <span class='mono'>&lt;job&gt;.lck</span> in the workflow directory and run again. It is left "
     "behind when an earlier run of the same job was killed."),
    (r"memory", r"(not sufficient|insufficient|too low|exceed|not enough|out of memory)", "bad", "Not enough memory",
     "Request more memory on the form. A datacheck run writes the memory the model needs to the .dat file "
     "(MEMORY TO MINIMIZE I/O)."),
    (r"oom-kill|out-of-memory|oom_kill", r"", "bad", "The job ran out of memory and was killed",
     "Request more memory on the form."),
    (r"problem during (compilation|linking)|error.*user subroutine|ifort: command not found|ifx: command not found|"
     r"icpc: command not found|undefined reference", r"", "bad", "The user subroutine did not compile or link",
     "The compiler messages are in <span class='mono'>%s.log</span>. Check the Fortran (fixed format: code from "
     "column 7, at most 72 columns), the INCLUDE 'aba_param.inc' line, and that the subroutine names and "
     "arguments match the Abaqus manual. \"command not found\" means the compiler module did not load."),
    (r"oldjob|global model|globalmodel", r"(not found|cannot|does not exist|unable)", "bad",
     "Files of the previous job or global model are missing",
     "Abaqus needs all files of the earlier job (.res, .mdl, .stt, .prt, .odb) next to the file you selected, "
     "and the earlier job must have *RESTART, WRITE."),
    (r"DUE TO TIME LIMIT", r"", "bad", "The job reached its time limit",
     "Request more time. If the input has *RESTART, WRITE, Abaqus can continue from the restart file instead of "
     "starting over (abaqus job=new oldjob=%s with *RESTART, READ)."),
]


def problems(jobdir, job, jobid):
    found, seen = [], set()

    def check(line):
        for i, (first, second, level, title, text) in enumerate(CHECKS):
            if re.search(first, line, re.IGNORECASE) and (not second or re.search(second, line, re.IGNORECASE)):
                if i not in seen:
                    seen.add(i)
                    found.append(hint(level, title, text.replace("%s", e(job)), code=line.strip()[:300]))
                return True
        return False

    logs = "\n".join(t for t in (tail(os.path.join(jobdir, n), 128 * 1024) for n in
                                 ("out.%s" % jobid, "error.%s" % jobid, "%s.log" % job)) if t)
    for line in logs.splitlines():
        check(line)
    # ***ERROR messages from the .msg and the .dat (pre-processor): explained when known, else the first two shown
    errors, warnings = [], 0
    for ext in ("msg", "dat"):
        text = tail(os.path.join(jobdir, "%s.%s" % (job, ext))) or ""
        if ext == "msg":
            warnings = len(re.findall(r"\*\*\*WARNING", text))
        for m in re.finditer(r"\*\*\*ERROR:?(.*?)(?:\n\s*\n|\Z)", text, re.DOTALL):
            msg = " ".join(m.group(1).split())
            if msg and msg not in errors and not check(msg):
                errors.append(msg)
    for msg in errors[:2]:
        found.append(hint("bad", "Abaqus error", code=msg[:600]))
    if len(errors) > 2:
        found.append("<div class='muted'>%d more errors in %s.msg / %s.dat.</div>" % (len(errors) - 2, e(job), e(job)))
    return found, warnings


def bar(pct, colour):
    return ("<div style='display:flex;align-items:center;gap:10px;margin-top:4px'>"
            "<div style='flex:1;height:8px;border-radius:4px;background:var(--b);overflow:hidden'>"
            "<div style='height:100%%;width:%.1f%%;background:%s'></div></div>"
            "<span class='mono' style='white-space:nowrap'>%.0f%%</span></div>" % (pct, colour, pct))


def fmt(x):
    return "?" if x is None else ("%.4g" % x)


def main():
    if len(sys.argv) < 3 or not sys.argv[1] or not sys.argv[2] or sys.argv[1].startswith("$"):
        print("<div class='muted'>Select a job to see its Abaqus progress.</div>")
        return
    jobdir, jobid = sys.argv[1], sys.argv[2]
    status = (sys.argv[3] if len(sys.argv) > 3 else "").upper()
    finished = status in DONE_STATES
    job, inp = job_and_input(jobdir)
    out = ["<div id='dc-abq'>", STYLE]
    if not job:
        out.append("<div class='muted'>No abaqus line found in abaqus-job.slurm.</div></div>")
        print("\n".join(out))
        return

    sta = read_sta(os.path.join(jobdir, job + ".sta"))
    periods = step_periods(inp if os.path.isabs(inp) else os.path.join(jobdir, inp)) if inp else []
    out.append("<h4>Progress</h4>")
    if not sta:
        out.append("<div class='muted'>%s</div>" % (
            "No %s.sta: Abaqus stopped before the analysis started (see below)." % e(job) if finished else
            "Waiting for %s.sta; it appears when the analysis starts, after the pre-processor." % e(job)))
    else:
        nsteps = len(periods)
        step, inc = sta["step"], sta["inc"]
        period = periods[step - 1] if 0 < step <= nsteps else None
        # overall progress: finished steps plus the fraction of the current one
        frac = (sta["step_time"] / period) if period and sta["step_time"] is not None else 0.0
        frac = min(max(frac, 0.0), 1.0)
        pct = 100.0 if sta["done"] else (100.0 * ((step - 1) + frac) / nsteps if nsteps and step else 0.0)
        if sta["done"]:
            colour, label = "#047857", "Analysis completed successfully"
        elif sta["not_done"] or finished:
            colour, label = "#b91c1c", "Analysis did not complete"
        else:
            colour, label = "#2563eb", "Running"
        out.append("<div><b style='color:%s'>%s</b> <span class='muted'>Abaqus/%s</span></div>"
                   % (colour, e(label), e(sta["kind"])))
        out.append(bar(pct, colour))
        rows = [("Step", "%d of %d" % (step, nsteps) if nsteps else str(step)),
                ("Increment", str(inc)),
                ("Step time", "%s of %s" % (fmt(sta["step_time"]), fmt(period)) if period else fmt(sta["step_time"])),
                ("Total time", fmt(sta["total_time"]))]
        if sta["kind"] == "Standard":
            rows.append(("Cutbacks", "%d (%d in the last 20 increments)" % (sta["cutbacks"], sta["recent_cutbacks"])))
        elif sta["stable"]:
            rows.append(("Stable increment", "%s (first %s)" % (fmt(sta["stable"][-1]), fmt(sta["stable"][0]))))
        age = int(time.time() - sta["mtime"])
        if not finished and not sta["done"]:
            rows.append(("Last update", "%d min ago" % (age // 60) if age >= 120 else "%d s ago" % age))
        out.append("<table>%s</table>" % "".join("<tr><td style='width:30%%'>%s</td><td class='mono'>%s</td></tr>"
                                                  % (e(k), e(v)) for k, v in rows))
        if sta["kind"] == "Standard" and sta["recent_cutbacks"] >= 5 and not sta["done"]:
            out.append(hint("warn", "Many cutbacks recently",
                            "Abaqus keeps reducing the increment because it doesn't converge. Check the .msg file "
                            "for the cause (contact, large distortion, material instability)."))
        if sta["kind"] == "Explicit" and len(sta["stable"]) > 2 and sta["stable"][0] and \
                sta["stable"][-1] < sta["stable"][0] / 10:
            out.append(hint("warn", "The stable increment dropped more than 10x",
                            "Some elements are distorting badly, so the analysis slows down. The critical element "
                            "is in the .sta file."))
        if not finished and not sta["done"] and age > 1800:
            out.append(hint("info", "No new increment for %d minutes" % (age // 60),
                            "Large increments can take long; if it stays like this, check the job's CPU use below."))

    found, warnings = problems(jobdir, job, jobid)
    out.append("<h4>Messages</h4>")
    out.extend(found)
    if warnings:
        out.append("<div class='muted'>%d warnings in the last part of %s.msg.</div>" % (warnings, e(job)))
    if not found and not warnings:
        out.append("<div class='ok'>No errors or warnings so far.</div>")
    out.append("</div>")
    print("\n".join(out))


if __name__ == "__main__":
    main()
