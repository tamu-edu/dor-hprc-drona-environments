#!/usr/bin/env python3
"""Progress card for tamulauncher runs in a workflow directory.

Reads .tamulauncher-log/<md5 of commands file>/ (the files `tamulauncher --status` reads):
commands (all commands), done + log.* (finished commands, one line each), RUNNING.<host> (active nodes).
Usage: tamulauncher_progress.py JOBDIR [STATUS]"""
import glob
import html
import os
import sys

KNOWN = {"commands", "done", "node_list", "00_LOCK", "released", "todo"}
KNOWN_PREFIXES = ("log.", "todo.", "RUNNING.", "signal.")

FONT = "font-family:'Inter',system-ui,sans-serif;font-size:13px;color:#1e293b"


def count_lines(paths):
    n = 0
    for p in paths:
        try:
            with open(p, "rb") as f:
                n += sum(1 for line in f if line.strip())
        except OSError:
            pass
    return n


def run_info(logdir):
    names = os.listdir(logdir)
    inputs = [n for n in names if n not in KNOWN and not n.startswith(KNOWN_PREFIXES)]
    return {
        "name": inputs[0] if inputs else os.path.basename(logdir.rstrip("/"))[:12],
        "total": count_lines([os.path.join(logdir, "commands")]),
        "done": count_lines([os.path.join(logdir, "done")] + glob.glob(os.path.join(logdir, "log.*"))),
        "nodes": sorted(n[len("RUNNING."):] for n in names if n.startswith("RUNNING.")),
        "mtime": os.path.getmtime(logdir),
    }


def render(run, status):
    total, done = run["total"], min(run["done"], run["total"])
    pct = 100.0 * done / total if total else 0.0
    left = total - done
    finished = status == "DONE"
    if total and left == 0:
        colour, label = "#047857", "All commands finished"
    elif not finished and run["nodes"]:
        colour, label = "#2563eb", "Running"
    elif finished:
        colour, label = "#b45309", "Incomplete: rerun the workflow to run the remaining %d command%s" % (
            left, "" if left == 1 else "s")
    else:
        colour, label = "#64748b", "Waiting"
    nodes = ""
    if run["nodes"] and not finished:
        nodes = "<span style='color:#64748b'>&middot; active on %s</span>" % html.escape(", ".join(run["nodes"]))
    return (
        "<div style=\"%s;padding:8px 12px;border:1px solid #e2e8f0;border-radius:10px;background:#f8fafc;margin-bottom:6px\">"
        "<div style='display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px'>"
        "<span style='display:inline-flex;align-items:center;gap:6px;font-weight:700;color:%s'>"
        "<span style='width:9px;height:9px;border-radius:50%%;background:%s;display:inline-block'></span>%s</span>"
        "<span style='font-family:monospace'>%s</span>%s</div>"
        "<div style='display:flex;align-items:center;gap:10px;margin-top:6px'>"
        "<div style='flex:1;height:8px;border-radius:4px;background:#e2e8f0;overflow:hidden'>"
        "<div style='height:100%%;width:%.1f%%;background:%s'></div></div>"
        "<span style='font-family:monospace;white-space:nowrap'>%d / %d (%.0f%%)</span></div></div>"
        % (FONT, colour, colour, html.escape(label), html.escape(run["name"]), nodes, pct, colour, done, total, pct))


def main():
    jobdir = sys.argv[1] if len(sys.argv) > 1 else ""
    status = sys.argv[2].strip().upper() if len(sys.argv) > 2 else ""
    if not jobdir or jobdir.startswith("$"):
        return
    logdirs = [d for d in glob.glob(os.path.join(jobdir, ".tamulauncher-log", "*")) if os.path.isdir(d)]
    if not logdirs:
        print("<div style=\"%s;color:#64748b\">No tamulauncher progress yet. It appears once the job starts.</div>" % FONT)
        return
    runs = sorted((run_info(d) for d in logdirs), key=lambda r: r["mtime"], reverse=True)
    print("".join(render(r, status) for r in runs))


if __name__ == "__main__":
    main()
