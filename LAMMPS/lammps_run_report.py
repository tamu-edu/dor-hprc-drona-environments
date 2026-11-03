#!/usr/bin/env python3
"""LAMMPS run report: setup, performance and warnings/errors from out.<jobid>
(and error.<jobid>), rendered as one HTML card for the Manage page.

Usage: lammps_run_report.py JOBDIR JOBID [--json]

One streaming pass over the output file. Thermo rows ("  - [...]") are skipped
with a byte-prefix check, so a 1M-row file costs ~0.25 s; only the few hundred
other lines are decoded and parsed. Warnings are grouped (numbers normalised)
and capped, so a warning printed every step cannot blow up the output.
Standalone: must not import Drona modules.
"""
import argparse
import glob
import html
import json
import os
import re
import subprocess
import sys
import ast

THERMO_PREFIX = b"  - ["
MAX_DISTINCT_MESSAGES = 40
MAX_RUNS_SHOWN = 10
MIN_LOOP_FOR_TIMING_HINTS = 5.0  # seconds

# ---------------------------------------------------------------------------
# Known messages: substring -> (severity override or None, one-line explanation)
# ---------------------------------------------------------------------------
KNOWN_MESSAGES = [
    ("Lost atoms", None,
     "Atoms moved too far in one step (the system blew up). Reduce the timestep, "
     "minimize first to remove overlaps, or check fixes/thermostats."),
    ("Out of range atoms", None,
     "Atoms left the box or the PPPM grid, usually because the system blew up. "
     "Reduce the timestep or minimize first."),
    ("Bond atom missing", None,
     "A bonded partner is not on this processor: usually a blow-up, or the "
     "communication cutoff is shorter than a bond (comm_modify cutoff)."),
    ("Neighbor list overflow", None,
     "Too many neighbors per atom. Increase neigh_modify one/page, or check for a "
     "collapsing (too dense) system."),
    ("Non-numeric", None,
     "NaN/Inf in the simulation, almost always a blow-up. Reduce the timestep or "
     "check the force field parameters."),
    ("Too many neighbor bins", None,
     "The box grew very large (blow-up) or the neighbor binsize is too small."),
    ("is shorter than a bond", None,
     "Communication cutoff is shorter than a bond length; set comm_modify cutoff."),
    ("Inconsistent image flags", None,
     "Bonded atoms have image flags that put them far apart; check the data file."),
    ("New thermo_style command", None,
     "Harmless: thermo_modify came before thermo_style, so it was reset. Put "
     "thermo_modify after thermo_style."),
    ("neigh_modify every 1 delay 0 check yes", None,
     "Harmless: LAMMPS uses safe neighbor settings during minimization."),
    ("GPU-aware MPI", None,
     "GPU-aware MPI is off, so halo exchange goes through host memory and is slower. "
     "Use an MPI build with GPU support and -pk kokkos gpu/aware on."),
    ("Failed to allocate", "error",
     "Out of memory. Request more memory, use more nodes/GPUs, or reduce the system size."),
    ("DUE TO TIME LIMIT", "error",
     "The job hit its wall-time limit. Request more time, or restart from the last "
     "restart file."),
    ("oom-kill", "error",
     "Slurm killed the job for exceeding its memory request. Increase --mem."),
    ("Out Of Memory", "error",
     "Slurm killed the job for exceeding its memory request. Increase --mem."),
    ("CANCELLED", "error", "The job was cancelled (by the user, an admin or Slurm)."),
    ("Segmentation fault", "error",
     "LAMMPS crashed. Often a blow-up or an unsupported style/package combination; "
     "check the last thermo rows and earlier warnings."),
    ("forcing job termination", None,
     "Slurm stopped the remaining tasks after one exited. Usually follows an earlier "
     "error; harmless if the run finished."),
]

SLURM_PATTERNS = ("DUE TO TIME LIMIT", "oom-kill", "Out Of Memory", "CANCELLED",
                  "Segmentation fault", "forcing job termination", "BAD TERMINATION")

NUM_RE = re.compile(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?")
SRC_RE = re.compile(r"\s*\([^()]*\.(cpp|h):\d+\)\s*$")


def explain(msg):
    for key, sev, text in KNOWN_MESSAGES:
        if key in msg:
            return sev, text
    return None, ""


class Messages:
    """Group warning/error lines by a number-normalised key, capped."""

    def __init__(self):
        self.groups = {}
        self.order = []
        self.dropped = 0

    def add(self, severity, text, source, lineno):
        text = text.strip()
        key = (severity, NUM_RE.sub("#", SRC_RE.sub("", text)))
        g = self.groups.get(key)
        if g is None:
            if len(self.order) >= MAX_DISTINCT_MESSAGES:
                self.dropped += 1
                return
            g = {"severity": severity, "message": SRC_RE.sub("", text), "count": 0,
                 "source": source, "first_line": lineno, "last_line": lineno}
            self.groups[key] = g
            self.order.append(key)
        g["count"] += 1
        g["last_line"] = lineno

    def items(self):
        out = []
        for k in self.order:
            g = dict(self.groups[k])
            sev, text = explain(g["message"])
            if sev:
                g["severity"] = sev
            g["hint"] = text
            out.append(g)
        out.sort(key=lambda g: (0 if g["severity"] == "error" else 1, g["first_line"]))
        return out


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
LOOP_RE = re.compile(r"^Loop time of (\S+) on (\d+) procs for (\d+) steps with (\d+) atoms")
CPU_RE = re.compile(r"^([\d.]+)% CPU use with (\d+) MPI tasks x (\d+) OpenMP threads")
SECTION_RE = re.compile(r"^(\w+)\s*\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|\s*([\d.]+)\s*$")
NLOCAL_RE = re.compile(r"^Nlocal:\s+(\S+) ave\s+(\S+) max\s+(\S+) min")
GRID_RE = re.compile(r"^\s*(\d+) by (\d+) by (\d+) MPI processor grid")
CREATED_ATOMS_RE = re.compile(r"^Created (\d+) atoms")
READ_ATOMS_RE = re.compile(r"^\s+(\d+) atoms\s*$")
BOX_RE = re.compile(r"^(?:Created )?\s*(orthogonal|triclinic) box = (.*)$")
MEM_RE = re.compile(r"^Per MPI rank memory allocation \(min/avg/max\) = (\S+) \| (\S+) \| (\S+) Mbytes")
PAIR_RE = re.compile(r"^\s+\(\d+\) pair (\S+?),")
PAIR_BUILD_RE = re.compile(r"^\s+pair build: (\S+)")
NEIGH_UPDATE_RE = re.compile(r"^\s+update: every = (\d+) steps, delay = (\d+) steps, check = (\w+)")
KOKKOS_GPUS_RE = re.compile(r"will use up to (\d+) GPU\(s\) per node")
OMP_RE = re.compile(r"using (\d+) OpenMP thread\(s\) per MPI task")


def new_run(index):
    return {"index": index, "loop_time": None, "procs": None, "steps": None, "atoms": None,
            "performance": [], "cpu_use": None, "mpi_tasks": None, "omp_threads": None,
            "sections": [], "nlocal": None, "neighbor_builds": None, "dangerous": None,
            "finished": False}


class Thermo:
    """Tracks the thermo keywords, and the first and last row of the latest block (raw bytes)."""

    def __init__(self):
        self.keywords = None
        self.first = None
        self.last = None
        self._need_first = False

    def header(self, line):
        body = line[len("keywords:"):].strip()
        try:
            kws = [str(k) for k in ast.literal_eval(body)]
        except Exception:
            kws = re.findall(r"'((?:[^']|'(?!\s*[,\]]))*)'", body)  # colnames may hold an apostrophe
        if kws:
            self.keywords = kws
            self._need_first = True

    def row(self, raw):
        if self._need_first:
            self.first = raw
            self._need_first = False
        self.last = raw

    def values(self, raw):
        """Validated {keyword: float} for a raw row, or None."""
        if raw is None or not self.keywords:
            return None
        txt = raw.decode("utf-8", "replace").strip()
        if not (txt.startswith("- [") and txt.endswith("]")):
            return None
        parts = [x.strip() for x in txt[3:-1].split(",")]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        if len(parts) != len(self.keywords):
            return None
        out = {}
        for k, v in zip(self.keywords, parts):
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = None
        return out


def iter_lines(path, thermo=None):
    """(lineno, text) for every non-thermo line; thermo rows go to thermo.row()."""
    with open(path, "rb") as f:
        for lineno, raw in enumerate(f, 1):
            if raw.startswith(THERMO_PREFIX):
                if thermo is not None:
                    thermo.row(raw)
                continue
            yield lineno, raw.decode("utf-8", "replace").rstrip("\n")


def parse_out(path, msgs, thermo=None):
    setup = {"version": None, "accelerator": [], "gpus_used": 0, "grid": None, "atoms": None,
             "box": None, "units": None, "timestep": None, "timesteps_seen": [],
             "memory_mb": None, "pair_styles": [], "pair_builds": [], "neigh_update": None}
    runs = []
    cur = None
    in_sections = False
    wall = None
    in_setup_block = False

    for lineno, line in iter_lines(path, thermo):
        if line.startswith("WARNING"):
            msgs.add("warning", line[len("WARNING:"):] if line.startswith("WARNING:") else line, "out", lineno)
            continue
        if line.startswith("ERROR"):
            msgs.add("error", line, "out", lineno)
            continue
        if thermo is not None and line.startswith("keywords:"):
            thermo.header(line)
            continue

        if setup["version"] is None and line.startswith("LAMMPS ("):
            setup["version"] = line[len("LAMMPS ("):].rstrip(")")
            continue

        # accelerator detection (first occurrence of each line)
        if "KOKKOS mode" in line or "Intel Package" in line or "Using acceleration for" in line:
            txt = SRC_RE.sub("", line).strip(" -")
            if txt not in setup["accelerator"]:
                setup["accelerator"].append(txt)
            continue
        m = KOKKOS_GPUS_RE.search(line)
        if m:
            setup["gpus_used"] = max(setup["gpus_used"], int(m.group(1)))
            setup["accelerator"].append("KOKKOS: up to %s GPU(s) per node" % m.group(1))
            continue
        if line.startswith("Device ") and ":" in line:
            setup["gpus_used"] += 1
            setup["accelerator"].append(line.strip())
            continue
        m = OMP_RE.search(line)
        if m and int(m.group(1)) > 1:
            txt = "OpenMP: %s thread(s) per MPI task" % m.group(1)
            if txt not in setup["accelerator"]:
                setup["accelerator"].append(txt)
            continue

        m = GRID_RE.match(line)
        if m and setup["grid"] is None:
            setup["grid"] = "%s × %s × %s" % m.groups()
            continue
        m = CREATED_ATOMS_RE.match(line) or READ_ATOMS_RE.match(line)
        if m and setup["atoms"] is None:
            setup["atoms"] = int(m.group(1))
            continue
        m = BOX_RE.match(line)
        if m and setup["box"] is None and "using lattice units" not in line:
            setup["box"] = "%s: %s" % (m.group(1), m.group(2))
            continue
        m = MEM_RE.match(line)
        if m:
            try:
                mx = float(m.group(3))
                setup["memory_mb"] = max(setup["memory_mb"] or 0, mx)
            except ValueError:
                pass
            in_setup_block = False
            continue
        m = PAIR_RE.match(line)
        if m:
            if m.group(1) not in setup["pair_styles"]:
                setup["pair_styles"].append(m.group(1))
            continue
        m = PAIR_BUILD_RE.match(line)
        if m:
            if m.group(1) not in setup["pair_builds"]:
                setup["pair_builds"].append(m.group(1))
            continue
        m = NEIGH_UPDATE_RE.match(line)
        if m and setup["neigh_update"] is None:
            setup["neigh_update"] = {"every": int(m.group(1)), "delay": int(m.group(2)),
                                     "check": m.group(3)}
            continue

        if line.startswith("Setting up ") and line.rstrip().endswith("run ..."):
            cur = new_run(len(runs) + 1)
            cur["kind"] = line[len("Setting up "):-len(" run ...")].strip()
            runs.append(cur)
            in_setup_block = True
            in_sections = False
            continue
        if in_setup_block:
            s = line.strip()
            if s.startswith("Unit style") and ":" in s:
                setup["units"] = setup["units"] or s.split(":", 1)[1].strip()
            elif s.startswith("Time step") and ":" in s:
                ts = s.split(":", 1)[1].strip()
                setup["timestep"] = setup["timestep"] or ts
                if ts not in setup["timesteps_seen"]:
                    setup["timesteps_seen"].append(ts)
            continue

        m = LOOP_RE.match(line)
        if m:
            if cur is None or cur["finished"]:
                cur = new_run(len(runs) + 1)
                cur["kind"] = "?"
                runs.append(cur)
            cur["loop_time"] = float(m.group(1))
            cur["procs"] = int(m.group(2))
            cur["steps"] = int(m.group(3))
            cur["atoms"] = int(m.group(4))
            cur["finished"] = True
            continue
        if cur is None:
            continue
        if line.startswith("Performance:"):
            cur["performance"] = [p.strip() for p in line[len("Performance:"):].split(",") if p.strip()]
            continue
        m = CPU_RE.match(line)
        if m:
            cur["cpu_use"] = float(m.group(1))
            cur["mpi_tasks"] = int(m.group(2))
            cur["omp_threads"] = int(m.group(3))
            continue
        if line.startswith("Section |"):
            in_sections = True
            continue
        if in_sections:
            m = SECTION_RE.match(line)
            if m:
                def num(x):
                    x = x.strip()
                    try:
                        return float(x)
                    except ValueError:
                        return None
                cur["sections"].append({"name": m.group(1), "avg": num(m.group(3)),
                                        "max": num(m.group(4)), "varavg": num(m.group(5)),
                                        "total": float(m.group(6))})
                continue
            if line.startswith("----"):
                continue
            in_sections = False
        m = NLOCAL_RE.match(line)
        if m:
            try:
                cur["nlocal"] = tuple(float(x) for x in m.groups())
            except ValueError:
                pass
            continue
        if line.startswith("Neighbor list builds ="):
            try:
                cur["neighbor_builds"] = int(line.split("=")[1])
            except ValueError:
                pass
            continue
        if line.startswith("Dangerous builds"):
            rest = line[len("Dangerous builds"):].strip()
            if rest.startswith("="):
                try:
                    cur["dangerous"] = int(rest[1:])
                except ValueError:
                    pass
            else:
                cur["dangerous"] = rest  # e.g. "not checked"
            continue
        if line.startswith("Total wall time:"):
            wall = line.split(":", 1)[1].strip()
            continue

    return setup, runs, wall


def parse_err(path, msgs):
    for lineno, line in iter_lines(path):
        if "ERROR" in line and not line.startswith("WARNING"):
            msgs.add("error", line, "err", lineno)
        elif line.startswith("WARNING"):
            msgs.add("warning", line[len("WARNING:"):] if line.startswith("WARNING:") else line, "err", lineno)
        elif any(p in line for p in SLURM_PATTERNS):
            msgs.add("error" if "forcing job termination" not in line else "info", line, "err", lineno)


def gpus_requested(jobdir):
    """True/False from the job script(s) in jobdir, None if unknown."""
    scripts = glob.glob(os.path.join(jobdir, "*.slurm")) + glob.glob(os.path.join(jobdir, "*.sh"))
    seen_sbatch = False
    for path in scripts:
        try:
            with open(path, "r", errors="replace") as f:
                for line in f:
                    if line.startswith("#SBATCH"):
                        seen_sbatch = True
                        if re.search(r"--gres[= ]\S*gpu|--gpus(-per-\w+)?[= ]|-G\s*\d", line):
                            return True
        except OSError:
            continue
    return False if seen_sbatch else None


# ---------------------------------------------------------------------------
# Hints
# ---------------------------------------------------------------------------
def setup_hints(setup, gpu_req):
    hints = []
    styles = setup["pair_styles"]
    kokkos = any("KOKKOS" in a for a in setup["accelerator"])
    on_gpu = run_on_gpu(setup)
    accel_suffix = ("/kk", "/gpu", "/intel", "/omp", "/opt")
    if gpu_req and not on_gpu:
        hints.append(("warn", "GPUs were requested, but LAMMPS reports no GPU in use. "
                      "Run with -k on g N -sf kk -pk kokkos (KOKKOS) or -sf gpu -pk gpu N (GPU package)."))
    if (gpu_req or kokkos) and styles:
        plain = [s for s in styles if not any(x in s for x in accel_suffix)]
        if plain:
            hints.append(("warn", "Pair style(s) without an accelerator suffix: %s. They run on the "
                          "CPU; add -sf kk (KOKKOS) or -sf gpu." % ", ".join(plain)))
    if kokkos and on_gpu and any("kk/host" in b for b in setup["pair_builds"]):
        hints.append(("warn", "KOKKOS is in GPU mode, but a neighbor list is built on the host (kk/host)."))
    if len(setup["timesteps_seen"]) > 1:
        hints.append(("info", "The timestep changes between runs: %s." % ", ".join(setup["timesteps_seen"])))
    return hints


def run_on_gpu(setup):
    return setup["gpus_used"] > 0 or any("kk/device" in b for b in setup["pair_builds"])


def run_hints(run, setup):
    hints = []
    if isinstance(run["dangerous"], int) and run["dangerous"] > 0:
        hints.append(("bad", "%d dangerous neighbor builds: lists were rebuilt on the first step atom "
                      "movement was checked, so some interactions may have been missed. Reduce the delay "
                      "(neigh_modify delay), e.g. delay 0 every 1 check yes." % run["dangerous"]))
    # The timing breakdown of very short runs is dominated by setup noise
    if run["loop_time"] is None or run["loop_time"] < MIN_LOOP_FOR_TIMING_HINTS:
        return hints
    # On GPUs the kernels run asynchronously, so their time lands in whichever section waits for
    # them (often Comm or Modify). LAMMPS docs (Speed_kokkos): an accurate breakdown needs
    # CUDA_LAUNCH_BLOCKING=1, which is slower and not for production. Skip section-based hints.
    if run_on_gpu(setup):
        hints.append(("info", "This run used GPUs: the section times above are not reliable, because GPU "
                      "work is asynchronous and is usually counted under Comm or Modify. Compare the "
                      "timesteps/s between runs (e.g. 1 vs 2 GPUs) to judge scaling instead."))
        if run["nlocal"]:
            ave, mx, mn = run["nlocal"]
            if ave > 0 and mx / ave > 1.2:
                hints.append(("warn", "Load imbalance: the busiest rank has %.0f%% more atoms than average. "
                              "Try the balance command or fix balance." % ((mx / ave - 1) * 100)))
        return hints
    sec = {s["name"]: s for s in run["sections"]}
    comm = sec.get("Comm", {}).get("total")
    neigh = sec.get("Neigh", {}).get("total")
    pair = sec.get("Pair", {}).get("total")
    if comm is not None and comm > 50:
        hints.append(("warn", "Communication-bound (%.0f%% Comm). The system may be too small for "
                      "this many ranks/GPUs; try fewer, use GPU-aware MPI (-pk kokkos gpu/aware on), "
                      "or simulate a larger system." % comm))
    nu = setup.get("neigh_update")
    if neigh is not None and neigh > 25:
        if nu and nu["check"] == "no":
            hints.append(("warn", "Neighbor builds take %.0f%% with check = no: lists are rebuilt every "
                          "%d steps whether needed or not. Try neigh_modify check yes." % (neigh, nu["every"])))
        else:
            hints.append(("info", "Neighbor builds take %.0f%%. A larger skin (neighbor command) "
                          "rebuilds less often." % neigh))
    if run["nlocal"]:
        ave, mx, mn = run["nlocal"]
        if ave > 0 and mx / ave > 1.2:
            hints.append(("warn", "Load imbalance: the busiest rank has %.0f%% more atoms than average. "
                          "Try the balance command or fix balance." % ((mx / ave - 1) * 100)))
    varavg = max((s["varavg"] or 0) for s in run["sections"]) if run["sections"] else 0
    if varavg > 10:
        hints.append(("warn", "Large timing variation across ranks (%%varavg %.0f): load imbalance." % varavg))
    accel = setup["gpus_used"] > 0 or any(x in s for s in setup["pair_styles"] for x in ("/kk", "/gpu", "/intel", "/omp"))
    if pair is not None and pair > 70 and not accel:
        hints.append(("info", "Pair forces take %.0f%% on plain CPU styles; a GPU, KOKKOS, INTEL or "
                      "OPENMP variant (-sf ...) could speed this up." % pair))
    return hints


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
STYLE = """
<style>
#dc-lmp-report { --t:#1e293b; --m:#64748b; --b:#e2e8f0; --d:#f1f5f9; color:var(--t);
  font-family:'Inter',system-ui,sans-serif; font-size:13px; width:100%; box-sizing:border-box; }
#dc-lmp-report h4 { font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--m);
  margin:14px 0 6px; font-weight:600; }
#dc-lmp-report h4:first-child { margin-top:0; }
#dc-lmp-report details > summary { cursor:pointer; font-size:11px; text-transform:uppercase; letter-spacing:.04em;
  color:var(--m); font-weight:600; margin:14px 0 6px; list-style-position:inside; }
#dc-lmp-report details.inner > summary { margin:8px 0 4px; text-transform:none; letter-spacing:0; font-size:12px; }
#dc-lmp-report details > summary:hover { color:var(--t); }
#dc-lmp-report .banner { padding:8px 10px; border-radius:8px; margin-bottom:10px; font-weight:600; }
#dc-lmp-report .banner.bad { background:#fef2f2; color:#b91c1c; border:1px solid #fecaca; }
#dc-lmp-report .banner.info { background:#f8fafc; color:var(--m); border:1px solid var(--b); }
#dc-lmp-report .kv { display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:6px 18px; }
#dc-lmp-report .kv div { border-bottom:1px solid var(--d); padding:3px 0; }
#dc-lmp-report .k { color:var(--m); font-size:11px; display:block; }
#dc-lmp-report .v { font-family:'JetBrains Mono',monospace; font-size:12px; word-break:break-word; }
#dc-lmp-report table { width:100%; border-collapse:collapse; }
#dc-lmp-report th { text-align:left; font-size:11px; color:var(--m); font-weight:600; padding:4px 6px; }
#dc-lmp-report td { padding:4px 6px; border-top:1px solid var(--d); vertical-align:top; }
#dc-lmp-report .mono { font-family:'JetBrains Mono',monospace; font-size:12px; }
#dc-lmp-report .bar-bg { background:var(--d); border-radius:999px; height:9px; min-width:120px; }
#dc-lmp-report .bar { height:100%; border-radius:999px; background:#3b82f6; }
#dc-lmp-report .bar.hot { background:#f59e0b; }
#dc-lmp-report .hint { padding:5px 8px; border-left:3px solid; margin:4px 0; border-radius:0 6px 6px 0; }
#dc-lmp-report .hint.warn { border-color:#f59e0b; background:#fffbeb; }
#dc-lmp-report .hint.bad { border-color:#ef4444; background:#fef2f2; }
#dc-lmp-report .hint.info { border-color:#94a3b8; background:#f8fafc; }
#dc-lmp-report .ok { color:#047857; }
#dc-lmp-report .pill { display:inline-block; font-size:10px; font-weight:700; padding:1px 7px;
  border-radius:999px; text-transform:uppercase; }
#dc-lmp-report .pill.error { background:#fee2e2; color:#b91c1c; }
#dc-lmp-report .pill.warning { background:#fef3c7; color:#92400e; }
#dc-lmp-report .pill.info { background:#e2e8f0; color:#475569; }
#dc-lmp-report .muted { color:var(--m); font-size:11px; }
#dc-lmp-report .run { border:1px solid var(--b); border-radius:10px; padding:8px 10px; margin-bottom:8px; }
</style>
"""

e = html.escape


def kv(label, value):
    if value in (None, "", []):
        return ""
    return "<div><span class='k'>%s</span><span class='v'>%s</span></div>" % (e(label), e(str(value)))


def hints_html(hints):
    return "".join("<div class='hint %s'>%s</div>" % (lvl, e(text)) for lvl, text in hints)


def fmt_seconds(s):
    if s is None:
        return None
    if s < 120:
        return "%.3g s" % s
    h, rem = divmod(int(round(s)), 3600)
    m, sec = divmod(rem, 60)
    return "%d:%02d:%02d" % (h, m, sec)


def render(report):
    setup, runs, msgs, wall = report["setup"], report["runs"], report["messages"], report["wall"]
    out = ["<div id='dc-lmp-report'>", STYLE]

    n_err = sum(1 for g in msgs if g["severity"] == "error")
    if n_err:
        out.append("<div class='banner bad'>%d error message%s found, see Warnings &amp; errors below.</div>"
                   % (n_err, "" if n_err == 1 else "s"))

    # --- Setup
    out.append(hints_html(report["setup_hints"]))
    out.append("<details><summary>Setup</summary><div class='kv'>")
    out.append(kv("LAMMPS version", setup["version"]))
    out.append(kv("Accelerator", "; ".join(setup["accelerator"]) or "none (plain CPU)"))
    out.append(kv("MPI processor grid", setup["grid"]))
    out.append(kv("Atoms", "{:,}".format(setup["atoms"]) if setup["atoms"] else None))
    out.append(kv("Box", setup["box"]))
    out.append(kv("Units / timestep", " / ".join(x for x in (setup["units"], setup["timestep"]) if x)))
    out.append(kv("Memory per rank (max)", "%.4g MB" % setup["memory_mb"] if setup["memory_mb"] else None))
    out.append(kv("Pair style", ", ".join(setup["pair_styles"])))
    out.append(kv("Neighbor build", ", ".join(setup["pair_builds"])))
    nu = setup["neigh_update"]
    if nu:
        out.append(kv("Neighbor update", "every %d, delay %d, check %s" % (nu["every"], nu["delay"], nu["check"])))
    out.append(kv("GPUs requested (job script)", {True: "yes", False: "no", None: "unknown"}[report["gpu_requested"]]))
    out.append("</div></details>")

    # --- Performance
    out.append("<h4>Performance</h4>")
    finished = [r for r in runs if r["finished"]]
    started_only = len(runs) - len(finished)
    if not runs:
        out.append("<div class='muted'>No run has started yet.</div>")
    elif not finished:
        out.append("<div class='muted'>Run 1 is in progress; the performance summary appears when it finishes.</div>")
    else:
        if started_only:
            out.append("<div class='muted'>Run %d is in progress.</div>" % len(runs))
        shown = finished[-MAX_RUNS_SHOWN:]
        if len(finished) > len(shown):
            out.append("<div class='muted'>Showing the last %d of %d finished runs.</div>" % (len(shown), len(finished)))
        for r in shown:
            title = "Run %d" % r["index"]
            if r.get("kind") and r["kind"] != "?":
                title += " (%s)" % r["kind"]
            out.append("<div class='run'><div style='font-weight:600;margin-bottom:4px'>%s</div><div class='kv'>" % e(title))
            out.append(kv("Loop time", fmt_seconds(r["loop_time"])))
            out.append(kv("Steps", "{:,}".format(r["steps"])))
            out.append(kv("Atoms", "{:,}".format(r["atoms"])))
            for p in r["performance"]:
                parts = p.split(" ", 1)
                if len(parts) == 2:
                    out.append(kv(parts[1], parts[0]))
            if r["mpi_tasks"] is not None:
                out.append(kv("Ranks × threads", "%d × %d" % (r["mpi_tasks"], r["omp_threads"])))
                out.append(kv("CPU use", "%.1f%%" % r["cpu_use"]))
            if r["neighbor_builds"] is not None:
                out.append(kv("Neighbor builds", r["neighbor_builds"]))
            if r["dangerous"] is not None:
                out.append(kv("Dangerous builds", r["dangerous"]))
            out.append("</div>")
            if r["sections"]:
                out.append("<details class='inner'><summary>Time breakdown by section</summary>")
                out.append("<table style='margin-top:6px'><tr><th>Section</th><th style='width:55%'>% of loop time</th>"
                           "<th>avg time</th><th>%varavg</th></tr>")
                top = max(s["total"] for s in r["sections"])
                for s in r["sections"]:
                    cls = "bar hot" if s["total"] == top and top > 50 and not run_on_gpu(setup) else "bar"
                    out.append("<tr><td>%s</td><td><div style='display:flex;align-items:center;gap:8px'>"
                               "<div class='bar-bg' style='flex:1'><div class='%s' style='width:%.1f%%'></div></div>"
                               "<span class='mono' style='width:48px;text-align:right'>%.1f%%</span></div></td>"
                               "<td class='mono'>%s</td><td class='mono'>%s</td></tr>"
                               % (e(s["name"]), cls, min(s["total"], 100), s["total"],
                                  "" if s["avg"] is None else "%.4g s" % s["avg"],
                                  "" if s["varavg"] is None else "%.1f" % s["varavg"]))
                out.append("</table></details>")
            out.append(hints_html(r["hints"]))
            out.append("</div>")
    if wall:
        out.append("<div class='muted'>Total wall time: %s</div>" % e(wall))

    # --- Warnings & errors
    out.append("<h4>Warnings &amp; errors</h4>")
    if not msgs:
        out.append("<div class='ok'>No warnings or errors.</div>")
    else:
        out.append("<table><tr><th>Type</th><th>Message</th><th>Count</th><th>Where</th></tr>")
        for g in msgs:
            where = "%s line %d" % ("out" if g["source"] == "out" else "error", g["first_line"])
            if g["count"] > 1:
                where += "–%d" % g["last_line"]
            hint = "<div class='muted' style='margin-top:2px'>%s</div>" % e(g["hint"]) if g["hint"] else ""
            out.append("<tr><td><span class='pill %s'>%s</span></td><td><span class='mono'>%s</span>%s</td>"
                       "<td class='mono'>%s</td><td class='muted'>%s</td></tr>"
                       % (g["severity"], g["severity"], e(g["message"]), hint,
                          "{:,}".format(g["count"]), e(where)))
        out.append("</table>")
        if report["messages_dropped"]:
            out.append("<div class='muted'>%d more lines with other messages not shown.</div>"
                       % report["messages_dropped"])

    out.append("</div>")
    return "".join(out)


def build_report(jobdir, jobid, thermo=None):
    out_path = os.path.join(jobdir, "out.%s" % jobid)
    err_path = os.path.join(jobdir, "error.%s" % jobid)
    msgs = Messages()
    if not os.path.exists(out_path):
        return None
    setup, runs, wall = parse_out(out_path, msgs, thermo)
    if os.path.exists(err_path):
        parse_err(err_path, msgs)
    gpu_req = gpus_requested(jobdir)
    for r in runs:
        r["hints"] = run_hints(r, setup) if r["finished"] else []
    return {"setup": setup, "runs": runs, "wall": wall, "messages": msgs.items(),
            "messages_dropped": msgs.dropped, "gpu_requested": gpu_req,
            "setup_hints": setup_hints(setup, gpu_req)}

# ---------------------------------------------------------------------------
# Status strip: one line on top of the Manage page
# ---------------------------------------------------------------------------
STRIP_KEYWORDS = ("Temp", "Press", "PotEng", "TotEng", "Volume", "Density")
STATE_STYLE = {  # Slurm state -> (dot/text colour, label)
    "PENDING": ("#64748b", "Pending"), "CONFIGURING": ("#64748b", "Starting"),
    "RUNNING": ("#10b981", "Running"), "COMPLETING": ("#10b981", "Completing"),
    "COMPLETED": ("#047857", "Completed"), "FAILED": ("#dc2626", "Failed"),
    "TIMEOUT": ("#dc2626", "Hit time limit"), "CANCELLED": ("#b45309", "Cancelled"),
    "OUT_OF_MEMORY": ("#dc2626", "Out of memory"), "NODE_FAIL": ("#dc2626", "Node failure"),
    "PREEMPTED": ("#b45309", "Preempted"),
}


def parse_slurm_time(t):
    """Slurm [D-]HH:MM:SS / MM:SS -> seconds, None for UNLIMITED/unknown."""
    t = (t or "").strip()
    if not t or t in ("UNLIMITED", "INVALID", "Partition_Limit"):
        return None
    days = 0
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d)
    parts = [int(x) for x in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + sec


def fmt_hms(sec):
    if sec is None:
        return "?"
    d, rem = divmod(int(sec), 86400)
    h, rem = divmod(rem, 3600)
    m, s_ = divmod(rem, 60)
    return ("%d-" % d if d else "") + "%d:%02d:%02d" % (h, m, s_)


def slurm_info(jobid):
    """{'state', 'elapsed', 'limit'} from squeue, then sacct; None if Slurm can't tell."""
    def run(cmd):
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               universal_newlines=True, timeout=8)
            return p.stdout.strip()
        except Exception:
            return ""
    out = run(["squeue", "-j", str(jobid), "-h", "-o", "%T|%M|%l"])
    if not out:
        out = run(["sacct", "-j", str(jobid), "-n", "-X", "-P", "--format=State,Elapsed,Timelimit"])
    if not out:
        return None
    fields = out.splitlines()[0].split("|")
    if len(fields) < 3:
        return None
    state = fields[0].split()[0] if fields[0] else "?"   # "CANCELLED by 123" -> CANCELLED
    try:
        elapsed, limit = parse_slurm_time(fields[1]), parse_slurm_time(fields[2])
    except ValueError:
        elapsed = limit = None
    return {"state": state, "elapsed": elapsed, "limit": limit}


def fmt_val(v):
    if v is None:
        return "nan"
    if v == 0 or 1e-3 <= abs(v) < 1e6:
        return ("%.4g" % v)
    return "%.3e" % v


def build_status(jobdir, jobid, slurm=None):
    thermo = Thermo()
    report = build_report(jobdir, jobid, thermo)
    st = {"slurm": slurm, "report": report is not None, "step": None, "values": [],
          "rate": None, "rate_kind": None, "run_no": None, "runs_total": None,
          "errors": 0, "warnings": 0, "first_error": None}
    if report is None:
        return st
    runs = report["runs"]
    st["runs_total"] = len(runs)
    msgs = report["messages"]
    errs = [m for m in msgs if m["severity"] == "error"]
    st["errors"] = sum(m["count"] for m in errs)
    st["warnings"] = sum(m["count"] for m in msgs if m["severity"] == "warning")
    st["first_error"] = errs[0]["message"] if errs else None

    last = thermo.values(thermo.last)
    first = thermo.values(thermo.first)
    if last:
        xkey = thermo.keywords[0]
        st["step"] = last.get(xkey)
        st["values"] = [(k, last[k]) for k in STRIP_KEYWORDS if k in last][:4]
    # Rate: the finished run's own number; else CPU keyword (seconds since run start); else
    # an average over the job's elapsed time, only for the first run (no earlier runs mixed in).
    latest = runs[-1] if runs else None
    if latest and latest["finished"]:
        for p in latest["performance"]:
            if p.endswith("timesteps/s"):
                st["rate"], st["rate_kind"] = float(p.split()[0]), "run"
    elif last and first and st["step"] is not None:
        xkey = thermo.keywords[0]
        done = (last.get(xkey) or 0) - (first.get(xkey) or 0)
        cpu = last.get("CPU")
        if cpu and cpu > 0 and done > 0:
            st["rate"], st["rate_kind"] = done / cpu, "live"
        elif slurm and slurm.get("elapsed") and slurm["elapsed"] > 120 and len(runs) == 1 and done > 0:
            st["rate"], st["rate_kind"] = done / slurm["elapsed"], "avg"
    if runs:
        st["run_no"] = len(runs)
    return st


def render_status(st):
    sl = st["slurm"] or {}
    state = sl.get("state", "?")
    colour, label = STATE_STYLE.get(state, ("#64748b", state.title() if state != "?" else "Unknown"))
    bad = colour == "#dc2626"
    items = []
    # elapsed / limit, amber above 90 %
    if sl.get("elapsed") is not None:
        txt = fmt_hms(sl["elapsed"])
        warn = False
        if sl.get("limit"):
            txt += " / " + fmt_hms(sl["limit"])
            warn = state in ("RUNNING", "COMPLETING") and sl["elapsed"] > 0.9 * sl["limit"]
        items.append(("<span title='Elapsed / time limit'%s>&#9201; %s</span>"
                      % (" style='color:#b45309;font-weight:600'" if warn else "", e(txt))))
    if st["step"] is not None:
        run_txt = ""
        if st["runs_total"] and st["runs_total"] > 1:
            run_txt = " (run %d)" % st["run_no"]
        items.append("<span title='Last thermo step'>step %s%s</span>" % ("{:,}".format(int(st["step"])), run_txt))
    if st["rate"]:
        tip = {"run": "From the LAMMPS performance summary of the last finished run",
               "live": "From the CPU thermo keyword",
               "avg": "Average since the job started (includes setup time)"}[st["rate_kind"]]
        items.append("<span title='%s'>%s%s steps/s</span>"
                     % (e(tip), "&asymp; " if st["rate_kind"] == "avg" else "", "{:,.0f}".format(st["rate"])
                        if st["rate"] >= 100 else "%.3g" % st["rate"]))
    for k, v in st["values"]:
        items.append("<span title='Last thermo value'>%s <b>%s</b></span>" % (e(k), e(fmt_val(v))))
    if st["errors"]:
        items.append("<span style='color:#dc2626;font-weight:600'>&#10006; %d error%s</span>"
                     % (st["errors"], "" if st["errors"] == 1 else "s"))
    if st["warnings"]:
        items.append("<span style='color:#b45309'>&#9888; %d warning%s</span>"
                     % (st["warnings"], "" if st["warnings"] == 1 else "s"))
    if not st["errors"] and not st["warnings"] and st["report"]:
        items.append("<span style='color:#047857'>no warnings</span>")
    if not st["report"]:
        items.append("<span style='color:#64748b'>no output yet</span>")

    sep = "<span style='color:#cbd5e1'>&middot;</span>"
    html_ = ["<div style=\"display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;"
             "font-family:'Inter',system-ui,sans-serif;font-size:13px;color:#1e293b;padding:8px 12px;"
             "border:1px solid %s;border-radius:10px;background:%s\">"
             % ("#fecaca" if bad or st["errors"] else "#e2e8f0", "#fef2f2" if bad or st["errors"] else "#f8fafc"),
             "<span style='display:inline-flex;align-items:center;gap:6px;font-weight:700;color:%s'>"
             "<span style='width:9px;height:9px;border-radius:50%%;background:%s;display:inline-block'></span>%s</span>"
             % (colour, colour, e(label))]
    if not st["slurm"]:
        html_.pop()  # Slurm could not tell: no state label rather than "Unknown"
        items_html = sep.join(items)
    else:
        items_html = "".join(sep + it for it in items)
    html_.append(items_html)
    html_.append("</div>")
    if st["first_error"]:
        html_.append("<div style='margin-top:4px;font-size:12px;color:#b91c1c;font-family:monospace'>%s</div>"
                     % e(st["first_error"][:200]))
    return "".join(html_)



def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jobdir")
    ap.add_argument("jobid")
    ap.add_argument("--json", action="store_true", help="print the parsed report as JSON")
    ap.add_argument("--status-line", action="store_true", help="print the one-line status strip instead")
    args = ap.parse_args()
    if not args.jobid:
        print("<div class='muted'>No job selected.</div>")
        return
    if args.status_line:
        try:
            st = build_status(args.jobdir, args.jobid, slurm_info(args.jobid))
            print(json.dumps(st, default=str) if args.json else render_status(st))
        except Exception as ex:  # never break the page
            print("<div style='color:#64748b'>Status unavailable: %s</div>" % e(str(ex)))
            print("lammps_run_report --status-line: %r" % ex, file=sys.stderr)
        return
    try:
        report = build_report(args.jobdir, args.jobid)
    except Exception as ex:  # never break the page
        print("<div>Could not read the LAMMPS output: %s</div>" % e(str(ex)))
        print("lammps_run_report: %r" % ex, file=sys.stderr)
        return
    if report is None:
        print("<div style='color:#64748b'>No output file yet (out.%s).</div>" % e(args.jobid))
        return
    if args.json:
        print(json.dumps(report, indent=1, default=str))
    else:
        print(render(report))


if __name__ == "__main__":
    main()
