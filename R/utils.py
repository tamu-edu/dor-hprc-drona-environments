import getpass
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from drona_utils import (drona_add_additional_file, drona_add_error, drona_add_mapping, drona_add_note,
                         drona_add_warning)
from drona_runtime_utils import cluster_max_cores, retrieve_slurm_params

# A version is the full "module load" argument, e.g. "GCC/14.3.0 OpenMPI/5.0.8 R_tamu/4.5.2",
# from module_versions/<cluster>.json (see configuration/README)
MODULE_VERSIONS_DIR = Path(__file__).resolve().parent / "module_versions"
MODULE_LIST = re.compile(r"^[A-Za-z0-9_.+-]+(/[A-Za-z0-9_.+-]+)*( +[A-Za-z0-9_.+-]+(/[A-Za-z0-9_.+-]+)*)*$")


def retrieve_driver_contents(mode, cancel_jobs=""):
    """Create mode submits jobscripts/r-job.slurm with the shared sbatch driver, manage mode runs
    the cancel/delete actions. An unselected mode (None, "" or the unmapped "$name") gets the create driver."""
    base = Path(__file__).resolve().parent

    if mode == "manage":
        file_path = base / "drivers/driver-manage.sh"
    else:
        file_path = base / "drivers/driver-create.sh"
        drona_add_additional_file("jobscripts/r-job.slurm", "Job Script")
        drona_add_additional_file("jobscripts/drona_r_monitor.py", "R Worker Monitor", preview_order=-1,
                                  name=".drona/drona_r_monitor.py")

    with open(file_path, 'r') as file:
        return file.read()


def _int(value, default):
    value = str(value or "").strip()
    return int(value) if value.isdigit() else default


def _clean(value):
    value = str(value or "").strip()
    return "" if value.startswith("$") else value


def _many_layout(runnodes, runspernode, threads, nruns, warn=True):
    """MANY RUNS: (nodes, runs at the same time per node, threads per run). Runs per node defaults to
    as many as fit, but no more than the runs need; threads are reduced when runs x threads does not
    fit on a node; no more nodes than runs."""
    n = max(_int(_clean(runnodes), 1), 1)
    t = max(_int(_clean(threads), 1), 1)
    c = _int(_clean(runspernode), 0)
    say = drona_add_warning if warn else (lambda msg: None)
    if nruns and n > nruns:
        say(f"#nodes reduced from {n} to {nruns}: there are only {nruns} runs.")
        n = nruns
    need = -(-nruns // n) if nruns else 0          # runs per node when all run at once
    maxcores = cluster_max_cores()
    if maxcores:
        if t > maxcores:
            say(f"Threads per R process reduced from {t} to {maxcores} ({maxcores} cores per node).")
            t = maxcores
        if c > maxcores:
            say(f"Runs at the same time per node reduced from {c} to {maxcores} ({maxcores} cores per node).")
            c = maxcores
        if c == 0:
            c = max(1, min(maxcores // t, need or maxcores))
        elif c * t > maxcores:
            newt = max(maxcores // c, 1)
            say(f"Runs at the same time per node x threads per R process cannot exceed {maxcores} cores per node: "
                f"threads reduced from {t} to {newt}.")
            t = newt
    elif c == 0:
        say("Cores per node unknown on this cluster: running 1 run at a time per node. "
            "Set Runs at the same time per node to run more.")
        c = 1
    if need and c > need:
        say(f"Runs at the same time per node reduced from {c} to {need}: that runs all {nruns} runs at once.")
        c = need
    return n, c, t


def retrieve_r_slurm_params(parallel, workers, threads, runnodes, runspernode, memory, walltime, account, extra,
                            runs="", runargs="", runfile=""):
    """Slurm layout per parallel cluster type, with t = threads per R process (BLAS/OpenMP threads):
       none        1 node, 1 task, t cpus
       sock/other  1 node, 1 task, (workers+1)*t cpus (the R main process plus the workers, all on one node)
       mpi         workers+1 tasks with t cpus each, as many nodes as needed; R starts with mpirun -np 1
                   and spawns the workers into the remaining slots
       many        tamulauncher: #nodes tasks, one per node, each with (runs per node)*t cpus
    Adds R_WORKERS, R_PARALLEL, R_THREADS and RSBATCH for the job script (the run line itself comes
    from retrieve_main); the shared cluster checks add the Slurm mappings."""
    parallel = _clean(parallel)
    if parallel == "many":
        n, c, t = _many_layout(runnodes, runspernode, threads, len(_run_args(runs, runargs, runfile)[0]))
        drona_add_mapping("R_WORKERS", "0")
        drona_add_mapping("R_PARALLEL", "many")
        drona_add_mapping("R_THREADS", str(t))
        drona_add_mapping("RSBATCH", "#SBATCH --ntasks-per-node=1   # one tamulauncher per node")
        return retrieve_slurm_params(str(n), str(n), str(c * t), memory, "none", "", walltime, account, extra)

    w = max(_int(workers, 1), 1) if parallel in ("sock", "mpi", "other") else 0
    t = max(_int(threads, 1), 1)

    maxcores = cluster_max_cores()
    if maxcores and t > maxcores:
        drona_add_warning(f"Threads per R process reduced from {t} to {maxcores} ({maxcores} cores per node).")
        t = maxcores

    if parallel == "mpi":
        nodes, tasks, cpus = "", str(w + 1), str(t)
    else:
        if w and maxcores and (w + 1) * t > maxcores:
            fit = maxcores // t - 1
            if fit < 1:
                t = max(maxcores // 2, 1)
                fit = maxcores // t - 1
                drona_add_warning(f"Threads per R process reduced to {t}, so at least one worker fits on a node.")
            drona_add_warning(f"A {parallel.upper()} cluster runs on a single node: #workers reduced from {w} to {fit} "
                              f"({maxcores} cores per node, {t} per R process including the main process). "
                              f"Use MPI for more workers.")
            w = fit
        nodes, tasks, cpus = "1", "1", str((w + 1) * t)

    drona_add_mapping("R_WORKERS", str(w))
    drona_add_mapping("R_PARALLEL", parallel or "none")
    drona_add_mapping("R_THREADS", str(t))
    drona_add_mapping("RSBATCH", "")
    if w and t > 1:
        drona_add_note(f"Each of the {w + 1} R processes may use {t} threads ({(w + 1) * t} cores in total). "
                       f"Keep threads at 1 unless every worker does heavy matrix math.")

    # CPU only: R_tamu has no GPU-enabled packages, so the form offers no GPU
    return retrieve_slurm_params(nodes, tasks, cpus, memory, "none", "", walltime, account, extra)


def _cluster_versions():
    cluster = subprocess.run(["/sw/local/bin/clustername"], capture_output=True, text=True, check=False).stdout.strip()
    try:
        with open(MODULE_VERSIONS_DIR / (cluster + ".json")) as f:
            return [o["value"] for o in json.load(f) if o.get("value")]
    except (OSError, ValueError, TypeError, KeyError):
        return []


def retrieve_r_module(version=""):
    """Modules for "module load": the selected dropdown value, or the newest version of this
    cluster when none was selected."""
    version = " ".join((version or "").split())
    if version.startswith("$"):
        version = ""
    if not version:
        versions = _cluster_versions()
        if not versions:
            drona_add_error("No R versions are configured for this cluster. Ask HPRC to run "
                            "configuration/generate_module_versions.py for the R environment.")
            return ""
        version = versions[0]
    if not MODULE_LIST.match(version):
        drona_add_error(f"Invalid R version: {version!r}")
        return ""
    return version


def retrieve_main(mainscript="", version="", rlibs="", parallel="", threads="",
                  runs="", runargs="", runfile="", runnodes="", runspernode=""):
    """The job script's run line (RRUN): Rscript <script>, with mpirun -np 1 for MPI, or tamulauncher
    over r-runs.commands for MANY RUNS. Uploaded scripts arrive as a bare file name in the workflow
    directory, cluster picks as a full path; the job runs in the workflow directory, so both work as given.
    A script picked on the cluster is also checked (see check_script): uploads reach the server only at
    submit, so they can't be."""
    mainscript = (mainscript or "").strip()
    parallel = _clean(parallel)
    if not mainscript or mainscript.startswith("$"):
        drona_add_warning("No main R script selected. Select one, or edit the Rscript line in the job script preview.")
        script = ""
    else:
        script = shlex.quote(mainscript)
        if os.path.isabs(mainscript):
            try:
                check_script(mainscript, version, rlibs, parallel, threads)
            except Exception:   # a check must never stop the job script from being generated
                pass

    if parallel == "many":
        drona_add_mapping("RRUN", many_runs(script, runs, runargs, runfile, runnodes, runspernode, threads))
    else:
        drona_add_mapping("RRUN", ("mpirun -np 1 " if parallel == "mpi" else "") + "Rscript " + script)
    return script


RUNS_FILE = "r-runs.commands"
RUNS_LOG_DIR = "r-runs"
MAX_RUNS = 100000


def _arg_lines(text):
    return [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]


def _run_args(runs, runargs, runfile):
    """(argument line per run, where they came from, error or ""); no messages, so both map functions can call it."""
    runfile, runargs = _clean(runfile), _clean(runargs)
    if runfile:
        source = "the arguments file " + os.path.basename(runfile)
        try:
            with open(runfile, errors="replace") as f:
                args = _arg_lines(f.read())
        except OSError:
            return [], source, f"Cannot read the arguments file {runfile}."
        return args, source, "" if args else f"No runs: {source} has no lines with arguments."
    if runargs:
        args = _arg_lines(runargs)
        return args, "Arguments per run", "" if args else "No runs: Arguments per run has no lines with arguments."
    n = _int(_clean(runs), 0)
    return [""] * n, "#runs", "" if n else "Set #runs, or give arguments per run."


def many_runs(script, runs, runargs, runfile, runnodes, runspernode, threads):
    """Write r-runs.commands (one Rscript per run, run number in DRONA_R_RUN, output in r-runs/run-<n>.log)
    and return the job script lines that run it with tamulauncher."""
    args, source, error = _run_args(runs, runargs, runfile)
    if error:
        drona_add_error(error)
    if len(args) > MAX_RUNS:
        drona_add_error(f"{len(args)} runs is more than the {MAX_RUNS} this form supports; split them over several jobs.")
        args = args[:MAX_RUNS]
    if any("\\" in a for a in args):
        drona_add_warning("tamulauncher removes backslashes (\\) from commands; avoid them in the arguments.")
    _, per_node, _ = _many_layout(runnodes, runspernode, threads, len(args), warn=False)

    lines = ["DRONA_R_RUN=%d Rscript %s%s > %s/run-%d.log 2>&1" % (i, script, (" " + a) if a else "", RUNS_LOG_DIR, i)
             for i, a in enumerate(args, 1)]
    drona_add_additional_file(content="\n".join(lines) + "\n", name=RUNS_FILE,
                              preview_name="Runs (%d)" % len(lines), preview_order=1)
    if lines:
        drona_add_note(f"{len(lines)} runs from {source}, {per_node} at the same time per node.")
    return ("# MANY RUNS: tamulauncher runs every line of %s, %d at the same time per node. Each run's output\n"
            "# goes to %s/run-<n>.log. Finished runs are recorded in .tamulauncher-log/, so submitting this script again skips them.\n"
            "mkdir -p %s\n"
            "tamulauncher --commands-pernode %d %s" % (RUNS_FILE, per_node, RUNS_LOG_DIR, RUNS_LOG_DIR, per_node, RUNS_FILE))


# ---------------------------------------------------------------- script checks (preview)
# Static checks of the main script, cheap enough for the web server: one file read and a
# directory listing per library; no module load, no R.

MAX_SCRIPT_BYTES = 2 * 1024 * 1024
NAME = r"[A-Za-z][A-Za-z0-9.]*"
LOAD_CALL = re.compile(r"\b(?:library|require|requireNamespace|loadNamespace)\s*\(\s*(?:package\s*=\s*)?"
                       r"[\"']?(" + NAME + r")[\"']?\s*([,)])")
NAMESPACE = re.compile(r"(?<![A-Za-z0-9._])(" + NAME + r"):::?(?=[A-Za-z._`])")
PARALLEL_CODE = re.compile(
    r"\b(makeCluster|makePSOCKcluster|makeForkCluster|makeMPIcluster|startMPIcluster|registerDoParallel|"
    r"registerDoMC|registerDoSNOW|registerDoMPI|registerDoFuture|mclapply|mcmapply|pvec|parLapply|parSapply|"
    r"parApply|clusterApply|clusterMap|bplapply|MulticoreParam|SnowParam|future_lapply|future_sapply|"
    r"future_map\w*|future_pmap\w*)\s*\(|\bplan\s*\(\s*(?:future::)?(multisession|multicore|cluster)\b")
REGISTER = re.compile(r"\bregisterDo\w+\s*\(")
MPI_CODE = re.compile(r"\b(Rmpi|doMPI|pbdMPI|makeMPIcluster|startMPIcluster|mpi\.spawn\.Rslaves)\b|"
                      r"type\s*=\s*[\"']MPI[\"']")


def _strip_line(line, keep_strings):
    """The line without its comment; string contents blanked unless keep_strings."""
    out, quote, i = [], None, 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\":
                out.append(c + line[i + 1:i + 2] if keep_strings else "  ")
                i += 2
                continue
            if c == quote:
                quote = None
                out.append(c)
            else:
                out.append(c if keep_strings else " ")
        elif c in "\"'":
            quote = c
            out.append(c)
        elif c == "#":
            break
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _first_lines(pattern, lines, group=1):
    """{match: first line number} for a regex over numbered lines."""
    found = {}
    for n, line in enumerate(lines, 1):
        for m in pattern.finditer(line):
            key = next((g for g in m.groups() if g), m.group(0)) if group is None else m.group(group)
            found.setdefault(key, n)
    return found


def _library_table(version):
    """Library directories and personal library for an R version, from module_versions/<cluster>.libs.json
    (configuration/generate_r_libraries.py); None when the file or the version is missing."""
    cluster = subprocess.run(["/sw/local/bin/clustername"], capture_output=True, text=True, check=False).stdout.strip()
    try:
        with open(MODULE_VERSIONS_DIR / (cluster + ".libs.json")) as f:
            table = json.load(f)
    except (OSError, ValueError):
        return None
    return table.get(version)


def _installed(dirs):
    names = set()
    for d in dirs:
        try:
            names.update(os.listdir(d))
        except OSError:
            pass
    return names


def _cores(threads):
    n = max(_int(threads, 1), 1)
    return "1 core" if n == 1 else "%d cores" % n


def _line_list(found, limit=6):
    items = ["%s (line %d)" % (k, v) for k, v in sorted(found.items(), key=lambda kv: kv[1])]
    return ", ".join(items[:limit]) + (", ..." if len(items) > limit else "")


def check_script(path, version="", rlibs="", parallel="", threads=""):
    """Warn in the preview about packages that are not installed and about parallel code that does
    not match the form."""
    if not os.path.isfile(path) or os.path.getsize(path) > MAX_SCRIPT_BYTES:
        return
    with open(path, errors="replace") as f:
        raw = f.read().splitlines()
    code = [_strip_line(l, keep_strings=True) for l in raw]
    bare = [_strip_line(l, keep_strings=False) for l in raw]
    name = os.path.basename(path)
    sources = any(re.search(r"\bsource\s*\(", l) for l in bare)

    # 1. packages
    version = " ".join((version or "").split())
    version = "" if version.startswith("$") else version
    if not version:
        versions = _cluster_versions()
        version = versions[0] if versions else ""
    libs = _library_table(version)
    if libs is not None:
        used = {}
        for n, line in enumerate(code, 1):
            for m in LOAD_CALL.finditer(line):
                if m.group(2) == "," and "character.only" in line[m.end():]:
                    continue   # library(pkg, character.only = TRUE): the name is in a variable
                used.setdefault(m.group(1), n)
        for pkg, n in _first_lines(NAMESPACE, bare).items():
            used.setdefault(pkg, n)
        if used:
            rl = (rlibs or "").strip()
            extra = [] if rl.startswith("$") else [d for d in rl.split(":") if d]
            scratch = os.environ.get("SCRATCH") or "/scratch/user/" + getpass.getuser()
            personal = [os.path.join(scratch, libs["user_lib"])] if libs.get("user_lib") else []
            have = _installed(libs.get("libs", []) + extra + personal)
            missing = {p: n for p, n in used.items() if p not in have}
            if missing:
                label = version.split()[-1] if version else "the selected R version"
                drona_add_warning(
                    f"{name} uses packages that are not installed for {label}: {_line_list(missing)}. "
                    f"Install them first (load the same R module, start R and run "
                    f"install.packages(\"{sorted(missing, key=missing.get)[0]}\"); Bioconductor packages with "
                    f"BiocManager::install), or add the directory that has them under Additional R_LIBS_USER paths. "
                    f"Otherwise the job stops with \"there is no package called ...\".")

    # 2. parallel code vs. the form
    parallel = (parallel or "").strip()
    parallel = "none" if not parallel or parallel.startswith("$") else parallel
    par = _first_lines(PARALLEL_CODE, bare, group=None)
    dopar = next((n for n, l in enumerate(bare, 1) if "%dopar%" in l), 0)
    registered = any(REGISTER.search(l) for l in bare)

    detect = next((n for n, l in enumerate(bare, 1) if re.search(r"\bdetectCores\s*\(", l)), 0)
    if detect:
        drona_add_warning(
            f"{name} line {detect}: detectCores() returns all cores of the node, not the cores of your job, so R "
            f"starts more workers than the job has and they slow each other down. Use "
            + ("as.integer(Sys.getenv(\"DRONA_R_THREADS\")) instead (the threads per run on this form)."
               if parallel == "many" else
               "as.integer(Sys.getenv(\"DRONA_R_WORKERS\")) instead (the #workers on this form)."))

    if parallel == "many":
        if par or (dopar and registered):
            first = min(par.items(), key=lambda kv: kv[1]) if par else ("%dopar%", dopar)
            drona_add_warning(
                f"{name} starts its own parallel workers ({first[0]} on line {first[1]}), but with MANY RUNS each run "
                f"gets only {_cores(threads)}, so its workers share them. Make the script serial (the runs are the "
                f"parallelism), or raise Threads per R process and size the workers with "
                f"as.integer(Sys.getenv(\"DRONA_R_THREADS\")).")
    elif parallel == "none" and (par or dopar):
        first = min(par.items(), key=lambda kv: kv[1]) if par else ("%dopar%", dopar)
        drona_add_warning(
            f"{name} looks parallel ({first[0]} on line {first[1]}), but \"How your script runs in parallel\" is NONE, so all of it "
            f"runs on {_cores(threads)}. Select SOCK (makeCluster/parLapply, doParallel, future "
            f"multisession), OTHER (mclapply, doMC, future multicore) or MPI, and set #workers.")
    elif parallel != "none" and not par and not dopar:
        drona_add_note(
            f"No parallel code found in {name} (makeCluster, mclapply, foreach %dopar%, future, BiocParallel, ...), "
            f"so the #workers cores may stay idle"
            + (" unless the parallel code is in a sourced file." if sources else ". If the script runs serially, "
               "select NONE, or MANY RUNS to run it many times."))
    elif parallel == "mpi" and par and not any(MPI_CODE.search(l) for l in code):
        drona_add_note(f"\"How your script runs in parallel\" is MPI, but {name} has no MPI code (Rmpi, doMPI, makeCluster(type = \"MPI\")). "
                       f"A cluster from makeCluster(n) or mclapply runs on one node: SOCK or OTHER fits it better.")

    if dopar and not registered and not sources:
        drona_add_warning(
            f"{name} line {dopar}: foreach %dopar% is used, but no parallel backend is registered "
            f"(registerDoParallel, registerDoFuture, ...), so the loop runs one iteration at a time.")


def retrieve_r_libs(rlibs=""):
    """R_LIBS_USER = the selected directories (the picker returns several as one colon-joined string)
    added to R_LIBS_USER. R_tamu's R_LIBS and its Rscript wrapper still come first in .libPaths()."""
    rlibs = (rlibs or "").strip()
    if rlibs.startswith("$"):
        return ""
    dirs = [d for d in rlibs.split(":") if d]
    dirs = [d for i, d in enumerate(dirs) if d not in dirs[:i]]
    if not dirs:
        return ""
    return "export R_LIBS_USER=" + shlex.quote(":".join(dirs)) + "${R_LIBS_USER:+:$R_LIBS_USER}"
