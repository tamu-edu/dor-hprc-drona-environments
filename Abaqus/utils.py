import grp
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from drona_utils import drona_add_additional_file, drona_add_error, drona_add_mapping, drona_add_note, drona_add_warning
from drona_runtime_utils import cluster_max_cores, retrieve_slurm_params

# A version is the full "module load" argument, e.g. "ABAQUS/2025", from module_versions/<cluster>.json
# (built by an admin with configuration/generate_module_versions.py ABAQUS)
MODULE_VERSIONS_DIR = Path(__file__).resolve().parent / "module_versions"
MODULE_LIST = re.compile(r"^[A-Za-z0-9_.+-]+(/[A-Za-z0-9_.+-]+)*( +[A-Za-z0-9_.+-]+(/[A-Za-z0-9_.+-]+)*)*$")

# Abaqus is licensed software: the module only loads for members of this group
ACCESS_GROUP = "abaqus"

# "any" GPU: run the cluster checks with a GPU type that has the same partition and the
# strictest limits on that cluster, then drop the type from --gres (see _any_gpu_filter)
ANY_GPU_PROXY = {"aces": "a30", "grace": "a40", "faster": "a30", "launch": "a30"}

MAX_INPUT_BYTES = 50 * 1024 * 1024   # only the first part of a larger input file is scanned


def _clean(value):
    value = str(value or "").strip()
    return "" if value.startswith("$") else value


def _int(value, default):
    value = _clean(value)
    return int(value) if value.isdigit() else default


def _cluster():
    return subprocess.run(["/sw/local/bin/clustername"], capture_output=True, text=True, check=False).stdout.strip()


def retrieve_driver_contents(mode, cancel_jobs=""):
    """Create mode submits jobscripts/abaqus-job.slurm with the shared sbatch driver, manage mode runs
    the cancel/delete actions. An unselected mode (None, "" or the unmapped "$name") gets the create driver."""
    base = Path(__file__).resolve().parent
    if mode == "manage":
        file_path = base / "drivers/driver-manage.sh"
    else:
        file_path = base / "drivers/driver-create.sh"
        drona_add_additional_file("jobscripts/abaqus-job.slurm", "Job Script")
    with open(file_path) as f:
        return f.read()


# ---------------------------------------------------------------- version and access

def _cluster_versions():
    try:
        with open(MODULE_VERSIONS_DIR / (_cluster() + ".json")) as f:
            return [o["value"] for o in json.load(f) if o.get("value")]
    except (OSError, ValueError, TypeError, KeyError):
        return []


def _has_access():
    """True when the user is in the Abaqus group (or the group doesn't exist on this cluster)."""
    try:
        group = grp.getgrnam(ACCESS_GROUP)
    except KeyError:
        return True
    return group.gr_gid in os.getgroups() or os.getenv("USER", "") in group.gr_mem


def retrieve_abaqus_module(version=""):
    """Module for "module load": the selected dropdown value, or the newest version of this cluster."""
    if not _has_access():
        drona_add_warning("Abaqus is restricted software: you are not in the access list, so the job will fail "
                          "at \"module load\". Contact the HPRC helpdesk to request access.")
    version = " ".join(_clean(version).split())
    if not version:
        versions = _cluster_versions()
        if not versions:
            drona_add_error("No Abaqus versions are configured for this cluster. Ask HPRC to run "
                            "configuration/generate_module_versions.py ABAQUS for the Abaqus environment.")
            return ""
        version = versions[0]
    if not MODULE_LIST.match(version):
        drona_add_error(f"Invalid Abaqus version: {version!r}")
        return ""
    return version


# ---------------------------------------------------------------- input file

def _job_name(path):
    """Abaqus job name from the input file name: letters, digits, _ and -; starting with a letter."""
    name = os.path.basename(path)
    if name.lower().endswith(".inp"):
        name = name[:-4]
    name = re.sub(r"[^A-Za-z0-9_-]", "_", name) or "job"
    return name if name[0].isalpha() else "job_" + name


# Keywords that only work with a user subroutine: *USER MATERIAL (UMAT/VUMAT), *USER ELEMENT (UEL/VUEL),
# *USER DEFINED FIELD (USDFLD), *USER OUTPUT VARIABLES (UVARM), and USER as a parameter value
# (e.g. *BOUNDARY, USER -> DISP; *PLASTIC, HARDENING=USER -> UHARD; *INITIAL CONDITIONS, ..., USER -> SIGINI/SDVINI)
USER_KEYWORDS = re.compile(r"^\*(user ?material|user ?element|user ?defined ?field|user ?output ?variables)", re.I)
# Files an input file reads: *INCLUDE, INPUT=...; *AMPLITUDE, INPUT=...; *NODE, INPUT=...; ... FILE=...
FILE_PARAM = re.compile(r",\s*(input|file)\s*=\s*(\"[^\"]*\"|[^,]+)", re.I)
USER_PARAM = re.compile(r",\s*(\w+\s*=\s*)?user\s*(,|$)", re.I)


def _analysis(path):
    """Quick scan of a cluster input file: {"kind": "explicit" | "standard", "includes": [...],
    "user": [keywords needing a user subroutine], "restart": *RESTART, READ, "import": *IMPORT,
    "submodel": *SUBMODEL}. Empty dict when the file can't be read here (an upload arrives only at submit)."""
    if not os.path.isabs(path) or not os.path.isfile(path):
        return {}
    try:
        with open(path, errors="replace") as f:
            text = f.read(MAX_INPUT_BYTES)
    except OSError:
        return {}
    info = {"kind": "standard", "includes": [], "user": [], "restart": False, "import": False, "submodel": False}
    for line in text.splitlines():
        if not line.startswith("*") or line.startswith("**"):
            continue
        key = line.replace(" ", "").lower()
        word = key.split(",")[0]
        if word == "*dynamic" and "explicit" in key.split(",")[1:]:
            info["kind"] = "explicit"
        if word == "*restart" and "read" in key.split(",")[1:]:
            info["restart"] = True
        info["import"] |= word == "*import"
        info["submodel"] |= word == "*submodel"
        if (USER_KEYWORDS.match(line.strip()) or USER_PARAM.search(line.strip())) and len(info["user"]) < 5:
            name = line.split(",")[0].strip().upper()
            if name not in info["user"]:
                info["user"].append(name)
        for m in FILE_PARAM.finditer(line.strip()):
            ref = m.group(2).strip().strip('"')
            if ref and ref not in info["includes"]:
                info["includes"].append(ref)
    return info


def _referenced_files(inputfile):
    """(relative references found next to the input file, references not found) for a cluster input file,
    following *INCLUDE files up to a few levels. Relative paths are resolved against the input's folder."""
    folder = os.path.dirname(inputfile)
    found, missing, todo, seen = [], [], [inputfile], set()
    while todo and len(seen) < 50:
        path = todo.pop(0)
        if path in seen:
            continue
        seen.add(path)
        for ref in _analysis(path).get("includes", []):
            full = ref if os.path.isabs(ref) else os.path.join(folder, ref)
            if not os.path.exists(full):
                missing.append(ref)
                continue
            if not os.path.isabs(ref) and ref not in found:
                found.append(ref)
            if full.lower().endswith(".inp"):
                todo.append(full)
    return found, missing


def retrieve_input(inputfile=""):
    """input= for the abaqus command (quoted) and job= (ABQJOB mapping). Uploaded files arrive as a bare
    name in the workflow directory, cluster picks as a full path; the job runs in the workflow directory."""
    inputfile = _clean(inputfile)
    if not inputfile:
        drona_add_warning("No input file selected. Select an Abaqus input file (.inp), or edit the abaqus line "
                          "in the job script preview.")
        drona_add_mapping("ABQJOB", "job")
        drona_add_mapping("ABQ_INPUT_LINKS", "")
        return ""
    if not inputfile.lower().endswith(".inp"):
        drona_add_warning(f"{os.path.basename(inputfile)} does not end in .inp; Abaqus expects an input file.")
    drona_add_mapping("ABQJOB", _job_name(inputfile))
    links = ""
    if os.path.isabs(inputfile):
        found, missing = _referenced_files(inputfile)
        # the job runs in the workflow directory: link what the input reads (the first part of each relative
        # path, so mesh/part.inp links the folder mesh) from the input's folder, unless an upload is there
        tops = sorted({Path(r).parts[0] for r in found if Path(r).parts and Path(r).parts[0] != ".."})
        if tops:
            folder = shlex.quote(os.path.dirname(inputfile))
            links = ("# Files the input file reads (*INCLUDE, INPUT=, FILE=), linked from its folder\n" +
                     "".join(f"[ -e {shlex.quote(t)} ] || ln -s {folder}/{shlex.quote(t)} .\n" for t in tops))
            drona_add_note("Files the input file reads are linked into the workflow directory: " +
                           ", ".join(tops[:5]) + (" ..." if len(tops) > 5 else ""))
        if any(Path(r).parts[:1] == ("..",) for r in found):
            drona_add_warning("The input file reads files through ../ paths; these don't resolve from the workflow "
                              "directory. Use full paths in the input file.")
        if missing:
            drona_add_warning("The input file reads files that were not found: " + ", ".join(missing[:5]) +
                              ". Upload them under Other model files, or fix the paths in the input file.")
    drona_add_mapping("ABQ_INPUT_LINKS", links)
    return shlex.quote(inputfile)


# ---------------------------------------------------------------- resources

def _any_gpu_filter(key, value):
    # --gres=gpu:<type>:<n> -> --gres=gpu:<n>
    if key == "PARTITION":
        return re.sub(r"--gres=gpu:[^:\s]+:", "--gres=gpu:", value)
    return value


def license_tokens(cores):
    """Abaqus analysis tokens for a job on `cores` cores: int(5 x cores^0.422)."""
    return int(5 * max(cores, 1) ** 0.422)


def retrieve_abaqus_slurm_params(parallel, cores, nodes, inputfile, memory, gpu, numgpu, walltime, account, extra):
    """Slurm layout and abaqus options per parallel mode (auto: serial for 1 core, threads up to a node,
    else mpi; nodes, when not given, are the fewest that hold the cores):
       serial   1 node, 1 task, 1 cpu
       threads  1 node, 1 task, `cores` cpus; abaqus mp_mode=threads (shared memory, one node)
       mpi      `nodes` nodes, `cores` tasks of 1 cpu; abaqus mp_mode=mpi with the job's hosts in abaqus_v6.env
    GPUs (Abaqus/Standard only) add gpus=<n>. Adds ABQ_CPUS, ABQ_MPMODE, ABQ_GPUS and ABQ_HOSTS for
    the job script; the shared cluster checks add the Slurm mappings."""
    parallel = _clean(parallel) or "auto"
    maxcores = cluster_max_cores()
    if parallel == "auto":   # 1 core serial, up to a node threads, more MPI over the fewest nodes
        want = max(_int(cores, 1), 1)
        parallel = "serial" if want == 1 else "mpi" if maxcores and want > maxcores else "threads"
    elif parallel == "threads" and max(_int(cores, 1), 1) == 1:
        parallel = "serial"
    gpu = _clean(gpu)
    gpu = "" if gpu == "none" else gpu
    ngpu = max(_int(numgpu, 1), 1) if gpu else 0
    analysis = _analysis(_clean(inputfile)).get("kind", "")

    if parallel == "serial":
        n, nodes_s, tasks, cpus = 1, "1", "1", "1"
    elif parallel == "threads":
        n = max(_int(cores, 1), 1)
        if maxcores and n > maxcores:
            drona_add_warning(f"#cores reduced from {n} to {maxcores}: threads mode runs on a single node "
                              f"({maxcores} cores). Use MPI for more cores.")
            n = maxcores
        nodes_s, tasks, cpus = "1", "1", str(n)
    else:   # mpi
        n = max(_int(cores, 2), 1)
        nn = max(_int(nodes, 1), 1)
        if nn > n:
            drona_add_warning(f"#nodes reduced from {nn} to {n}: there are only {n} cores.")
            nn = n
        if maxcores and n > nn * maxcores:
            need = -(-n // maxcores)
            if _clean(nodes):
                drona_add_warning(f"#nodes raised from {nn} to {need}: {n} cores need {need} nodes of {maxcores} cores.")
            else:
                drona_add_note(f"{n} cores need {need} nodes of {maxcores} cores: Abaqus runs with MPI over {need} nodes.")
            nn = need
        nodes_s, tasks, cpus = str(nn), str(n), "1"

    drona_add_mapping("ABQ_CPUS", str(n))
    drona_add_mapping("ABQ_MPMODE", "" if parallel == "serial" else f"mp_mode={'threads' if parallel == 'threads' else 'mpi'} ")
    drona_add_mapping("ABQ_GPUS", f"gpus={ngpu} " if ngpu else "")
    drona_add_mapping("ABQ_HOSTS", MPI_HOSTS if parallel == "mpi" else "")

    if ngpu and analysis == "explicit":
        drona_add_warning("This is an Abaqus/Explicit analysis (*DYNAMIC, EXPLICIT): GPUs are only used by "
                          "Abaqus/Standard, so the GPU would stay idle. Set Advanced > Use GPU to no GPU.")
    if analysis:
        drona_add_note(f"Abaqus/{analysis.capitalize()} analysis on {n} core{'s' if n != 1 else ''}"
                       f"{f' and {ngpu} GPU' + ('s' if ngpu != 1 else '') if ngpu else ''}: "
                       f"about {license_tokens(n)} license tokens while it runs.")
    else:
        drona_add_note(f"{n} core{'s' if n != 1 else ''}: about {license_tokens(n)} license tokens while it runs.")

    mapping_filter = None
    if gpu == "any":
        gpu = ANY_GPU_PROXY.get(_cluster(), "any")
        mapping_filter = _any_gpu_filter
    return retrieve_slurm_params(nodes_s, tasks, cpus, _clean(memory), gpu or "none", str(ngpu) if ngpu else "",
                                 _clean(walltime), _clean(account), _clean(extra), mapping_filter=mapping_filter)


# MPI mode: Abaqus starts its processes on the hosts in mp_host_list. Written at run time from the
# nodes Slurm gave the job, into abaqus_v6.env in the workflow directory (lines from an earlier run
# are replaced; other settings in the file are kept).
MPI_HOSTS = r'''# MPI: tell Abaqus which nodes the job got and how many cores on each (abaqus_v6.env)
ABQ_HOSTLIST=$(srun --overlap --ntasks="$SLURM_NTASKS" hostname -s | sort | uniq -c | awk '{printf "%s[\"%s\",%d]", (NR>1?",":""), $2, $1}')
touch abaqus_v6.env && sed -i '/# drona mp_host_list$/d' abaqus_v6.env
echo "mp_host_list=[${ABQ_HOSTLIST}]  # drona mp_host_list" >> abaqus_v6.env
'''


UNSAFE = re.compile(r"[;&|`$<>(){}\\\n]")


def retrieve_options(options=""):
    """Extra abaqus command options (e.g. double=both user=mysub.f), as typed, followed by a space."""
    options = " ".join(_clean(options).split())
    if not options:
        return ""
    if UNSAFE.search(options):
        drona_add_error("Additional Abaqus options may only hold options like double=both or user=sub.f "
                        "(no ; & | $ ` < > or brackets).")
        return ""
    reserved = [o for o in options.split() if o.split("=")[0].lower() in
                ("job", "input", "cpus", "mp_mode", "gpus", "memory", "scratch", "interactive",
                                               "user", "double", "oldjob", "globalmodel", "datacheck", "syntaxcheck")]
    if reserved:
        drona_add_warning("These options are already set from the form and should not be repeated: " +
                          " ".join(reserved))
    return options + " "


# ---------------------------------------------------------------- analysis options

# Compiler for user subroutines: module_versions/<cluster>.compilers.json maps an Abaqus module to the
# compiler module its abaqus_v6.env (compile_fortran / compile_cpp / link_sl) expects; "default" is the
# fallback. Abaqus 2024 and 2025 on Linux use Intel Fortran (ifort/ifx); intel-compilers/2023.2.1 has both.
DEFAULT_COMPILER = "intel-compilers/2023.2.1"
SUBROUTINE_TYPES = (".f", ".for", ".f90", ".c", ".cc", ".cpp", ".cxx", ".o")
RESULT_TYPES = (".odb", ".res", ".mdl", ".stt", ".prt", ".sim", ".fil", ".inp", ".sta", ".msg", ".dat")
JOB_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def _compiler(version):
    try:
        with open(MODULE_VERSIONS_DIR / (_cluster() + ".compilers.json")) as f:
            table = json.load(f)
        module = table.get(version) or table.get("default") or DEFAULT_COMPILER
    except (OSError, ValueError, AttributeError):
        module = DEFAULT_COMPILER
    return module if MODULE_LIST.match(module) else DEFAULT_COMPILER


def _link_job_files(path, what):
    """(name, shell lines) to link <name>.* from the directory of `path` into the workflow directory, where
    Abaqus looks for oldjob= and globalmodel= files. Cluster files only (a restart needs several files)."""
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    if ext.lower() not in RESULT_TYPES:
        name = base
    if not os.path.isabs(path):
        drona_add_error(f"{what}: select the file on the cluster (its other files must be next to it).")
        return "", ""
    if not JOB_NAME.match(name):
        drona_add_error(f"{what}: Abaqus job names must start with a letter and hold only letters, digits, _ and - "
                        f"({name!r}).")
        return "", ""
    folder = os.path.dirname(path)
    if not any(n.startswith(name + ".") for n in (os.listdir(folder) if os.path.isdir(folder) else [])):
        drona_add_warning(f"{what}: no files {name}.* found in {folder}.")
    lines = (f"for f in {shlex.quote(folder)}/{name}.*; do\n"
             f"    case \"$f\" in *.lck) continue ;; esac\n"
             f"    [ -e \"$(basename \"$f\")\" ] || ln -s \"$f\" .\n"
             f"done\n")
    return name, lines


def retrieve_abaqus_extras(version="", inputfile="", usersub="", precision="", runtype="", oldjob="", globalmodel=""):
    """Analysis options for the abaqus line (ABQ_EXTRA, ending in a space) and the mappings ABQ_COMPILER
    (module load for a user subroutine) and ABQ_LINKS (links to the files of oldjob=/globalmodel=)."""
    inputfile, usersub, oldjob, globalmodel = map(_clean, (inputfile, usersub, oldjob, globalmodel))
    precision, runtype = _clean(precision), _clean(runtype)
    info = _analysis(inputfile)
    options, links, compiler = [], [], ""

    # user subroutine
    if usersub:
        ext = os.path.splitext(usersub)[1].lower()
        if ext not in SUBROUTINE_TYPES:
            drona_add_warning(f"{os.path.basename(usersub)}: Abaqus expects a user subroutine in Fortran (.f, .for), "
                              "C/C++ (.c, .cpp) or an object file (.o).")
        if os.path.isabs(usersub) and not os.path.isfile(usersub):
            drona_add_error(f"User subroutine not found: {usersub}")
        options.append("user=" + shlex.quote(usersub))
        compiler = _compiler(_clean(version))
        drona_add_note(f"User subroutine {os.path.basename(usersub)}: Abaqus compiles and links it at the start of "
                       f"the job with {compiler}. Compile errors are in <job>.log.")
    elif info.get("user"):
        drona_add_warning("The input file uses keywords that need a user subroutine (" + ", ".join(info["user"]) +
                          "), but no user subroutine is selected.")

    # precision (Abaqus/Explicit; Standard always runs in double precision)
    if precision and precision != "default":
        if info.get("kind") == "standard":
            drona_add_warning("Precision applies to Abaqus/Explicit only; this is an Abaqus/Standard analysis, "
                              "which always runs in double precision.")
        options.append("double=" + precision)

    # restart / import from an earlier job
    if oldjob:
        name, lines = _link_job_files(oldjob, "Previous job")
        if name:
            job = _job_name(inputfile) if inputfile else ""
            if name == job:
                drona_add_error(f"The previous job ({name}) has the same name as this job; rename this input file "
                                "(Abaqus would overwrite the files it restarts from).")
            if info and not (info.get("restart") or info.get("import")):
                drona_add_warning("A previous job is selected, but the input file has no *RESTART, READ or *IMPORT, "
                                  "so Abaqus won't use it.")
            options.append("oldjob=" + name)
            links.append(f"# Files of the previous job (oldjob={name})\n" + lines)
    elif info.get("restart") or info.get("import"):
        drona_add_warning("The input file has *RESTART, READ or *IMPORT: select the previous job "
                          "(any of its files, e.g. the .odb or .res) under Advanced > Restart / import from job.")

    # submodeling
    if globalmodel:
        name, lines = _link_job_files(globalmodel, "Global model")
        if name:
            if info and not info.get("submodel"):
                drona_add_warning("A global model is selected, but the input file has no *SUBMODEL.")
            options.append("globalmodel=" + name)
            links.append(f"# Results of the global model (globalmodel={name})\n" + lines)
    elif info.get("submodel"):
        drona_add_warning("The input file has *SUBMODEL: select the global model's results (.odb) under Advanced > Global model.")

    # run type
    if runtype in ("datacheck", "syntaxcheck"):
        options.append(runtype)
        drona_add_note("Data check: Abaqus checks the model and writes the .dat file (with the memory the model "
                       "needs) without running the analysis. A short time and a few cores are enough."
                       if runtype == "datacheck" else
                       "Syntax check: Abaqus only checks the input file syntax (no license tokens, no analysis).")

    drona_add_mapping("ABQ_COMPILER", f"module load {compiler}" if compiler else "")
    drona_add_mapping("ABQ_LINKS", "".join(links))
    return " ".join(options) + " " if options else ""
