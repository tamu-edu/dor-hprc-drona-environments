import shlex
from pathlib import Path

from drona_utils import drona_add_additional_file, drona_add_mapping, drona_add_message
from drona_runtime_utils import cluster_max_cores, retrieve_slurm_params


def _value(v):
    """Field value, or "" when the field is unset (None, "" or the unmapped "$name")."""
    if v is None:
        return ""
    v = str(v).strip()
    return "" if v.startswith("$") else v


def _int(v, default):
    v = _value(v)
    return int(v) if v.isdigit() else default


def retrieve_driver_contents(mode, cancel_jobs=""):
    """Create mode submits jobscripts/tamulauncher-job.slurm with the shared sbatch driver,
    manage mode runs the cancel/delete actions."""
    base = Path(__file__).resolve().parent

    if _value(mode) == "":
        return 'echo "No workflow mode selected: choose Create or Manage." >&2\nexit 1\n'
    if mode == "create":
        file_path = base / "drivers/driver-create.sh"
        drona_add_additional_file("jobscripts/tamulauncher-job.slurm", "Job Script")
    else:
        file_path = base / "drivers/driver-manage.sh"

    with open(file_path, 'r') as file:
        return file.read()


def retrieve_tamulauncher_slurm_params(nodes, commandspernode, threads, memory, walltime, account, extra):
    """One launcher task per node with commands-per-node x threads-per-command cores.

    Commands per node defaults to as many as fit on a node. Threads are capped at the cores per node,
    and reduced when commands x threads does not fit. Adds CPN/CPUPT for the job script; the shared
    cluster checks add TASKS/NODES/CPUS/MEM/TIME/PARTITION."""
    n = max(_int(nodes, 1), 1)
    t = max(_int(threads, 1), 1)
    c = _int(commandspernode, 0)

    maxcores = cluster_max_cores()
    if maxcores:
        if t > maxcores:
            drona_add_message(f"#threads per command reduced from {t} to {maxcores} (cores per node).", "warning")
            t = maxcores
        if c > maxcores:
            drona_add_message(f"#commands per node reduced from {c} to {maxcores} (cores per node).", "warning")
            c = maxcores
        if c == 0:
            c = maxcores // t
        elif c * t > maxcores:
            newt = max(maxcores // c, 1)
            drona_add_message(f"commands per node x threads per command cannot exceed {maxcores} cores per node. "
                              f"Reducing #threads per command from {t} to {newt}.", "warning")
            t = newt
    elif c == 0:
        drona_add_message("Cores per node unknown on this cluster: running 1 command per node. "
                          "Set #commands per node to run more.", "warning")
        c = 1

    drona_add_mapping("CPN", str(c))
    drona_add_mapping("CPUPT", str(t))

    return retrieve_slurm_params(str(n), str(n), str(c * t), _value(memory), "", "", _value(walltime),
                                 _value(account), _value(extra))


def retrieve_commandsfile(cfile=""):
    """Commands file path for tamulauncher (quoted), with a check that it exists and is not empty."""
    cfile = _value(cfile)
    if cfile == "":
        drona_add_message("No commands file selected. The job will fail; select one in the form or edit the job script.", "error")
        return ""
    try:
        with open(cfile, errors="replace") as f:
            ncommands = sum(1 for line in f if line.strip())
        if ncommands == 0:
            drona_add_message(f"Commands file {cfile} is empty.", "error")
    except OSError:
        drona_add_message(f"Cannot read commands file {cfile}. Check that it exists on the cluster.", "warning")
    return shlex.quote(cfile)


def retrieve_norestart(norestart=""):
    return "--norestart" if _value(norestart) else ""


def retrieve_module_lines(modules=""):
    """The job script's module load line: WebProxy plus the modules picked in the form.

    The picker's list starts with the chosen toolchain, if any (e.g.
    "foss/2024a OpenMPI/4.1.6"), so one `module load` loads them as one stack."""
    return " ".join(["module load WebProxy"] + _value(modules).split())
