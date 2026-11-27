import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

from drona_utils import drona_add_additional_file, drona_add_message
from drona_runtime_utils import (drona_cluster_name, retrieve_gpu_monitor_start_nvml,
                                 retrieve_gpu_monitor_stop, retrieve_slurm_params)

from lammps_report import main

# Defaults used when the Resources fields are left empty (same as the 1.x environment)
DEFAULT_MEMORY = "40G"
DEFAULT_WALLTIME = "02:00"
# Accelerators per cluster: accelerators/<cluster>.json, built by an admin with
# configuration/generate_accelerators.py (also the hardware filter options, retrieve_accelerators).
# "vendor": the shared GPU monitor uses NVML, so it only runs for nvidia (not PVC).
# "constraints": GPU count -> Slurm --constraint (e.g. PVC Xe Link: 2 -> xelink2).
ACCELERATORS_DIR = Path(__file__).resolve().parent / "accelerators"


@lru_cache(maxsize=None)
def _accelerators():
    """{value: entry} from accelerators/<cluster>.json; {} without a file (CPU only, see retrieve_accelerators)."""
    try:
        with open(ACCELERATORS_DIR / (drona_cluster_name() + ".json")) as f:
            return {o["value"]: o for o in json.load(f) if isinstance(o, dict) and o.get("value")}
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def _is_nvidia(gpu_type):
    return _accelerators().get((gpu_type or "").lower(), {}).get("vendor") == "nvidia"


def retrieve_driver_contents(mode, cancel_jobs=""):
    """Create mode submits jobscripts/lammps-job.slurm with the shared sbatch driver, manage mode runs
    the cancel/delete actions. An unselected mode (None, "" or the unmapped "$name") gets the create driver."""
    base = Path(__file__).resolve().parent

    if mode == "manage":
        file_path = base / "drivers/driver-manage.sh"
    else:
        file_path = base / "drivers/driver-create.sh"
        drona_add_additional_file("jobscripts/lammps-job.slurm", "Job Script")
        drona_add_message("Add your required package-specific and accelerator-specific optimizations to the lammps command", "note")

    with open(file_path, 'r') as file:
        return file.read()


def _count(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _accelerator(json_input):
    if not json_input:
        return ""
    return (main("data", json_input).get("accelerator") or "").lower()


def retrieve_lammps_slurm_params(mode, json_input, gpu_count, cpu_count, memory, walltime, account, extra):
    """Turn the selected acceleration strategy into Slurm resources and let the shared cluster checks
    add the mappings (TASKS, NODES, CPUS, MEM, TIME, PARTITION, EXTRA).
    GPU strategies: one MPI task per CPU count (default: one per GPU). CPU strategies: one task with
    cpu_count threads. A GPU count listed under "constraints" for the accelerator (e.g. PVC 2/4:
    xelink2/xelink4) adds that --constraint.
    Manage mode (cancel/remove) submits no job script, so nothing is checked there."""
    if mode == "manage":
        return ""
    if not json_input:
        drona_add_message("No acceleration strategy selected.", "error")
        return ""

    x = _count(gpu_count, 1)
    y = _count(cpu_count, x)
    accelerator = _accelerator(json_input)

    gpu, numgpu = "", ""
    tasks, cpus = y, 1
    if accelerator == "cpu" or not accelerator or accelerator == "n/a":
        tasks, cpus = 1, y
    else:
        gpu, numgpu = accelerator, str(x)
        constraint = (_accelerators().get(accelerator, {}).get("constraints") or {}).get(str(x))
        if constraint:
            extra = f"{extra} --constraint={constraint}".strip()

    return retrieve_slurm_params("1", str(tasks), str(cpus), memory or DEFAULT_MEMORY, gpu, numgpu,
                                 walltime or DEFAULT_WALLTIME, account, extra)


def retrieve_lammps_gpu_monitor_start(gpu_type, monitor_enabled):
    if not _is_nvidia(gpu_type):
        return ""
    return retrieve_gpu_monitor_start_nvml(gpu_type, monitor_enabled)


def retrieve_lammps_gpu_monitor_stop(gpu_type, monitor_enabled):
    if not _is_nvidia(gpu_type):
        return ""
    return retrieve_gpu_monitor_stop(gpu_type, monitor_enabled)


# LAMMPS writes log.lammps in the directory it runs in (the input file's directory, see
# compute_workdir), which may be read-only: always write it to the job directory instead.
LOG_OPT = '-log "$DRONA_SUBMIT_DIR/log.lammps"'


def get_selected_lmp_cmd(json_input, gpu_count, cpu_count):
    if not json_input:
        return "lmp command: Error (No acceleration strategy detected)"
    data = main("data", json_input)
    x = _count(gpu_count, 1)
    y = _count(cpu_count, x)
    mpi_exe = data.get("mpi_exe") or "mpirun"
    lmp_exe = data.get("lmp_exe") or "lmp"
    prefix = data["prefix"] + " " if data.get("prefix") else ""
    framework = data.get("framework")
    if framework == "kokkos":
        mpi_cmd = f"{mpi_exe} -np {y} " if y > 1 else ""
        return f"{prefix}{mpi_cmd}{lmp_exe} -k on g {x} -sf kk -pk kokkos gpu/aware on {LOG_OPT} -in $INFILE"
    if framework == "gpu":
        mpi_cmd = f"{mpi_exe} -np {x} " if x > 1 else ""
        return f"{prefix}{mpi_cmd}{lmp_exe} -sf gpu -pk gpu {x} {LOG_OPT} -in $INFILE"
    if framework == "openmp":
        return f"{prefix}{lmp_exe} -sf omp -pk omp {y} {LOG_OPT} -in $INFILE"
    return f"{prefix}{lmp_exe} {LOG_OPT} -in $INFILE # (no acceleration framework detected)"


def get_selected_setup(json_input):
    if not json_input:
        return ": Error (No acceleration strategy detected)"
    data = main("data", json_input)
    if data.get("setup"):
        return data["setup"]
    if data.get("module"):
        return "module load LAMMPS/" + data["module"]
    return "# (no software module detected)"


def get_selected_summary(json_input):
    if not json_input:
        return "module: Error (No acceleration strategy detected)"
    return main("summary-txt", json_input)


def compute_workdir(path_to_file):
    """LAMMPS runs in the input file's directory, so relative read_data/include paths work."""
    workdir = os.path.dirname(path_to_file or "")
    if workdir and os.path.isdir(workdir) and not os.access(workdir, os.W_OK | os.X_OK):
        drona_add_message(
            "LAMMPS runs in the input file's directory, %s, and you cannot write there: dump, restart "
            "and write_data output of your input will fail. Copy the input (and the files it reads) "
            "to a directory you own and select that copy. log.lammps goes to the job directory." % workdir,
            "warning")
    return workdir


def generate_full_report(mode, filename):
    """Acceleration report for the Create preview; skipped in manage mode (no input file there)."""
    if mode == "manage":
        return ""
    drona_add_additional_file("report.txt", preview_name="Acceleration report")
    try:
        data = subprocess.run(
            ["/usr/bin/bash", "-c",
             "source /sw/hprc/sw/lammps-validator/env.sh && "
             '/sw/hprc/sw/lammps-validator/main.py --format None --verbose Info --file "$1"',
             "bash", filename],
            capture_output=True, text=True)
    except Exception as e:
        return f"Error, unable to generate report because:\n{e}"
    if data.stderr:
        return "stderr:\n" + data.stderr + "\nstdout:\n" + data.stdout
    return data.stdout
