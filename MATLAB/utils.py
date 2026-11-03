import os
import re
import subprocess
from pathlib import Path

from drona_utils import drona_add_additional_file, drona_add_mapping, drona_add_warning
from drona_runtime_utils import cluster_max_cores, retrieve_slurm_params

# "any" GPU: run the cluster checks with a GPU type that has the same partition and the
# strictest limits on that cluster, then drop the type from --gres (see _any_gpu_filter)
ANY_GPU_PROXY = {"aces": "a30", "grace": "a40", "faster": "a30", "launch": "a30"}


def retrieve_cluster():
    result = subprocess.run(['bash', '-c', 'clustername'], capture_output=True, text=True, check=False)
    return result.stdout.strip()


def retrieve_driver_contents(mode, cancel_jobs=""):
    """Create mode submits jobscripts/matlab-job.slurm with the shared sbatch driver, manage mode runs
    the cancel/delete actions. An unselected mode (None, "" or the unmapped "$name") gets the create driver."""
    base = Path(__file__).resolve().parent

    if mode == "manage":
        file_path = base / "drivers/driver-manage.sh"
    else:
        file_path = base / "drivers/driver-create.sh"
        drona_add_additional_file("jobscripts/matlab-job.slurm", "Job Script")
        drona_add_additional_file("jobscripts/drona_matlab_wrapper.m", "MATLAB Wrapper", preview_order=-1)

    with open(file_path, 'r') as file:
        return file.read()


def _any_gpu_filter(key, value):
    # --gres=gpu:<type>:<n> -> --gres=gpu:<n>
    if key == "PARTITION":
        return re.sub(r"--gres=gpu:[^:\s]+:", "--gres=gpu:", value)
    return value


def retrieve_matlab_slurm_params(workers, threads, memory, gpu, numgpu, walltime, account, extra):
    """Single-node MATLAB job: the client plus `workers` pool workers, each with `threads` threads.
    ntasks is workers+1, or workers when the pool alone fills the node (the client then shares a core).
    Adds MATLAB_WORKERS/MATLAB_THREADS for the job script; the shared cluster checks add the Slurm mappings."""
    if not allow_matlab():
        drona_add_warning("Matlab is restricted software on ACES. If you are associated with an educational organization, please Contact HPRC to be added to the access list.")

    w = int(workers) if str(workers).isdigit() else 0
    t = max(int(threads) if str(threads).isdigit() else 1, 1)

    # a local pool is single node, so cap before the cluster checks (they would add nodes instead)
    maxcores = cluster_max_cores()
    if maxcores:
        if t > maxcores:
            drona_add_warning(f"#comp threads reduced from {t} to {maxcores} (cores per node)")
            t = maxcores
        maxtasks = maxcores // t
        if w > maxtasks:
            drona_add_warning(f"#workers reduced from {w} to {maxtasks} to fit on a single node ({maxcores} cores, {t} threads each)")
            w = maxtasks
        tasks = min(w + 1, maxtasks)
    else:
        tasks = w + 1

    drona_add_mapping("MATLAB_WORKERS", str(w))
    drona_add_mapping("MATLAB_THREADS", str(t))

    mapping_filter = None
    if gpu == "any":
        gpu = ANY_GPU_PROXY.get(retrieve_cluster(), "any")
        mapping_filter = _any_gpu_filter

    return retrieve_slurm_params("1", str(tasks), str(t), memory, gpu, numgpu, walltime, account, extra,
                                 mapping_filter=mapping_filter)


def retrieve_main(mainscript=""):
    if mainscript:
        return os.path.basename(mainscript)
    return ""


def retrieve_matlabpath(mainscript, matlabroot):
    """MATLABPATH = directory of the main script plus the selected directories
    (the picker returns several as one colon-joined string)."""
    dirs = []
    if mainscript:
        dirs.append(os.path.dirname(mainscript))
    if matlabroot:
        dirs.extend(d for d in matlabroot.split(":") if d)
    dirs = [d for i, d in enumerate(dirs) if d and d not in dirs[:i]]
    if not dirs:
        return ""
    return "export MATLABPATH=" + ":".join(dirs) + ":$MATLABPATH"


def allow_matlab():
    # On ACES, matlab is group protected. We will check if this Matlab env is on ACES and if so
    # check if the user is in the matlab group

    #check if user is in the MATLAB group on ACES
    cluster = retrieve_cluster();
    if cluster != "aces":
        return True
    else:
        matlab_group = "matlab"
        user_name = os.getenv("USER")

        #fetch group information for the matlab group
        try:
            group_info = subprocess.check_output(f"grep {matlab_group} /etc/group", shell=True).decode("utf-8")
        except subprocess.CalledProcessError:
            raise ValueError(f"Group '{matlab_group}' not found in /etc/group")

        #output is usually in the format: group_name:x:group_id:user1,user2,...
        group_members = group_info.split(":")[-1].strip().split(",")

        #check if the current user is in the MATLAB group
        if user_name in group_members:
            return True
        else:
            return False
