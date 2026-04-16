import math
import os
import importlib
import subprocess
import sys
from pathlib import Path

def  retrieve_cluster_info():
   cluster_module = None
   module_directory = Path(__file__).resolve().parent
   if module_directory not in sys.path:
      sys.path.insert(0,f"{module_directory}")
   
   #TODO 
   cluster = subprocess.check_output(["/sw/local/bin/clustername"], text=True).strip()
   if importlib.util.find_spec(f"clusters.{cluster}") is not  None:
      cluster_module = importlib.import_module(f"clusters.{cluster}")
   else:
      cluster_module = importlib.import_module(f"clusters.defaultcluster")
   return cluster, cluster_module


def retrieve_tasks_and_other_resources(nodes,tasks,cpus,mem,gpu,numgpu,walltime,account,extra,advancedbox):

   # if advancedbox was not set, there will be no value for nodes and cpus, so set to ""
   if advancedbox != "Yes":
       nodes=""
       cpus=""

   # if gpu not set, variable numgpu does not exit. 
   numgpunum=1
   if gpu != "" and gpu !=  "none":
      # if numgpu not set, default to 1
      numgpunum=1 if numgpu=="" else int(numgpu)

   tasknum = int(tasks)
   nodenum  = 0 if nodes == "" else int(nodes)
   cpunum = 1 if cpus == "" else int(cpus)

   totalmemnum = 0 if mem =="" else int(mem[:-1])
   timestring = "02:00" if walltime == "" else walltime

   # compute the number of hours requested
   times=timestring.split(':')
   total_hours = (int(times[0])+int(times[1])/60)


   # dynamic disoatch to correct module to check the provided info
   cluster, cluster_module  = retrieve_cluster_info()
   cluster_module.cluster_slurm_checks(nodenum,tasknum,cpunum,totalmemnum,gpu,numgpunum,timestring,account,extra,drona_add_mapping,drona_add_message)

   return f""



def retrieve_loaded_modules(modules=""):
    if modules == "":
        return f""
    else:

        return f"" +modules  



def retrieve_driver_contents(mode):
    base = Path(__file__).resolve().parent
    
    if mode == "$mode":
        file_path = base / "configuration/configure.sh"
    elif mode == "create" or mode == "":
        file_path = base / "drivers/driver-create.sh"
        drona_add_additional_file("jobscripts/generic-job.slurm", "Job Script")
    else:
        file_path = base / "drivers/driver-manage.sh"
    
    with open(file_path, 'r') as file:
        return file.read()
        



def retrieve_manage_action(mode, action, allworkflows, jobs, status, delete_workdir, job_dir):

    import json, re
    from views.utils import get_runtime_dir
    from drona_utils.core import drona_add_warning, drona_add_error

    if not mode or mode != "manage":
        return "echo 'No action taken.'"

    # Normalize action
    try:
        action = json.loads(action).get("value", action)
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # Normalize delete_workdir
    try:
        parsed = json.loads(delete_workdir)
        delete_workdir = parsed.get("value") if isinstance(parsed, dict) else parsed
    except Exception:
        pass
    delete_workdir = str(delete_workdir).lower() if delete_workdir else ""

    runtime_dir = get_runtime_dir()
    db_retriever = os.path.join(runtime_dir, "db_access", "drona_db_retriever.py")
    cleanup = f"python3 {db_retriever} --delete -i $DRONA_WF_ID && echo 'Manage workflow record cleaned up.' || echo 'Cleanup failed.'"

    if not action or action == "none" or action.startswith("$"):
        return f"echo 'No action taken.'\n{cleanup}"

    if not jobs:
        return f"echo 'No jobs found.'\n{cleanup}"

    # Normalize job_dir
    match = re.search(r'>(/[^<]+)<', job_dir or "")
    job_dir = match.group(1) if match else (job_dir or "").strip()

    if action == "cancel":
        if status == "DONE":
            return f"echo 'Job is already completed, nothing to cancel.'\n{cleanup}"
        elif not status or status.startswith("$"):
            return f"echo 'Could not determine job status, cancellation aborted.'\n{cleanup}"
        else:
            return (
                f"scancel {jobs} && echo 'Cancelled job(s): {jobs}.' || echo 'scancel returned an error.'\n"
                f"{cleanup}"
            )

    if action == "delete":
        if status != "DONE":
            drona_add_warning(
                f"Job(s) {jobs} are currently {status}! Workflow {allworkflows} is going to be deleted."
            )

        delete_cmd = (
            f"python3 {db_retriever} --delete -i {allworkflows} && echo 'Workflow {allworkflows} deleted.' || echo 'Delete failed.'\n"
        )

        if delete_workdir == "yes" and job_dir:
            drona_add_warning(
                f"The working directory {job_dir} and ALL of its contents will be permanently deleted. "
                "This cannot be undone."
            )
            delete_cmd += f"\nrm -rf {job_dir} && echo 'Working directory {job_dir} deleted.' || echo 'Directory deletion failed.'\n"
        elif delete_workdir == "yes" and not job_dir:
            drona_add_error("Working directory deletion was requested but no directory path was found. Directory will not be deleted.")

        delete_cmd += cleanup
        return delete_cmd

    return f"echo 'No action taken.'\n{cleanup}"


def import_slurm_script(importscript):
    if importscript == "" or importscript == "$importscript":
        return ""
    else:
        content = ""
        with open(importscript, 'r', encoding='utf-8') as f:
            content = f.read()

        prefix = "# START OF IMPORTED FILE ---\n"
        suffix = "\n# END OF IMPORTED FILE ---\n\n"

        return  f"{prefix}{content}{suffix}"


