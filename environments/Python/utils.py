import math
import importlib
import subprocess
import sys
from pathlib import Path

def  retrieve_cluster_info():
   cluster_module = None
   module_directory = Path(__file__).resolve().parent
   if module_directory not in sys.path:
      sys.path.insert(0,f"{module_directory}")

   
   cluster = subprocess.check_output(["/sw/local/bin/clustername"], text=True).strip()
   if importlib.util.find_spec(f"clusters.{cluster}") is not  None:
      cluster_module = importlib.import_module(f"clusters.{cluster}")
   else:
      cluster_module = importlib.import_module(f"clusters.defaultcluster")
   return cluster, cluster_module




def setup_python_env(penv, pythonVersionDropdown, createEnvName, currentEnvDropdown, sharedEnvDropdown):
    if penv == "module":
        # defined statically, not a good solution  TODO retrieve modules dynamically 
        return f"# Load latest Python module\nmodule load GCCcore/13.3.0 Python/3.12.3 "
    elif penv == "private":
        return f"#Setup private virtual environment\n"+currentEnvDropdown
    elif penv == "create":
        if createEnvName=="":
            return f""
        else:
            return f"#Create new virtual env\n"+pythonVersionDropdown+"\ncreate_venv " + createEnvName + f"\nsource activate_venv " + createEnvName
    elif penv == "shared":
        return f"#Setup shared virtual environment\n"+sharedEnvDropdown
    else:
        return f""


def retrieve_tasks_and_other_resources(nodes,tasks,cpus,mem,gpu,numgpu,walltime,account,extra,slurmBox):
   # if advancedbox was not set, there will be no value for nodes and cpus, so set to ""
   if slurmBox != "Yes":
      tasks="1"
      nodes="1"
      cpus="1"
      mem=""
      gpu=""
      walltime=""
      account=""
      extra=""

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


