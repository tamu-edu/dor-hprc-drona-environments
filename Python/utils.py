from pathlib import Path

# Slurm checks (retrieve_slurm_params), job cancel, workflow actions and CPU/GPU monitoring come from
# runtime_support/drona_runtime_utils through the map.json "$include" entries.


def retrieve_driver_contents(mode, cancel_jobs=""):
    base = Path(__file__).resolve().parent

    # No mode selected arrives as "" (None / "$drona_create_manage_mode" on older Drona)
    if mode is None or mode == "$drona_create_manage_mode" or mode == "":
        return 'echo "No workflow mode selected: choose Create or Manage." >&2\nexit 1\n'
    elif mode == "create":
        file_path = base / "drivers/driver-create.sh"
        drona_add_additional_file("jobscripts/python-job.slurm", "Job Script")
    else:
        file_path = base / "drivers/driver-manage.sh"

    with open(file_path, 'r') as file:
        return file.read()


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
