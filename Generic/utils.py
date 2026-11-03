import math
import os
import importlib
import subprocess
import sys
from pathlib import Path

def retrieve_loaded_modules(modules=""):
    if modules == "":
        return f""
    else:

        return f"" +modules  


def retrieve_driver_contents(mode, cancel_jobs=""):
    base = Path(__file__).resolve().parent
    
    # No mode selected arrives as "" (None / "$drona_create_manage_mode" on older Drona)
    if mode is None or mode == "$drona_create_manage_mode" or mode == "":
        return 'echo "No workflow mode selected: choose Create or Manage." >&2\nexit 1\n'
    elif mode == "create":
        file_path = base / "drivers/driver-create.sh"
        drona_add_additional_file("jobscripts/generic-job.slurm", "Job Script")
    else:
        file_path = base / "drivers/driver-manage.sh"
    
    with open(file_path, 'r') as file:
        return file.read()


def import_slurm_script(importscript):
    if importscript == "" or importscript == "$drona_importscript":
        return ""
    else:
        content = ""
        with open(importscript, 'r', encoding='utf-8') as f:
            content = f.read()

        prefix = "# START OF IMPORTED FILE ---\n"
        suffix = "\n# END OF IMPORTED FILE ---\n\n"

        return  f"{prefix}{content}{suffix}"


