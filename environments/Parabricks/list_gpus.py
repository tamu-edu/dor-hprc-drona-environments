#!/usr/bin/env python3
import subprocess
import json

def get_gpus():

    cluster = subprocess.check_output(["/sw/local/bin/clustername"], text=True).strip()

    gpu_set = {}

    if cluster == "aces":

        gpu_set = {"a30":"A30","h100":"H100"}
    
    elif cluster == "grace":

        gpu_set = {"t4":"T4", "a100":"A100"}

    elif cluster == "faster":

        gpu_set = {"t4":"T4", "a10":"A10", "a30":"A30", "a40":"A40"}

    options = [{"value": key, "label": label} for key, label in gpu_set.items()]

    return options

# Entry point of the script

if __name__ == "__main__":
    result = get_gpus()
    print(json.dumps(result))
