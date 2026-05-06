#!/usr/bin/env python3
"""
SLURM-related utility functions for job configuration and resource management.
"""

from drona_utils import drona_add_mapping, drona_add_additional_file, drona_add_note


def setup_execution_method(job_type="bash", operation_cmd="", operation="unknown", walltime="2:00", memory="16G"):
    """
    Set up execution method for any operation with 3 modes: bash, slurm_interactive, slurm.

    Args:
        job_type: Execution mode (bash, slurm_interactive, slurm)
        operation_cmd: The command to execute (e.g., "python3 download.py --model-id [MODEL_ID]...")
        operation: Operation name for logging
        walltime: Job walltime
        memory: Job memory
    """

    if job_type == "bash":
        # Branch 1: Direct bash execution
        # VENV_SETUP is now included in driver.sh template
        drona_add_mapping("DRIVERCOMMAND", operation_cmd)

    elif job_type == "slurm_interactive":
        # Branch 2: SLURM interactive with log streaming
        drona_add_additional_file("additional_files/main.py", "main.py")
        drona_add_additional_file("additional_files/run_operation_slurm.sh", "run_operation_slurm.sh")
        drona_add_mapping("DRIVER", "#!/bin/bash")
        drona_add_mapping("DRIVERCOMMAND", "python3 main.py")

        # Set SLURM job script name for main.py
        drona_add_mapping("SLURM_SCRIPT", "run_operation_slurm.sh")

    elif job_type == "slurm":
        # Branch 3: Direct SLURM batch job
        drona_add_additional_file("additional_files/run_operation_slurm.sh", "run_operation_slurm.sh")
        drona_add_mapping("DRIVER", "#!/bin/bash")
        drona_add_mapping("DRIVERCOMMAND", "sbatch run_operation_slurm.sh")

    else:
        drona_add_note(f"DEBUG: UNKNOWN job_type={job_type} - this should not happen!")

    return ""


def setup_gpu_resources(gpu="none", numgpu="1"):
    """Set up GPU resource allocation for SLURM based on accelerator selection."""
    # Check for undefined variables (happens when form fields don't exist)
    # Return empty string - operation-specific code handles GPU allocation instead
    if str(gpu).startswith("$"):
        return ""

    if not gpu or gpu == "none" or gpu.strip() == "":
        return ""

    if not numgpu or str(numgpu).startswith("$"):
        numgpu = "1"

    # GPU value comes from retrieve_gpus retriever
    # Format: #SBATCH --gres=gpu:type:count
    try:
        num = int(numgpu) if numgpu else 1
        return f"#SBATCH --gres=gpu:{gpu}:{num}"
    except:
        return f"#SBATCH --gres=gpu:{gpu}:1"


def setup_account(account=""):
    """Set up SLURM account if provided."""
    if not account or account.strip() == "":
        return ""
    return f"#SBATCH --account={account}"


def setup_slurm_tasks_line(tasks):
    """Generate the --ntasks line, or comment it out if not needed.

    For DDP mode, we don't want --ntasks because SLURM calculates it from
    nodes × ntasks-per-node. Operation code will set tasks to "SKIP" to skip this line.
    """
    if tasks == "SKIP":
        return "# --ntasks auto-calculated by SLURM from nodes × ntasks-per-node"
    elif tasks is None or str(tasks).startswith("$"):
        return "#SBATCH --ntasks=1"
    else:
        return f"#SBATCH --ntasks={tasks}"


def setup_default(value, default):
    """Return value if it exists and is not empty, otherwise return default.

    If value was explicitly set to empty string by operation code, respect that (return empty).
    Only use default if value is None or a placeholder like "$tasks".
    """
    # Check if value is a literal placeholder like "$tasks" - use default
    if value is not None and str(value).startswith("$"):
        return default

    # Check if value is None - use default
    if value is None:
        return default

    # If value is empty string, return it (operation code explicitly set it to empty)
    # Otherwise return the value
    return str(value)


def setup_partition(gpu="none"):
    """Set up SLURM partition based on GPU/accelerator selection.

    Note: For fine-tuning operations, PARTITION is set by operation-specific code.
    If we receive a placeholder like "$gpu", return empty string so operation code's
    setting is preserved.
    """
    # If placeholder or not set, return empty (operation code will have set PARTITION already)
    if not gpu or str(gpu).startswith("$") or gpu.strip() == "":
        return ""

    # Otherwise set partition based on gpu value
    if gpu == "none":
        return "#SBATCH --partition=cpu"
    else:
        return "#SBATCH --partition=gpu"
