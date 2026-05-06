#!/usr/bin/env python3

# Import modular utils
from utils_scripts.model_download import setup_model_download
from utils_scripts.dataset_download import setup_dataset_download
from utils_scripts.inference_setup_final import setup_inference
from utils_scripts.inference_setup_ipu import setup_inference_ipu
from utils_scripts.finetuning_setup import setup_finetuning
from utils_scripts.finetuning_setup_ipu import setup_finetuning_ipu
from utils_scripts.constants import SLURM_SCRIPT, SLURM_MULTINODE_SCRIPT, VENV_BASE, VENV_INFERENCE, VENV_BASH, VENV_IPU, IPU_RUNCOMMAND, IPU_DRIVER
from utils_scripts.slurm import setup_gpu_resources, setup_account, setup_default, setup_partition, setup_execution_method, setup_slurm_tasks_line
from drona_utils import drona_add_mapping, drona_add_additional_file, drona_add_note, drona_add_warning, drona_add_error



def setup_hf_operation(operation, model_id="", dataset_id="", cache_option="", hf_token="", location="", input_text="", max_samples="",
                      gpu_type="auto", num_gpus="1", nodes="1", precision="auto", model_download_path="model_cache",
                      job_type="slurm", gpu="none", walltime="2:00", memory="16G", dataset_download_path="dataset_cache", dataset_source="hub", model_source="hub", accelerator_type="gpu",
                      ipu_type="mk2_pod4", ipu_use_pipeline=True,
                      input_column="", target_column="", learning_rate=2e-4, num_train_epochs=3, per_device_train_batch_size=4,
                      lora_r=8, lora_alpha=16, lora_dropout=0.05, custom_finetune_params=None, test_size=0.1,
                      distributed_type="DDP", ddp_bucket_cap_mb=25, use_lora=False,
                      enable_wandb=False, wandb_api_key="", wandb_project="hf-finetuning",
                      wandb_run_name="", wandb_tags="", wandb_notes="",
                      task_type="causal_lm", text_column="text", label_column="label", num_labels=2):
    # Set up environment variables mapping
    # Use separate variables for bash vs SLURM venv setup
    if job_type == "bash":
        drona_add_mapping("VENV_SETUP_BASH", VENV_BASH)
        drona_add_mapping("VENV_SETUP_SLURM", "")
    else:
        drona_add_mapping("VENV_SETUP_BASH", "")
        drona_add_mapping("VENV_SETUP_SLURM", VENV_BASE)
    drona_add_mapping("OUTPUT_DIR", location)

    if operation == "model_management":
        # Set up model management specific driver and resources
        setup_model_management_driver(job_type, gpu, walltime, memory, nodes)

        # Set up the actual model download logic
        setup_model_download(model_id, cache_option, hf_token, location, model_download_path)
        return ""

    elif operation == "dataset_management":
        # Set up dataset management specific driver and resources
        setup_dataset_driver(job_type, gpu, walltime, memory, nodes)

        # Set up the actual dataset download logic
        setup_dataset_download(dataset_id, cache_option, hf_token, location, dataset_download_path)
        return ""
        
    elif operation == "dataset":
        # Add dataset download script
        drona_add_additional_file("scripts/download_dataset.py", "download_dataset.py")
        
        # Set up dataset download command
        token_flag = f"--token {hf_token}" if hf_token else ""
        
        dataset_cmd = f"python download_dataset.py --dataset_id {dataset_id} {token_flag}"
        drona_add_mapping("OPERATION_CMD", dataset_cmd)
        drona_add_mapping("OPERATION_DESC", f"Downloading dataset: {dataset_id}")
        
    elif operation == "inference":
        # Handle accelerator_type that might be a dict {"value": "ipu"} or string "ipu"
        if isinstance(accelerator_type, dict):
            accel_value = accelerator_type.get("value", "gpu")
        else:
            accel_value = accelerator_type if accelerator_type else "gpu"


        if accel_value == "ipu":
            drona_add_note("=== IPU WORKFLOW DETECTED ===")
            # IPU uses SSH to poplar2, not SLURM
            # Normalize ipu parameters
            ipu_type_value = ipu_type.get("value", "mk2_pod4") if isinstance(ipu_type, dict) else ipu_type
            ipu_pipeline_value = ipu_use_pipeline if isinstance(ipu_use_pipeline, bool) else (ipu_use_pipeline == "Yes" if isinstance(ipu_use_pipeline, str) else True)

            drona_add_note(f"Calling setup_ipu_driver()...")
            setup_ipu_driver()
            drona_add_note(f"Calling setup_inference_ipu()...")
            setup_inference_ipu(
                operation=operation,
                model_id=model_id,
                dataset_id=dataset_id,
                dataset_source=dataset_source,
                input_text=input_text,
                max_samples=max_samples,
                hf_token=hf_token,
                location=location,
                model_source=model_source,
                ipu_type=ipu_type_value,
                ipu_use_pipeline=ipu_pipeline_value
            )
        else:
            # GPU/CPU use SLURM/bash
            # Normalize GPU parameters
            gpu_type_value = gpu_type.get("value", "auto") if isinstance(gpu_type, dict) else gpu_type
            precision_value = precision.get("value", "auto") if isinstance(precision, dict) else precision

            setup_inference_driver(job_type, gpu, walltime, memory, nodes)
            setup_inference(
                operation=operation,
                model_id=model_id,
                dataset_id=dataset_id,
                dataset_source=dataset_source,
                input_text=input_text,
                max_samples=max_samples,
                hf_token=hf_token,
                location=location,
                model_source=model_source,
                accelerator_type=accel_value,
                gpu_type=gpu_type_value,
                num_gpus=num_gpus,
                precision=precision_value
            )
        return ""

    elif operation == "finetune_lora":

        # Handle accelerator_type that might be a dict {"value": "ipu"} or string "ipu"
        if isinstance(accelerator_type, dict):
            accel_value = accelerator_type.get("value", "gpu")
        else:
            accel_value = accelerator_type if accelerator_type else "gpu"


        # Sanitize num_gpus - if placeholder or not set, default based on accelerator
        if str(num_gpus).startswith("$") or not num_gpus:
            if accel_value == "cpu":
                num_gpus = "0"
            else:
                num_gpus = "1"

        # Normalize distributed_type if it's a dict
        if isinstance(distributed_type, dict):
            dist_type_value = distributed_type.get("value", "DDP")
        elif str(distributed_type).startswith("$") or not distributed_type:
            # Placeholder or missing - set based on accelerator type
            if accel_value == "cpu":
                dist_type_value = "NONE"
            else:
                dist_type_value = "DDP"
        else:
            dist_type_value = distributed_type if distributed_type else "DDP"


        # Auto-disable DDP in bash mode (bash can't run distributed training)
        if job_type == "bash" and dist_type_value in ["DDP", "DEEPSPEED", "FSDP"]:
            dist_type_value = "NONE"
            drona_add_note(f"Bash mode: Distributed training disabled (was: {distributed_type})")

        # Auto-disable DDP for single GPU (wasteful overhead)
        if int(num_gpus) == 1 and dist_type_value == "DDP":
            dist_type_value = "NONE"
            drona_add_note("Single GPU detected: Using simple mode (DDP disabled)")

        if accel_value == "ipu":
            drona_add_note("=== IPU FINE-TUNING WORKFLOW DETECTED ===")
            # IPU uses SSH to poplar2, not SLURM
            # Normalize ipu parameters
            ipu_type_value = ipu_type.get("value", "mk2_pod4") if isinstance(ipu_type, dict) else ipu_type
            ipu_pipeline_value = ipu_use_pipeline if isinstance(ipu_use_pipeline, bool) else (ipu_use_pipeline == "Yes" if isinstance(ipu_use_pipeline, str) else False)

            drona_add_note(f"Calling setup_ipu_finetuning_driver()...")
            setup_ipu_finetuning_driver()
            drona_add_note(f"Calling setup_finetuning_ipu()...")
            setup_finetuning_ipu(
                operation=operation,
                model_id=model_id,
                dataset_id=dataset_id,
                dataset_source=dataset_source,
                input_column=input_column,
                target_column=target_column,
                max_samples=max_samples,
                hf_token=hf_token,
                location=location,
                model_source=model_source,
                ipu_type=ipu_type_value,
                ipu_use_pipeline=ipu_pipeline_value,
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                custom_finetune_params=custom_finetune_params,
                test_size=test_size
            )
        else:
            # GPU/CPU fine-tuning
            # Normalize GPU parameters
            if isinstance(gpu_type, dict):
                gpu_type_value = gpu_type.get("value", "auto")
            elif str(gpu_type).startswith("$") or not gpu_type:
                # Placeholder - set based on accelerator
                gpu_type_value = "none" if accel_value == "cpu" else "auto"
            else:
                gpu_type_value = gpu_type

            # For CPU mode, ensure gpu_type is "none"
            if accel_value == "cpu":
                gpu_type_value = "none"
                # Also set $gpu variable so map.json's setup_partition gets the right value
                drona_add_mapping("gpu", "none")
            else:
                # Set $gpu to match gpu_type so map.json partition logic works
                drona_add_mapping("gpu", gpu_type_value)

            setup_finetuning_driver(job_type, gpu, walltime, memory, nodes, gpu_type_value, num_gpus, dist_type_value)
            setup_finetuning(
                operation=operation,
                model_id=model_id,
                dataset_id=dataset_id,
                dataset_source=dataset_source,
                input_column=input_column,
                target_column=target_column,
                max_samples=max_samples,
                hf_token=hf_token,
                location=location,
                model_source=model_source,
                gpu_type=gpu_type_value,
                num_gpus=num_gpus,
                nodes=nodes,
                distributed_type=dist_type_value,
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                custom_finetune_params=custom_finetune_params,
                test_size=test_size,
                ddp_bucket_cap_mb=ddp_bucket_cap_mb,
                use_lora=use_lora,
                enable_wandb=enable_wandb,
                wandb_api_key=wandb_api_key,
                wandb_project=wandb_project,
                wandb_run_name=wandb_run_name,
                wandb_tags=wandb_tags,
                wandb_notes=wandb_notes,
                task_type=task_type,
                text_column=text_column,
                label_column=label_column,
                num_labels=num_labels
            )
        return ""

    else:
        # Unsupported operation
        drona_add_mapping("OPERATION_CMD", "echo 'Unsupported operation'")
        drona_add_mapping("OPERATION_DESC", f"Unsupported operation: {operation}")

    return ""


def setup_memory(memory="16G", gpu_type="auto", num_gpus="1", nodes="1"):
    """Set up memory allocation based on GPU configuration."""
    if gpu_type == "none":
        return memory if memory else "16G"
    elif gpu_type in ["h100", "a100_80"]:
        # High-memory GPUs need more system memory
        return memory if memory else "64G"
    elif gpu_type in ["a100", "rtx8000"]:
        # Medium-memory GPUs
        return memory if memory else "48G"
    else:
        # Standard GPUs or auto
        return memory if memory else "32G"

def setup_ntasks_per_node(num_gpus="1", nodes="1"):
    """Set up ntasks-per-node for multinode distributed training."""
    if int(nodes) > 1:
        return f"#SBATCH --ntasks-per-node={num_gpus}"
    else:
        return ""

def setup_modules(gpu_type="auto"):
    """Set up environment modules based on GPU configuration."""
    modules = ["GCCcore/12.2.0", "Python/3.10.8"]

    if gpu_type != "none":
        modules.extend(["CUDA/12.1.1", "cuDNN/8.9.2.26-CUDA-12.1.1"])

    return " ".join(modules)

def setup_model_management_driver(job_type="bash", gpu="none", walltime="2:00", memory="16G", nodes="1"):
    """Set up driver specifically for model management operations."""
    operation_cmd = "python3 download.py --model-id [MODEL_ID] --download-path [DOWNLOAD_PATH] [TOKEN_FLAG]"
    return setup_execution_method(job_type, operation_cmd, "Model download", walltime, memory)

def setup_dataset_driver(job_type="bash", gpu="none", walltime="2:00", memory="16G", nodes="1"):
    """Set up driver specifically for dataset management operations."""
    operation_cmd = "[OPERATION_CMD]"
    return setup_execution_method(job_type, operation_cmd, "Dataset download", walltime, memory)

def setup_inference_driver(job_type="slurm", gpu="none", walltime="2:00", memory="16G", nodes="1"):
    """Set up driver specifically for inference operations."""
    operation_cmd = "[OPERATION_CMD]"
    return setup_execution_method(job_type, operation_cmd, "Inference", walltime, memory)

def setup_finetuning_driver(job_type="slurm", gpu="none", walltime="8:00", memory="32G", nodes="1", gpu_type="auto", num_gpus="1", distributed_type="DDP"):
    """Set up driver specifically for fine-tuning operations."""

    # Sanitize nodes parameter - if it's a placeholder like "$nodes", use default
    if str(nodes).startswith("$"):
        nodes = "1"

    # Sanitize num_gpus parameter - if it's a placeholder, use default
    if str(num_gpus).startswith("$"):
        num_gpus = "1"

    operation_cmd = "[OPERATION_CMD]"

    if job_type != "bash":

        # Set PARTITION
        if gpu_type and gpu_type not in ["none", ""]:
            drona_add_mapping("PARTITION", "#SBATCH --partition=gpu")
        else:
            drona_add_mapping("PARTITION", "#SBATCH --partition=cpu")

        # Set GPU_RESOURCES (new placeholder to avoid conflict with map.json's GRES)
        if gpu_type and gpu_type not in ["none", ""]:
            if gpu_type == "auto":
                gpu_resources = f"#SBATCH --gres=gpu:{num_gpus}"
            else:
                gpu_resources = f"#SBATCH --gres=gpu:{gpu_type}:{num_gpus}"
            drona_add_mapping("GPU_RESOURCES", gpu_resources)
            drona_add_note(f"GPU: {num_gpus}x {gpu_type}")
        else:
            drona_add_mapping("GPU_RESOURCES", "")


        # Set NTASKS_PER_NODE for multi-GPU
        if distributed_type == "DDP":
            # For DDP, only set ntasks-per-node (SLURM will calculate total tasks = nodes × ntasks-per-node)
            # Setting all three (ntasks, ntasks-per-node, nodes) causes SLURM rejection
            drona_add_mapping("tasks", "SKIP")  # Signal to comment out --ntasks line
            drona_add_mapping("NTASKS_PER_NODE", f"#SBATCH --ntasks-per-node={num_gpus}")
            total_tasks = int(nodes) * int(num_gpus)
            drona_add_note(f"DDP: {total_tasks} total tasks ({nodes} nodes × {num_gpus} GPUs)")
        elif distributed_type == "MULTI_GPU" and int(num_gpus) > 1:
            drona_add_mapping("NTASKS_PER_NODE", "#SBATCH --ntasks-per-node=1")
            drona_add_note(f"Multi-GPU: {num_gpus} GPUs via DataParallel")
        else:
            drona_add_mapping("NTASKS_PER_NODE", "")
    else:
        # Bash mode
        drona_add_mapping("PARTITION", "")
        drona_add_mapping("GPU_RESOURCES", "")
        drona_add_mapping("NTASKS_PER_NODE", "")

    return setup_execution_method(job_type, operation_cmd, "Fine-tuning", walltime, memory)

def setup_ipu_driver():
    """Set up driver for IPU inference via SSH to poplar2."""
    import datetime
    job_name = f"job_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Disable local venv setup for IPU (everything runs on poplar2)
    drona_add_mapping("VENV_SETUP_BASH", "")
    drona_add_mapping("VENV_SETUP_SLURM", "")

    # Set up IPU driver commands (SSH to poplar2)
    ipu_commands = f"""
# Create job directory on poplar2
JOB_NAME="{job_name}"
ssh poplar2 "mkdir -p /localdata/${{USER}}/drona_ipu_jobs/${{JOB_NAME}}"

# Copy files to IPU machine
scp inference_ipu.py poplar2:/localdata/${{USER}}/drona_ipu_jobs/${{JOB_NAME}}/
scp runcommand.sh poplar2:/localdata/${{USER}}/drona_ipu_jobs/${{JOB_NAME}}/

# Execute on IPU
echo "Running inference on IPU (poplar2)..."
ssh poplar2 "bash /localdata/${{USER}}/drona_ipu_jobs/${{JOB_NAME}}/runcommand.sh"
"""
    drona_add_mapping("DRIVERCOMMAND", ipu_commands)

    # Create runcommand.sh for IPU (runs on poplar2)
    runcommand_content = f"""#!/bin/bash
source /etc/profile
source /usr/local/bin/source.poplar.sh

cd /localdata/${{USER}}/drona_ipu_jobs/{job_name}

{VENV_IPU}

python3 inference_ipu.py
"""

    # Write runcommand.sh as additional file
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(base_path, "additional_files")
    os.makedirs(temp_dir, exist_ok=True)

    runcommand_path = os.path.join(temp_dir, "runcommand.sh")
    with open(runcommand_path, 'w') as f:
        f.write(runcommand_content)

    drona_add_additional_file(runcommand_path)

    return ""

def setup_ipu_finetuning_driver():
    """Set up driver for IPU fine-tuning via SSH to poplar2."""
    import datetime
    job_name = f"finetune_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Disable local venv setup for IPU (everything runs on poplar2)
    drona_add_mapping("VENV_SETUP_BASH", "")
    drona_add_mapping("VENV_SETUP_SLURM", "")

    # Set up IPU driver commands (SSH to poplar2)
    ipu_commands = f"""
# Create job directory on poplar2
JOB_NAME="{job_name}"
ssh poplar2 "mkdir -p /localdata/${{USER}}/drona_ipu_jobs/${{JOB_NAME}}"

# Copy files to IPU machine
scp finetuning_ipu.py poplar2:/localdata/${{USER}}/drona_ipu_jobs/${{JOB_NAME}}/
scp runcommand.sh poplar2:/localdata/${{USER}}/drona_ipu_jobs/${{JOB_NAME}}/

# Execute on IPU
echo "Running fine-tuning on IPU (poplar2)..."
ssh poplar2 "bash /localdata/${{USER}}/drona_ipu_jobs/${{JOB_NAME}}/runcommand.sh"
"""
    drona_add_mapping("DRIVERCOMMAND", ipu_commands)

    # Create runcommand.sh for IPU (runs on poplar2)
    runcommand_content = f"""#!/bin/bash
source /etc/profile
source /usr/local/bin/source.poplar.sh

cd /localdata/${{USER}}/drona_ipu_jobs/{job_name}

{VENV_IPU}

python3 finetuning_ipu.py
"""

    # Write runcommand.sh as additional file
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(base_path, "additional_files")
    os.makedirs(temp_dir, exist_ok=True)

    runcommand_path = os.path.join(temp_dir, "runcommand.sh")
    with open(runcommand_path, 'w') as f:
        f.write(runcommand_content)

    drona_add_additional_file(runcommand_path)
    return ""

def setup_model_management_resources():
    """Set up resource allocation defaults for model management."""
    # Model downloads typically don't need GPU resources
    drona_add_mapping("PARTITION", "")
    drona_add_mapping("GRES", "")
    drona_add_mapping("CONSTRAINT", "")
    drona_add_mapping("NTASKS_PER_NODE", "")
    return ""


