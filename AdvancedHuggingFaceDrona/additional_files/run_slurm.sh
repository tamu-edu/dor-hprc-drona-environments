#!/bin/bash
#SBATCH --job-name=tinyllama_finetune
#SBATCH --output=logs/finetune_output_%j.log
#SBATCH --error=logs/finetune_output_%j.log
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1    # We launch ONE torchrun task per node
#SBATCH --cpus-per-task=16   # Give all CPUs to the torchrun manager
#SBATCH --gres=gpu:h100:2      # It will manage these 2 GPUs
#SBATCH --partition=gpu
#SBATCH --time=00:10:00

echo "============================================="
echo "        Setting up Job Environment           "
echo "============================================="
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Allocated Nodes: $SLURM_JOB_NODELIST"
echo "============================================="

mkdir -p logs

# Get the hostname of the first node in the SLURM allocation. This is the master.
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

export HF_HOME="$SCRATCH/.cache/huggingface" 
export HF_DATASETS_CACHE="$SCRATCH/.cache/huggingface/datasets"

echo "MASTER_ADDR set to: $MASTER_ADDR"

# srun will launch one instance of this command block on each of the 2 nodes.
srun bash -c '
  echo "--- Activating environment on $(hostname) ---"
  
  module purge
  module load WebProxy GCCcore/12.2.0 Python/3.10.8 CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1

  VENV_DIR="${SCRATCH}/virtual_envs/drona_huggingface"
  source ${VENV_DIR}/bin/activate

  # ====================================================================
  # LAUNCHING WITH TORCHRUN - THE NATIVE PYTORCH LAUNCHER
  # --nproc_per_node=2: We have 2 H100s per node
  # --nnodes=2: We have 2 nodes in total
  # --node_rank=$SLURM_NODEID: SLURM provides the rank (0 or 1)
  # --rdzv_id=$SLURM_JOB_ID: A unique ID for the job
  # --rdzv_backend=c10d: Standard backend
  # --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT: Where to find the master
  # ====================================================================
  echo "--- Launching torchrun on $(hostname) with rank $SLURM_NODEID ---"
  
  torchrun \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=$SLURM_NODEID \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    run_finetuning.py
'

echo "============================================="
echo "SLURM job script finished."
echo "============================================="
