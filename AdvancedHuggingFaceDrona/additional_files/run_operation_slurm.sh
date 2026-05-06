#!/bin/bash
#SBATCH --job-name=[JOBNAME]
#SBATCH --output=out.%j
#SBATCH --error=error.%j
#SBATCH --nodes=[SLURM_NODES]
[SLURM_TASKS_LINE]
#SBATCH --cpus-per-task=[SLURM_CPUS]
[PARTITION]
#SBATCH --time=[SLURM_TIME]
#SBATCH --mem=[SLURM_MEMORY]

[NTASKS_PER_NODE]
[GPU_RESOURCES]
[GRES]
[SLURM_ACCOUNT]
[EXTRA_SLURM]

source /etc/profile

echo "============================================="
echo "    HuggingFace Operation: [MODEL_JOB_TYPE]"
echo "============================================="
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "[OPERATION_DESC]"
echo "============================================="

module purge
module load WebProxy [MODULES]

[VENV_SETUP_SLURM]

echo "============================================="
echo "Starting Operation"
echo "============================================="

[OPERATION_CMD]

echo "============================================="
echo "Operation completed at $(date)"
echo "Results available in: [OUTPUT_DIR]"
echo "============================================="
