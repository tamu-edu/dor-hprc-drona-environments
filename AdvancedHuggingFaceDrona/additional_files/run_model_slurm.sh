#!/bin/bash
#SBATCH --job-name=[JOBNAME]
#SBATCH --time=[MODEL_TIME]
#SBATCH --mem=[MODEL_MEMORY]  
#SBATCH --nodes=[MODEL_NODES]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=[MODEL_PARTITION]
#SBATCH --output=out.%j 
#SBATCH --error=error.%j
[ACCOUNT]

echo "============================================="
echo "    Model Download Job"
echo "============================================="
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Model: [MODEL_ID]"
echo "Download path: [MODEL_DOWNLOAD_PATH]" 
echo "============================================="

module purge
module load WebProxy [MODULES]

[VENV_SETUP]

echo "============================================="
echo "Starting Model Download"
echo "[OPERATION_DESC]"
echo "============================================="

[OPERATION_CMD]

echo "============================================="
echo "Model download completed at $(date)"
echo "Results available in: [OUTPUT_DIR]"
echo "============================================="