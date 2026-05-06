SLURM_SCRIPT = """
JOB_ID=$(/sw/local/bin/sbatch runcommand.sh | grep -o '[0-9]*')
# Wait for output file to be created and follow it
OUTPUT_FILE="out.${JOB_ID}"
echo "Waiting for job to start..."
# Wait for file to exist
while [ ! -f "$OUTPUT_FILE" ]; do
    sleep 1
done
tail -f "$OUTPUT_FILE" &
TAIL_PID=$!
while squeue -j $JOB_ID &>/dev/null; do
    sleep 2
done
kill $TAIL_PID 2>/dev/null
"""

SLURM_MULTINODE_SCRIPT = """
# For multinode jobs, we need to use srun to launch the distributed process
# The SLURM job script will handle the distributed execution automatically
JOB_ID=$(/sw/local/bin/sbatch runcommand.sh | grep -o '[0-9]*')
# Wait for output file to be created and follow it
OUTPUT_FILE="out.${JOB_ID}"
echo "Waiting for multinode job to start..."
# Wait for file to exist
while [ ! -f "$OUTPUT_FILE" ]; do
    sleep 1
done
tail -f "$OUTPUT_FILE" &
TAIL_PID=$!
while squeue -j $JOB_ID &>/dev/null; do
    sleep 2
done
kill $TAIL_PID 2>/dev/null
"""



VENV_BASE  = """
# Set up virtual environment
if [[ ! -d "${SCRATCH}/virtual_envs/huggingface" ]]; then
   echo "Creating new virtual environment for Hugging Face CLI..."
   create_venv -d "Hugging Face CLI environment" huggingface
   source activate_venv huggingface
   pip install --upgrade huggingface_hub transformers datasets torch accelerate
else
   echo "Using existing Hugging Face CLI environment..."
   source activate_venv huggingface
fi
"""

VENV_INFERENCE = """
# Set up virtual environment
if [[ ! -d "${SCRATCH}/virtual_envs/huggingface" ]]; then
   echo "Creating new virtual environment for Hugging Face CLI..."
   create_venv -d "Hugging Face CLI environment" huggingface
   source activate_venv huggingface
   pip install --upgrade huggingface_hub transformers datasets torch accelerate
   pip install torch accelerate
else
   echo "Using existing Hugging Face CLI environment..."
   source activate_venv huggingface
fi
"""

VENV_BASH = """
# Re-execute as login shell if not already one
if ! shopt -q login_shell; then
    exec bash -l "$0" "$@"
fi

# Set up virtual environment
if [[ ! -d "${SCRATCH}/virtual_envs/huggingface" ]]; then
   echo "Creating new virtual environment for Hugging Face..."
   create_venv -d "Hugging Face environment" huggingface
   source activate_venv huggingface
   pip install --upgrade huggingface_hub transformers datasets torch accelerate
else
   echo "Using existing Hugging Face environment..."
   source activate_venv huggingface
fi
"""

# IPU Virtual Environment Setup
VENV_IPU = """
# Set up IPU virtual environment (similar to IPUTutorial pattern)
cd /localdata/${USER}
mkdir -p ipu_labs
mkdir -p huggingface_cache

# Set HuggingFace cache directories to avoid home directory space issues
export HF_HOME="/localdata/${USER}/huggingface_cache"
export TRANSFORMERS_CACHE="/localdata/${USER}/huggingface_cache/transformers"
export HF_DATASETS_CACHE="/localdata/${USER}/huggingface_cache/datasets"
export TORCH_HOME="/localdata/${USER}/huggingface_cache/torch"

if [[ ! -d "optimum_ipu_venv" ]]; then
   echo "Creating new virtual environment for Optimum Graphcore IPU..."
   virtualenv -p python3 optimum_ipu_venv
   source optimum_ipu_venv/bin/activate

   # Install poptorch from Poplar SDK (required for optimum-graphcore)
   echo "Installing poptorch from Poplar SDK..."
   python -m pip install /opt/gc/poplar/poplar_sdk-ubuntu_20_04-3.3.0+1403-208993bbb7/poptorch-3.3.0+113432_960e9c294b_ubuntu_20_04-cp38-cp38-linux_x86_64.whl

   # Install optimum-graphcore and dependencies from source
   echo "Installing optimum-graphcore from source..."
   pip install --upgrade pip
   pip install git+https://github.com/huggingface/optimum.git@v1.6.1-release
   pip install git+https://github.com/huggingface/optimum-graphcore.git
   pip install transformers datasets huggingface_hub
else
   echo "Using existing Optimum Graphcore IPU environment..."
   source optimum_ipu_venv/bin/activate
fi
"""

# IPU runcommand script template
IPU_RUNCOMMAND = """#!/bin/bash
source /etc/profile
source /usr/local/bin/source.poplar.sh

cd /localdata/${USER}/drona_ipu_jobs/[JOB_NAME]

[VENV_IPU_SETUP]

python inference_ipu.py
"""

# IPU driver script template
IPU_DRIVER = """#!/bin/bash
source /etc/profile

# Create job directory on poplar2
JOB_NAME="job_$(date +%Y%m%d_%H%M%S)"
ssh poplar2 "mkdir -p /localdata/${USER}/drona_ipu_jobs/${JOB_NAME}"

# Copy files to IPU machine
cd [flocation]
scp inference_ipu.py poplar2:/localdata/${USER}/drona_ipu_jobs/${JOB_NAME}/
scp runcommand.sh poplar2:/localdata/${USER}/drona_ipu_jobs/${JOB_NAME}/

# Execute on IPU
echo "Running inference on IPU (poplar2)..."
ssh poplar2 "bash /localdata/${USER}/drona_ipu_jobs/${JOB_NAME}/runcommand.sh"
"""
