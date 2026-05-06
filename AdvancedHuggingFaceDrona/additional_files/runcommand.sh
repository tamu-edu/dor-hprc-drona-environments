#!/bin/bash
source /etc/profile
source /usr/local/bin/source.poplar.sh

cd /localdata/${USER}/drona_ipu_jobs/finetune_20251028_111210


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


python3 finetuning_ipu.py
