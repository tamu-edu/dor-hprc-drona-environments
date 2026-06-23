#!/bin/bash
source /etc/profile
cd [flocation]

module purge
module load WebProxy

if [ -f prefetch_data.py ]; then
    python3 prefetch_data.py || exit 1
fi

/sw/local/bin/sbatch [job-file-name]
