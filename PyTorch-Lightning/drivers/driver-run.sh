#!/bin/bash
source /etc/profile

if [ -f /sw/lmod/lmod/init/bash ]; then
    source /sw/lmod/lmod/init/bash
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
fi

cd [flocation]

module purge
module load WebProxy

[setupEnv]

if [ -f prefetch_data.py ]; then
    python3 prefetch_data.py || exit 1
fi

/sw/local/bin/sbatch [job-file-name]
