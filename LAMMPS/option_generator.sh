# check if file is readable. If it isn't, stop here.
python3 read_file.py $FILE || exit 0
source /sw/hprc/sw/lammps-validator/env.sh
acceleratoropts=""
if [ "$FILTER_H100" = "Yes" ]; then
    acceleratoropts="$acceleratoropts,h100"
fi
if [ "$FILTER_A30" = "Yes" ]; then
    acceleratoropts="$acceleratoropts,a30"
fi
if [ "$FILTER_PVC" = "Yes" ]; then
    acceleratoropts="$acceleratoropts,pvc"
fi
if [ "$FILTER_CPU" = "Yes" ]; then
    acceleratoropts="$acceleratoropts,cpu"
fi
if [ "${acceleratoropts:0:1}" = "," ]; then
    acceleratoropts=${acceleratoropts#,}
fi
packageopts=""
if [ "$FILTER_KK" = "Yes" ]; then
    packageopts="$packageopts,kokkos"
fi
if [ "$FILTER_GPU" = "Yes" ]; then
    packageopts="$packageopts,gpu"
fi
if [ "$FILTER_OMP" = "Yes" ]; then
    packageopts="$packageopts,omp"
fi
if [ "$FILTER_INT" = "Yes" ]; then
    packageopts="$packageopts,intel"
fi
if [ "$FILTER_OPT" = "Yes" ]; then
    packageopts="$packageopts,opt"
fi
if [ "${packageopts:0:1}" = "," ]; then
    packageopts=${packageopts#,}
fi
/sw/hprc/sw/lammps-validator/main.py --file $FILE --accelerators "$acceleratoropts" --packages "$packageopts" 
