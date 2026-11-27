# check if file is readable. If it isn't, stop here.
python3 read_file.py $FILE || exit 0
# FILTER_ACCEL (filter_accel: h100, pvc, cpu, ...) and FILTER_PKG (filter_pkg: kokkos, gpu, openmp,
# intel) are the checked filters. The app passes a list as Python text, e.g. "['h100', 'pvc']"
# (JSON and a comma-separated string work too) -> h100,pvc. Parsed before env.sh changes PATH.
# The validator matches packages exactly against a build's framework name.
to_csv() { python3 -c 'import sys; from accel_info import parse_list; print(",".join(parse_list(sys.argv[1])))' "$1" 2>/dev/null; }
acceleratoropts=$(to_csv "$FILTER_ACCEL")
packageopts=$(to_csv "$FILTER_PKG")
source /sw/hprc/sw/lammps-validator/env.sh
/sw/hprc/sw/lammps-validator/main.py --file $FILE --accelerators "$acceleratoropts" --packages "$packageopts"
