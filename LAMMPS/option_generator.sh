# check if file is readable. If it isn't, stop here.
python3 read_file.py $FILE || exit 0
source /sw/hprc/sw/lammps-validator/env.sh
/sw/hprc/sw/lammps-validator/main.py --file $FILE
