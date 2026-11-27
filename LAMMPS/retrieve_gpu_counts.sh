# Options of the GPU count dropdown: the "counts" of the strategy's GPU in accelerators/<cluster>.json.
[[ -v REPORT_JSON && -n "$REPORT_JSON" ]] || { echo '[]'; exit 0; }
python3 lammps_report.py --tool gpu-counts --input "$REPORT_JSON"
