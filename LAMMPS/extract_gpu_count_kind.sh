# "list" when accelerators/<cluster>.json gives "counts" for the strategy's GPU (dropdown), "number" for
# any other GPU, empty for CPU. Picks which GPU count field the create form shows.
[[ -v REPORT_JSON && -n "$REPORT_JSON" ]] || exit 0
python3 lammps_report.py --tool gpu-count-kind --input "$REPORT_JSON"
