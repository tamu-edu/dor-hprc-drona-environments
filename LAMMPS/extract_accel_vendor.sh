[[ -v REPORT_JSON && -n "$REPORT_JSON" ]] || exit 0
python3 lammps_report.py --tool accel-vendor --input "$REPORT_JSON"
