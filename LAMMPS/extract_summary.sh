[[ -v REPORT_JSON && -n "$REPORT_JSON" ]] ||  echo "No acceleration strategy selected." 
[[ -v REPORT_JSON && -n "$REPORT_JSON" ]] ||  exit 0 
python3 lammps_report.py --tool summary-html --input "$REPORT_JSON"