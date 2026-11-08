#!/bin/bash
# Download card (HTML): link to the results zip made by make_af_bundle.py (*_results.zip in the
# workflow directory), with size and date, or a hint how to make one.
# Params: LOCATION (workflow directory). Stdlib Python only, Python 3.6 compatible.

python3 - <<'PYEOF'
import glob, html, os, time
from urllib.parse import quote

LOCATION = os.environ.get("LOCATION", "").strip()
FILES_URL = "/pun/sys/dashboard/files/fs"

zips = sorted(glob.glob(os.path.join(LOCATION, "*_results.zip")), key=os.path.getmtime, reverse=True)
if not zips:
    print("<div style='color:#64748b;font-size:0.9em'>No download bundle yet. It is made at the end of a successful "
          "prediction. For runs from before that, or after a failure, tick <b>Create bundle</b> and click Generate, "
          "then Run: it zips the structures, scores and plots (a few MB) without the large intermediate files.</div>")
    raise SystemExit(0)

rows = []
for z in zips:
    # new tab + ?download (OOD Files app sends it as an attachment); the HTML download attribute fails here
    url = FILES_URL + quote(z) + "?download=" + str(int(time.time()))
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(z)))
    rows.append("<div style='margin:2px 0'><a href='%s' target='_blank' rel='noopener noreferrer' style='font-weight:600'>&#11015; %s</a> "
                "<span style='color:#64748b;font-size:0.85em'>%.1f MB · %s</span></div>"
                % (url, html.escape(os.path.basename(z)), os.path.getsize(z) / 1e6, stamp))
print("".join(rows))
print("<div style='color:#64748b;font-size:0.8em;margin-top:4px'>Structures, confidence scores and plots, without "
      "the large intermediate files. A successful re-run remakes it; if the date above is older than the re-run, "
      "use Create bundle.</div>")
PYEOF
