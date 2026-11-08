#!/bin/bash
# AlphaPickle plots card (HTML) for a finished AlphaFold 2 run: thumbnails of the PNGs AlphaPickle
# wrote next to ranking_debug.json (ranked_*_pLDDT.png, PAE plots for monomer_ptm / multimer),
# each linking to the full-size image in the Open OnDemand Files app.
# Params: LOCATION (workflow directory). Stdlib Python only, Python 3.6 compatible.

PYTHONPATH="$(dirname "$(readlink -f "$0")")${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PYEOF'
import glob, html, os, re
from af_results import af2_result_dir
from urllib.parse import quote

LOCATION = os.environ.get("LOCATION", "").strip()
FILES_URL = "/pun/sys/dashboard/files/fs"


def note(msg):
    print("<div style='color:#64748b;font-size:0.9em'>" + html.escape(msg) + "</div>")
    raise SystemExit(0)


result_dir = af2_result_dir(LOCATION)
if not result_dir:
    note("No AlphaFold 2 results yet.")
pngs = glob.glob(os.path.join(result_dir, "*.png"))
if not pngs:
    note("No AlphaPickle plots in " + result_dir + ". Enable \"AlphaPickle plots\" in Create for new runs.")


def order(path):
    """ranked_0 first; pLDDT before PAE for the same model."""
    name = os.path.basename(path)
    m = re.match(r"ranked_(\d+)", name)
    return (int(m.group(1)) if m else 99, 0 if "plddt" in name.lower() else 1, name)


def caption(name):
    stem = os.path.splitext(name)[0]
    m = re.match(r"ranked_(\d+)_(.*)", stem)
    if m:
        kind = m.group(2).replace("_", " ")
        return "Rank %d · %s" % (int(m.group(1)), kind)
    return stem.replace("_", " ")


cells = []
for p in sorted(pngs, key=order):
    url = FILES_URL + quote(p)
    name = os.path.basename(p)
    cells.append(
        "<a href='%s' target='_blank' rel='noopener noreferrer' title='%s' "
        "style='display:block;text-decoration:none;color:#334155;text-align:center;font-size:0.8em'>"
        "<img src='%s' loading='lazy' alt='%s' "
        "style='width:100%%;border:1px solid #e2e8f0;border-radius:4px;background:#fff'>"
        "<div style='margin-top:2px'>%s</div></a>"
        % (url, html.escape(name), url, html.escape(name), html.escape(caption(name))))

print("<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px'>"
      + "".join(cells) + "</div>")
print("<div style='color:#64748b;font-size:0.8em;margin-top:6px'>Click a plot to open it full size. "
      "Files: " + html.escape(result_dir) + "</div>")
PYEOF
