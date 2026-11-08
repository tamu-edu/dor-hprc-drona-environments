#!/bin/bash
# FREE UP SPACE card (HTML): size of the intermediate files clean_af_intermediates.py would remove, or when
# they were removed. Params: LOCATION (workflow directory), AF_VERSION. Stdlib Python only, Python 3.6 compatible.
HERE="$(dirname "$(readlink -f "$0")")"
LOCATION="$LOCATION" AF_VERSION="$AF_VERSION" HELPER="$HERE/../clean_af_intermediates.py" python3 - <<'PYEOF'
import json, os, subprocess
loc, version = os.environ.get("LOCATION", "").strip(), os.environ.get("AF_VERSION", "").strip()
grey = "<div style='color:#64748b;font-size:0.9em'>%s</div>"
marker = os.path.join(loc, ".drona", "af_cleaned")
done = None
try:
    with open(marker) as f:
        done = json.load(f)
except (OSError, ValueError):
    pass
try:
    listing = json.loads(subprocess.check_output(["python3", os.environ["HELPER"], loc, version, "--list"],
                                                 universal_newlines=True, stderr=subprocess.DEVNULL))
except Exception:
    listing = {"files": [], "bytes": 0}
mb = listing.get("bytes", 0) / 1e6
if done and not listing["files"]:
    print(grey % ("Intermediate files removed on %s (%.1f MB freed). Structures, scores, plots and the download "
                  "bundle are kept; the GPU stage can no longer be re-run." % (done.get("date", "?"), done.get("bytes", 0) / 1e6)))
elif not listing["files"]:
    print(grey % "No intermediate files to remove.")
else:
    what = ("model pickles (result_model_*.pkl), features.pkl and the MSAs" if version == "alphafold2"
            else "the MSA/template data (*_data.json) and embeddings")
    print(grey % ("<b>%.1f MB</b> of intermediate files: %s. Tick <b>Remove intermediate files</b>, click Generate "
                  "to see the list, then Run. Structures, scores, plots and the download bundle are kept (the bundle "
                  "and the viewer data are saved first); afterwards the GPU stage cannot be re-run." % (mb, what)))
PYEOF
