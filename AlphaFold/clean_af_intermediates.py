#!/usr/bin/env python3
"""Remove the large intermediate files of a finished AlphaFold run (Manage "FREE UP SPACE" card).

    clean_af_intermediates.py <workflow dir> <alphafold2|alphafold3> [--list]

Removed:
  AlphaFold 2: result_model_*.pkl, features.pkl and msas/ in the result directory.
  AlphaFold 3: <name>_data.json (MSA + templates, in the CPU-stage folder and each prediction folder)
               and embeddings.
Kept: structures, confidence/score files, AlphaPickle plots, logs, job files, the download bundle.

Before removing, the download bundle is made if missing and (AF2) the viewer data is cached in .drona/,
so the Manage cards keep working. Afterwards a re-run of the GPU stage is no longer possible; the marker
.drona/af_cleaned records the date and the space freed.
--list prints what would be removed and exits. Stdlib only, Python 3.6 compatible.
"""
import fnmatch
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import functools

print = functools.partial(print, flush=True)  # keep order with the helper scripts it calls

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "retriever_scripts"))
from af_results import af2_result_dirs, af3_output_base  # noqa: E402

AF2_PATTERNS = ["result_model_*.pkl", "features.pkl", "msas"]
AF3_PATTERNS = ["*_data.json", "*embeddings*"]
MARKER = os.path.join(".drona", "af_cleaned")


def _size(path):
    if os.path.isdir(path) and not os.path.islink(path):
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += os.lstat(os.path.join(root, f)).st_size
                except OSError:
                    pass
        return total
    try:
        return os.lstat(path).st_size
    except OSError:
        return 0


def targets(location, version):
    """[(path, bytes)] of the intermediate files of the workflow."""
    found = []
    if version == "alphafold2":
        for d in af2_result_dirs(location):
            for name in sorted(os.listdir(d)):
                if any(fnmatch.fnmatch(name, p) for p in AF2_PATTERNS):
                    found.append(os.path.join(d, name))
    else:
        base = af3_output_base(location)
        if base:
            # <base>/<name>/ (CPU stage) and <base>/<name>_<timestamp>/ (each prediction), plus sample folders
            for root, dirs, files in os.walk(base):
                if root[len(base):].count(os.sep) > 2:
                    continue
                for name in sorted(dirs + files):
                    if any(fnmatch.fnmatch(name, p) for p in AF3_PATTERNS):
                        found.append(os.path.join(root, name))
                dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, p) for p in AF3_PATTERNS)]
    location = os.path.realpath(location)
    safe = [p for p in found if os.path.realpath(p).startswith(location + os.sep)]
    return [(p, _size(p)) for p in safe]


def main(argv):
    if len(argv) < 3 or argv[2] not in ("alphafold2", "alphafold3"):
        print("usage: clean_af_intermediates.py <workflow dir> <alphafold2|alphafold3> [--list]")
        return 2
    location, version = os.path.abspath(argv[1]), argv[2]
    found = targets(location, version)
    total = sum(s for _, s in found)
    if "--list" in argv:
        print(json.dumps({"files": [[os.path.relpath(p, location), s] for p, s in found], "bytes": total}))
        return 0
    if not found:
        print("Nothing to remove: no intermediate files in " + location)
        return 0

    # Keep the Manage cards working: bundle first, then the AF2 viewer cache (reads the best model's pkl)
    if not glob.glob(os.path.join(location, "*_results.zip")):
        print("==> Making the download bundle first")
        if subprocess.call(["python3", os.path.join(HERE, "make_af_bundle.py"), location, version]) != 0:
            print("ERROR: could not make the download bundle; nothing was removed.")
            return 1
    if version == "alphafold2":
        env = dict(os.environ, LOCATION=location)
        try:
            out = subprocess.check_output(["bash", os.path.join(HERE, "retriever_scripts", "retrieve_af_combined.sh")],
                                          env=env, universal_newlines=True)
            ok = os.path.isfile(os.path.join(location, ".drona", "af2_viewer_cache.json")) and '"error"' not in out[:200]
        except (OSError, subprocess.CalledProcessError):
            ok = False
        if not ok:
            print("ERROR: could not save the viewer data (.drona/af2_viewer_cache.json); nothing was removed.")
            return 1

    freed = 0
    for path, size in found:
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            freed += size
            print("removed %s (%.1f MB)" % (os.path.relpath(path, location), size / 1e6))
        except OSError as e:
            print("could not remove %s: %s" % (path, e.strerror or e))
    os.makedirs(os.path.join(location, ".drona"), exist_ok=True)
    previous = 0
    try:
        with open(os.path.join(location, MARKER)) as f:
            previous = json.load(f).get("bytes", 0)
    except (OSError, ValueError):
        pass
    with open(os.path.join(location, MARKER), "w") as f:
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M"), "bytes": previous + freed}, f)
    print("Freed %.1f MB" % (freed / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
