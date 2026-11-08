#!/usr/bin/env python3
"""Zip the results of a finished AlphaFold run for download, without the large intermediate files.

    make_af_bundle.py <workflow dir> <alphafold2|alphafold3>

Writes <workflow dir>/<prediction name>_results.zip (replaced if it exists):
  AlphaFold 2: ranked_*.pdb, ranking_debug.json, timings.json, relax_metrics.json, AlphaPickle plots and
               CSV/TXT files; not result_model_*.pkl, features.pkl, msas/ or the (un)relaxed_model_* copies.
  AlphaFold 3: the newest prediction directory (model.cif, confidences, ranking_scores.csv, all samples);
               not <name>_data.json (MSA/templates) or embeddings.
plus the input (FASTA / alphafold3-input.json). Stdlib only, Python 3.6 compatible.
"""
import fnmatch
import glob
import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "retriever_scripts"))
from af_results import af2_result_dir, af3_prediction_dir  # noqa: E402

AF2_KEEP = ["ranked_*.pdb", "ranking_debug.json", "timings.json", "relax_metrics.json",
            "*.png", "*.csv", "*.txt"]
AF3_SKIP = ["*_data.json", "*embeddings*", "*.npz"]


def fail(msg):
    print("ERROR: " + msg)
    sys.exit(1)


def af2_files(location):
    result_dir = af2_result_dir(location)
    if not result_dir:
        fail("no AlphaFold 2 results (ranking_debug.json) in " + location)
    files = [os.path.join(result_dir, f) for f in sorted(os.listdir(result_dir))
             if os.path.isfile(os.path.join(result_dir, f)) and any(fnmatch.fnmatch(f, p) for p in AF2_KEEP)]
    inputs = []
    for job in ("parafold-cpu.job", "reduced_dbs.job"):
        try:
            with open(os.path.join(location, job)) as f:
                for line in f:
                    for key in ("protein_fasta=", "--fasta_paths="):
                        if line.strip().startswith(key):
                            inputs.append(line.strip().split("=", 1)[1].strip().strip('"\''))
        except OSError:
            pass
    return os.path.basename(result_dir), result_dir, files, [p for p in inputs if os.path.isfile(p)][:1]


def af3_files(location):
    pred = af3_prediction_dir(location)
    if not pred:
        fail("no AlphaFold 3 prediction results in " + location)
    files = []
    for root, _, names in os.walk(pred):
        for n in sorted(names):
            if not any(fnmatch.fnmatch(n, p) for p in AF3_SKIP):
                files.append(os.path.join(root, n))
    inp = os.path.join(location, "alphafold3-input.json")
    return os.path.basename(pred), pred, files, [inp] if os.path.isfile(inp) else []


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in ("alphafold2", "alphafold3"):
        fail("usage: make_af_bundle.py <workflow dir> <alphafold2|alphafold3>")
    location = os.path.abspath(sys.argv[1])
    name, result_dir, files, inputs = (af2_files if sys.argv[2] == "alphafold2" else af3_files)(location)
    if not files:
        fail("no result files found in " + result_dir)
    target = os.path.join(location, name + "_results.zip")
    tmp = target + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, os.path.join(name, os.path.relpath(f, result_dir)))
        for f in inputs:
            z.write(f, os.path.join(name, "input", os.path.basename(f)))
    os.replace(tmp, target)
    size = os.path.getsize(target)
    print("Wrote %s (%d files, %.1f MB)" % (target, len(files) + len(inputs), size / 1e6))


if __name__ == "__main__":
    main()
