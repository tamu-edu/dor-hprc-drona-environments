#!/usr/bin/env python3
"""Submit-time check of the AlphaFold 3 input, run by the create driver before sbatch.

The preview cannot see edits made in the alphafold3-input.json tab, so the checks run here, on the
files as submitted. Without them an unedited skeleton or a broken file only fails when the CPU job
starts, after the queue wait.

Usage: check_af3_input.py <input.json> [<gpu job file>]   (exit 1 = do not submit)
"""
import json
import re
import sys

PLACEHOLDER = "REPLACE_WITH_PROTEIN_SEQUENCE"
POLYMERS = ("protein", "rna", "dna")


def problems(input_path, gpu_job=None):
    try:
        with open(input_path) as f:
            text = f.read()
    except OSError as e:
        return ["cannot read %s: %s" % (input_path, e.strerror or e)]
    if PLACEHOLDER in text:
        return ["%s still contains the placeholder %s. Edit the alphafold3-input.json tab (or select an input "
                "file) and submit again." % (input_path, PLACEHOLDER)]
    try:
        data = json.loads(text)
    except ValueError as e:
        return ["%s is not valid JSON: %s" % (input_path, e)]
    if isinstance(data, list):          # alphafoldserver dialect: AlphaFold 3 checks it itself
        return []
    if not isinstance(data, dict):
        return ["%s must contain a JSON object" % input_path]
    found = []
    if not str(data.get("name", "")).strip():
        found.append('"name" is empty; AlphaFold 3 names the output folder after it')
    seqs = data.get("sequences")
    if not seqs:
        found.append('"sequences" is empty')
    for i, seq in enumerate(seqs or [], 1):
        for kind, entry in (seq.items() if isinstance(seq, dict) else []):
            if kind in POLYMERS and isinstance(entry, dict):
                s = str(entry.get("sequence", ""))
                if not s.strip():
                    found.append("sequence %d (%s) is empty" % (i, kind))
                elif kind == "protein" and not re.fullmatch(r"[A-Za-z]+", s):
                    found.append("sequence %d (protein) has characters other than letters" % i)
    seeds = data.get("modelSeeds") or []
    if not seeds:
        found.append('"modelSeeds" is empty; list at least one seed, e.g. [1]')
    if gpu_job and len(seeds) > 1:
        try:
            with open(gpu_job) as f:
                if re.search(r"--num_seeds=\d+", f.read()):
                    found.append("the input lists %d seeds and the GPU job sets --num_seeds; AlphaFold 3 accepts "
                                 "--num_seeds only with one seed in the input" % len(seeds))
        except OSError:
            pass
    return found


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    found = problems(argv[1], argv[2] if len(argv) > 2 else None)
    if not found:
        print("AlphaFold 3 input check: OK")
        return 0
    print("ERROR: AlphaFold 3 input check failed, nothing was submitted:")
    for p in found:
        print("  - " + p)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
