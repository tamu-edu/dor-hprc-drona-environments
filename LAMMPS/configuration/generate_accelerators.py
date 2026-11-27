#!/usr/bin/env python3
"""Build the accelerator list for the LAMMPS hardware filter (admin task).

Run once on each cluster where the environment is offered, and again after GPU
types are added or removed there:

    ./generate_accelerators.py

It asks Slurm for the GPU types on the cluster (GRES, "sinfo -o %G"), adds CPU, and
writes
../accelerators/<cluster>.json (cluster from DRONA_CLUSTER or Slurm's ClusterName,
or --cluster):

    [ {"value": "h100", "label": "H100 GPU", "vendor": "nvidia"},
      {"value": "pvc", "label": "PVC (Intel Max 1100)", "vendor": "intel",
       "constraints": {"2": "xelink2", "4": "xelink4"}},
      {"value": "cpu", "label": "CPU", "vendor": "cpu"} ]

value/label are the checkbox options (retrieve_accelerators). utils.py reads vendor
(the NVML GPU monitor only runs for nvidia) and constraints (#GPUs -> --constraint),
which are found from node features named xelink<N>. An optional "counts" list (e.g.
[1, 2, 4, 8]) limits the GPU count users can request for a type to those values (a
dropdown instead of a number field); add it by hand.

A re-run keeps label, vendor, constraints and counts you edited by hand for types it finds
again, adds new types and drops types Slurm no longer has. The file is replaced in
one step. If no GPU type is found while the old file has some, it is left unchanged
(a CPU-only cluster has no file, or one written with --cpu-only).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# Defaults for known type names: (label, vendor). Unknown names get "<NAME> GPU" and vendor "unknown".
KNOWN = {
    "h200": ("H200 GPU", "nvidia"), "h100": ("H100 GPU", "nvidia"), "a100": ("A100 GPU", "nvidia"),
    "l40s": ("L40S GPU", "nvidia"), "l40": ("L40 GPU", "nvidia"), "a40": ("A40 GPU", "nvidia"),
    "a30": ("A30 GPU", "nvidia"), "a10": ("A10 GPU", "nvidia"), "l4": ("L4 GPU", "nvidia"),
    "v100": ("V100 GPU", "nvidia"), "t4": ("T4 GPU", "nvidia"), "rtx": ("RTX GPU", "nvidia"),
    "pvc": ("PVC (Intel Max 1100)", "intel"),
    "mi300a": ("MI300A GPU", "amd"), "mi250": ("MI250 GPU", "amd"), "mi210": ("MI210 GPU", "amd"),
}
CPU = {"value": "cpu", "label": "CPU", "vendor": "cpu"}
KEPT_KEYS = ("label", "vendor", "constraints", "counts")
XELINK = re.compile(r"^xelink(\d+)$")


def cluster_name():
    name = os.environ.get("DRONA_CLUSTER", "").strip()
    if not name:
        out = subprocess.run(["scontrol", "show", "config"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout
        m = re.search(r"^ClusterName\s*=\s*(\S+)", out, re.M)
        name = m.group(1) if m else ""
    return name.lower()


def slurm_gpu_types():
    """{type: set of xelink GPU counts} from "sinfo -h -o %G|%f" (e.g. gpu:pvc:4(S:0) | gen5_fabric,xelink4)."""
    out = subprocess.run(["sinfo", "-h", "-o", "%G|%f"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True).stdout
    types = {}
    for line in out.splitlines():
        gres, _, features = line.partition("|")
        for item in gres.split(","):
            parts = re.sub(r"\(.*?\)", "", item).split(":")
            if len(parts) >= 3 and parts[0] == "gpu":   # gpu:<type>:<count>; untyped gpu:<count> is skipped
                links = types.setdefault(parts[1].lower(), set())
                for feature in features.split(","):
                    m = XELINK.match(feature.strip())
                    if m:
                        links.add(m.group(1))
    return types


def validator_accelerators(validator):
    """Accelerator names the LAMMPS validator has builds for (None in the archive means CPU), or None if unreadable."""
    files = glob.glob(os.path.join(validator, "archive", "accelerator_archive_*.json"))
    if not files:
        return None
    found = set()
    for path in files:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict):
                found.add((item.get("accelerator") or "cpu").lower())
    return found


def load_existing(path):
    try:
        with open(path) as f:
            return {o["value"]: o for o in json.load(f) if isinstance(o, dict) and o.get("value")}
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--outdir", default=os.path.join(HERE, "..", "accelerators"),
                        help="directory for <cluster>.json (default: ../accelerators)")
    parser.add_argument("--cluster", default=None, help="cluster name (default: DRONA_CLUSTER or Slurm ClusterName)")
    parser.add_argument("--validator", default="/sw/hprc/sw/lammps-validator",
                        help="LAMMPS validator directory, for its build archive (default: %(default)s)")
    parser.add_argument("--require-build", action="store_true",
                        help="leave out GPU types the validator has no LAMMPS builds for (default: keep every type)")
    parser.add_argument("--exclude", default="", help="comma-separated GPU types to leave out, e.g. ve")
    parser.add_argument("--cpu-only", action="store_true", help="write a CPU-only list even if the old file has GPUs")
    parser.add_argument("-n", "--dry-run", action="store_true", help="print the result, write nothing")
    args = parser.parse_args()

    cluster = (args.cluster or cluster_name()).lower()
    if not cluster:
        sys.exit("Cannot determine the cluster name; pass --cluster")

    try:
        types = slurm_gpu_types()
    except (OSError, subprocess.CalledProcessError) as e:
        sys.exit(f"sinfo failed: {e}")
    for name in filter(None, (x.strip().lower() for x in args.exclude.split(","))):
        types.pop(name, None)

    if args.require_build:
        supported = validator_accelerators(args.validator)
        if supported is None:
            print(f"warning: no build archive in {args.validator}/archive; keeping all GPU types", file=sys.stderr)
        else:
            for name in sorted(set(types) - supported):
                print(f"skipped  {name:8} (no LAMMPS build in the validator archive)")
                types.pop(name)

    target = os.path.join(args.outdir, cluster + ".json")
    existing = load_existing(target)

    order = list(KNOWN)
    options = []
    for name in sorted(types, key=lambda n: (order.index(n) if n in order else len(order), n)):
        label, vendor = KNOWN.get(name, (name.upper() + " GPU", "unknown"))
        entry = {"value": name, "label": label, "vendor": vendor}
        if types[name]:
            entry["constraints"] = {n: "xelink" + n for n in sorted(types[name], key=int)}
        old = existing.get(name)
        if old:
            entry.update({k: old[k] for k in KEPT_KEYS if k in old})
            print(f"kept     {name:8} {entry['label']}")
        else:
            print(f"added    {name:8} {entry['label']}")
        if entry["vendor"] == "unknown":
            print(f"warning: vendor of '{name}' unknown; set \"vendor\" (nvidia/intel/amd) in {target}", file=sys.stderr)
        options.append(entry)
    cpu = dict(CPU)
    cpu.update({k: existing["cpu"][k] for k in ("label",) if k in existing.get("cpu", {})})
    options.append(cpu)
    for name in sorted(set(existing) - set(types) - {"cpu"}):
        print(f"dropped  {name:8} (no longer in Slurm, or skipped)")

    if len(options) == 1 and set(existing) - {"cpu"} and not args.cpu_only and not args.dry_run:
        sys.exit(f"No GPU types found; {target} left unchanged (use --cpu-only to write CPU only)")

    text = json.dumps(options, indent=2) + "\n"
    if args.dry_run:
        print(text, end="")
        return

    os.makedirs(args.outdir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=args.outdir, prefix="." + cluster + ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.chmod(tmp, 0o644)
    os.replace(tmp, target)
    print(f"Wrote {len(options)} accelerators to {os.path.normpath(target)}")


if __name__ == "__main__":
    main()
