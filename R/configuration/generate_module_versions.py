#!/usr/bin/env python3
"""Build the version list for a module, for an environment's version dropdown (admin task).

Run once on each cluster (on a node where "module spider" shows what users of that
cluster see), and again after the module is installed or removed there:

    ./generate_module_versions.py R_tamu -o ../module_versions

It asks Lmod for every version of the module and the modules each one needs loaded
first, and writes <outdir>/<cluster>.json (cluster from /sw/local/bin/clustername,
or --cluster) as dynamicSelect options, newest first:

    [ {"value": "GCC/14.3.0 OpenMPI/5.0.8 R_tamu/4.5.2", "label": "R_tamu/4.5.2"}, ... ]

The value is the full "module load" argument, so the environment needs no table of
toolchains. Forms read the file through the retrieve_module_versions retriever, so
users never wait for "module spider". The file is replaced in one step, and only if
at least one version was found.

Nothing here is specific to R: any environment that loads one particular module can
use this script and the retriever with its own module name.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile


def lmod(args, timeout):
    """Run "module <args>" in a login shell (module is a shell function); return stdout+stderr."""
    cmd = "module " + " ".join(args)
    result = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
    return result.stdout + result.stderr


def versions(name, timeout):
    out = lmod(["-t", "spider", name], timeout)
    found = [line.strip() for line in out.splitlines() if line.strip().startswith(name + "/")]
    return list(dict.fromkeys(found))


def prerequisites(fullname, timeout):
    """Modules to load before fullname: the first line of the "You will need to load" block
    (Lmod lists alternatives one per line), or [] when it can be loaded directly."""
    lines = lmod(["spider", fullname], timeout).splitlines()
    for i, line in enumerate(lines):
        if "You will need to load all module(s)" in line:
            for candidate in lines[i + 1:]:
                if candidate.strip():
                    return candidate.split()
    return []


def version_key(fullname):
    # "R_tamu/4.10.1" sorts after "R_tamu/4.9.2"; non-numeric parts sort as text
    return [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in re.split(r"[./-]", fullname.split("/", 1)[1])]


def clustername():
    try:
        return subprocess.run(["/sw/local/bin/clustername"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("module", help="module name, e.g. R_tamu")
    parser.add_argument("-o", "--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "module_versions"),
                        help="directory for <cluster>.json (default: ../module_versions)")
    parser.add_argument("--cluster", default=None, help="cluster name (default: output of clustername)")
    parser.add_argument("--timeout", type=int, default=600, help="seconds allowed per module spider call (default 600)")
    args = parser.parse_args()

    cluster = args.cluster or clustername()
    if not cluster:
        sys.exit("Cannot determine the cluster name; pass --cluster")

    names = sorted(versions(args.module, args.timeout), key=version_key, reverse=True)
    if not names:
        sys.exit(f"No versions of {args.module} found on {cluster}; {args.outdir}/{cluster}.json left unchanged")

    options = []
    for fullname in names:
        options.append({"value": " ".join(prerequisites(fullname, args.timeout) + [fullname]), "label": fullname})
        print(f"{fullname:24} module load {options[-1]['value']}")

    os.makedirs(args.outdir, exist_ok=True)
    target = os.path.join(args.outdir, cluster + ".json")
    fd, tmp = tempfile.mkstemp(dir=args.outdir, prefix="." + cluster + ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(options, f, indent=2)
        f.write("\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, target)
    print(f"Wrote {len(options)} versions to {os.path.normpath(target)}")


if __name__ == "__main__":
    main()
