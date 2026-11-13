#!/usr/bin/env python3
"""Record the package library directories of every R version in the dropdown (admin task).

Run after generate_module_versions.py, on the same cluster:

    ./generate_r_libraries.py

For each version in ../module_versions/<cluster>.json it loads the modules once and writes
../module_versions/<cluster>.libs.json:

    { "GCC/14.3.0 OpenMPI/5.0.8 R_tamu/4.5.2": {
        "libs": ["/sw/hprc/sw/dor-hprc-tools-Rtamu/R_LIBS/4.5.2-gfbf-2025b", "/sw/eb/sw/R-bundle-CRAN/...", ...],
        "user_lib": "R_LIBS_USER/4.5.2-gfbf-2025b" }, ... }

libs are R_tamu's library, the site bundles (R_LIBS_SITE) and base R's library; user_lib is
the R_tamu personal library under $SCRATCH (see the R_tamu Rscript wrapper). The form's
package check only lists these directories, so the preview never loads a module or starts R.
Without this file the package check is skipped.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def clustername():
    try:
        return subprocess.run(["/sw/local/bin/clustername"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def library_dirs(modules, timeout):
    script = ("module purge >/dev/null 2>&1; module load %s >/dev/null 2>&1 || exit 3; "
              "echo \"R_LIBS=$R_LIBS\"; echo \"R_LIBS_SITE=$R_LIBS_SITE\"; echo \"EBROOTR=$EBROOTR\"; "
              "echo \"RPACKAGEDIR=$RPACKAGEDIR\"; echo \"R_TAMU_ARCH=$R_TAMU_ARCH\"" % modules)
    result = subprocess.run(["bash", "-lc", script], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        return None
    env = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    libs = [d for d in env.get("R_LIBS", "").split(":") + env.get("R_LIBS_SITE", "").split(":") if d]
    for sub in ("lib64", "lib"):
        base = os.path.join(env.get("EBROOTR", ""), sub, "R", "library")
        if env.get("EBROOTR") and os.path.isdir(base):
            libs.append(base)
            break
    libs = [d for i, d in enumerate(libs) if d not in libs[:i] and os.path.isdir(d)]
    ext = os.path.basename(env.get("RPACKAGEDIR", "").rstrip("/"))
    user_lib = "R_LIBS_USER%s/%s" % (env.get("R_TAMU_ARCH", ""), ext) if ext else ""
    return {"libs": libs, "user_lib": user_lib}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-d", "--dir", default=os.path.join(HERE, "..", "module_versions"),
                        help="module_versions directory (default: ../module_versions)")
    parser.add_argument("--cluster", default=None, help="cluster name (default: output of clustername)")
    parser.add_argument("--timeout", type=int, default=300, help="seconds allowed per module load (default 300)")
    args = parser.parse_args()

    cluster = args.cluster or clustername()
    if not cluster:
        sys.exit("Cannot determine the cluster name; pass --cluster")
    try:
        with open(os.path.join(args.dir, cluster + ".json")) as f:
            versions = [o["value"] for o in json.load(f) if o.get("value")]
    except (OSError, ValueError, KeyError) as err:
        sys.exit("Cannot read the version list (run generate_module_versions.py first): %s" % err)

    table = {}
    for modules in versions:
        info = library_dirs(modules, args.timeout)
        if info is None or not info["libs"]:
            print("%-45s could not be loaded; skipped" % modules)
            continue
        table[modules] = info
        print("%-45s %d library directories" % (modules, len(info["libs"])))
    if not table:
        sys.exit("No version could be loaded; nothing written")

    target = os.path.join(args.dir, cluster + ".libs.json")
    fd, tmp = tempfile.mkstemp(dir=args.dir, prefix="." + cluster + ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(table, f, indent=2)
        f.write("\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, target)
    print("Wrote %d versions to %s" % (len(table), os.path.normpath(target)))


if __name__ == "__main__":
    main()
