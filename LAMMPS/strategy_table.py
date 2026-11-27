#!/usr/bin/env python3
"""Rows for the "Acceleration strategy" table on the LAMMPS create page (a dynamicTable).

Reads the lammps-validator strategy list (option_generator.sh: [{"label", "value"}]) on stdin and
prints {"columns", "rows"}. Each row's value is the validator's JSON string, unchanged, so map.json
($reportjson), extract_summary.sh, extract_gpu_type.sh and saved workflows work as with the old
dropdown. Hardware labels come from accelerators/<cluster>.json, package names from lammps_report.
Entries without "data" (read_file.py messages such as "Please select a file") become one row with
the message in the build column.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accel_info import load_accelerators  # noqa: E402
from lammps_report import FRAMEWORKS  # noqa: E402

COLUMNS = ["Hardware", "Package", "LAMMPS build"]


def row(option, labels):
    value = option.get("value", "")
    try:
        strategy = json.loads(value)
    except (TypeError, ValueError):
        strategy = {}
    data = strategy.get("data") if isinstance(strategy, dict) else None
    if not isinstance(data, dict):
        return {"value": value, "label": ["", "", str(option.get("label", ""))]}
    accel = str(data.get("accelerator") or "").lower()
    accel = "cpu" if accel in ("", "n/a") else accel
    hardware = labels.get(accel, "CPU" if accel == "cpu" else accel.upper() + " GPU")
    framework = str(data.get("framework") or "")
    package = FRAMEWORKS.get(framework.lower(), framework)
    return {"value": value, "label": [hardware, package, str(data.get("module") or "")]}


def main():
    try:
        options = json.loads(sys.stdin.read() or "[]")
    except ValueError:
        options = []
    if not isinstance(options, list):
        options = []
    labels = {a["value"].lower(): a.get("label", a["value"]) for a in load_accelerators()}
    rows = [row(o, labels) for o in options if isinstance(o, dict)]
    return json.dumps({"columns": COLUMNS, "rows": rows})


if __name__ == "__main__":
    print(main())
