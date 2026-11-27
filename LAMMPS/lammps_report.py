#!/usr/bin/env python3
"""Read fields from a lammps-validator acceleration strategy (the JSON value of the
"Acceleration strategy" table (formerly a select)). Used by utils.py and, as a CLI, by the
extract_*.sh retrievers. Must not import Drona modules: the retrievers run it standalone."""
import argparse
import json
import os
import sys

FRAMEWORKS = {"kokkos": "Kokkos", "gpu": "GPU package", "openmp": "OpenMP", "intel": "Intel package",
              "opt": "OPT package"}


def setup_line(strategy):
    """The module load line of a strategy (the table row already shows hardware, package, build)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from accel_info import card, e
    data = strategy.get("data")
    if not isinstance(data, dict):
        return card("<div>%s</div>" % e(strategy.get("summary-txt", strategy.get("summary", ""))))
    setup = data.get("setup") or ("module load LAMMPS/%s" % data["module"] if data.get("module") else "")
    return card("<div class='muted'>Loads: <code style='white-space:normal'>%s</code></div>" % e(setup)) if setup else ""


def main(tool, input_data_str, verbose=False):
    if verbose:
        print(f"Tool: {tool}")
        print(f"Input data: {input_data_str}")
        print("=======")

    try:
        input_data = json.loads(input_data_str)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input.  Details: {e}")
        sys.exit(1)

    if "value" in input_data.keys():
        input_data = input_data["value"]

    if tool == 'setup-line':
        return setup_line(input_data)
    if tool in ('summary-html', 'summary-txt'):
        if tool in input_data.keys():
            return input_data[tool]
        print("Error: 'summary' field not found for the selected acceleration strategy.")
        sys.exit(1)
    if tool == 'gpu-type':
        if 'data' in input_data.keys():
            accelerator = input_data['data'].get("accelerator")
            if accelerator and accelerator != "n/a":
                return accelerator
            return "cpu"
        print("Error: 'data' field not found for the selected acceleration strategy.")
        sys.exit(1)
    if tool == 'accel-vendor':
        # nvidia / intel / amd / cpu, from accelerators/<cluster>.json; "other" for a GPU not in the file
        gpu_type = main('gpu-type', json.dumps(input_data))
        if gpu_type == "cpu":
            return "cpu"
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from accel_info import load_accelerators
        for o in load_accelerators():
            if str(o["value"]).lower() == str(gpu_type).lower():
                return (o.get("vendor") or "other").lower()
        return "other"
    if tool in ('gpu-count-kind', 'gpu-counts'):
        # "counts" of the strategy's accelerator in accelerators/<cluster>.json: the allowed GPU counts
        gpu_type = main('gpu-type', json.dumps(input_data))
        counts, name = [], str(gpu_type).upper()
        if gpu_type != "cpu":
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from accel_info import load_accelerators
            for o in load_accelerators():
                if str(o["value"]).lower() == str(gpu_type).lower() and isinstance(o.get("counts"), list):
                    counts = [str(c) for c in o["counts"]]
                    name = o.get("label") or name
        if tool == 'gpu-counts':
            return json.dumps([{"value": c, "label": "%s \u00d7 %s" % (c, name)} for c in counts])
        return "" if gpu_type == "cpu" else ("list" if counts else "number")
    if tool == 'data':
        if 'data' in input_data.keys():
            return input_data['data']
        print("Error: 'data' field not found for the selected acceleration strategy.")
        sys.exit(1)
    print(f"Error: Tool '{tool}' not recognized.")
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract specific fields from the input. Input is expected to be a JSON string.")
    parser.add_argument('--tool', dest='tool_name', type=str, required=True, help='The name of the tool to use.')
    parser.add_argument('--input', dest='input_data', type=str, required=True, help='The input data as a JSON string.')
    parser.add_argument('--verbose', dest='verbose', type=bool, required=False, help='Extra print statements for debugging.')
    args = parser.parse_args()
    print(main(args.tool_name, args.input_data, verbose=args.verbose))
