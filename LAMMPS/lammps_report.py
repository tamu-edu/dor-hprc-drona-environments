#!/usr/bin/env python3
"""Read fields from a lammps-validator acceleration strategy (the JSON value of the
"Module and Acceleration Strategy" select). Used by utils.py and, as a CLI, by the
extract_*.sh retrievers. Must not import Drona modules: the retrievers run it standalone."""
import argparse
import json
import sys


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
