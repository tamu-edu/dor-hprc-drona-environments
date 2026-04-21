#!/usr/bin/env python3
import json
import sys
import os
from subprocess import run

def main(verbose=False):
    try:
        input_text = sys.stdin.read()
    except Exception as e:
        return f"Error, unable to gather gpu availability data because:\n{e}"
    try:
        # Attempt to parse the input string as JSON.
        gpu_dict = json.loads(input_text)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input.  Details: {e}")
        sys.exit(1)
    any_row=False
    html = "<table border=1>\n"
    html+= "<tr><th>Nodes</th><th>GPU Type</th><th>GPU Count</th></tr>\n"
    for item in gpu_dict['avail(merged)']:
        if "VE" in item['identifier']:
            # vector engine doesn't count as a GPU for this purpose
            continue
        elif item['asterisk']:
            # excluding including partial nodes
            continue
        else:
            any_row = True
        (label, ngpus) = item['identifier'].split(":")
        html+="<tr><td>"+str(item['count'])+"</td><td>"+label+"</td><td>"+ngpus+"</td></tr>\n"
    html+="</table>"
    if not any_row:
        html+="<br>(all accelerators busy)</br>"
    return html
    
if __name__ == "__main__":
    result = main()
    print(result)