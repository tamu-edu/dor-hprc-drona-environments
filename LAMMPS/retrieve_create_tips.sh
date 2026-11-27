#!/bin/bash
# YAML thermo tip under the input file picker (create_text.py).
cd "$(dirname "$0")" || exit 1
python3 ./create_text.py tip
