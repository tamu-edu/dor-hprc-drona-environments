#!/bin/bash
# Intro card at the top of the LAMMPS page (create_text.py).
cd "$(dirname "$0")" || exit 1
python3 ./create_text.py intro
