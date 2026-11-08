#!/bin/bash
# AlphaFold version of the workflow in $JOB_DIR: "alphafold2" or "alphafold3".
# Written at submit time to .drona/af_version; for older runs, guessed from the files.

[ -n "$JOB_DIR" ] && [ -d "$JOB_DIR" ] || exit 0

if [ -s "$JOB_DIR/.drona/af_version" ]; then
    head -n 1 "$JOB_DIR/.drona/af_version" | tr -cd 'a-z0-9'
elif compgen -G "$JOB_DIR/alphafold3-*.job" > /dev/null; then
    echo -n alphafold3
elif compgen -G "$JOB_DIR/parafold-*.job" > /dev/null || [ -e "$JOB_DIR/reduced_dbs.job" ]; then
    echo -n alphafold2
fi
