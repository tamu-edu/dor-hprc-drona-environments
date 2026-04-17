#!/bin/bash
PDB_DIR="/scratch/user/u.ak202494/alphafold_run/parafold_output_dir/1L2Y"

printf "{\n"

first=true
for i in 0 1 2 3 4; do
    pdb_file="${PDB_DIR}/ranked_${i}.pdb"

    if [ ! -f "$pdb_file" ]; then
        continue
    fi

    rank=$((i + 1))

    if [ "$first" = true ]; then
        first=false
    else
        printf ",\n"
    fi

    printf "  \"rank_%d\": \"%s\"" "$rank" "$(base64 -w 0 < "$pdb_file")"
done

printf "\n}\n"
