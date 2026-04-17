#!/bin/bash
# Given $LOCATION, finds ranked_*.pdb files (AlphaFold2/parafold output) and
# returns them base64-encoded in the same JSON structure the protein viewer expects.

mapfile -t PDB_FILES < <(
    find "$LOCATION" -name "ranked_*.pdb" -type f 2>/dev/null | sort
)

if [ ${#PDB_FILES[@]} -eq 0 ]; then
    echo '{"error": "No ranked PDB files found. The job may not have completed yet."}'
    exit 0
fi

printf "{\n"

first=true
rank=1
for pdb_file in "${PDB_FILES[@]}"; do
    if [ "$first" = true ]; then
        first=false
    else
        printf ",\n"
    fi
    printf "  \"rank_%d\": \"%s\"" "$rank" "$(base64 -w 0 < "$pdb_file")"
    rank=$((rank + 1))
done

printf "\n}\n"
