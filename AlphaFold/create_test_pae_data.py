#!/usr/bin/env python3
"""
Create a synthetic AlphaFold2 test directory using a REAL prediction
downloaded from the EBI AlphaFold database.

This lets you test the PAE viewer without needing a local monomer_ptm/multimer run.

Usage:
    python3 create_test_pae_data.py [uniprot_id] [output_dir]

Defaults:
    uniprot_id  = P00698       (hen egg-white lysozyme, 148 aa — good PAE structure)
    output_dir  = /tmp/af_pae_test

The script only uses Python stdlib — no numpy needed.
The result PKL stores plddt and predicted_aligned_error as plain Python lists,
which the retrieve_af_*.sh scripts handle identically to numpy arrays.

After running, test with:
    LOCATION='<output_dir>' bash retrieve_af_combined.sh | python3 -m json.tool | head -60
"""

import sys, os, json, pickle, math, urllib.request, urllib.error

UNIPROT_ID = sys.argv[1] if len(sys.argv) > 1 else 'P00698'
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else '/tmp/af_pae_test'

EBI_API   = f'https://alphafold.ebi.ac.uk/api/prediction/{UNIPROT_ID}'

# ── Step 1: query EBI API to get file URLs ────────────────────────────────────
print(f'Querying EBI AlphaFold API for {UNIPROT_ID}...')
try:
    with urllib.request.urlopen(EBI_API, timeout=30) as r:
        entries = json.load(r)
except urllib.error.URLError as e:
    sys.exit(f'ERROR: Could not reach EBI API: {e}')

if not entries:
    sys.exit(f'ERROR: No entry found for {UNIPROT_ID}')

entry   = entries[0]
pdb_url = entry.get('pdbUrl')
pae_url = entry.get('paeDocUrl')

if not pdb_url or not pae_url:
    sys.exit(f'ERROR: Missing pdbUrl or paeDocUrl in API response: {entry.keys()}')

print(f'  PDB URL : {pdb_url}')
print(f'  PAE URL : {pae_url}')

# ── Step 2: download PDB ──────────────────────────────────────────────────────
print('Downloading PDB...')
try:
    with urllib.request.urlopen(pdb_url, timeout=60) as r:
        pdb_text = r.read().decode('utf-8')
except urllib.error.URLError as e:
    sys.exit(f'ERROR downloading PDB: {e}')

# ── Step 3: parse PDB — extract sequence, pLDDT (B-factor), coordinates ──────
AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}

sequence_chars = []
plddt_list     = []
chains_raw     = []         # list of (chain_id, res_num, res_name, x, y, z, bfac)
seen           = set()
prev_chain     = None
chain_start    = 0
chains         = []

for line in pdb_text.splitlines():
    if not line.startswith('ATOM'):
        continue
    if line[13:15].strip() != 'CA':
        continue
    chain_id = line[21]
    try:
        res_num = int(line[22:26])
        bfac    = float(line[60:66])
        x       = float(line[30:38])
        y       = float(line[38:46])
        z       = float(line[46:54])
    except ValueError:
        continue
    res_name = line[17:20].strip()
    key = (chain_id, res_num)
    if key in seen:
        continue
    seen.add(key)

    if chain_id != prev_chain:
        if prev_chain is not None:
            chains.append({'id': prev_chain,
                           'start': chain_start,
                           'end': len(sequence_chars) - 1})
        chain_start = len(sequence_chars)
        prev_chain  = chain_id

    sequence_chars.append(AA3TO1.get(res_name, 'X'))
    plddt_list.append(round(bfac, 1))
    chains_raw.append((chain_id, res_num, res_name, x, y, z, bfac))

if prev_chain is not None:
    chains.append({'id': prev_chain,
                   'start': chain_start,
                   'end': len(sequence_chars) - 1})

sequence = ''.join(sequence_chars)
N = len(sequence_chars)
print(f'  Parsed {N} residues, {len(chains)} chain(s): {[c["id"] for c in chains]}')
print(f'  Sequence: {sequence[:40]}{"..." if N > 40 else ""}')
print(f'  Mean pLDDT: {sum(plddt_list)/len(plddt_list):.1f}')

# ── Step 4: download PAE JSON ─────────────────────────────────────────────────
print('Downloading PAE JSON...')
try:
    with urllib.request.urlopen(pae_url, timeout=60) as r:
        pae_data = json.load(r)
except urllib.error.URLError as e:
    sys.exit(f'ERROR downloading PAE: {e}')

# EBI format: [{"predicted_aligned_error": [[...]], "max_predicted_aligned_error": 31.75}]
if isinstance(pae_data, list):
    pae_data = pae_data[0]

pae_matrix = pae_data.get('predicted_aligned_error')
matrix_max = float(pae_data.get('max_predicted_aligned_error', 31.75))

if pae_matrix is None:
    sys.exit('ERROR: No predicted_aligned_error in PAE JSON')

pae_n = len(pae_matrix)
print(f'  PAE matrix: {pae_n}×{pae_n}, max={matrix_max}')

if pae_n != N:
    print(f'  WARNING: PAE size ({pae_n}) != residue count ({N}); using PAE size')
    N = pae_n
    # Trim plddt_list if needed
    plddt_list = plddt_list[:N]

# ── Step 5: build the output directory ───────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'\nWriting test data to: {OUTPUT_DIR}')

# ── ranking_debug.json ────────────────────────────────────────────────────────
model_names = [f'model_{i+1}_ptm_pred_0' for i in range(5)]
mean_plddt  = sum(plddt_list) / len(plddt_list)
# Give each model a slightly lower pLDDT so ranking makes sense
ranking = {
    'order': model_names,
    'plddts': {
        name: round(mean_plddt - i * 1.5, 2)
        for i, name in enumerate(model_names)
    },
}
with open(os.path.join(OUTPUT_DIR, 'ranking_debug.json'), 'w') as f:
    json.dump(ranking, f)
print('  wrote ranking_debug.json')

# ── result PKL (plain Python lists — no numpy needed) ────────────────────────
# pae_matrix from EBI is already a nested list of ints/floats
result = {
    'plddt':                    plddt_list,
    'predicted_aligned_error':  pae_matrix,
    'ptm':                      0.87,
}
pkl_path = os.path.join(OUTPUT_DIR, f'result_{model_names[0]}.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(result, f)
print(f'  wrote {os.path.basename(pkl_path)}')

# ── ranked PDB files (rank_0 = original, others get slightly adjusted B-factors) ──
def write_pdb(path, pdb_lines, bfac_offset=0.0):
    out = []
    for line in pdb_lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            try:
                orig = float(line[60:66])
                new  = max(0.0, min(100.0, orig + bfac_offset))
                line = line[:60] + f'{new:6.2f}' + line[66:]
            except (ValueError, IndexError):
                pass
        out.append(line)
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')

pdb_lines = pdb_text.splitlines()
for rank_idx in range(5):
    pdb_path = os.path.join(OUTPUT_DIR, f'ranked_{rank_idx}.pdb')
    write_pdb(pdb_path, pdb_lines, bfac_offset=-rank_idx * 1.5)
print('  wrote ranked_0.pdb … ranked_4.pdb')

# ── fake slurm_jobids.txt (so status scripts don't crash) ────────────────────
with open(os.path.join(OUTPUT_DIR, 'slurm_jobids.txt'), 'w') as f:
    f.write('99999999\n')
print('  wrote slurm_jobids.txt')

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"""
Done!  Test directory: {OUTPUT_DIR}
Protein: {UNIPROT_ID}  ({N} residues, true PAE {pae_n}×{pae_n})

Test retrieve_af_pae.sh:
    cd {os.path.abspath(os.path.dirname(__file__))}
    LOCATION='{OUTPUT_DIR}' bash retrieve_af_pae.sh | python3 -m json.tool | head -40

Test retrieve_af_combined.sh:
    LOCATION='{OUTPUT_DIR}' bash retrieve_af_combined.sh | python3 -m json.tool | head -40
""")
