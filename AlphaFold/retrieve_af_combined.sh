#!/bin/bash
# Combined retriever: PAE/distance matrix + ranked PDB structures + sequence/chain info.
# Used by the integrated 3D + PAE viewer.

AF_BASE=/sw/eb/sw/Python/3.10.4-GCCcore-11.3.0
AF_PYTHON="$AF_BASE/bin/python3"
export LD_LIBRARY_PATH="$AF_BASE/lib:/sw/eb/sw/GCCcore/11.3.0/lib64:/sw/eb/sw/FlexiBLAS/3.2.0-GCC-11.3.0/lib:/sw/eb/sw/OpenBLAS/0.3.20-GCC-11.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/sw/eb/sw/SciPy-bundle/2022.05-foss-2022a/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"

"$AF_PYTHON" << 'PYEOF'
import os, glob, json, pickle, base64
import numpy as np

LOCATION = os.environ.get('LOCATION', '').strip()
if not LOCATION:
    print(json.dumps({"error": "LOCATION not set"}))
    raise SystemExit(0)

# ── Locate ranking_debug.json ────────────────────────────────────────────────
ranking_file = None
protein_dir  = None
for f in glob.glob(os.path.join(LOCATION, '**', 'ranking_debug.json'), recursive=True):
    ranking_file = f
    protein_dir  = os.path.dirname(f)
    break

if not ranking_file:
    print(json.dumps({"error": "ranking_debug.json not found — run may not be complete yet"}))
    raise SystemExit(0)

with open(ranking_file) as f:
    rd = json.load(f)

order  = rd.get('order', [])
plddts = rd.get('plddts', {})

if not order:
    print(json.dumps({"error": "No model ranking in ranking_debug.json"}))
    raise SystemExit(0)

best_model = order[0]
pkl_path   = os.path.join(protein_dir, f'result_{best_model}.pkl')

if not os.path.exists(pkl_path):
    print(json.dumps({"error": f"PKL file not found: result_{best_model}.pkl"}))
    raise SystemExit(0)

with open(pkl_path, 'rb') as f:
    result = pickle.load(f)

# ── Per-residue pLDDT ────────────────────────────────────────────────────────
plddt_arr = result.get('plddt')
plddt_list = [round(float(v), 1) for v in plddt_arr] if plddt_arr is not None else []

def to_scalar(v):
    try:    return round(float(v), 4)
    except: return None

ptm  = to_scalar(result.get('ptm'))
iptm = to_scalar(result.get('iptm'))

# ── PAE matrix or expected distance from distogram ───────────────────────────
pae_raw = result.get('predicted_aligned_error')

if pae_raw is not None:
    matrix       = [[round(float(v), 2) for v in row] for row in pae_raw]
    data_type    = 'pae'
    matrix_max   = 30.0
    matrix_label = 'Predicted Aligned Error (Å)'
else:
    dg        = result.get('distogram', {})
    logits    = dg.get('logits')
    bin_edges = dg.get('bin_edges')

    if logits is None or bin_edges is None:
        print(json.dumps({"error": "Neither predicted_aligned_error nor distogram found in pkl"}))
        raise SystemExit(0)

    step    = float(bin_edges[1] - bin_edges[0])
    centres = np.concatenate([
        [bin_edges[0] / 2],
        (bin_edges[:-1] + bin_edges[1:]) / 2,
        [float(bin_edges[-1]) + step / 2],
    ])
    l        = logits - logits.max(axis=-1, keepdims=True)
    probs    = np.exp(l) / np.exp(l).sum(axis=-1, keepdims=True)
    exp_dist = (probs * centres).sum(axis=-1)

    matrix       = [[round(float(v), 2) for v in row] for row in exp_dist]
    data_type    = 'distance_map'
    matrix_max   = round(float(bin_edges[-1]) + step, 1)
    matrix_label = 'Expected Cβ Distance (Å)'

n = len(matrix)

# ── PDB files (base64-encoded) ───────────────────────────────────────────────
pdb_files = sorted(glob.glob(os.path.join(protein_dir, 'ranked_*.pdb')))
pdbs = {}
for pf in pdb_files:
    try:
        rank_num = int(os.path.basename(pf).replace('ranked_', '').replace('.pdb', '')) + 1
        with open(pf, 'rb') as f:
            pdbs[f'rank_{rank_num}'] = base64.b64encode(f.read()).decode()
    except Exception:
        pass

# ── Sequence & chain boundaries from ranked_0.pdb ────────────────────────────
AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}

sequence_chars = []
chains = []
prev_chain = None
chain_start = 0
seen_residues = set()

if pdb_files:
    try:
        with open(pdb_files[0]) as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                if line[13:15].strip() != 'CA':
                    continue
                chain_id = line[21]
                try:
                    res_num = int(line[22:26])
                except ValueError:
                    continue
                res_name = line[17:20].strip()
                key = (chain_id, res_num)
                if key in seen_residues:
                    continue
                seen_residues.add(key)
                if chain_id != prev_chain:
                    if prev_chain is not None:
                        chains.append({'id': prev_chain, 'start': chain_start,
                                       'end': len(sequence_chars) - 1})
                    chain_start = len(sequence_chars)
                    prev_chain = chain_id
                sequence_chars.append(AA3TO1.get(res_name, 'X'))
        if prev_chain is not None:
            chains.append({'id': prev_chain, 'start': chain_start,
                           'end': len(sequence_chars) - 1})
    except Exception:
        pass

sequence = ''.join(sequence_chars)

# ── Output ────────────────────────────────────────────────────────────────────
output = {
    "data_type":         data_type,
    "matrix_label":      matrix_label,
    "matrix_max":        matrix_max,
    "matrix":            matrix,
    "model":             best_model,
    "rank":              1,
    "plddt_mean":        round(float(plddts.get(best_model, 0)), 2),
    "n_residues":        n,
    "plddt_per_residue": plddt_list,
    "pdbs":              pdbs,
    "sequence":          sequence,
    "chains":            chains,
    "all_models": [
        {"name": m, "rank": i + 1, "plddt": round(float(plddts.get(m, 0)), 2)}
        for i, m in enumerate(order)
    ],
}
if ptm  is not None: output["ptm"]  = ptm
if iptm is not None: output["iptm"] = iptm

print(json.dumps(output))
PYEOF
