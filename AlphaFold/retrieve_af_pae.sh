#!/bin/bash
# Extract confidence metrics from AlphaFold2 result pkl files.
# - monomer_ptm / multimer presets: outputs true PAE matrix
# - monomer preset:                 outputs expected-distance map from distogram
# Loads the AlphaFold module to get numpy.

AF_BASE=/sw/eb/sw/Python/3.10.4-GCCcore-11.3.0
AF_PYTHON="$AF_BASE/bin/python3"
export LD_LIBRARY_PATH="$AF_BASE/lib:/sw/eb/sw/GCCcore/11.3.0/lib64:/sw/eb/sw/FlexiBLAS/3.2.0-GCC-11.3.0/lib:/sw/eb/sw/OpenBLAS/0.3.20-GCC-11.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/sw/eb/sw/SciPy-bundle/2022.05-foss-2022a/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"

"$AF_PYTHON" << 'PYEOF'
import os, glob, json, pickle
import numpy as np

LOCATION = os.environ.get('LOCATION', '').strip()
if not LOCATION:
    print(json.dumps({"error": "LOCATION not set"}))
    raise SystemExit(0)

# ── Locate output files ───────────────────────────────────────────────────────
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

# ── Per-residue pLDDT (always present) ───────────────────────────────────────
plddt = result.get('plddt')
plddt_list = [round(float(v), 1) for v in plddt] if plddt is not None else []

def to_scalar(v):
    try:    return round(float(v), 4)
    except: return None

ptm  = to_scalar(result.get('ptm'))
iptm = to_scalar(result.get('iptm'))

# ── Matrix: PAE if available, else expected distance from distogram ───────────
pae_raw = result.get('predicted_aligned_error')

if pae_raw is not None:
    # True PAE (monomer_ptm / multimer)
    matrix     = [[round(float(v), 2) for v in row] for row in pae_raw]
    data_type  = 'pae'
    matrix_max = 30.0
    matrix_label = 'Predicted Aligned Error (Å)'
else:
    # Fall back to expected distance from distogram (all presets)
    dg         = result.get('distogram', {})
    logits     = dg.get('logits')     # (N, N, 64)
    bin_edges  = dg.get('bin_edges')  # (63,)  — break points between 64 bins

    if logits is None or bin_edges is None:
        print(json.dumps({"error": "Neither predicted_aligned_error nor distogram found in pkl"}))
        raise SystemExit(0)

    # Build bin centres for 64 bins
    step = float(bin_edges[1] - bin_edges[0])
    centres = np.concatenate([
        [bin_edges[0] / 2],                                  # bin 0: 0 → edge[0]
        (bin_edges[:-1] + bin_edges[1:]) / 2,                # bins 1-62
        [float(bin_edges[-1]) + step / 2],                   # bin 63: beyond last edge
    ])

    # Stable softmax then expected distance
    l = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(l) / np.exp(l).sum(axis=-1, keepdims=True)
    exp_dist = (probs * centres).sum(axis=-1)                # (N, N)

    max_val   = float(bin_edges[-1]) + step
    matrix    = [[round(float(v), 2) for v in row] for row in exp_dist]
    data_type = 'distance_map'
    matrix_max = round(max_val, 1)
    matrix_label = 'Expected Cβ Distance (Å)'

n = len(matrix)

output = {
    "data_type":          data_type,
    "matrix_label":       matrix_label,
    "matrix_max":         matrix_max,
    "model":              best_model,
    "rank":               1,
    "plddt_mean":         round(float(plddts.get(best_model, 0)), 2),
    "n_residues":         n,
    "matrix":             matrix,
    "plddt_per_residue":  plddt_list,
    "all_models": [
        {"name": m, "rank": i + 1, "plddt": round(float(plddts.get(m, 0)), 2)}
        for i, m in enumerate(order)
    ],
}
if ptm  is not None: output["ptm"]  = ptm
if iptm is not None: output["iptm"] = iptm

print(json.dumps(output))
PYEOF
