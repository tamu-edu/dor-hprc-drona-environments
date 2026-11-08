#!/bin/bash
# Combined retriever: PAE/distance matrix + ranked PDB structures + sequence/chain info.
# Used by the integrated 3D + PAE viewer.

AF_BASE=/sw/eb/sw/Python/3.10.4-GCCcore-11.3.0
AF_PYTHON="$AF_BASE/bin/python3"
export LD_LIBRARY_PATH="$AF_BASE/lib:/sw/eb/sw/GCCcore/11.3.0/lib64:/sw/eb/sw/FlexiBLAS/3.2.0-GCC-11.3.0/lib:/sw/eb/sw/OpenBLAS/0.3.20-GCC-11.3.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/sw/eb/sw/SciPy-bundle/2022.05-foss-2022a/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"

PYTHONPATH="$(dirname "$(readlink -f "$0")")${PYTHONPATH:+:$PYTHONPATH}" "$AF_PYTHON" << 'PYEOF'
import os, glob, json, pickle
from af_results import af2_result_dir, shrink_matrix

LOCATION = os.environ.get('LOCATION', '').strip()
if not LOCATION:
    print(json.dumps({"error": "LOCATION not set"}))
    raise SystemExit(0)

# ── Locate ranking_debug.json ────────────────────────────────────────────────
protein_dir  = af2_result_dir(LOCATION)          # newest result directory, same as the summary card
ranking_file = os.path.join(protein_dir, 'ranking_debug.json') if protein_dir else None

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

# ── Values from the best model's pkl, cached in .drona/ ──────────────────────
# Unpickling result_model_*.pkl takes seconds for large proteins, so the derived values are written once
# and reused while the pkl is unchanged (or after "Remove intermediate files" deleted it).
cache_path = os.path.join(LOCATION, '.drona', 'af2_viewer_cache.json')
pkl_mtime  = os.path.getmtime(pkl_path) if os.path.exists(pkl_path) else None
cached = None
try:
    with open(cache_path) as f:
        c = json.load(f)
    if c.get('model') == best_model and c.get('result_dir') == protein_dir and (pkl_mtime is None or c.get('pkl_mtime') == pkl_mtime):
        cached = c['values']
except (OSError, ValueError, KeyError, TypeError):
    pass

def to_scalar(v):
    try:    return round(float(v), 4)
    except: return None

def values_from_pkl():
    import numpy as np
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    plddt_arr = result.get('plddt')
    v = {
        'plddt_per_residue': [round(float(x), 1) for x in plddt_arr] if plddt_arr is not None else [],
        'ptm':  to_scalar(result.get('ptm')),
        'iptm': to_scalar(result.get('iptm')),
    }
    pae_raw = result.get('predicted_aligned_error')
    if pae_raw is not None:
        v['n_full'] = len(pae_raw)
        v['matrix'], v['matrix_axis'], v['matrix_step'] = shrink_matrix(pae_raw)
        v.update(data_type='pae', matrix_max=30.0, matrix_label='Predicted Aligned Error (Å)')
        return v
    dg        = result.get('distogram', {})
    logits    = dg.get('logits')
    bin_edges = dg.get('bin_edges')
    if logits is None or bin_edges is None:
        return None
    step    = float(bin_edges[1] - bin_edges[0])
    centres = np.concatenate([
        [bin_edges[0] / 2],
        (bin_edges[:-1] + bin_edges[1:]) / 2,
        [float(bin_edges[-1]) + step / 2],
    ])
    l        = logits - logits.max(axis=-1, keepdims=True)
    probs    = np.exp(l) / np.exp(l).sum(axis=-1, keepdims=True)
    exp_dist = (probs * centres).sum(axis=-1)
    v['n_full'] = len(exp_dist)
    v['matrix'], v['matrix_axis'], v['matrix_step'] = shrink_matrix(exp_dist)
    v.update(data_type='distance_map', matrix_max=round(float(bin_edges[-1]) + step, 1),
             matrix_label='Expected Cβ Distance (Å)')
    return v

if cached is None:
    if pkl_mtime is None:
        print(json.dumps({"error": f"PKL file not found: result_{best_model}.pkl (and no cached viewer data)"}))
        raise SystemExit(0)
    cached = values_from_pkl()
    if cached is None:
        print(json.dumps({"error": "Neither predicted_aligned_error nor distogram found in pkl"}))
        raise SystemExit(0)
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path + '.part'
        with open(tmp, 'w') as f:
            json.dump({'model': best_model, 'result_dir': protein_dir, 'pkl_mtime': pkl_mtime, 'values': cached}, f)
        os.replace(tmp, cache_path)
    except OSError:
        pass

plddt_list   = cached['plddt_per_residue']
ptm, iptm    = cached.get('ptm'), cached.get('iptm')
matrix, matrix_axis, matrix_step = cached['matrix'], cached['matrix_axis'], cached['matrix_step']
data_type, matrix_max, matrix_label = cached['data_type'], cached['matrix_max'], cached['matrix_label']
n = cached['n_full']

# ── PDB files (plain text, structure_encoding: text) ─────────────────────────
# All ranked structures as plain text, plus per-rank pLDDT (CA B-factors) for the rankings plot.
# (Fetching ranks on demand needs DynamicViewer support: the sandboxed iframe may not connect to the server.)
def pdb_ca_plddt(path):
    vals, seen = [], set()
    with open(path, errors='replace') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                key = (line[21], line[22:27])
                if key not in seen:
                    seen.add(key)
                    try:
                        vals.append(round(float(line[60:66]), 1))
                    except ValueError:
                        pass
    return vals

def rank_of(path):
    try:
        return int(os.path.basename(path)[len('ranked_'):-len('.pdb')]) + 1
    except ValueError:
        return None

pdb_files = sorted((p for p in glob.glob(os.path.join(protein_dir, 'ranked_*.pdb')) if rank_of(p)), key=rank_of)
pdbs, rank_plddts = {}, {}
for pf in pdb_files:
    key = f'rank_{rank_of(pf)}'
    try:
        rank_plddts[key] = pdb_ca_plddt(pf)
        with open(pf, errors='replace') as f:
            pdbs[key] = f.read()
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
    "matrix_axis":       matrix_axis,   # residue numbers of the cells when shrunk, else None
    "matrix_step":       matrix_step,   # residues per cell (1 = full matrix)
    "structure_encoding": "text",
    "model":             best_model,
    "rank":              1,
    "plddt_mean":        round(float(plddts.get(best_model, 0)), 2),
    "n_residues":        n,
    "plddt_per_residue": plddt_list,
    "pdbs":              pdbs,
    "rank_plddts":       rank_plddts,
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
