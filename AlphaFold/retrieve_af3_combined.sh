#!/bin/bash
# AlphaFold 3 combined retriever: PAE matrix + ranked CIF structures + per-residue pLDDT.
# Uses stdlib Python only (no numpy needed — AF3 confidence data is plain JSON).

python3 << 'PYEOF'
import os, glob, json, base64, re

LOCATION = os.environ.get('LOCATION', '').strip()
if not LOCATION:
    print(json.dumps({"error": "LOCATION not set"}))
    raise SystemExit(0)

# ── Find output directory ──────────────────────────────────────────────────────
output_base = None
try:
    with open(os.path.join(LOCATION, 'af3_output_dir.txt')) as f:
        c = f.read().strip()
    if c and os.path.isdir(c):
        output_base = c
except Exception:
    pass
if not output_base:
    dirs = sorted(glob.glob(os.path.join(LOCATION, 'output_*')))
    output_base = dirs[0] if dirs else None

if not output_base:
    print(json.dumps({"error": "No AlphaFold 3 output directory found"}))
    raise SystemExit(0)

# ── Find timestamped GPU output subdir (contains ranking_scores.csv) ──────────
timestamped_dir = None
try:
    for entry in os.listdir(output_base):
        full = os.path.join(output_base, entry)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, 'ranking_scores.csv')):
            timestamped_dir = full
            break
except Exception:
    pass

if not timestamped_dir:
    print(json.dumps({"error": "AlphaFold 3 GPU output not ready yet"}))
    raise SystemExit(0)

# ── Read ranking scores ────────────────────────────────────────────────────────
ranking_scores = {}   # (seed, sample) -> float
try:
    with open(os.path.join(timestamped_dir, 'ranking_scores.csv')) as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                ranking_scores[(int(parts[0]), int(parts[1]))] = float(parts[2])
except Exception:
    pass

# ── Enumerate sample directories in order ────────────────────────────────────
sample_keys = []
try:
    for entry in sorted(os.listdir(timestamped_dir)):
        m = re.match(r'seed-(\d+)_sample-(\d+)$', entry)
        if m:
            sample_keys.append((int(m.group(1)), int(m.group(2))))
except Exception:
    pass

if not sample_keys:
    print(json.dumps({"error": "No sample directories found in output"}))
    raise SystemExit(0)

# ── CIF parser: per-residue pLDDT from CA B-factors ───────────────────────────
def parse_cif_ca_plddts(cif_text):
    """Extract per-residue pLDDT by reading CA B-factors from mmCIF."""
    cols = []
    atom_col = bfac_col = chain_col = seq_col = None
    plddts = []
    seen = set()
    for line in cif_text.splitlines():
        s = line.strip()
        if s == 'loop_':
            cols = []; atom_col = bfac_col = chain_col = seq_col = None; seen = set()
            continue
        if s.startswith('_atom_site.'):
            cols.append(s.split('.', 1)[1])
            continue
        if cols and atom_col is None and (s.startswith('ATOM') or s.startswith('HETATM')):
            try:
                atom_col  = cols.index('label_atom_id')
                bfac_col  = cols.index('B_iso_or_equiv')
                chain_col = cols.index('label_asym_id')
                seq_col   = cols.index('label_seq_id')
            except ValueError:
                continue
        if atom_col is not None and (s.startswith('ATOM') or s.startswith('HETATM')):
            parts = s.split()
            if len(parts) > max(atom_col, bfac_col, chain_col, seq_col):
                if parts[atom_col] == 'CA' and parts[seq_col] not in ('.', '?'):
                    key = (parts[chain_col], parts[seq_col])
                    if key not in seen:
                        seen.add(key)
                        try:
                            plddts.append(round(float(parts[bfac_col]), 2))
                        except ValueError:
                            pass
    return plddts

# ── Process each sample ────────────────────────────────────────────────────────
cifs    = {}   # "sample_N" -> base64 CIF
samples = []

for i, key in enumerate(sample_keys):
    seed, sample_idx = key
    path = os.path.join(timestamped_dir, f'seed-{seed}_sample-{sample_idx}')

    cif_b64 = cif_text = None
    try:
        with open(os.path.join(path, 'model.cif'), 'rb') as f:
            raw = f.read()
        cif_b64  = base64.b64encode(raw).decode()
        cif_text = raw.decode('utf-8', errors='replace')
    except Exception:
        pass

    if cif_b64:
        cifs[f'sample_{i}'] = cif_b64

    plddt = parse_cif_ca_plddts(cif_text) if cif_text else []

    sc = {}
    try:
        with open(os.path.join(path, 'summary_confidences.json')) as f:
            sc = json.load(f)
    except Exception:
        pass

    rs = ranking_scores.get(key, sc.get('ranking_score'))
    samples.append({
        'idx':           i,
        'label':         f'Sample {i + 1}',
        'ranking_score': round(rs, 4) if rs is not None else None,
        'ptm':           round(sc['ptm'],  4) if sc.get('ptm')  is not None else None,
        'iptm':          round(sc['iptm'], 4) if sc.get('iptm') is not None else None,
        'plddt':         plddt,
    })

# ── Best sample ────────────────────────────────────────────────────────────────
best_idx = max(range(len(samples)), key=lambda i: samples[i].get('ranking_score') or 0)
best_key = sample_keys[best_idx]

# ── PAE matrix from best sample's confidences.json ───────────────────────────
matrix = []
n_residues = 0
matrix_max = 31.75

seed, sample_idx = best_key
conf_path = os.path.join(timestamped_dir, f'seed-{seed}_sample-{sample_idx}', 'confidences.json')
try:
    with open(conf_path) as f:
        conf = json.load(f)
    pae = conf.get('pae', [])
    n_tokens = len(conf.get('token_res_ids', []))
    # AF3 pae is a nested list (N x N) — handle both nested and flat
    if pae and isinstance(pae[0], list):
        n_residues = len(pae)
        matrix     = [[round(v, 2) for v in row] for row in pae]
        all_vals   = [v for row in pae for v in row]
        matrix_max = round(max(all_vals), 1) if all_vals else 31.75
    elif n_tokens and len(pae) == n_tokens * n_tokens:
        n_residues = n_tokens
        matrix     = [[round(pae[r * n_tokens + c], 2) for c in range(n_tokens)]
                      for r in range(n_tokens)]
        matrix_max = round(max(pae), 1) if pae else 31.75
except Exception:
    pass

if n_residues == 0:
    n_residues = len(samples[best_idx].get('plddt', []))

best = samples[best_idx]

print(json.dumps({
    "data_type":      "pae",
    "matrix_label":   "Predicted Aligned Error (Å)",
    "matrix_max":     matrix_max,
    "matrix":         matrix,
    "best_sample":    best_idx,
    "n_residues":     n_residues,
    "cifs":           cifs,
    "samples":        samples,
    "ranking_score":  best.get('ranking_score'),
    "ptm":            best.get('ptm'),
    "iptm":           best.get('iptm'),
}))
PYEOF
