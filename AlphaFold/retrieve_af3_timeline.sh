#!/bin/bash
# AlphaFold 3 pipeline timeline.
# Phase 1: CPU data pipeline (MSA + template search)
# Phase 2: GPU structure prediction (per-seed progress + ranking scores)

python3 << 'PYEOF'
import os, re, glob, json, subprocess

LOCATION = os.environ.get('LOCATION', '').strip()
if not LOCATION:
    print("<div class='text-muted' style='font-size:0.85em;padding:8px'>No location available.</div>")
    raise SystemExit(0)

# ── Job IDs ───────────────────────────────────────────────────────────────────
cpu_jobid = gpu_jobid = None
try:
    with open(os.path.join(LOCATION, 'slurm_jobids.txt')) as f:
        ids = [l.strip() for l in f if l.strip()]
    if ids:       cpu_jobid = ids[0]
    if len(ids) > 1: gpu_jobid = ids[1]
except Exception:
    pass

# ── SLURM helpers ─────────────────────────────────────────────────────────────
def job_state_elapsed(jobid):
    """Returns (state, elapsed_str) or (None, None)."""
    if not jobid:
        return None, None
    r = subprocess.run(['squeue', '-j', jobid, '-h', '-o', '%T|%M'],
                       capture_output=True, text=True)
    line = r.stdout.strip()
    if line:
        parts = line.split('|')
        return parts[0], (parts[1] if len(parts) > 1 else None)
    r = subprocess.run(['sacct', '-j', jobid, '--noheader',
                        '--format=State,Elapsed', '--parsable2'],
                       capture_output=True, text=True)
    line = (r.stdout.strip().split('\n')[0] if r.stdout.strip() else '')
    if line:
        parts = line.split('|')
        return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else None)
    return None, None

def parse_elapsed(s):
    """SLURM elapsed string → seconds (int) or None."""
    if not s:
        return None
    try:
        days = 0
        if '-' in s:
            d, s = s.split('-', 1)
            days = int(d)
        parts = list(map(int, s.split(':')))
        if len(parts) == 2: return days * 86400 + parts[0] * 60 + parts[1]
        if len(parts) == 3: return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    except Exception:
        pass
    return None

def fmt_dur(s):
    if s is None: return '—'
    if s < 60:    return f'{s:.0f}s'
    if s < 3600:  return f'{s/60:.1f}m'
    return f'{s/3600:.1f}h'

FAILED_STATES = {'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY'}

cpu_state, cpu_elapsed_str = job_state_elapsed(cpu_jobid)
gpu_state, gpu_elapsed_str = job_state_elapsed(gpu_jobid)

cpu_done   = cpu_state == 'COMPLETED'
cpu_failed = cpu_state in FAILED_STATES if cpu_state else False
gpu_running = gpu_state == 'RUNNING'
gpu_done    = gpu_state == 'COMPLETED'
gpu_failed  = gpu_state in FAILED_STATES if gpu_state else False
any_failed  = cpu_failed or gpu_failed

cpu_elapsed_s = parse_elapsed(cpu_elapsed_str)
gpu_elapsed_s = parse_elapsed(gpu_elapsed_str)

# ── Output directory evidence ─────────────────────────────────────────────────
# CPU job writes the exact output path to af3_output_dir.txt; use it if present
output_dirs = []
af3_out_file = os.path.join(LOCATION, 'af3_output_dir.txt')
try:
    with open(af3_out_file) as f:
        af3_out = f.read().strip()
    if af3_out and os.path.isdir(af3_out):
        output_dirs = [af3_out]
except Exception:
    pass
if not output_dirs:
    output_dirs = sorted(glob.glob(os.path.join(LOCATION, 'output_*')))

# CPU done: plain {name}/ subdir with *_data.json exists (distinct from timestamped GPU subdir)
msa_done = any(
    glob.glob(os.path.join(od, '*', '*_data.json'))
    for od in output_dirs
)

# GPU output lives in a timestamped subdir: {name}_{YYYYMMDD_HHMMSS}/
# Identified by the presence of ranking_scores.csv inside it.
timestamped_dirs = []
for od in output_dirs:
    try:
        for entry in os.listdir(od):
            full = os.path.join(od, entry)
            if os.path.isdir(full) and os.path.exists(os.path.join(full, 'ranking_scores.csv')):
                timestamped_dirs.append(full)
    except Exception:
        pass

# Ranking scores from CSV: seed,sample,ranking_score
ranking_scores = {}   # (seed, sample) -> float
for td in timestamped_dirs:
    try:
        with open(os.path.join(td, 'ranking_scores.csv')) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    ranking_scores[(int(parts[0]), int(parts[1]))] = float(parts[2])
    except Exception:
        pass

# Each completed sample is a subdir: seed-N_sample-M/
sample_dirs   = {}   # (seed, sample) -> path
sample_scores = {}   # (seed, sample) -> {ranking_score, ptm, iptm}
for td in timestamped_dirs:
    try:
        for entry in sorted(os.listdir(td)):
            m = re.match(r'seed-(\d+)_sample-(\d+)$', entry)
            if m:
                key  = (int(m.group(1)), int(m.group(2)))
                path = os.path.join(td, entry)
                sample_dirs[key] = path
                try:
                    with open(os.path.join(path, 'summary_confidences.json')) as f:
                        d = json.load(f)
                    sample_scores[key] = {
                        'ranking_score': ranking_scores.get(key, d.get('ranking_score')),
                        'ptm':  d.get('ptm'),
                        'iptm': d.get('iptm'),
                    }
                except Exception:
                    pass
    except Exception:
        pass

num_samples_done = len(sample_dirs)
expected_samples = max(5, num_samples_done)

best_sample = None
if sample_scores:
    best_sample = max(sample_scores, key=lambda k: sample_scores[k].get('ranking_score') or 0)

# ── HTML helpers ──────────────────────────────────────────────────────────────
BAR_COLORS = ['#51cf66', '#94d82d', '#ffd43b', '#ff922b', '#f06595']

section_hdr = lambda title: (
    f'<div style="font-size:0.83em;font-weight:600;color:#495057;margin-bottom:8px;'
    f'padding:3px 8px;background:#f8f9fa;border-radius:4px;border-left:3px solid #adb5bd">'
    f'{title}</div>'
)

def bar_row(label, pct, color, suffix=''):
    """pct: 0–100."""
    inner = f'<span style="padding-left:6px;font-size:0.8em;color:white;font-weight:500">{pct:.0f}%</span>'
    return (
        f'<div style="display:flex;align-items:center;margin-bottom:6px;font-size:0.82em">'
        f'  <div style="width:170px;min-width:170px;text-align:right;padding-right:10px;'
        f'       color:#555;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{label}</div>'
        f'  <div style="flex:1;background:#e9ecef;border-radius:3px;height:20px;overflow:hidden">'
        f'    <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:3px;'
        f'         display:flex;align-items:center">{inner}</div>'
        f'  </div>'
        f'  <div style="margin-left:8px;white-space:nowrap">{suffix}</div>'
        f'</div>'
    )

def elapsed_bar(label, elapsed_str, color, suffix=''):
    """Single full-width bar just for showing an elapsed time."""
    inner = f'<span style="padding-left:6px;font-size:0.8em;color:white;font-weight:500">{elapsed_str or "—"}</span>'
    return (
        f'<div style="display:flex;align-items:center;margin-bottom:6px;font-size:0.82em">'
        f'  <div style="width:170px;min-width:170px;text-align:right;padding-right:10px;'
        f'       color:#555;white-space:nowrap">{label}</div>'
        f'  <div style="flex:1;background:#e9ecef;border-radius:3px;height:20px;overflow:hidden">'
        f'    <div style="width:100%;height:100%;background:{color};border-radius:3px;'
        f'         display:flex;align-items:center">{inner}</div>'
        f'  </div>'
        f'  <div style="margin-left:8px;white-space:nowrap">{suffix}</div>'
        f'</div>'
    )

def running_row(label):
    if any_failed:
        indicator = '<span style="font-style:italic;color:#dc3545">failed</span>'
    else:
        indicator = '<span style="font-style:italic;color:#6c757d">running&hellip;</span>'
    return (
        f'<div style="display:flex;align-items:center;margin-bottom:6px;font-size:0.82em">'
        f'  <div style="width:170px;min-width:170px;text-align:right;padding-right:10px;'
        f'       color:#555">{label}</div>'
        f'  {indicator}'
        f'</div>'
    )

def pending_row(label):
    return (
        f'<div style="display:flex;align-items:center;margin-bottom:6px;font-size:0.82em">'
        f'  <div style="width:170px;min-width:170px;text-align:right;padding-right:10px;'
        f'       color:#adb5bd">{label}</div>'
        f'  <span style="font-style:italic;color:#adb5bd;font-size:0.9em">pending</span>'
        f'</div>'
    )

# ── Output ────────────────────────────────────────────────────────────────────
out = ['<div style="font-family:system-ui,sans-serif">']

# ── Phase 1: CPU ──────────────────────────────────────────────────────────────
out.append('<div style="margin-bottom:16px">')
out.append(section_hdr('Phase 1 &mdash; Data Pipeline (CPU)'))

if msa_done or cpu_done:
    out.append(elapsed_bar('MSA &amp; Template Search', cpu_elapsed_str, '#4dabf7'))
elif cpu_state == 'RUNNING':
    out.append(running_row('MSA &amp; Template Search'))
elif cpu_state == 'PENDING':
    out.append(pending_row('MSA &amp; Template Search'))
elif cpu_failed:
    out.append(running_row('MSA &amp; Template Search'))  # running_row shows "failed" when any_failed
else:
    out.append(
        '<div style="color:#6c757d;font-size:0.85em;padding:4px 8px">'
        'Waiting for job to start&hellip;</div>'
    )

out.append('</div>')

# ── Phase 2: GPU ──────────────────────────────────────────────────────────────
out.append('<div style="margin-bottom:16px">')
out.append(section_hdr('Phase 2 &mdash; Structure Prediction (GPU)'))

if not msa_done and not sample_dirs and not gpu_running and not gpu_done:
    out.append(
        '<div style="color:#6c757d;font-size:0.85em;padding:4px 8px">'
        'Waiting for data pipeline&hellip;</div>'
    )
else:
    max_score = max((sample_scores[k].get('ranking_score') or 0) for k in sample_scores) if sample_scores else 1.0

    for i in range(expected_samples):
        # Samples are (seed=1, sample=0..N-1) by default
        key   = (1, i)
        label = f'Sample {i + 1}'
        color = BAR_COLORS[i % len(BAR_COLORS)]

        if key in sample_scores:
            sc = sample_scores[key]
            rs = sc.get('ranking_score')

            suffix_parts = []
            if rs is not None:
                bg = '#d3f9d8' if key == best_sample else '#dee2e6'
                fg = '#2f9e44' if key == best_sample else '#495057'
                suffix_parts.append(
                    f'<span style="font-size:0.75em;padding:1px 6px;border-radius:10px;'
                    f'background:{bg};color:{fg};margin-left:4px">'
                    f'Score {rs:.3f}</span>'
                )
            if sc.get('ptm') is not None:
                suffix_parts.append(
                    f'<span style="font-size:0.75em;padding:1px 6px;border-radius:10px;'
                    f'background:#fff9db;color:#e67700;margin-left:4px">'
                    f'pTM {sc["ptm"]:.3f}</span>'
                )
            if sc.get('iptm') is not None:
                suffix_parts.append(
                    f'<span style="font-size:0.75em;padding:1px 6px;border-radius:10px;'
                    f'background:#fff0f6;color:#c2255c;margin-left:4px">'
                    f'ipTM {sc["iptm"]:.3f}</span>'
                )
            out.append(elapsed_bar(label, gpu_elapsed_str, color, suffix=''.join(suffix_parts)))

        elif key in sample_dirs:
            out.append(running_row(label))
        elif gpu_running or gpu_state == 'PENDING':
            if i == num_samples_done:
                out.append(running_row(label))
            else:
                out.append(pending_row(label))
        elif gpu_failed:
            out.append(running_row(label))

out.append('</div>')
out.append('</div>')
print('\n'.join(out))
PYEOF
