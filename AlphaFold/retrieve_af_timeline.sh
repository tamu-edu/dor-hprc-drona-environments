#!/bin/bash
# Parse AlphaFold log files and emit an HTML Gantt-style timeline.
# Requires only stdlib Python (no numpy/conda env needed).

python3 << 'PYEOF'
import os, re, ast, glob, json, subprocess
from datetime import datetime

LOCATION = os.environ.get('LOCATION', '').strip()
if not LOCATION:
    print("<div class='text-muted' style='font-size:0.85em;padding:8px'>No location available.</div>")
    raise SystemExit(0)

# ── Overall SLURM status ──────────────────────────────────────────────────────
def get_overall_status():
    jobids_file = os.path.join(LOCATION, 'slurm_jobids.txt')
    try:
        with open(jobids_file) as f:
            jobids = [l.strip() for l in f if l.strip()]
    except Exception:
        return 'UNKNOWN'
    for jid in jobids:
        r = subprocess.run(['squeue', '-j', jid, '-h', '-o', '%T'], capture_output=True, text=True)
        status = r.stdout.strip()
        if not status:
            r = subprocess.run(['sacct', '-j', jid, '--noheader', '--format=State', '--parsable2'],
                               capture_output=True, text=True)
            status = r.stdout.strip().split('\n')[0].split('|')[0].strip()
        if status in ('FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY'):
            return 'FAILED'
    return 'OK'

JOB_FAILED = get_overall_status() == 'FAILED'

# ── Log parsing ──────────────────────────────────────────────────────────────
LOG_RE = re.compile(r'^I(\d{2})(\d{2}) (\d{2}):(\d{2}):(\d{2})\.\d+\s+\S+\s+\S+\]\s+(.+)$')

def parse_log(filepath):
    events = []
    try:
        mtime = os.path.getmtime(filepath)
        year = datetime.fromtimestamp(mtime).year
        with open(filepath, errors='replace') as f:
            for line in f:
                m = LOG_RE.match(line.rstrip())
                if m:
                    mo, day, hh, mm, ss, msg = m.groups()
                    try:
                        ts = datetime(year, int(mo), int(day), int(hh), int(mm), int(ss))
                        events.append((ts, msg.strip()))
                    except ValueError:
                        pass
    except Exception:
        pass
    return events

def find_pair(events, start_pat, end_pat):
    start_ts = end_ts = duration = None
    for ts, msg in events:
        if start_pat in msg and start_ts is None:
            start_ts = ts
        elif end_pat in msg and start_ts is not None and end_ts is None:
            end_ts = ts
            duration = (end_ts - start_ts).total_seconds()
            break
    return start_ts, end_ts, duration

def fmt_dur(s):
    if s is None: return '—'
    if s < 60:    return f"{s:.0f}s"
    if s < 3600:  return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"

# ── Find log files ───────────────────────────────────────────────────────────
cpu_logs = sorted(glob.glob(os.path.join(LOCATION, 'out-alphafold.*')))
gpu_logs = sorted(glob.glob(os.path.join(LOCATION, 'out-parafold.*')))
# AF3 / generic fallback
if not cpu_logs and not gpu_logs:
    all_logs = sorted(glob.glob(os.path.join(LOCATION, 'slurm-*.out')))
    cpu_logs = all_logs[:1]
    gpu_logs = all_logs[1:2]

cpu_events = parse_log(cpu_logs[0]) if cpu_logs else []
gpu_events = parse_log(gpu_logs[0]) if gpu_logs else []

# ── CPU stage extraction ─────────────────────────────────────────────────────
cpu_stages = []

pairs = [
    ('Jackhmmer (UniRef90)',    'Started Jackhmmer (uniref90',      'Finished Jackhmmer (uniref90'),
    ('Jackhmmer (MGnify)',      'Started Jackhmmer (mgy_clusters',  'Finished Jackhmmer (mgy_clusters'),
    ('HHsearch (Templates)',    'Started HHsearch query',            'Finished HHsearch query'),
    ('HHblits (BFD)',           'Started HHblits query',             'Finished HHblits query'),
]
for label, s_pat, e_pat in pairs:
    ts0, ts1, dur = find_pair(cpu_events, s_pat, e_pat)
    if ts0:
        cpu_stages.append({'name': label, 'start': ts0, 'end': ts1, 'dur': dur})

# MSA stats
msa_stats = {}
for _, msg in cpu_events:
    for pat, key in [
        (r'Uniref90 MSA size: (\d+)',               'uniref90'),
        (r'BFD MSA size: (\d+)',                     'bfd'),
        (r'MGnify MSA size: (\d+)',                  'mgnify'),
        (r'Final \(deduplicated\) MSA size: (\d+)', 'final'),
        (r'Total number of templates.*?: (\d+)',     'templates'),
    ]:
        m = re.search(pat, msg)
        if m:
            msa_stats[key] = int(m.group(1))

# ── GPU stage extraction ─────────────────────────────────────────────────────
gpu_model_timings  = {}   # model_name -> seconds (float) or None if in-progress
gpu_relax_timings  = {}   # model_name -> seconds
current_running    = None

# Prefer the "Final timings" dict — exact and already computed
for _, msg in gpu_events:
    m = re.search(r'Final timings for \S+: (.+)$', msg)
    if m:
        try:
            td = ast.literal_eval(m.group(1))
            for k, v in td.items():
                if k.startswith('predict_and_compile_model_'):
                    gpu_model_timings[k.replace('predict_and_compile_', '')] = float(v)
                elif k.startswith('relax_model_'):
                    gpu_relax_timings[k.replace('relax_', '')] = float(v)
        except Exception:
            pass

# If job is still running, fall back to per-line parsing
if not gpu_model_timings:
    for _, msg in gpu_events:
        m = re.search(r'Running model (model_\d+_pred_0) on', msg)
        if m:
            name = m.group(1)
            if name not in gpu_model_timings:
                gpu_model_timings[name] = None   # in-progress marker
                current_running = name
        m = re.search(r'Total JAX model (model_\d+_pred_0) .* predict time.*?: ([\d.]+)s', msg)
        if m:
            gpu_model_timings[m.group(1)] = float(m.group(2))

# ── Ranking info ─────────────────────────────────────────────────────────────
ranking_plddts = {}
ranking_order  = []
for rfile in glob.glob(os.path.join(LOCATION, '**', 'ranking_debug.json'), recursive=True):
    try:
        with open(rfile) as f:
            rd = json.load(f)
        ranking_plddts = rd.get('plddts', {})
        ranking_order  = rd.get('order', [])
        break
    except Exception:
        pass

# ── HTML helpers ─────────────────────────────────────────────────────────────
BAR_COLORS_CPU = ['#4dabf7', '#74c0fc', '#a5d8ff', '#339af0']
BAR_COLORS_GPU = ['#51cf66', '#94d82d', '#ffd43b', '#ff922b', '#f06595']

def bar_row(label, dur, max_dur, color, suffix=''):
    pct = f"{min(100, dur / max(max_dur, 1) * 100):.1f}"
    dur_label = fmt_dur(dur)
    inner = f'<span style="padding-left:6px;font-size:0.8em;color:white;font-weight:500">{dur_label}</span>'
    return (
        f'<div style="display:flex;align-items:center;margin-bottom:6px;font-size:0.82em">'
        f'  <div style="width:170px;min-width:170px;text-align:right;padding-right:10px;'
        f'       color:#555;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{label}</div>'
        f'  <div style="flex:1;background:#e9ecef;border-radius:3px;height:20px;overflow:hidden">'
        f'    <div style="width:{pct}%;height:100%;background:{color};border-radius:3px;'
        f'         display:flex;align-items:center">{inner}</div>'
        f'  </div>'
        f'  <div style="margin-left:8px;white-space:nowrap">{suffix}</div>'
        f'</div>'
    )

def running_row(label):
    if JOB_FAILED:
        indicator = '<span style="font-style:italic;color:#dc3545">failed</span>'
    else:
        indicator = '<span style="font-style:italic;color:#6c757d">running…</span>'
    return (
        f'<div style="display:flex;align-items:center;margin-bottom:6px;font-size:0.82em">'
        f'  <div style="width:170px;min-width:170px;text-align:right;padding-right:10px;'
        f'       color:#555;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{label}</div>'
        f'  {indicator}'
        f'</div>'
    )

# ── Output ────────────────────────────────────────────────────────────────────
out = ['<div style="font-family:system-ui,sans-serif">']

section_hdr = lambda title: (
    f'<div style="font-size:0.83em;font-weight:600;color:#495057;margin-bottom:8px;'
    f'     padding:3px 8px;background:#f8f9fa;border-radius:4px;border-left:3px solid #adb5bd">'
    f'{title}</div>'
)

# ── Phase 1: CPU ──────────────────────────────────────────────────────────────
if cpu_stages:
    max_dur = max((s['dur'] or 0) for s in cpu_stages) or 1
    out.append('<div style="margin-bottom:16px">')
    out.append(section_hdr('Phase 1 &mdash; Data Pipeline (CPU)'))
    for i, s in enumerate(cpu_stages):
        if s['end'] is None:
            out.append(running_row(s['name']))
        else:
            out.append(bar_row(s['name'], s['dur'], max_dur, BAR_COLORS_CPU[i % 4]))
    if msa_stats:
        parts = []
        for key, label in [('uniref90','UniRef90'),('mgnify','MGnify'),('bfd','BFD')]:
            if key in msa_stats:
                parts.append(f'{label}: <strong>{msa_stats[key]:,}</strong> seq')
        if 'final' in msa_stats:
            parts.append(f'Final MSA: <strong>{msa_stats["final"]:,}</strong> seq')
        if 'templates' in msa_stats:
            parts.append(f'Templates: <strong>{msa_stats["templates"]}</strong>')
        out.append(
            f'<div style="font-size:0.78em;color:#6c757d;margin-top:4px;padding:4px 8px;'
            f'     background:#f8f9fa;border-radius:4px">{" &nbsp;|&nbsp; ".join(parts)}</div>'
        )
    out.append('</div>')

# ── Phase 2: GPU ──────────────────────────────────────────────────────────────
if gpu_model_timings:
    models = sorted(gpu_model_timings.keys())
    max_dur = max((v or 0) for v in gpu_model_timings.values()) or 1
    out.append('<div style="margin-bottom:16px">')
    out.append(section_hdr('Phase 2 &mdash; Model Inference (GPU)'))
    for i, mname in enumerate(models):
        dur = gpu_model_timings[mname]
        in_prog = dur is None and mname == current_running
        color = BAR_COLORS_GPU[i % 5]
        label = f'Model {i+1}'
        if in_prog:
            out.append(running_row(label))
        else:
            # rank badge
            suffix = ''
            if mname in ranking_order:
                rank = ranking_order.index(mname) + 1
                plddt = ranking_plddts.get(mname, 0)
                if rank == 1:
                    suffix += (f'<span style="font-size:0.75em;padding:1px 6px;border-radius:10px;'
                               f'background:#d3f9d8;color:#2f9e44;margin-left:4px">#{rank} &mdash; {plddt:.1f}</span>')
                else:
                    suffix += (f'<span style="font-size:0.75em;padding:1px 6px;border-radius:10px;'
                               f'background:#dee2e6;color:#495057;margin-left:4px">#{rank} &mdash; {plddt:.1f}</span>')
            # relax badge
            rdur = gpu_relax_timings.get(mname)
            if rdur:
                suffix += (f'<span style="font-size:0.75em;padding:1px 6px;border-radius:10px;'
                           f'background:#e7f5ff;color:#1971c2;margin-left:4px">relax {fmt_dur(rdur)}</span>')
            out.append(bar_row(label, dur, max_dur, color, suffix=suffix))
    out.append('</div>')

if not cpu_stages and not gpu_model_timings:
    out.append(
        '<div style="color:#6c757d;font-size:0.85em;padding:10px">'
        'Waiting for pipeline to start&hellip;</div>'
    )

out.append('</div>')
print('\n'.join(out))
PYEOF
