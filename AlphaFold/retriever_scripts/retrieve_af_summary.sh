#!/bin/bash
# Results summary card (HTML) for a finished AlphaFold 2 / 3 run:
# best model + confidence scores (from the result files) and the run time of each Slurm job.
# Params: LOCATION (workflow directory), DRONA_ID (workflow record), AF_VERSION (alphafold2|alphafold3).
# Stdlib Python only, Python 3.6 compatible.

PYTHONPATH="$(dirname "$(readlink -f "$0")")${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PYEOF'
import csv, glob, html, json, os, subprocess
from af_results import af2_result_dir, af3_output_base, af3_prediction_dirs

LOCATION = os.environ.get("LOCATION", "").strip()
DRONA_ID = os.environ.get("DRONA_ID", "").strip()
VERSION = os.environ.get("AF_VERSION", "").strip()
RUNTIME = os.environ.get("DRONA_RUNTIME_DIR", "")


def esc(v):
    return html.escape(str(v))


def note(msg):
    print("<div style='color:#64748b;font-size:0.9em'>" + esc(msg) + "</div>")
    raise SystemExit(0)


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ── confidence bands (AlphaFold DB colours) ───────────────────────────────────
def plddt_band(v):
    if v is None:
        return ("", "#64748b")
    if v >= 90:
        return ("very high", "#0053D6")
    if v >= 70:
        return ("confident", "#3b9ae1")
    if v >= 50:
        return ("low", "#b58900")
    return ("very low", "#e0662f")


def tm_band(v):
    """pTM / ipTM: > 0.8 high-quality, 0.6-0.8 grey zone, < 0.6 likely wrong (AF3 guidance)."""
    if v is None:
        return ("", "#64748b")
    if v > 0.8:
        return ("high", "#0053D6")
    if v >= 0.6:
        return ("grey zone", "#b58900")
    return ("low", "#e0662f")


def metric(label, value, fmt, band=None, hint=""):
    if value is None:
        return ""
    word, colour = band(value) if band else ("", "#0f172a")
    sub = "<div style='font-size:0.75em;color:%s'>%s</div>" % (colour, esc(word)) if word else ""
    title = " title='%s'" % esc(hint) if hint else ""
    return ("<div%s style='min-width:7.5rem;padding:6px 10px;border:1px solid #e2e8f0;border-radius:8px;background:#fff'>"
            "<div style='font-size:0.72em;color:#64748b;text-transform:uppercase;letter-spacing:0.04em'>%s</div>"
            "<div style='font-size:1.25em;font-weight:600;color:%s'>%s</div>%s</div>"
            % (title, esc(label), colour if word else "#0f172a", esc(fmt % value), sub))


# ── Slurm run times from the workflow record ─────────────────────────────────
def job_rows():
    if not DRONA_ID or not RUNTIME:
        return []
    try:
        out = subprocess.check_output(
            ["python3", os.path.join(RUNTIME, "db_access", "drona_db_retriever.py"), "-i", DRONA_ID],
            universal_newlines=True, stderr=subprocess.DEVNULL)
        ids = [str(j["id"]) for j in json.loads(out)["runtime_meta"]["jobinfo"]]
    except Exception:
        return []
    if not ids:
        return []
    try:
        out = subprocess.check_output(
            ["sacct", "-X", "-n", "-P", "-j", ",".join(ids), "--format=JobID,JobName,State,Elapsed,AllocTRES"],
            universal_newlines=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    rows = {}
    for line in out.splitlines():
        p = line.split("|")
        if len(p) >= 5:
            rows[p[0]] = p
    return [rows[i] for i in ids if i in rows]


def stage_name(jobname, tres):
    n = jobname.lower()
    if "gpu" in n or "parafold" in n or "gres/gpu" in tres:
        return "GPU stage (prediction)"
    return "CPU stage (MSA, templates)"


def jobs_html():
    rows = job_rows()
    if not rows:
        return ""
    cells = []
    for jid, name, state, elapsed, tres in rows:
        colour = "#10b981" if state.startswith("COMPLETED") else "#e0662f" if state.split()[0] in (
            "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL") else "#64748b"
        cells.append("<span style='margin-right:1.2rem'>%s <code>%s</code>: %s "
                     "<span style='color:%s'>%s</span></span>"
                     % (esc(stage_name(name, tres)), esc(jid), esc(elapsed), colour, esc(state.split()[0].lower())))
    return "<div style='margin-top:8px;font-size:0.85em;color:#334155'>" + "".join(cells) + "</div>"


def card(title_html, metrics, extra=""):
    print("<div style='font-size:0.95em'>")
    print("<div style='margin-bottom:6px'>" + title_html + "</div>")
    print("<div style='display:flex;flex-wrap:wrap;gap:8px'>" + "".join(m for m in metrics if m) + "</div>")
    print(extra + jobs_html())
    print("</div>")


# ── AlphaFold 3 ──────────────────────────────────────────────────────────────
def af3():
    base = af3_output_base(LOCATION)
    if not base:
        note("No AlphaFold 3 output directory found.")
    # newest prediction directory (a re-run writes a new <name>_<timestamp> directory)
    preds = af3_prediction_dirs(base)
    if not preds:
        note("No AlphaFold 3 prediction results yet.")
    pred = preds[0]
    summary = None
    for f in glob.glob(os.path.join(pred, "*_summary_confidences.json")):
        summary = load_json(f)
        break
    summary = summary or {}
    scores = []
    try:
        with open(os.path.join(pred, "ranking_scores.csv")) as f:
            for r in csv.DictReader(f):
                scores.append((float(r["ranking_score"]), r["seed"], r["sample"]))
    except Exception:
        pass
    best = max(scores) if scores else None
    plddt = None
    for f in glob.glob(os.path.join(pred, "*_confidences.json")):
        if f.endswith("_summary_confidences.json"):
            continue
        d = load_json(f) or {}
        a = d.get("atom_plddts") or []
        if a:
            plddt = sum(a) / len(a)
        break
    chains = len(summary.get("chain_ptm") or [])
    title = ("<b>%s</b> · best of %d samples: seed %s, sample %s · %d chain%s"
             % (esc(os.path.basename(pred)), len(scores), esc(best[1]) if best else "?",
                esc(best[2]) if best else "?", chains, "" if chains == 1 else "s"))
    iptm = summary.get("iptm") if chains > 1 else None
    clash = summary.get("has_clash")
    extra = ""
    if clash:
        extra = "<div style='margin-top:6px;color:#e0662f;font-size:0.85em'>The top model has clashing chains (has_clash).</div>"
    card(title, [
        metric("Ranking score", summary.get("ranking_score", best[0] if best else None), "%.2f",
               hint="AlphaFold 3 ranking score of the top model (0-1)"),
        metric("Mean pLDDT", plddt, "%.1f", plddt_band, "Average per-atom confidence of the top model (0-100)"),
        metric("pTM", summary.get("ptm"), "%.2f", tm_band, "Predicted TM-score of the whole structure"),
        metric("ipTM", iptm, "%.2f", tm_band, "Interface pTM: confidence in the relative position of the chains"),
        metric("Disordered", summary.get("fraction_disordered"), "%.2f",
               hint="Fraction of the structure predicted as disordered"),
    ], extra)


# ── AlphaFold 2 ──────────────────────────────────────────────────────────────
def af2():
    result_dir = af2_result_dir(LOCATION)
    if not result_dir:
        note("No AlphaFold 2 results (ranking_debug.json) yet.")
    rfile = os.path.join(result_dir, "ranking_debug.json")
    rank = load_json(rfile) or {}
    order = rank.get("order") or []
    best = order[0] if order else None
    multimer = "iptm+ptm" in rank
    plddts = rank.get("plddts") or {}
    best_plddt = plddts.get(best) if best else None
    if best_plddt is None:
        conf = load_json(os.path.join(os.path.dirname(rfile), "confidence_%s.json" % best)) if best else None
        if conf and conf.get("confidenceScore"):
            v = conf["confidenceScore"]
            best_plddt = sum(v) / len(v)
    vals = list(plddts.values())
    spread = "%.1f-%.1f" % (min(vals), max(vals)) if len(vals) > 1 else ""
    timings = load_json(os.path.join(os.path.dirname(rfile), "timings.json")) or {}
    predict = sum(v for k, v in timings.items() if k.startswith("predict_and_compile"))
    title = ("<b>%s</b> · best model <code>%s</code> of %d%s"
             % (esc(os.path.basename(os.path.dirname(rfile))), esc(best or "?"), len(order),
                " · multimer" if multimer else ""))
    card(title, [
        metric("Model confidence", (rank.get("iptm+ptm") or {}).get(best) if multimer else None, "%.2f", tm_band,
               "0.8 x ipTM + 0.2 x pTM of the best model (multimer ranking)"),
        metric("Mean pLDDT", best_plddt, "%.1f", plddt_band, "Average per-residue confidence of the best model (0-100)"),
        metric("pLDDT range", None if not spread else spread, "%s",
               hint="Mean pLDDT of the worst and best model"),
        metric("Prediction", predict / 60.0 if predict else None, "%.1f min",
               hint="GPU time spent predicting all models (timings.json)"),
    ])


if not LOCATION or not os.path.isdir(LOCATION):
    note("Workflow directory not found.")
if VERSION == "alphafold3":
    af3()
elif VERSION == "alphafold2":
    af2()
else:
    note("AlphaFold version unknown.")
PYEOF
