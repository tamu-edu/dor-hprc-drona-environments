"""The cluster's accelerator list, accelerators/<cluster>.json (see configuration/README), and the
shared look of the LAMMPS create-page cards. Used by gputable.py and lammps_report.py.
Standalone: must not import Drona modules, because retrievers run it as a plain script."""
import html
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def cluster_name():
    name = os.environ.get("DRONA_CLUSTER", "")
    if not name:
        try:
            out = subprocess.run(["scontrol", "show", "config"], stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, universal_newlines=True, timeout=10).stdout
            for line in out.splitlines():
                if line.startswith("ClusterName"):
                    name = line.split("=", 1)[1].strip()
                    break
        except (OSError, subprocess.SubprocessError):
            pass
    return name.lower()


def load_accelerators():
    """List of {"value", "label", "vendor", ...} in the file's order, or [] without a file."""
    cluster = cluster_name()
    if not cluster:
        return []
    try:
        with open(os.path.join(HERE, "accelerators", cluster + ".json")) as f:
            return [o for o in json.load(f) if isinstance(o, dict) and o.get("value")]
    except (OSError, ValueError, TypeError):
        return []


def parse_list(raw):
    """A list-valued retriever param: the app passes "['h100', 'pvc']" (Python text); JSON and a
    comma-separated string work too."""
    import ast
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        v = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        v = raw.split(",")
    v = v if isinstance(v, (list, tuple)) else [v]
    return [x.strip().lower() for x in v if isinstance(x, str) and x.strip()]


def e(text):
    return html.escape(str(text), quote=True)


# Same palette as the app's cards (choiceStyles.js, workdir/status strips) and the --drona-* type
# scale (website/docs/environments/styling.md); the value after each comma is a fallback.
CSS = """
<style>
.dc-lmp-create { color:#1e293b; font-size:var(--drona-text-data, 13px); }
.dc-lmp-create .muted { color:#64748b; font-size:var(--drona-text-caption, 12px); }
.dc-lmp-create .chips { display:flex; flex-wrap:wrap; gap:4px; margin-top:6px; }
.dc-lmp-create .chip { display:inline-block; font-size:var(--drona-text-caption, 12px); padding:1px 8px;
  border-radius:999px; border:1px solid #e2e8f0; background:#f1f5f9; color:#334155; white-space:nowrap; }
.dc-lmp-create .chip.accent { border-color:#d9b8b8; background:#fdf8f8; color:#500000;
  font-weight:var(--drona-weight-strong, 600); }
.dc-lmp-create .mono { font-family:var(--drona-font-mono, monospace); font-size:var(--drona-text-caption, 12px); }
.dc-lmp-create .line { display:flex; flex-wrap:wrap; align-items:center; gap:6px; }
.dc-lmp-create .foot { margin-top:8px; }
.dc-lmp-create table.avail { border-collapse:separate; border-spacing:0; min-width:360px; max-width:560px;
  border:1px solid #e2e8f0; border-radius:8px; overflow:hidden; background:#ffffff; }
.dc-lmp-create .avail th { text-align:left; font-size:var(--drona-text-caption, 12px); font-weight:var(--drona-weight-strong, 600);
  color:#64748b; background:#f8fafc; padding:6px 12px; border-bottom:1px solid #e2e8f0; }
.dc-lmp-create .avail td { font-size:var(--drona-text-data, 13px); padding:5px 12px; vertical-align:top; }
.dc-lmp-create .avail tr.start td { border-top:1px solid #e2e8f0; }
.dc-lmp-create .avail tr.inner td { border-top:1px solid #f1f5f9; }
.dc-lmp-create .avail td.type { font-weight:var(--drona-weight-strong, 600); }
.dc-lmp-create .avail td.type .muted { font-weight:var(--drona-weight-normal, 400); }
.dc-lmp-create .avail .num { text-align:right; }
.dc-lmp-create .avail td b { color:#500000; font-weight:var(--drona-weight-strong, 600); }
.dc-lmp-create .avail tr.off td { opacity:.45; }
.dc-lmp-create .intro { background:#ffffff; border:1px solid #e2e8f0; border-left:3px solid #500000;
  border-radius:8px; padding:10px 14px; margin-bottom:10px; }
.dc-lmp-create .intro p { margin:0; }
.dc-lmp-create .warn { background:#fffbeb; border:1px solid #fde68a; border-left:3px solid #d97706;
  border-radius:8px; padding:8px 14px; margin-top:8px; color:#78350f; }
.dc-lmp-create code { font-family:var(--drona-font-mono, monospace); font-size:var(--drona-text-caption, 12px);
  background:#f1f5f9; border:1px solid #e2e8f0; border-radius:4px; padding:0 4px; white-space:nowrap; }
</style>
"""


NO_FILE_WARNING = (
    "<div class='warn'><b>GPU options are not set up for this cluster yet</b>, so only CPU strategies "
    "are offered. Administrators: run <code>LAMMPS/configuration/generate_accelerators.py</code> "
    "(see <code>configuration/README</code>).</div>"
)


def card(body):
    return "<div class='dc-lmp-create'>" + CSS + body + "</div>"
