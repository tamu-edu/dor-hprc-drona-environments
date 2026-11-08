"""Move the settings block of a generated train.py into config.yaml.

The generators still produce a train.py with constants such as
    MAX_EPOCHS = 3            # Maximum number of training epochs
externalize_settings() turns every top-level constant with a literal value
(before the first class/def) into a config.yaml entry and replaces the line with
    MAX_EPOCHS = CFG["training"]["max_epochs"]            # Maximum number of training epochs
so the rest of the generated code is unchanged. A small loader is inserted
into train.py: python train.py [--config FILE] [--set training.max_epochs=20 ...]
"""
import ast
import json
import re

# Section of each setting in config.yaml; anything not listed goes to "model".
_SECTIONS = {
    "data": ["DATA_DIR", "DATA_PATH", "CSV_PATH", "TARGET_COLUMN", "BATCH_SIZE", "NUM_WORKERS",
             "SAMPLE_RATE", "DURATION_SEC", "SEQ_LEN", "PRED_LEN", "MAX_SEQ_LEN"],
    "training": ["LR", "MAX_EPOCHS", "ACCELERATOR", "PRECISION", "SEED"],
    "logging": ["LOG_DIR", "EXPERIMENT_NAME", "LOG_EVERY_N_STEPS"],
}
_SECTION_ORDER = ["data", "model", "training", "logging"]
_SECTION_TITLES = {
    "data": "Data: where it comes from and how it is loaded",
    "model": "Model: architecture and task",
    "training": "Training: optimizer, length, hardware, reproducibility",
    "logging": "Logging: TensorBoard logs and checkpoints",
}
# Config key names that differ from the lower-cased constant
_KEY_NAMES = {"LR": "learning_rate"}
# Stay as code: they come from Slurm or are computed from other settings
_KEEP_AS_CODE = {"DEVICES", "NUM_NODES", "NUM_SAMPLES"}
_DERIVED = {"FLAT_SIZE": "IN_CHANNELS * IMAGE_SIZE * IMAGE_SIZE"}

_ASSIGN = re.compile(r"^([A-Z][A-Z0-9_]*)(\s*=\s*)(.*)$")

LOADER = '''
# ── Settings: read from config.yaml ───────────────────────────────────────────
# Change settings in config.yaml, or for one run on the command line:
#   python train.py --set training.max_epochs=20 --set data.batch_size=64
import argparse
import ast as _ast


def _parse_scalar(text):
    text = text.strip()
    if text in ("", "null", "~"):
        return None
    if text in ("true", "false"):
        return text == "true"
    try:
        return _ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def load_config(path, overrides=()):
    """Read config.yaml (PyYAML if available, else the simple two-level layout Drona writes)."""
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except ImportError:
        cfg, section = {}, None
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip()
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                key, _, value = line.strip().partition(":")
                if not line.startswith((" ", "\\t")):
                    section = cfg.setdefault(key.strip(), {})
                elif section is not None:
                    section[key.strip()] = _parse_scalar(value)
    for item in overrides:
        key, _, value = item.partition("=")
        group, _, name = key.strip().partition(".")
        if not name:
            raise SystemExit("--set expects SECTION.KEY=VALUE, got: " + item)
        cfg.setdefault(group, {})[name] = _parse_scalar(value)
    return cfg


_cli = argparse.ArgumentParser(description="Train with the settings in config.yaml")
_cli.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
                  help="settings file (default: config.yaml next to train.py)")
_cli.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                  help="override one setting for this run; repeat as needed")
_args, _ = _cli.parse_known_args()
CFG = load_config(_args.config, _args.set)
'''


def _split_value_comment(rest):
    """'3   # Max epochs' -> ('3', '# Max epochs'); a '#' inside a string literal is kept."""
    for i, ch in enumerate(rest):
        if ch == "#":
            head = rest[:i].rstrip()
            try:
                ast.literal_eval(head)
                return head, rest[i:]
            except (ValueError, SyntaxError):
                continue
    return rest.strip(), ""


def _yaml_value(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        mantissa, e, exp = text.partition("e")
        if e and "." not in mantissa:  # YAML 1.1 (PyYAML) reads 1e-05 as a string
            text = mantissa + ".0e" + exp
        return text
    return json.dumps(str(value))


def externalize_settings(train_script, header=""):
    """Return (train_script, config_yaml), or (train_script, None) if there is nothing to move."""
    lines = train_script.split("\n")
    entries = {}  # section -> [(key, yaml value, comment)]
    first_setting = None
    for i, line in enumerate(lines):
        if line.startswith(("class ", "def ", "@")):
            break
        m = _ASSIGN.match(line)
        if not m:
            continue
        name, eq, rest = m.groups()
        value_text, comment = _split_value_comment(rest)
        if name in _DERIVED:
            lines[i] = name + eq + _DERIVED[name] + ("  " + comment if comment else "")
            continue
        if name in _KEEP_AS_CODE:
            continue
        try:
            value = ast.literal_eval(value_text)
        except (ValueError, SyntaxError):
            continue  # an expression: leave it in the code
        if not isinstance(value, (str, int, float, bool, type(None))):
            continue
        section = next((s for s, names in _SECTIONS.items() if name in names), "model")
        key = _KEY_NAMES.get(name, name.lower())
        entries.setdefault(section, []).append((key, _yaml_value(value), comment.lstrip("# ").strip()))
        new = f'{name}{eq}CFG["{section}"]["{key}"]'
        lines[i] = new + (" " * max(1, len(m.group(0)) - len(comment) - len(new)) + comment if comment else "")
        if first_setting is None:
            first_setting = i

    if first_setting is None:
        return train_script, None

    # Put the loader just before the settings (after the section comment above them).
    insert_at = first_setting
    while insert_at > 0 and lines[insert_at - 1].startswith("#"):
        insert_at -= 1
    lines[insert_at:insert_at] = LOADER.strip("\n").split("\n") + [""]

    out = [header.rstrip("\n")] if header else []
    for section in _SECTION_ORDER:
        if section not in entries:
            continue
        out.append("")
        out.append(f"# {_SECTION_TITLES[section]}")
        out.append(f"{section}:")
        for key, value, comment in entries[section]:
            if comment:
                out.append(f"  # {comment}")
            out.append(f"  {key}: {value}")
    return "\n".join(lines), "\n".join(out).lstrip("\n") + "\n"
