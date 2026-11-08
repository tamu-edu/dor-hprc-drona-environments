"""Split a generated single-file training script into data.py and model.py.

The generators still build one script (imports, settings, dataset classes, LitDataModule,
model classes, build_loggers, main). This module takes that script apart:

  data.py   imports + settings it uses + everything up to and including LitDataModule
            + build_datamodule() (the datamodule lines from main)
  model.py  imports + settings it uses + the classes after LitDataModule
            + build_model(datamodule) (the model lines from main)

train.py (files/train.py) and .drona/pl_support.py are static and replace the rest
(Lightning import shim, config loader, loggers, Trainer, test, .pt copy, ONNX export).

Works on Python 3.6+ (no ast end_lineno): a statement's text runs up to the next statement,
with comment lines directly above a statement kept with that statement.
"""
import ast
import re

_LOADER_NAMES = {"_parse_scalar", "load_config", "_cli", "_args", "CFG"}
_SUPPORT_NAMES = ["CFG", "L", "TensorBoardLogger", "ModelCheckpoint"]  # from pl_support
_CONST = re.compile(r"^[A-Z][A-Z0-9_]*$")
_STOP_IN_MAIN = {"loggers", "callbacks", "trainer"}
_NOISE = ("# Injected dynamically based on your dataset choice.",)


class SplitError(ValueError):
    pass


def _is_comment_or_blank(line):
    s = line.strip()
    return not s or s.startswith("#")


def _segments(lines, stmts, indent, limit):
    """[(stmt, text)] for consecutive statements at one indent. limit: 1-based line after the last."""
    starts = []
    for i, n in enumerate(stmts):
        s = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
        floor = starts[-1][0] + 1 if starts else 1
        while s - 1 >= floor and (not lines[s - 2].strip() or (
                lines[s - 2].startswith(indent + "#") and not lines[s - 2].startswith(indent + " "))):
            s -= 1
        while s < n.lineno and not lines[s - 1].strip():
            s += 1  # no leading blank lines
        starts.append((s, n))
    out = []
    for i, (s, n) in enumerate(starts):
        e = starts[i + 1][0] if i + 1 < len(starts) else limit
        seg = lines[s - 1:e - 1]
        while seg and _is_comment_or_blank(seg[-1]) and not seg[-1].startswith(indent + " "):
            seg.pop()  # trailing comments belong to the next statement
        out.append((n, seg))
    return out


def _names_used(text):
    tree = ast.parse(text)
    used = set()
    for x in ast.walk(tree):
        if isinstance(x, ast.Name):
            used.add(x.id)
    return used


def _bound_names(node):
    """Names an import (or try/except import block) binds."""
    names = set()
    for x in ast.walk(node):
        if isinstance(x, (ast.Import, ast.ImportFrom)):
            for a in x.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(x, ast.Assign):
            names.update(t.id for t in x.targets if isinstance(t, ast.Name))
    return names


def _is_import_block(node):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return True
    if isinstance(node, ast.Try):
        stmts = list(node.body) + [s for h in node.handlers for s in h.body]
        return all(isinstance(s, (ast.Import, ast.ImportFrom, ast.Assign, ast.Pass)) for s in stmts)
    return False


def _imports_lightning(node):
    return any(isinstance(x, (ast.Import, ast.ImportFrom)) and
               any(m in ast.dump(x) for m in ("pytorch_lightning", "'lightning")) for x in ast.walk(node))


def _import_text(node, seg, used):
    """Import statement text with only the names that are used (None if none is)."""
    if isinstance(node, ast.ImportFrom):
        keep = [a for a in node.names if (a.asname or a.name) in used]
        if not keep:
            return None
        if len(keep) == len(node.names):
            return "\n".join(seg)
        parts = [a.name + (" as " + a.asname if a.asname else "") for a in keep]
        return "from {}{} import {}".format("." * node.level, node.module or "", ", ".join(parts))
    if isinstance(node, ast.Import):
        keep = [a for a in node.names if (a.asname or a.name).split(".")[0] in used]
        if not keep:
            return None
        if len(keep) == len(node.names):
            return "\n".join(seg)
        return "import " + ", ".join(a.name + (" as " + a.asname if a.asname else "") for a in keep)
    return "\n".join(seg) if _bound_names(node) & used else None


def _assigned(node):
    if isinstance(node, ast.Assign):
        out = set()
        for t in node.targets:
            for x in ast.walk(t):
                if isinstance(x, ast.Name):
                    out.add(x.id)
        return out
    return set()


def _task_title(tree):
    doc = ast.get_docstring(tree) or ""
    first = doc.strip().split("\n")[0] if doc.strip() else ""
    m = re.search(r"[—-]\s*(.+?)\.?$", first)
    return m.group(1).strip() if m else "PyTorch Lightning"


def split_script(src):
    """Return (data_py, model_py). Raises SplitError if the script has an unexpected shape."""
    lines = src.split("\n")
    tree = ast.parse(src)
    title = _task_title(tree)
    body = list(tree.body)
    if body and ast.get_docstring(tree) is not None:
        body = body[1:]  # the old module docstring; data.py and model.py get their own

    imports, consts, data_defs, model_defs = [], [], [], []
    main = None
    seen_datamodule = False
    for node, seg in _segments(lines, body, "", len(lines) + 1):
        names = _assigned(node)
        if _is_import_block(node):
            if not _imports_lightning(node):
                imports.append((node, seg))
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name in _LOADER_NAMES or node.name == "build_loggers":
                continue
            if node.name == "main":
                main = node
            elif not seen_datamodule:
                data_defs.append((node, seg))
                seen_datamodule = node.name == "LitDataModule"
            else:
                model_defs.append((node, seg))
        elif names and names <= _LOADER_NAMES | {"_"}:
            continue
        elif isinstance(node, ast.Expr) and "_cli" in _names_used("\n".join(s.strip() for s in seg) or "0"):
            continue
        elif isinstance(node, ast.Assign) and len(names) == 1 and _CONST.match(next(iter(names))):
            code = [l for l in seg if not _is_comment_or_blank(l) or l.startswith(" ")]
            consts.append((next(iter(names)), code))
        elif isinstance(node, ast.If) and "__name__" in _names_used("\n".join(seg)):
            continue
        else:
            raise SplitError("unexpected top-level statement at line {}".format(node.lineno))
    if main is None or not seen_datamodule or not model_defs:
        raise SplitError("script has no main(), LitDataModule or model class")

    # main(): the datamodule lines go to build_datamodule(), the model lines to build_model()
    next_top = [n.lineno for n in tree.body if n.lineno > main.lineno]
    main_segs = _segments(lines, main.body, "    ", next_top[0] if next_top else len(lines) + 1)
    data_lines, model_lines, phase = [], [], None
    for node, seg in main_segs:
        names = _assigned(node)
        if phase is None:
            if "datamodule" in names:
                phase = "data"
            else:
                continue
        elif phase == "data" and not (isinstance(node, ast.Expr) and
                                      "datamodule" in _names_used("\n".join(s.strip() for s in seg))):
            phase = "model"
        if names & _STOP_IN_MAIN or "build_loggers" in _names_used("\n".join(s.strip() for s in seg)):
            break
        (data_lines if phase == "data" else model_lines).extend(seg)
    if not data_lines or not any("model" in l and "=" in l for l in model_lines):
        raise SplitError("could not find the datamodule and model lines in main()")
    data_lines = [l for l in data_lines if not l.strip().startswith("# Initialize data")]

    data_fn = ["def build_datamodule():",
               '    """Called by train.py: create the LightningDataModule (and prepare the data)."""']
    data_fn += data_lines + ["    return datamodule"]
    model_fn = ["def build_model(datamodule):",
                '    """Called by train.py: create the LightningModule; the datamodule gives data-dependent sizes."""']
    model_fn += model_lines + ["    return model"]

    data_code = "\n\n\n".join(["\n".join(s) for _, s in data_defs] + ["\n".join(data_fn)])
    model_code = "\n\n\n".join(["\n".join(s) for _, s in model_defs] + ["\n".join(model_fn)])

    data_names = {n.name for n, _ in data_defs}
    model_from_data = sorted(_names_used(model_code) & data_names)

    header_data = ('"""Data for this training run: {}.\n\n'
                   "build_datamodule() is called by train.py and returns a LightningDataModule.\n"
                   "Change the dataset, transforms or train/validation split here. Settings such as the\n"
                   'data path and batch_size are in config.yaml (section data:)."""').format(title)
    header_model = ('"""Model for this training run: {}.\n\n'
                    "build_model(datamodule) is called by train.py and returns a LightningModule.\n"
                    "Change the network, loss, metrics or optimizer here. Settings such as the\n"
                    'learning rate are in config.yaml (sections model: and training:)."""').format(title)

    data_py = _assemble(header_data, data_code, imports, consts, [])
    model_py = _assemble(header_model, model_code, imports, consts,
                         ["from data import " + ", ".join(model_from_data)] if model_from_data else [])
    for text in (data_py, model_py):
        ast.parse(text)  # must be valid Python
    return data_py, model_py


def _assemble(header, code, imports, consts, extra_imports):
    code = "\n".join(l for l in code.split("\n") if l.strip() not in _NOISE)
    # settings used, plus the settings those depend on (e.g. FLAT_SIZE = IN_CHANNELS * ...)
    const_map = dict((name, text) for name, text in consts)
    used = _names_used(code)
    wanted, todo = set(), [n for n in const_map if n in used]
    while todo:
        n = todo.pop()
        if n in wanted:
            continue
        wanted.add(n)
        todo.extend(d for d in _names_used("\n".join(l.strip() for l in const_map[n])) if d in const_map)
    const_lines = []
    for name, text in consts:
        if name in wanted:
            const_lines.extend(text)

    used_all = used | _names_used("\n".join(const_lines) or "0")
    import_lines = []
    for node, seg in imports:
        text = _import_text(node, seg, used_all)
        if text and text not in import_lines:
            import_lines.append(text)
    support = [n for n in _SUPPORT_NAMES if n in used_all]
    if support:
        import_lines.append("from pl_support import " + ", ".join(support))
    import_lines.extend(extra_imports)

    parts = [header, "\n".join(import_lines)]
    if const_lines:
        parts.append("# ── Settings from config.yaml ─────────────────────────────────────────────────\n"
                     + "\n".join(const_lines))
    parts.append(code)
    return "\n\n".join(p for p in parts if p.strip()) + "\n"
