#!/usr/bin/env python3
"""Accelerator availability card on the LAMMPS create page: one table, a row group per accelerator type
with the number of free accelerators per node and how many nodes have that many.

Reads gpuavail JSON (gpuavail_to_json.py) on stdin: "config" lists every GPU node as
(count x TYPE:gpus), "avail(merged)" the nodes with free GPUs as (count x TYPE:free gpus).
Partly used nodes ("asterisk") are not counted, as before. Types and labels come from
accelerators/<cluster>.json, in its order, so the rows match the hardware filter; without that
file only a short note is shown (the page is CPU-only then). Types with nothing free get no rows. FILTER_ACCEL (the checked hardware filter) dims the other rows.
"""
import json
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accel_info import card, e, load_accelerators, parse_list  # noqa: E402

SKIP = {"ve"}   # vector engine: not a GPU for LAMMPS


def parse_rows(rows):
    """[(type, gpus per node, node count)] from gpuavail rows, skipping partial nodes."""
    out = []
    for item in rows or []:
        if item.get("asterisk"):
            continue
        ident = str(item.get("identifier", ""))
        if ":" not in ident:
            continue
        name, gpus = ident.split(":", 1)
        try:
            out.append((name.strip().lower(), int(gpus), int(item.get("count", 0))))
        except ValueError:
            continue
    return out


def plural(n, word):
    return "%d %s%s" % (n, word, "" if n == 1 else "s")


def render(data, accelerators, checked):
    totals = {}
    for name, _gpus, count in parse_rows(data.get("config")):
        totals[name] = totals.get(name, 0) + count
    free = OrderedDict()
    for name, gpus, count in parse_rows(data.get("avail(merged)")):
        free.setdefault(name, {})
        free[name][gpus] = free[name].get(gpus, 0) + count

    if accelerators:
        types = [(a["value"].lower(), a.get("label", a["value"])) for a in accelerators
                 if a["value"].lower() != "cpu"]
    else:
        names = list(totals) + [n for n in free if n not in totals]
        types = [(n, n.upper() + " GPU") for n in names if n not in SKIP]

    rows = []
    for name, label in types:
        by_gpus = free.get(name, {})
        nodes = sum(by_gpus.values())
        if not nodes:
            continue
        off = bool(checked) and name not in checked
        total = totals.get(name)
        sub = "%d of %d nodes free" % (nodes, total) if total else plural(nodes, "node") + " free"
        counts = sorted(by_gpus.items(), key=lambda kv: -kv[0])
        for i, (per_node, n) in enumerate(counts):
            first = ("<td rowspan='%d' class='type'>%s<div class='muted'>%s</div></td>"
                     % (len(counts), e(label), e(sub))) if i == 0 else ""
            cls = [c for c in ("off" if off else "", "start" if i == 0 and rows else "",
                                          "inner" if i else "") if c]
            attr = " class='%s'" % " ".join(cls) if cls else ""
            rows.append("<tr%s>%s<td class='num'><b>%d</b></td><td class='num'>%d</td></tr>"
                        % (attr, first, per_node, n))

    if not types:
        return card("<div class='muted'>No accelerator nodes on this cluster.</div>")
    if rows:
        body = ("<table class='avail'><tr><th>Accelerator</th><th class='num'>Free per node</th>"
                "<th class='num'>Nodes</th></tr>%s</table>" % "".join(rows))
    else:
        body = "<div>No accelerator nodes are free right now.</div>"
    notes = []
    notes.append("Partly used nodes are not counted.")
    if checked and rows:
        notes.append("Dimmed: not checked in the hardware filter.")
    return card("%s<div class='muted foot'>%s</div>" % (body, " ".join(notes)))


def main():
    try:
        data = json.loads(sys.stdin.read())
    except ValueError as err:
        return card("<div class='muted'>Accelerator availability is not available right now (%s).</div>" % e(err))
    accelerators = load_accelerators()
    if not accelerators:
        return card("<div class='muted'>GPU availability is shown once the accelerator list for this "
                    "cluster is set up (see the note at the top of the page).</div>")
    checked = set(parse_list(os.environ.get("FILTER_ACCEL", ""))) - {"cpu"}
    return render(data, accelerators, checked)


if __name__ == "__main__":
    print(main())
