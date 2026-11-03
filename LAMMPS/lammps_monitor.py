import re
import yaml
import sys
import os
import argparse
import json
import math
import bisect

try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader

def generate_svg_plot(xdata, ydata, xlabel, ylabel):
    if not xdata or not ydata or len(xdata) != len(ydata):
        return "<p>Invalid data for plot</p>"
    
    width = 400
    height = 300
    padding = 50
    
    xmin, xmax = min(xdata), max(xdata)
    
    # Ensure the y-axis always includes 0
    ymin = min(0, min(ydata))
    ymax = max(0, max(ydata))
    
    if xmax == xmin: xmax = xmin + 1
    if ymax == ymin: ymax = ymin + 1
    
    def scale_x(x):
        return padding + (x - xmin) / (xmax - xmin) * (width - 2 * padding)
    
    def scale_y(y):
        return height - (padding + (y - ymin) / (ymax - ymin) * (height - 2 * padding))
    
    points = []
    markers = []
    for x, y in zip(xdata, ydata):
        sx = scale_x(x)
        sy = scale_y(y)
        points.append(f"{sx},{sy}")
        # Add a circular marker for each data point
        markers.append(f'<circle cx="{sx}" cy="{sy}" r="4" fill="#003c71" />')
    
    polyline = f'<polyline fill="none" stroke="#003c71" stroke-width="2" points="{" ".join(points)}" />'
    markers_svg = "\n        ".join(markers)
    
    # Find exactly where Y=0 sits on the canvas
    zero_y = scale_y(0)
    
    # Place labels at the top or bottom of the graph boundary, never floating in the middle
    if zero_y < height / 2:
        # 0 line is in the top half; place labels above the plot area
        x_tick_y = padding - 10
        x_title_y = padding - 25
    else:
        # 0 line is in the bottom half (or middle); place labels below the plot area
        x_tick_y = height - padding + 20
        x_title_y = height - padding + 35
    
    svg = f"""
    <svg width="{width}" height="{height}" style="border: 1px solid #ccc; margin: 5px;">
        <text x="{width/2}" y="{x_title_y}" text-anchor="middle">{xlabel}</text>
        <text x="15" y="{height/2}" text-anchor="middle" transform="rotate(-90 15,{height/2})">{ylabel}</text>
        {polyline}
        {markers_svg}
        <!-- Axes -->
        <line x1="{padding}" y1="{zero_y}" x2="{width-padding}" y2="{zero_y}" stroke="black" />
        <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="black" />
        <!-- Labels -->
        <text x="{padding}" y="{x_tick_y}" font-size="10">{int(xmin)}</text>
        <text x="{width-padding}" y="{x_tick_y}" font-size="10" text-anchor="end">{int(xmax)}</text>
        <text x="{padding-5}" y="{height-padding}" font-size="10" text-anchor="end">{ymin:.2g}</text>
        <text x="{padding-5}" y="{padding}" font-size="10" text-anchor="end" dominant-baseline="hanging">{ymax:.2g}</text>
    </svg>
    """
    return svg

def read_yaml_blocks(filepath, stop_after_keywords=False):
    """
    Reads the LAMMPS output file and extracts YAML blocks.
    If stop_after_keywords is True, aborts reading immediately after finding the header.
    """
    docs = ""
    with open(filepath, "r") as f:
        for line in f:
            m = re.search(r"^(keywords:.*$|data:$|---$|\.\.\.$|  - \[.*\]$)", line)
            if m:
                matched_line = m.group(0)
                docs += matched_line + '\n'
                
                # CPU Optimization: Abort early if we just need the options/keywords
                if stop_after_keywords and matched_line.startswith("keywords:"):
                    break

    # Handle incomplete last block
    stripped_docs = docs.strip()
    if stripped_docs and not stripped_docs.endswith("..."):
        docs += "...\n"

    return list(yaml.load_all(docs, Loader=Loader))


# ---------------------------------------------------------------------------
# Thermo parser used by --chart-data and --get-opts.
#
# LAMMPS writes every YAML thermo row from one place (Thermo::compute() in
# src/thermo.cpp) as "  - [" + one printf-formatted number per keyword + ", " each
# + "]", on rank 0 only. So rows are split on "," instead of being parsed as YAML,
# which is ~30x faster and keeps memory bounded. Only the "keywords:" header goes
# through YAML (with a fallback, since LAMMPS does not escape quotes in colnames).
#
# Rows are validated, because user input can break the structure (verified with
# test runs, see lammps-monitoring-costs.md in the composer repo):
#   - value count must equal the keyword count: a `thermo_modify format` with a
#     comma, or `print`/`fix print` output that looks like a row, is rejected
#     instead of shifting columns or adding fake points;
#   - the x value (first column) must be numeric and must not go backwards
#     within a run;
#   - values that are not finite numbers ("3 K", nan, inf) are left out of that row.
# ---------------------------------------------------------------------------

_QUOTED_KW_RE = re.compile(r"'(.*?)'(?=\s*,|\s*$)")


def _parse_keywords(line):
    """Keyword names from a "keywords: ['Step', 'Temp', ]" header line."""
    try:
        kws = yaml.load(line, Loader=Loader)["keywords"]
        if isinstance(kws, list):
            return [str(k) for k in kws]
    except Exception:
        pass
    # e.g. colname "it's PE" is written unescaped as 'it's PE' (invalid YAML)
    body = line[line.find("[") + 1:line.rfind("]")]
    return _QUOTED_KW_RE.findall(body)


def _to_number(tok):
    t = tok.strip()
    if t.lstrip("+-").isdigit():
        return int(t)
    try:
        v = float(t)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def iter_thermo(filepath, stats, stop_after_keywords=False):
    """
    Streams a LAMMPS output file. Yields ("keywords", names) at the start of each
    YAML thermo block and ("row", x, tokens) for each valid data row, where tokens
    are the raw value strings (one per keyword) and x is the parsed first value.
    Values are only converted when a row is actually used (see build_chart_rows),
    which keeps large files fast. Invalid rows are counted in stats["rejected"].
    """
    nkw = None
    with open(filepath, "r", errors="replace") as f:
        for line in f:
            if line.startswith("  - ["):
                if not nkw:
                    continue                   # row-like text outside a thermo block
                body = line.rstrip("\n")
                if not body.endswith("]"):     # e.g. a partially written last line
                    stats["rejected"] += 1
                    continue
                toks = body[5:-1].split(",")
                if toks and not toks[-1].strip():
                    toks.pop()                 # LAMMPS ends every row with ", ]"
                if len(toks) != nkw:
                    stats["rejected"] += 1
                    continue
                x = _to_number(toks[0])
                if x is None:
                    stats["rejected"] += 1
                    continue
                yield ("row", x, toks)
            elif line.startswith("keywords:"):
                kws = _parse_keywords(line.rstrip("\n"))
                nkw = len(kws) if len(kws) >= 2 else None
                if nkw:
                    yield ("keywords", kws, None)
                    if stop_after_keywords:
                        return
            elif line.startswith("..."):
                nkw = None


class _Decimator:
    """
    Bounded-memory, evenly strided sample of a stream of rows: take() says whether
    the next row is kept (every `stride`-th row); when 4*max_points rows are held,
    every other one is dropped and the stride doubles. rows() then picks the rows
    nearest to max_points evenly spaced x values; the first and the most recent row
    are always included. max_points <= 0 keeps every row.
    """

    def __init__(self, max_points):
        self.max = max_points if max_points and max_points > 0 else 0
        self.reset()

    def reset(self):
        self.kept, self.stride, self.count = [], 1, 0

    def take(self):
        keep = (not self.max) or self.count % self.stride == 0
        self.count += 1
        return keep

    def keep(self, row):
        self.kept.append(row)
        if self.max and len(self.kept) >= 4 * self.max:
            self.kept = self.kept[::2]
            self.stride *= 2

    def rows(self, xkey, last):
        rows = list(self.kept)
        if last is not None and (not rows or rows[-1] is not last):
            rows.append(last)
        n = len(rows)
        if not self.max or n <= self.max:
            return rows
        if self.max == 1:
            return [rows[-1]]
        xs = [r[xkey] for r in rows]          # increasing (enforced by the caller)
        x0, x1 = xs[0], xs[-1]
        picked = []
        for i in range(self.max):
            t = x0 + (x1 - x0) * i / (self.max - 1)
            j = bisect.bisect_left(xs, t)
            if j > 0 and (j == n or t - xs[j - 1] <= xs[j] - t):
                j -= 1
            if not picked or picked[-1] != j:
                picked.append(j)
        return [rows[j] for j in picked]


def build_chart_rows(filepath, filter_list=None, max_points=400, keep_constant=False, stats=None):
    """
    Row objects for the chart element: [{"Step": 0, "Temp": 3.0, ...}, ...]

    - Blocks from several `run` commands are concatenated. LAMMPS repeats the last
      step of one run as the first step of the next; such rows are merged.
    - If a new run starts at a lower step (reset_timestep), only rows from that run
      on are returned, so the x axis stays increasing (counted in stats["restarts"]).
      Within a run, a row whose step goes backwards is rejected.
    - The x key is the first keyword of the first block (normally Step).
    - filter_list: only these keywords are emitted. Without a filter, keywords whose
      printed value never changes (e.g. E_bond=0 in an LJ melt) are dropped unless
      keep_constant.
    - max_points > 0: at most max_points rows, evenly spaced, first and last kept.
    """
    if stats is None:
        stats = {}
    stats.setdefault("rejected", 0)
    stats.setdefault("restarts", 0)
    track_constant = not filter_list and not keep_constant
    dec = _Decimator(max_points)
    xkey = None
    cols = []            # (index, keyword) emitted for the current block
    cand = []            # (index, keyword) not yet seen to vary
    new_block = False
    first_seen, varying, nrows = {}, set(), 0
    last_x, last_obj, last_toks = None, None, None   # most recent row; built lazily

    def make(x, toks):
        obj = {xkey: x}
        for j, kw in cols:
            v = _to_number(toks[j])
            if v is not None:
                obj[kw] = v
        return obj

    for kind, x, toks in iter_thermo(filepath, stats):
        if kind == "keywords":
            keywords = x
            if xkey is None:
                xkey = keywords[0]
            cols = [(j, kw) for j, kw in enumerate(keywords) if j > 0 and kw != xkey
                    and (not filter_list or kw in filter_list)]
            cand = [(j, kw) for j, kw in cols if kw not in varying] if track_constant else []
            new_block = True
            continue

        if last_x is not None and x < last_x:
            if new_block:
                dec.reset()
                first_seen, varying, nrows = {}, set(), 0
                cand = list(cols) if track_constant else []
                last_x, last_obj, last_toks = None, None, None
                stats["restarts"] += 1
            else:
                stats["rejected"] += 1
                continue
        new_block = False

        if cand:
            changed = False
            for j, kw in cand:
                tok = toks[j].strip()
                if first_seen.setdefault(kw, tok) != tok:
                    varying.add(kw)
                    changed = True
            if changed:
                cand = [c for c in cand if c[1] not in varying]

        if last_x is not None and x == last_x:
            # same step repeated at a run boundary: merge into the previous row
            if last_obj is None:
                last_obj = make(last_x, last_toks)
            last_obj.update(make(x, toks))
            continue

        nrows += 1
        last_x = x
        if dec.take():
            last_obj, last_toks = make(x, toks), None
            dec.keep(last_obj)
        else:
            last_obj, last_toks = None, toks

    if last_x is not None and last_obj is None:
        last_obj = make(last_x, last_toks)
    rows = dec.rows(xkey, last_obj)
    if track_constant and nrows > 1:
        constant = set(first_seen) - varying
        if constant:
            rows = [{k: v for k, v in obj.items() if k not in constant} for obj in rows]
    return rows


def main():
    parser = argparse.ArgumentParser(description="LAMMPS output monitor and plot generator")
    parser.add_argument("jobdir", help="Directory containing the job output")
    parser.add_argument("jobid", help="The job ID")
    
    # Optional filter argument
    parser.add_argument("--filter", type=str, help="JSON encoded list of strings to filter y-labels by (e.g., '[\"temp\", \"press\"]')")
    
    # Require exactly one of these action flags
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate-plots", action="store_true", help="Output HTML/SVG plot grid")
    group.add_argument("--get-opts", action="store_true", help="Output plot options as a JSON array")
    group.add_argument("--chart-data", action="store_true", help="Output thermo rows as a JSON array for the chart element")

    parser.add_argument("--max-points", type=int, default=400,
                        help="--chart-data: downsample to at most N rows (0 = all rows)")
    parser.add_argument("--keep-constant", action="store_true",
                        help="--chart-data: keep keywords whose value never changes")

    args = parser.parse_args()
    json_out = args.get_opts or args.chart_data
    
    # Parse the filter list if provided
    filter_list = None
    if args.filter:
        try:
            # Replace single quotes with double quotes to ensure valid JSON parsing
            filter_str = args.filter.replace("'", '"')
            filter_list = json.loads(filter_str)
            if not isinstance(filter_list, list):
                raise ValueError("Parsed JSON is not a list")
        except Exception as e:
            if json_out:
                print("[]")
                print(f"Error parsing --filter JSON: {str(e)}", file=sys.stderr)
                return
            print(f"<p>Error parsing --filter JSON: {str(e)}</p>")
            sys.exit(1)

    filepath = os.path.join(args.jobdir, f"out.{args.jobid}")

    if not os.path.exists(filepath):
        if json_out:
            print("[]")
        else:
            print(f"<p>Output file not found: '{filepath}'</p>")
        return

    # Options for the "Show only" checkboxes: keywords of the first thermo block
    if args.get_opts:
        opts = []
        try:
            for _kind, keywords, _ in iter_thermo(filepath, {"rejected": 0}, stop_after_keywords=True):
                opts = [{"name": kw, "value": kw} for kw in keywords[1:]]
        except Exception as e:
            print(f"Error reading thermo keywords: {str(e)}", file=sys.stderr)
        print(json.dumps(opts))
        return

    # Rows for the chart element
    if args.chart_data:
        stats = {}
        try:
            rows = build_chart_rows(filepath, filter_list=filter_list,
                                    max_points=args.max_points,
                                    keep_constant=args.keep_constant, stats=stats)
        except Exception as e:
            print("[]")
            print(f"Error reading thermo output: {str(e)}", file=sys.stderr)
            return
        if stats.get("rejected") or stats.get("restarts"):
            print(f"thermo: {stats['rejected']} invalid rows skipped, "
                  f"{stats['restarts']} step resets", file=sys.stderr)
        print(json.dumps(rows, separators=(",", ":")))
        return

    # Legacy HTML/SVG route (old schemas): still uses the YAML reader
    try:
        blocks = read_yaml_blocks(filepath)
    except Exception as e:
        print(f"<p>Error parsing YAML: {str(e)}</p>")
        return

    # Handle the plot generation route
    if args.generate_plots:
        all_plots_html = '<div style="display: flex; flex-wrap: wrap;">'
        
        for block in blocks:
            if not block or 'keywords' not in block or 'data' not in block:
                continue
            
            keywords = block['keywords']
            data = block['data']
            
            if not keywords or not data:
                continue
                
            xlabel = keywords[0]
            xdata = [row[0] for row in data if len(row) > 0]
            
            for j in range(1, len(keywords)):
                ylabel = keywords[j]
                
                # Apply the filter if it exists
                if filter_list and ylabel not in filter_list:
                    continue
                
                ydata = [row[j] for row in data if len(row) > j]
                
                if len(xdata) == len(ydata):
                    all_plots_html += generate_svg_plot(xdata, ydata, xlabel, ylabel)
        
        all_plots_html += '</div>'
        
        if all_plots_html == '<div style="display: flex; flex-wrap: wrap;"></div>':
            if filter_list:
                print("<p>No data found that matches the selected options.</p>")
            else:
                print("<p>No data found.</p>")
        else:
            print(all_plots_html)

if __name__ == "__main__":
    main()