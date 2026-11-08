"""Shared helpers for the AlphaFold retrievers and make_af_bundle.py: where the results of a workflow are,
and how to shrink a large PAE / distance matrix for the viewers. Stdlib only (NumPy used when available),
Python 3.6 compatible.

Import from a retriever heredoc with:
    PYTHONPATH="$(dirname "$(readlink -f "$0")")${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PYEOF'
"""
import glob
import os

# AF2 result directory: <workflow>/parafold_output_dir/<FASTA name>/ (ParaFold and reduced_dbs). Shallow
# fallbacks for other layouts; no recursive glob (the workflow directory also holds msas/ and large trees).
_AF2_PATTERNS = ("parafold_output_dir/*/ranking_debug.json", "*/ranking_debug.json", "*/*/ranking_debug.json")


def _newest_first(paths):
    return sorted(set(paths), key=lambda p: os.path.getmtime(p), reverse=True)


def af2_result_dirs(location):
    """AF2 result directories (with ranking_debug.json), newest first."""
    for pattern in _AF2_PATTERNS:
        found = glob.glob(os.path.join(location, pattern))
        if found:
            return [os.path.dirname(f) for f in _newest_first(found)]
    return []


def af2_result_dir(location):
    """Newest AF2 result directory, or None."""
    dirs = af2_result_dirs(location)
    return dirs[0] if dirs else None


def af3_output_base(location):
    """AF3 output directory: the path the CPU job wrote to af3_output_dir.txt, else the newest
    <workflow>/output_* directory, or None."""
    try:
        with open(os.path.join(location, "af3_output_dir.txt")) as f:
            base = f.read().strip()
        if base and os.path.isdir(base):
            return base
    except OSError:
        pass
    dirs = [d for d in glob.glob(os.path.join(location, "output_*")) if os.path.isdir(d)]
    return _newest_first(dirs)[0] if dirs else None


def af3_prediction_dirs(base):
    """AF3 prediction directories in base (<name>_<timestamp>/ with ranking_scores.csv), newest first.
    A re-run of the GPU stage adds a new one."""
    if not base:
        return []
    return [os.path.dirname(f) for f in _newest_first(glob.glob(os.path.join(base, "*", "ranking_scores.csv")))]


def af3_prediction_dir(location):
    """Newest AF3 prediction directory of the workflow, or None."""
    dirs = af3_prediction_dirs(af3_output_base(location))
    return dirs[0] if dirs else None


# Viewer matrix size limit: above this many rows the matrix is shrunk by block maximum (for PAE the worst
# error in the block stays visible). The heatmap panel is ~450 px wide, so 500 x 500 cells (~1.3 MB of JSON)
# already shows more than one cell per pixel.
MAX_MATRIX = 500


def shrink_matrix(matrix, max_size=MAX_MATRIX, decimals=1):
    """(rows as rounded lists, axis, step). matrix: N x N list of lists or NumPy array.
    axis = 1-based residue number at the start of each block (None when not shrunk); step = block size."""
    n = len(matrix)
    step = max(1, -(-n // max_size))
    if step == 1:
        return [[round(float(v), decimals) for v in row] for row in matrix], None, 1
    starts = list(range(0, n, step))
    try:
        import numpy as np
        a = np.asarray(matrix, dtype=float)
        a = np.maximum.reduceat(np.maximum.reduceat(a, starts, axis=0), starts, axis=1)
        rows = [[round(float(v), decimals) for v in row] for row in a]
    except ImportError:
        rows = []
        for r0 in starts:
            block_rows = matrix[r0:r0 + step]
            rows.append([round(max(max(row[c0:c0 + step]) for row in block_rows), decimals) for c0 in starts])
    return rows, [s + 1 for s in starts], step
