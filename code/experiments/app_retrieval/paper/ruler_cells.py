"""The RULER generation cells the paper's figures read, and the style every one of them
draws them in.

One cell is one file out/gen/<task>-<hay>_<model>_T<ctx>.jsonl: a `_meta` first line and
then one record per prompt, whose "match" is RULER's string_match_all for that prompt and
whose "depth" is where in the haystack the needle sat, as a fraction of the prompt.

    from experiments.app_retrieval.paper.ruler_cells import GEN, MODELS, scan, depth_curve

`scan` returns the per-cell accuracy from `_meta["match"]`, the mean over the records,
without reading them. Only the files directly under out/gen are read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AR = HERE.parent                                              # app_retrieval/
GEN = AR / "out" / "gen"

# run name -> (architecture, size label, LEMA head size)
MODELS = {
    "rope1024_16k": ("rope", "309M", 0),
    "gdn1024_16k": ("gdn", "326M", 0),
    "lema1024_h8": ("lema", "309M", 8),
    "lema1024_h16": ("lema", "309M", 16),
    "lema1024_h32": ("lema", "309M", 32),
    "lema1024_h64": ("lema", "309M", 64),
    "lema1536_h64": ("lema", "834M", 64),
    "rope1536_16k": ("rope", "834M", 0),
    "gdn1536_16k": ("gdn", "892M", 0),
}
MARKER = {"rope": "s", "gdn": "^", "lema": "o"}
SOFTMAX_MAX_CTX = 16384     # the softmax transformers are evaluated up to their extended context


def style(run, color_of, head_color):
    """The line style of one run: its architecture's colour (LEMA shaded by head size)
    and its marker."""
    arch, size, head = MODELS[run]
    col = head_color[head] if arch == "lema" else color_of[arch]
    return dict(color=col, marker=MARKER[arch], ls="-")


def label(run, name_of, *, size=True):
    arch, sz, head = MODELS[run]
    s = name_of[arch] + (f" $d_h{{=}}{head}$" if arch == "lema" else "")
    if size:
        s += f", {sz}"
    return s


def path_of(cell, run, ctx, root=GEN):
    return Path(root) / f"{cell}_{run}_T{ctx}.jsonl"


def meta_of(f):
    with open(f) as fh:
        head = json.loads(fh.readline())
    return head.get("_meta")


def records(f):
    """Every prompt of a cell as (depth, match), depth a fraction of the prompt."""
    with open(f) as fh:
        fh.readline()
        out = []
        for line in fh:
            r = json.loads(line)
            if "match" in r:
                out.append((float(r["depth"]), float(r["match"])))
    return np.array(out) if out else np.zeros((0, 2))


def scan(cell, runs, ctxs, root=GEN):
    """{run: {ctx: (accuracy, n)}} for the cells that exist, and the list of the ones that
    do not, as "<run>@<ctx>"."""
    got, missing = {}, []
    for run in runs:
        for ctx in ctxs:
            if MODELS[run][0] == "rope" and ctx > SOFTMAX_MAX_CTX:
                continue
            f = path_of(cell, run, ctx, root)
            if not f.exists():
                missing.append(f"{run}@{ctx}")
                continue
            m = meta_of(f)
            acc, n = float(m["match"]), int(m["n"])
            got.setdefault(run, {})[ctx] = (acc, n)
    return got, missing


def depth_curve(cell, run, ctx, bins=3, root=GEN):
    """Accuracy in `bins` equal depth bins at one context length, as (labels, accuracy,
    n), or None if that cell is missing."""
    f = path_of(cell, run, ctx, root)
    if not f.exists():
        return None
    rec = records(f)
    edges = np.linspace(0, 1, bins + 1)
    lab, acc, n = [], [], []
    for b in range(bins):
        sel = (rec[:, 0] >= edges[b]) & (rec[:, 0] <= edges[b + 1] if b == bins - 1
                                         else rec[:, 0] < edges[b + 1])
        lab.append(f"{edges[b]:.2f}–{edges[b + 1]:.2f}")
        acc.append(float(rec[sel, 1].mean()) if sel.any() else np.nan)
        n.append(int(sel.sum()))
    return lab, np.array(acc), n
