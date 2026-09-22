"""Validation cross-entropy of every run before and after context extension.

    python ce_table.py [--latex] [--first-tokens N]

Reads the per-token evaluations of ../main_lm_bigrams/out/pertoken/, whose mean over the
whole held-out split is the validation cross-entropy at that context length, so the numbers
at 2048 and at 16384 come from the same passes the recall analysis uses. Those arrays are
too large to keep, so a pass that has none falls back to its mean retained in
out/eval/ce_summary.json. A dash means the pass does not exist: softmax at base 1e4 is not
evaluated beyond its training context unless it is forced. No GPU.

--first-tokens N averages only the first N scored tokens instead of the whole split. Both
scripts cut the split into consecutive (ctx+1)-blocks from token 0 and score every block
position but the first, so the leading N = 2**26 scored tokens are exactly the 2**26 / ctx
windows of ../main_lm/eval_lm.py: `--first-tokens $((2**26))` puts this table on the
protocol of the main LM table, at any context length.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
PERTOKEN = HERE.parent / "main_lm_bigrams" / "out" / "pertoken"
RETAINED = HERE / "out" / "eval" / "ce_summary.json"

# (label, base run, extended run): the rows of the paper's table
ROWS = [("softmax", "rope1536", "rope1536_16k"),
        ("GDN", "gdn1536", "gdn1536_16k"),
        ("LEMA", "lema1536_h64", "lema1536_h64_16k")]
CTXS = (2048, 16384)


def ce(run: str, ctx: int, first: int | None = None):
    """Mean CE of a pass, over the whole split or over its first `first` scored tokens.

    The stored array is [windows, ctx] in window order, so flattening it puts the scored
    tokens in the order the split is read and a prefix of it is a prefix of the pass. A
    file too short for the requested prefix is an error: silently averaging fewer tokens
    would produce a number that is not on the protocol it claims.
    """
    f = PERTOKEN / f"{run}_T{ctx}.npz"
    if not f.exists():
        return retained(run, ctx, first)
    a = np.asarray(np.load(f)["ce"]).reshape(-1)
    if first is not None:
        if a.size < first:
            raise SystemExit(f"{f.name} scores {a.size} tokens, fewer than the "
                             f"{first} asked for")
        a = a[:first]
    return float(a.mean(dtype=np.float64))


def retained(run: str, ctx: int, first: int | None):
    """The mean of a pass whose per-token array is not in this tree, from out/eval/ce_summary.json.

    The file keeps, per pass, one mean in float64 over the token count the tables average, so
    it answers that prefix and nothing else: any other prefix is an error rather than a number
    taken over the wrong tokens. A pass the file does not know is a pass that does not exist.
    """
    if not RETAINED.exists():
        raise SystemExit(f"neither {PERTOKEN}/{run}_T{ctx}.npz nor {RETAINED}")
    e = json.loads(RETAINED.read_text()).get(f"{run}_T{ctx}")
    if e is None:
        return None
    if first != e["tokens"]:
        raise SystemExit(f"{RETAINED.name} holds {run}_T{ctx} over {e['tokens']} scored "
                         f"tokens, not the "
                         f"{'whole split' if first is None else first} asked for")
    return float(e["ce"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latex", action="store_true")
    p.add_argument("--first-tokens", type=int, default=None, metavar="N",
                   help="average the first N scored tokens only (default: whole split)")
    a = p.parse_args()
    rows = []
    for label, base, ext in ROWS:
        rows.append((label, [ce(base, c, a.first_tokens) for c in CTXS],
                     [ce(ext, c, a.first_tokens) for c in CTXS]))
    fmt = lambda v: "-" if v is None else f"{v:.3f}"
    if not a.latex:
        print(f"first {a.first_tokens} scored tokens" if a.first_tokens
              else "whole validation split")
        print(f"{'':<28} {'as trained':>18}   {'after extension':>18}")
        print(f"{'':<28} {'2048':>8} {'16384':>9}   {'2048':>8} {'16384':>9}")
        for label, b, e in rows:
            print(f"{label:<28} {fmt(b[0]):>8} {fmt(b[1]):>9}   {fmt(e[0]):>8} {fmt(e[1]):>9}")
        return
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r" & \multicolumn{2}{c}{as trained} & \multicolumn{2}{c}{after extension} \\")
    print(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    print(r"model & $2048$ & $16384$ & $2048$ & $16384$ \\")
    print(r"\midrule")
    for label, b, e in rows:
        print(f"{label} & {fmt(b[0])} & {fmt(b[1])} & {fmt(e[0])} & {fmt(e[1])} " + r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
