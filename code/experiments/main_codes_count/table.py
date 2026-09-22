"""The state-growth tables: mean entries per head after t tokens, from out/*.json.

    python table.py --latex          rows for the paper's two tables
    python table.py                  the same, readable

Table 1: the d=1024 models at every head size (the models of the main figure).
Table 2: d_h=64 at every model size. Context lengths 2k ... 256k; a json whose grid lacks
a length is reported at its nearest grid point (marked with ~ in the readable output).
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LENGTHS = [2 ** k for k in range(11, 19)]                    # 2048 ... 262144


def load(run):
    f = HERE / "out" / f"{run}.json"
    return json.loads(f.read_text()) if f.exists() else None


def at(d, t):
    """Mean entries per head after t tokens; (value, exact?)."""
    L = np.array(d["lengths"])
    i = int(np.argmin(np.abs(np.log(L) - np.log(t))))
    return d["mean"][i], int(L[i]) == t


def rows(runs):
    out = []
    for run, name in runs:
        d = load(run)
        if d is None:
            out.append((name, None)); continue
        out.append((name, [at(d, t) for t in LENGTHS]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latex", action="store_true")
    a = p.parse_args()
    t1 = [(f"lema1024_h{h}", f"$d_h={h}$") for h in (8, 16, 32, 64)]
    PARAMS = {256: "29", 512: "77", 768: "162", 1024: "309", 1280: "525", 1536: "834"}   # M, as in tab:lm-models
    t2 = [(f"lema{d}_h64", f"${PARAMS[d]}$M") for d in (256, 512, 768, 1024, 1280, 1536)]
    head = ["2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k"]
    for title, first, runs in (("d=1024, by head size", "$d_h$", t1),
                               ("d_h=64, by model size", "params", t2)):
        if a.latex:
            print(f"% {title}")
            print(first + " & " + " & ".join(head) + r" \\")
            print(r"\midrule")
        else:
            print(f"--- {title}"); print(f"{first:<10}" + "".join(f"{h:>8}" for h in head))
        for name, vals in rows(runs):
            if vals is None:
                cells = ["--"] * len(LENGTHS)
            elif a.latex:
                cells = [f"{v:,.0f}".replace(",", r"\,") for v, _ in vals]
            else:
                cells = [f"{v:.0f}{'' if ex else '~'}" for v, ex in vals]
            print((name + " & " + " & ".join(cells) + r" \\") if a.latex
                  else f"{name:<10}" + "".join(f"{c:>8}" for c in cells))
        print()


if __name__ == "__main__":
    main()
