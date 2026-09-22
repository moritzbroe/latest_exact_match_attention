"""The appendix table: validation cross-entropy of the 77M models at 20 and at 100 tokens per
parameter, at context length 2048 as trained and at 16384, the two baselines after context
extension and LEMA as trained.

    python losses.py [--latex]

The 2048 numbers are ../main_lm/out/<run>/eval.json. The 16384 numbers are the mean of the
per-token evaluation of ../main_lm_bigrams over its first 2^26 scored tokens, which is exactly
the protocol of eval_lm.py at that context length (see ../app_lm_context_extension/ce_table.py).
No GPU.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.app_lm_context_extension.ce_table import ce            # noqa: E402

LM_OUT = HERE.parent / "main_lm" / "out"
ROWS = [("softmax", "rope512", "rope512_16k"),
        ("GDN", "gdn512", "gdn512_16k"),
        ("LEMA", "lema512_h64", "lema512_h64")]
FACTORS = [("", "20"), ("_5x", "100")]      # run suffix, tokens per parameter


def ce2048(run):
    f = LM_OUT / run / "eval.json"
    return json.loads(f.read_text())["2048"]["ce"] if f.exists() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latex", action="store_true")
    a = p.parse_args()
    fmt = lambda v: "-" if v is None else f"{v:.3f}"
    rows = []
    for label, base, long in ROWS:
        cells = []
        for suffix, _ in FACTORS:
            b = base + suffix
            l = long.replace(base, b) if long != base else b
            cells += [ce2048(b), ce(l, 16384, 2 ** 26)]
        rows.append((label, cells))
    if a.latex:
        print(r"\begin{tabular}{lcccc}"); print(r"\toprule")
        print(r" & \multicolumn{2}{c}{$20$ tokens per parameter} & \multicolumn{2}{c}{$100$ tokens per parameter} \\")
        print(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
        print(r"model & $2048$ & $16\,384$ & $2048$ & $16\,384$ \\"); print(r"\midrule")
        for label, cells in rows:
            print(f"{label} & " + " & ".join(fmt(c) for c in cells) + r" \\")
        print(r"\bottomrule"); print(r"\end{tabular}")
    else:
        print(f"{'':<16} {'20/param 2048':>14} {'16384':>8}   {'100/param 2048':>15} {'16384':>8}")
        for label, cells in rows:
            print(f"{label:<16} {fmt(cells[0]):>14} {fmt(cells[1]):>8}   {fmt(cells[2]):>15} {fmt(cells[3]):>8}")


if __name__ == "__main__":
    main()
