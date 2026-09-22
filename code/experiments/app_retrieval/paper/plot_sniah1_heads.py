"""S-NIAH-1 accuracy against context length for the 309M models, one curve per LEMA head
size next to the two context-extended baselines.

    python plot_sniah1_heads.py            # -> ../out/sniah1_heads_309M.pdf

The appendix's head-size figure at 309M, colours by d_h. Style, cells and
colours are `plot_ruler.py`'s -- the shared reader `ruler_cells.py` gives the per-cell
accuracy, LEMA is shaded blue by head size, softmax orange and GDN green,
and the baselines are the checkpoints extended to 16k, the ones that reach past 2048
tokens. Cells that do not exist yet are skipped and listed on stdout. No GPU.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]                                                # code/
sys.path.insert(0, str(CODE))
from experiments.app_retrieval.paper import plot_ruler as pr           # noqa: E402
from experiments.app_retrieval.paper import ruler_cells as rc          # noqa: E402
from experiments.style import COLOR, HEAD_COLOR, NAME, paper           # noqa: E402

RUNS = ["rope1024_16k", "gdn1024_16k",
        "lema1024_h8", "lema1024_h16", "lema1024_h32", "lema1024_h64"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=pr.FIGS / "sniah1_heads_309M.pdf")
    a = p.parse_args()
    paper()
    plt.rcParams.update({"axes.labelsize": 7, "axes.titlesize": 7.5,
                         "xtick.labelsize": 6, "ytick.labelsize": 6})
    got, missing = rc.scan(pr.SNIAH1, RUNS, pr.CTXS)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    hs, ls = [], []
    for run in RUNS:
        pts = sorted(got.get(run, {}).items())
        if run.startswith("rope"):
            # the softmax transformer is extended to 16k, beyond that it gets no point
            pts = [(c, v) for c, v in pts if c <= 16384]
        if not pts:
            continue
        line, = ax.plot([c for c, _ in pts], [acc for _, (acc, _) in pts], ms=2.6, lw=1.1,
                        **rc.style(run, COLOR, HEAD_COLOR))
        hs.append(line)
        ls.append(rc.label(run, NAME, size=False))
    pr.length_axis(ax, pr.CTXS)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("accuracy")
    pr.legend_rows(fig, hs, ls)
    pr.save(fig, "sniah1_heads_309M", a.out)
    for run in RUNS:
        row = got.get(run, {})
        print(f"{run:<14} " + "  ".join(f"{c // 1024}k={row[c][0]:.3f}"
                                        for c in pr.CTXS if c in row))
    if missing:
        print("missing:", " ".join(missing))


if __name__ == "__main__":
    main()
