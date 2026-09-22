"""The S-NIAH-1 depth figure of the appendix, from the generation cells in ../out/gen,
plus the plotting helpers shared by the other RULER figures in this folder.

    python plot_ruler.py                      # -> ../out/sniah1_depth.pdf

sniah1_depth  S-NIAH-1 at 16384 tokens, accuracy against needle depth in three equal
              bins of the prompt.

The metric is RULER's string_match_all on greedy generation, n=500 prompts per cell; a
cell is one out/gen/<task>-<hay>_<model>_T<ctx>.jsonl. No GPU.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]                                                # code/
sys.path.insert(0, str(CODE))
from experiments.app_retrieval.paper import ruler_cells as rc          # noqa: E402
from experiments.style import COLOR, HEAD_COLOR, NAME, paper           # noqa: E402

FIGS = HERE.parent / "out"
CTXS = [2048, 4096, 8192, 16384, 32768, 65536]
SNIAH1 = "niah_single_1-noise"
DEPTH = ["rope1024_16k", "gdn1024_16k", "lema1024_h64", "lema1536_h64"]


def st(run):
    s = rc.style(run, COLOR, HEAD_COLOR)
    if run.startswith(("rope1536", "gdn1536", "lema1536")):
        s["ls"] = "--"                       # smaller models solid, the largest dashed
    return s


def lab(run):
    return rc.label(run, NAME)


def legend_rows(fig, handles, labels, fs=6.0, pad=0.10):
    """Pack the entries into rows that fit the figure's width, in the order given."""
    rows, row, used = [], [], 0.0
    for h, t in zip(handles, labels):
        wd = 0.30 + fs / 72 * 0.52 * len(t)
        if row and used + wd > fig.get_figwidth() - 0.1:
            rows.append(row)
            row, used = [], 0.0
        row.append((h, t))
        used += wd
    rows.append(row)
    row_h, figh = 0.115, fig.get_figheight()
    fig.tight_layout(pad=0.3, w_pad=0.8,
                     rect=(0, 0, 1, 1 - (pad + row_h * len(rows)) / figh))
    for j, r in enumerate(rows):
        fig.legend([h for h, _ in r], [t for _, t in r], frameon=False, fontsize=fs,
                   ncol=len(r), loc="upper center",
                   bbox_to_anchor=(0.5, 1 - (0.005 + row_h * j) / figh),
                   handlelength=1.8, columnspacing=1.1, handletextpad=0.4)



def length_axis(ax, ctxs):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ctxs)
    ax.set_xticklabels([f"{c // 1024}k" for c in ctxs], rotation=45, ha="right")
    ax.minorticks_off()
    ax.set_xlabel("context (tokens)")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.tick_params(labelsize=6, length=2, pad=1.5)


def save(fig, stem, out=None):
    dst = out or FIGS / f"{stem}.pdf"
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=200)
    print(f"-> {dst}")
    plt.close(fig)


# ---------------------------------------------------------------- the figure

def sniah1_depth(ctx=16384, out=None):
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    hs, ls, table, missing = [], [], {}, []
    labels = None
    for run in DEPTH:
        c = rc.depth_curve(SNIAH1, run, ctx)
        if c is None:
            missing.append(f"{run}@{ctx}")
            continue
        labels, acc, n = c
        line, = ax.plot(range(len(labels)), acc, ms=2.6, lw=1.1, **st(run))
        hs.append(line)
        ls.append(lab(run))
        table[run] = dict(acc=[round(v, 3) for v in acc], n=n)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("needle depth (fraction of the prompt)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("accuracy")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.tick_params(labelsize=6, length=2, pad=1.5)
    legend_rows(fig, hs, ls)
    save(fig, "sniah1_depth", out)
    return {"bins": labels, "acc": table, "missing": missing}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    paper()
    plt.rcParams.update({"axes.labelsize": 7, "axes.titlesize": 7.5,
                         "xtick.labelsize": 6, "ytick.labelsize": 6})
    print(sniah1_depth(out=a.out))


if __name__ == "__main__":
    main()
