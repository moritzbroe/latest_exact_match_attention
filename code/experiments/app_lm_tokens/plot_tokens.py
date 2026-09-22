"""The appendix figure: recall of repeated rare bigrams at 20 and at 100 tokens per parameter,
one panel per architecture at 77M parameters.

    python plot_tokens.py [--out out/lm_tokens.pdf]

Each panel shows the loss on the second token of a repeated rare bigram against the distance to
the earlier occurrence (../main_lm_bigrams) for the main run (dashed, grey) and the run on five
times the tokens (solid), softmax and GDN after context extension to 16k and
LEMA as trained, each with its no-recall baseline dotted in the same colour. No GPU.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.style import COLOR, NAME, paper                       # noqa: E402
from experiments.main_lm_bigrams.plot_recall_panel import (           # noqa: E402
    EDGES, bucket_label, iv_means, load_set, loss_means, nice_top)

PANELS = [("rope", "rope512_16k", "rope512_5x_16k", "s"),
          ("gdn", "gdn512_16k", "gdn512_5x_16k", "^"),
          ("lema", "lema512_h64", "lema512_h64_5x", "o")]
GREY = "0.45"
TOKEN_BUDGETS = ("1.5B tokens", "7.7B tokens")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--freq", type=int, default=100)
    p.add_argument("--out", type=Path, default=HERE / "out" / "lm_tokens.pdf")
    a = p.parse_args()
    lo_hi = [(lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:]) if lo < a.ctx]
    mid = [np.sqrt(lo * hi) for lo, hi in lo_hi]
    s = load_set(a.ctx, a.freq, False, lo_hi)
    paper()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharex=True)
    top = 0.0
    for ax, (arch, base, long, marker) in zip(axes, PANELS):
        for run, colour, ls, name in ((base, GREY, "--", TOKEN_BUDGETS[0]),
                                      (long, COLOR[arch], "-", TOKEN_BUDGETS[1])):
            c = loss_means(run, s, a.ctx, a.freq, lo_hi)
            if c is None:
                print(f"  (missing scores for {run})")
                continue
            b = iv_means(run, s, a.ctx, a.freq, lo_hi)
            ax.plot(mid, c, color=colour, ls=ls, marker=marker, ms=2.8, lw=1.3, zorder=3)
            top = max(top, c.max())
            if b is not None:
                ax.plot(mid, b, color=colour, ls=":", lw=0.9, zorder=2)
                top = max(top, b.max())
            print(f"{run:<18} " + "  ".join(f"{v:5.2f}" for v in c)
                  + ("" if b is None else "   | " + "  ".join(f"{v:5.2f}" for v in b)))
        ax.set_title(NAME[arch], fontsize=7.5, pad=2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(mid)
        ax.set_xticklabels([bucket_label(lo, hi) for lo, hi in lo_hi], rotation=45,
                           ha="right", fontsize=5.5)
        ax.minorticks_off()
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.tick_params(labelsize=6, length=2, pad=1.5)
        ax.set_xlabel("distance (tokens)", fontsize=7)
    for ax in axes:
        ax.set_ylim(0, nice_top(top))
    axes[0].set_ylabel("cross-entropy (nats)", fontsize=7)
    handles = [Line2D([], [], color=GREY, ls="--", lw=1.3, label=TOKEN_BUDGETS[0]),
               Line2D([], [], color="black", ls="-", lw=1.3, label=TOKEN_BUDGETS[1]),
               Line2D([], [], color="black", ls=":", lw=0.9,
                      label="earlier bigram occurrences replaced")]
    fig.legend(handles=handles, frameon=False, fontsize=7, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), handlelength=2.2, columnspacing=1.6)
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.92))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
