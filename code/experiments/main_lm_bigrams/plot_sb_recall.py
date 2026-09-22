"""The stick-breaking ablation's recall panel: one 77M panel in the style of
plot_recall_panel.py, with the plain stick-breaking run drawn next to the two extended
baselines and LEMA.

    python plot_sb_recall.py --out out/sb_recall_77M.pdf

Buckets, colours, markers, tick labels, the legend above the panel and the dotted no-recall
level measured by intervention are plot_recall_panel.py's helpers, imported rather than
copied, so this panel and the appendix's head-size panel read the same. The two baselines
are the context-extended rope<d>_16k and gdn<d>_16k checkpoints, the only ones that reach
16384 tokens; sb<d> is drawn as trained, with no context extension, and has no intervention
file, so it carries no dotted line.

No GPU.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.style import COLOR, HEAD_COLOR, NAME, paper          # noqa: E402
from plot_recall_panel import (EDGES, IV_LABEL, bucket_label, iv_means,  # noqa: E402
                               load_set, loss_means, nice_top)

SB_COLOR = "#7570b3"          # neither LEMA's blue, RoPE's orange nor GDN's green


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--head", type=int, default=64)
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--freq", type=int, default=100)
    p.add_argument("--figsize", default="3.4,2.6")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--png", type=Path, default=None)
    a = p.parse_args()
    lo_hi = [(lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:]) if lo < a.ctx]
    mid = [np.sqrt(lo * hi) for lo, hi in lo_hi]
    s = load_set(a.ctx, a.freq, False, lo_hi)

    runs = [(f"rope{a.dim}_16k", dict(color=COLOR["rope"], marker="s", label=NAME["rope"])),
            (f"gdn{a.dim}_16k", dict(color=COLOR["gdn"], marker="^", label=NAME["gdn"])),
            (f"lema{a.dim}_h{a.head}", dict(color=HEAD_COLOR[a.head], marker="o",
                                            label=f"LEMA $d_h{{=}}{a.head}$")),
            (f"sb{a.dim}", dict(color=SB_COLOR, marker="D", label=NAME["sb"]))]

    paper()
    fig, ax = plt.subplots(figsize=tuple(float(v) for v in a.figsize.split(",")))
    ymax, drawn, missing, curves = 0.0, {}, [], {}
    for run, st in runs:
        c = loss_means(run, s, a.ctx, a.freq, lo_hi)
        if c is None:
            missing.append(run)
            continue
        b = iv_means(run, s, a.ctx, a.freq, lo_hi)
        ax.plot(mid, c, ms=2.8, lw=1.3, zorder=3, **st)
        drawn.setdefault(st["label"], Line2D([], [], lw=1.3, ms=2.8, **st))
        curves[run] = c
        ymax = max(ymax, c.max())
        if b is None:
            missing.append(f"{run} (intervention)")
        else:
            ax.plot(mid, b, ls=":", lw=0.9, color=st["color"], zorder=2)
            drawn.setdefault(IV_LABEL, Line2D([], [], ls=":", lw=0.9, color="black",
                                              label=IV_LABEL))
            ymax = max(ymax, b.max())
        print(f"{run:<16} " + "  ".join(f"{v:5.2f}" for v in c)
              + ("" if b is None else "   | " + "  ".join(f"{v:5.2f}" for v in b)))
    ax.set_xscale("log", base=2)
    ax.set_xticks(mid)
    ax.set_xticklabels([bucket_label(lo, hi) for lo, hi in lo_hi], rotation=45, ha="right",
                       fontsize=5.5)
    ax.minorticks_off()
    ax.set_ylim(0, nice_top(ymax))
    ax.set_xlabel("distance (tokens)", fontsize=7)
    ax.set_ylabel("cross-entropy (nats)", fontsize=7)
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.tick_params(labelsize=6, length=2, pad=1.5)

    # the legend above the panel, packed by estimated width in draw order, as in
    # plot_recall_panel.py: over the panel it would cover the dotted baselines.
    fs = 6.5
    entries = ([v for k, v in drawn.items() if k != IV_LABEL]
               + [v for k, v in drawn.items() if k == IV_LABEL])   # the dotted line last
    rows, row, used = [], [], 0.0
    for h in entries:
        wd = 0.30 + fs / 72 * 0.52 * len(h.get_label())      # handle plus label
        if row and used + wd > fig.get_figwidth() - 0.1:
            rows.append(row)
            row, used = [], 0.0
        row.append(h)
        used += wd
    rows.append(row)
    row_h, figh = 0.13, fig.get_figheight()
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 1 - (0.03 + row_h * len(rows)) / figh))
    for j, r in enumerate(rows):
        fig.legend(r, [h.get_label() for h in r], frameon=False, fontsize=fs, ncol=len(r),
                   loc="upper center", bbox_to_anchor=(0.5, 1 - (0.005 + row_h * j) / figh),
                   handlelength=1.8, columnspacing=1.1, handletextpad=0.4)
    for f in [a.out] + ([a.png] if a.png else []):
        f.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f, dpi=200)
        print(f"-> {f}")
    if missing:
        print("  not drawn / no dotted line: " + ", ".join(missing))

    sb, rope = curves.get(f"sb{a.dim}"), curves.get(f"rope{a.dim}_16k")
    if sb is not None and rope is not None:
        print("sb - rope_16k  " + "  ".join(f"{v:+5.2f}" for v in sb - rope))
        ahead = [bucket_label(*lo_hi[i]) for i in range(len(lo_hi)) if sb[i] < rope[i]]
        print("  stick-breaking below extended softmax in: " + ", ".join(ahead))


if __name__ == "__main__":
    main()
