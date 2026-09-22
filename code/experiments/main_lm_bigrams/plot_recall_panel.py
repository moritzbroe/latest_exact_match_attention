"""One panel of plot_sizes_grid.py on its own, at 0.6 of the text width: the appendix's
head-size panel and its union-of-target-sets panel.

    python plot_recall_panel.py --dim 1024 --heads 8,16,32,64 \
        --out out/lm_recall_heads_309M.pdf
    python plot_recall_panel.py --dim 1536 --heads 64 --union \
        --out out/lm_recall_union_834M.pdf

Everything -- buckets, colours, markers, tick labels, the dotted no-recall level measured by
intervention -- is plot_sizes_grid.py's, with the legend above the panel instead of over the
whole figure. The baselines are the context-extended <run>_16k checkpoints, the only ones
that reach 16384 tokens.

--union draws Zoology's own target definition, the union of the target set the paper keeps
and the distracted one it drops. Per bucket the union mean is the population-weighted mean
of the two per-set means,

    mean_union = (n_u * mean_u + n_d * mean_d) / (n_u + n_d),

with n_u, n_d the full population of that bucket in each list; the solid curves' per-set
means are over the whole population, so this is the union mean exactly, and the dotted
baselines' are over each list's own sample, so the same weights combine the two estimates.

No GPU.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.main_lm_bigrams.measurements import means
from experiments.style import COLOR, HEAD_COLOR, NAME, paper           # noqa: E402

OUT = HERE / "out"
EDGES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
IV_LABEL = "earlier bigram occurrences replaced"


def bucket_label(lo, hi):
    if hi < 1024:
        return f"{lo}–{hi}"
    return f"{lo // 1024}k–{hi // 1024}k" if lo >= 1024 else f"{lo}–{hi // 1024}k"


def nice_top(ymax):
    y = ymax * 1.05
    if y <= 0:
        return 1.0
    step = 10.0 ** np.floor(np.log10(y)) / 4
    return float(np.ceil(y / step) * step)


def load_set(ctx, freq, distracted, lo_hi):
    tag = "_distracted" if distracted else ""
    d = np.load(OUT / f"targets_T{ctx}_f{freq}{tag}.npz")
    dist = d["dist"]
    cells = [(dist >= lo) & (dist < hi) for lo, hi in lo_hi]
    return dict(tag=tag, dist=dist, cells=cells, fingerprint=int(d["fingerprint"]),
                n=np.array([int(m.sum()) for m in cells], np.int64))


def loss_means(run, s, ctx, freq, lo_hi):
    return means(run, ctx, freq, bool(s["tag"]), bounds=lo_hi)


def iv_means(run, s, ctx, freq, lo_hi):
    """Mean intervention loss on the sampled targets in each bucket."""
    return means(run, ctx, freq, bool(s["tag"]), kind="intervention", bounds=lo_hi)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--freq", type=int, default=100)
    p.add_argument("--heads", default="8,16,32,64")
    p.add_argument("--union", action="store_true",
                   help="the union of the two target sets, population-weighted")
    p.add_argument("--figsize", default="3.4,2.4")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--png", type=Path, default=None)
    a = p.parse_args()
    heads = [int(h) for h in a.heads.split(",") if h]
    lo_hi = [(lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:]) if lo < a.ctx]
    mid = [np.sqrt(lo * hi) for lo, hi in lo_hi]
    sets = [load_set(a.ctx, a.freq, False, lo_hi)]
    if a.union:
        sets.append(load_set(a.ctx, a.freq, True, lo_hi))
    w = [s["n"] for s in sets]
    tot = sum(w)

    runs = [(f"rope{a.dim}_16k", dict(color=COLOR["rope"], marker="s", label=NAME["rope"])),
            (f"gdn{a.dim}_16k", dict(color=COLOR["gdn"], marker="^", label=NAME["gdn"]))]
    runs += [(f"lema{a.dim}_h{h}", dict(color=HEAD_COLOR[h], marker="o",
                                        label=f"LEMA $d_h{{=}}{h}$")) for h in heads]

    paper()
    fig, ax = plt.subplots(figsize=tuple(float(v) for v in a.figsize.split(",")))
    ymax, drawn, missing = 0.0, {}, []
    for run, st in runs:
        cs = [loss_means(run, s, a.ctx, a.freq, lo_hi) for s in sets]
        if any(c is None for c in cs):
            missing.append(run + (" (distracted scores)" if cs[0] is not None else ""))
            continue
        c = sum(wi * ci for wi, ci in zip(w, cs)) / tot
        bs = [iv_means(run, s, a.ctx, a.freq, lo_hi) for s in sets]
        b = (None if any(x is None for x in bs)
             else sum(wi * bi for wi, bi in zip(w, bs)) / tot)
        ax.plot(mid, c, ms=2.8, lw=1.3, zorder=3, **st)
        drawn.setdefault(st["label"], Line2D([], [], lw=1.3, ms=2.8, **st))
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

    # the legend above the panel, as in the grid: over the panel it would cover the dotted
    # baselines, which run across the top of it. A panel this narrow fits two or three
    # entries per row, so the entries are packed by their estimated width, in draw order.
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
    if a.union:
        print("population per bucket: n_u " + " ".join(f"{v:,}" for v in w[0])
              + " | n_d " + " ".join(f"{v:,}" for v in w[1]))


if __name__ == "__main__":
    main()
