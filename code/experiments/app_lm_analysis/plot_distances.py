"""Every head's attention-distance distribution of a LEMA language model, as one page.

    plot_distances.py out/lema1536_h64.json [--out out/distances_lema1536_h64.pdf]

One small panel per head, layers as rows: the distribution of the distance i - src (how
far back a query attends) over all positions of the held-out windows analyze_heads.py ran,
in its log-spaced bins 1, 2, 3-4, ..., 8k-16k, drawn as a filled curve scaled to the
panel's maximum. Within a layer the heads are sorted by their median distance, so the
short-range and the long-range heads of a layer stand out at a glance, and the fill
colour encodes the head's hit rate (the fraction of its queries that find a key): full
colour for a head that always matches, fading towards white as the hit rate drops, empty
for a head that never matched. No GPU.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.style import COLOR, paper                             # noqa: E402


def median_bin(frac):
    """Index of the bin holding the median of a normalised histogram; nan if empty."""
    if frac.sum() <= 0:
        return np.nan
    c = np.cumsum(frac) / frac.sum()
    return int(np.searchsorted(c, 0.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json", type=Path)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    d = json.loads(a.json.read_text())
    dist, hit = np.array(d["dist"], dtype=float), np.array(d["hit"], dtype=float)
    L, H, K = dist.shape
    assert K == len(d["dist_bins"])
    paper()
    lm, rm, tm, bm = 0.5, 0.55, 0.3, 0.5                         # margins, inches
    cell = min(0.205, (5.5 - lm - rm) / H)                        # panel size: text width
    W, Hh = lm + cell * H + rm, tm + cell * L + bm
    fig = plt.figure(figsize=(W, Hh))
    gs = fig.add_gridspec(L, H, left=lm / W, right=1 - rm / W, top=1 - tm / Hh,
                          bottom=bm / Hh, wspace=0.12, hspace=0.25)
    lema = np.array(mcolors.to_rgb(COLOR["lema"]))
    cmap = mcolors.LinearSegmentedColormap.from_list(   # white -> LEMA blue by hit rate
        "hit", [np.array([1.0, 1.0, 1.0]) * 0.92 + lema * 0.08, lema])
    x = np.arange(K)
    for li in range(L):
        meds = np.array([median_bin(dist[li, h]) for h in range(H)])
        order = np.argsort(np.where(np.isnan(meds), np.inf, meds), kind="stable")
        for col, h in enumerate(order):
            ax = fig.add_subplot(gs[li, col])
            f = dist[li, h]
            if f.sum() > 0:
                y = f / f.max()
                col_ = cmap(hit[li, h])
                ax.fill_between(x, 0, y, color=col_, lw=0)
                ax.plot(x, y, color=col_, lw=0.4)
            ax.set_xlim(-0.3, K - 0.7)
            ax.set_ylim(0, 1.08)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.3)
                s.set_color("0.75")
            if col == 0:
                ax.set_ylabel(f"{li + 1}", rotation=0, ha="right", va="center", fontsize=6,
                              labelpad=3)
    fig.text((lm - 0.38) / W, 0.5, "layer", rotation=90, ha="right", va="center", fontsize=7)
    fig.text(0.5, 1 - 0.08 / Hh, "heads of a layer, sorted by median attention distance",
             ha="center", va="top", fontsize=7)
    fig.text(0.5, 0.05 / Hh, "every panel: attention distance $i-\\ell_i$ from 1 (left) to 16k "
             "(right) in log-spaced bins 1, 2, 3\u20134, \u2026, 8k\u201316k\n"
             "height scaled to the panel's maximum", ha="center", va="bottom", fontsize=6,
             linespacing=1.4)
    # colour bar for the hit rate, at the right edge
    cax = fig.add_axes([1 - (rm - 0.1) / W, bm / Hh, 0.06 / W, 1 - (tm + bm) / Hh])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.set_label("hit rate of the head", fontsize=6, labelpad=2)
    cb.outline.set_linewidth(0.3)
    out = a.out or HERE / "out" / f"distances_{d['run']}.pdf"
    fig.savefig(out)
    print(f"-> {out}   ({L} layers x {H} heads, {int((dist.sum(-1) == 0).sum())} never matched, "
          f"mean hit rate {hit.mean():.2f})")


if __name__ == "__main__":
    main()
