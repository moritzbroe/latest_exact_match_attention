"""What context extension does to recall, one panel per model.

    python plot_extension.py [--metric loss] [--freq 100]

The curve of the recall analysis -- loss on the later occurrence of a repeated rare bigram
against the distance to the earlier one -- for each model as trained and after extension,
read from ../main_lm_bigrams/out/scores/.

`--metric benefit` plots the distance to the model's own no-recall baseline instead, which
is the quantity the main figure's dotted lines make visible.

Each panel gets its own y range, scaled to its own two curves: the figure is about what
extension does to one model, not about the levels across models. `--ymax` puts every panel
back on one range. No GPU.

`--runs` replaces the paper's three panels by explicit ones, each given as
"title:as-trained:extended".
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.main_lm_bigrams.plot_sizes_grid import bucket_label    # noqa: E402
from experiments.main_lm_bigrams.measurements import means
from experiments.style import paper                                     # noqa: E402

BIG = HERE.parent / "main_lm_bigrams"
EDGES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
STYLE = {"as trained": dict(color="0.45", ls="--", lw=1.3, marker="o"),
         "extended to $16384$": dict(color="#d1495b", ls="-", lw=1.5, marker="s")}


# (title, as-trained run, extended run) per panel: the paper's figure
PANELS = [("softmax", "rope1536", "rope1536_16k"),
          ("GDN", "gdn1536", "gdn1536_16k"),
          ("LEMA", "lema1536_h64", "lema1536_h64_16k")]


def parse_runs(specs):
    """Panels spelled out on the command line as "title:as-trained:extended", in the order
    they should appear. The title is taken literally, so it may say the model size."""
    out = []
    for s in specs:
        parts = [p.strip() for p in s.split(":")]
        if len(parts) != 3 or not all(parts):
            raise SystemExit(f"--runs wants title:as-trained:extended, got {s!r}")
        out.append(tuple(parts))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--freq", type=int, default=100)
    p.add_argument("--metric", choices=("loss", "benefit"), default="loss")
    p.add_argument("--ymax", type=float, default=None,
                   help="one y range for every panel instead of one per panel")
    p.add_argument("--runs", nargs="+", default=None, metavar="TITLE:BASE:EXT",
                   help="panels given explicitly as title:as-trained:extended, one per "
                        "panel, instead of the paper's three")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    pan = parse_runs(a.runs) if a.runs else PANELS
    ncol = min(3, len(pan))
    nrow = -(-len(pan) // ncol)
    d = np.load(BIG / "out" / f"targets_T{a.ctx}_f{a.freq}.npz")
    dist = d["dist"]
    lo_hi = [(lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:]) if lo < a.ctx]
    cell = [(dist >= lo) & (dist < hi) for lo, hi in lo_hi]
    mid = [np.sqrt(lo * hi) for lo, hi in lo_hi]

    def curve(run):
        if run is None:
            return None
        if a.metric == "benefit":
            loss = means(run, a.ctx, a.freq, kind="intervention", bounds=lo_hi)
            clean = means(run, a.ctx, a.freq, kind="intervention", key="clean", bounds=lo_hi)
            return None if loss is None or clean is None else loss - clean
        return means(run, a.ctx, a.freq, bounds=lo_hi)

    paper()
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 0.7 + 1.75 * nrow), sharex=True,
                             squeeze=False)
    for ax, (label, base, ext) in zip(axes.ravel(), pan):
        off, top = [], 0.0
        for run, name in ((base, "as trained"), (ext, "extended to $16384$")):
            c = curve(run)
            if c is None:
                if run is not None:
                    print(f"  (MISSING scores for {run})")
                continue
            ax.plot(mid, c, ms=3.0, zorder=3, label=name, **STYLE[name])
            if a.ymax is not None and c.min() > a.ymax:   # wholly above the panel, say so
                off.append((name, c.min(), c.max()))
            top = max(top, float(c.max()))
            print(f"{label:<16} {name:<20} " + "  ".join(f"{v:5.2f}" for v in c))
        # one y range per panel, a little above that panel's own highest curve: what the
        # figure shows is the gap between a model's two curves, and on one shared range
        # the models whose gap is small are flat lines
        ax.set_ylim(0, a.ymax or top * 1.06 or 1.0)
        ax.set_title(label, fontsize=7.5, pad=2)
        for k, (name, lo, hi) in enumerate(off):
            ax.text(0.5, 0.93 - 0.09 * k, f"{name}: {lo:.1f}–{hi:.1f}, above the panel",
                    transform=ax.transAxes, ha="center", va="top", fontsize=6,
                    color=STYLE[name]["color"])
        ax.set_xscale("log", base=2)
        ax.set_xticks(mid)
        ax.set_xticklabels([bucket_label(lo, hi) for lo, hi in lo_hi],
                           rotation=45, ha="right", fontsize=5.5)
        ax.minorticks_off()
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.tick_params(labelsize=6, length=2, pad=1.5)
    for ax in axes[-1]:
        ax.set_xlabel("distance (tokens)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel({"loss": "cross-entropy (nats)",
                       "benefit": "loss removed (nats)"}[a.metric], fontsize=7)
    handles, labels = [], []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    order = [labels.index(l) for l in ("as trained", "extended to $16384$") if l in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False,
               fontsize=7.5, handlelength=2.2, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.6,
                     rect=(0, 0, 1, 1 - 0.252 / fig.get_figheight()))
    out = a.out or HERE / "out" / f"extension_{a.metric}_f{a.freq}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
