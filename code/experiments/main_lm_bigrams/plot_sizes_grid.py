"""Recall against distance at every model size: one panel per size, one curve per
architecture and per LEMA head size, each against its own no-recall level.

    python plot_sizes_grid.py --ctx 2048                          # inside the training context
    python plot_sizes_grid.py --ctx 16384 --baselines extended     # beyond it: RoPE and GDN extended
    python plot_sizes_grid.py --ctx 16384 --baselines extended --metric benefit

Reads out/scores/<run>_T{ctx}_f{freq}.npz for rope<d>, gdn<d> and lema<d>_h<h> at every width
in SIZES and every head size in HEADS (a missing file leaves the curve out and is reported).
Buckets and the loss: the mean loss on the second token of a repeated rare
bigram, by the distance to the earlier occurrence.

--baselines picks which RoPE and GDN runs are drawn, since neither reaches 16384 tokens as
pretrained: "trained" the pretrained runs, "extended" the context-extended out/<run>_16k ones.

Dotted in each model's own colour is its no-recall level measured by intervention
(intervention_targets.py, eval_intervention.py): the same targets in the same place, with every
earlier copy of the bigram overwritten, read from out/intervention/<run>_T{ctx}_f{freq}.npz for
every curve that has such a file. The others are drawn without one and listed.

--metric benefit draws the gap between the two instead of the loss: per bucket the mean
intervention loss minus the mean clean loss of the SAME sampled targets, the nats the earlier
occurrence is worth to that model, with zero the floor for a model that cannot recall it.
Nothing is dotted there, and a curve without an intervention file is left out and listed.

--distracted draws the other target set instead, the one the definition of Arora et al. keeps
and ours drops.

Writes out/sizes_T{ctx}[_distracted][_benefit].pdf. No GPU.

--yaxes groups the panels that share a y range: "shared" one range for all six, "rows" one
per row, "first" the smallest model on its own and one range for the other five, "groups" the
smallest on its own, one range for the next two and one for the largest three. Each group is
scaled to its own highest line, and y tick labels go on the first panel of a group in a row.
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
from experiments.style import COLOR, HEAD_COLOR, NAME, paper            # noqa: E402

# The grid compares widths; parameter counts differ slightly by architecture, so use the
# common model dimension as the panel title.
SIZES = [(256, r"$d=256$"), (512, r"$d=512$"), (768, r"$d=768$"),
         (1024, r"$d=1024$"), (1280, r"$d=1280$"), (1536, r"$d=1536$")]
EDGES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
HEADS = (8, 16, 32, 64)
SUFFIX = {"trained": "", "extended": "_16k"}       # which RoPE and GDN runs --baselines draws
IV_LABEL = "earlier bigram occurrences replaced"
GROUPS = {"shared": [[0, 1, 2, 3, 4, 5]],          # panels that share one y range
          "rows":   [[0, 1, 2], [3, 4, 5]],
          "first":  [[0], [1, 2, 3, 4, 5]],
          "groups": [[0], [1, 2], [3, 4, 5]]}


def curves_for(width, archs, heads, suffix):
    """The runs of one panel, in legend order: the two baselines as selected, then LEMA from
    the smallest head size to the largest. Styles are shared with plot_recall_panel.py, so the figures read the
    same."""
    out = []
    if "rope" in archs:
        out.append((f"rope{width}{suffix}",
                    dict(color=COLOR["rope"], marker="s", label=NAME["rope"])))
    if "gdn" in archs:
        out.append((f"gdn{width}{suffix}",
                    dict(color=COLOR["gdn"], marker="^", label=NAME["gdn"])))
    if "lema" in archs:
        out += [(f"lema{width}_h{h}",
                 dict(color=HEAD_COLOR[h], marker="o", label=f"LEMA $d_h={h}$")) for h in heads]
    return out


def bucket_label(lo, hi):
    """A bucket's tick label, shared by the recall figures so that they read the
    same: the range rather than its lower edge, thousands as k."""
    if hi < 1024:
        return f"{lo}–{hi}"
    return f"{lo // 1024}k–{hi // 1024}k" if lo >= 1024 else f"{lo}–{hi // 1024}k"


def nice_top(ymax):
    """A tick-friendly bound a little above the highest line of a group."""
    y = ymax * 1.05
    if y <= 0:
        return 1.0
    step = 10.0 ** np.floor(np.log10(y)) / 4                           # a quarter of a decade
    return float(np.ceil(y / step) * step)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--freq", type=int, default=100)
    p.add_argument("--archs", default="rope,gdn,lema")
    p.add_argument("--metric", choices=("loss", "benefit"), default="loss",
                   help="the loss itself, or how much of it the earlier occurrence removes")
    p.add_argument("--heads", default=",".join(str(h) for h in HEADS),
                   help="the LEMA head sizes to draw, light to dark")
    p.add_argument("--baselines", choices=sorted(SUFFIX), default="trained",
                   help="which RoPE and GDN runs: as pretrained, or context-extended (_16k)")
    p.add_argument("--distracted", action="store_true",
                   help="the targets with competing continuations")
    p.add_argument("--yaxes", choices=sorted(GROUPS), default="rows")
    p.add_argument("--ymax", type=float, default=None, help="one range for every panel")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    archs = a.archs.split(",")
    heads = [int(h) for h in a.heads.split(",") if h]
    tag = "_distracted" if a.distracted else ""      # which of the two target sets is drawn
    d = np.load(HERE / "out" / f"targets_T{a.ctx}_f{a.freq}{tag}.npz")
    dist = d["dist"]
    lo_hi = [(lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:]) if lo < a.ctx]
    cell = [(dist >= lo) & (dist < hi) for lo, hi in lo_hi]
    mid = [np.sqrt(lo * hi) for lo, hi in lo_hi]
    no_score, no_iv = [], []

    def intervention(run):
        loss = means(run, a.ctx, a.freq, a.distracted, "intervention", "loss", lo_hi)
        clean = means(run, a.ctx, a.freq, a.distracted, "intervention", "clean", lo_hi)
        if loss is None or clean is None:
            no_iv.append(run)
            return None
        return np.array([loss, clean])

    paper()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
    panel = axes.ravel()
    ymax = [0.0] * len(SIZES)                        # the highest line of every panel
    drawn = {}                                       # label -> legend handle, in draw order
    for i, (ax, (width, label)) in enumerate(zip(panel, SIZES)):
        for run, st in curves_for(width, archs, heads, SUFFIX[a.baselines]):
            c = means(run, a.ctx, a.freq, a.distracted, bounds=lo_hi)
            if c is None:
                no_score.append(run)
                continue
            b = intervention(run)
            if a.metric == "benefit":
                if b is None:
                    continue
                c, b = b[0] - b[1], None
            ymax[i] = max(ymax[i], c.max())
            ax.plot(mid, c, ms=2.8, lw=1.3, zorder=3, **st)
            drawn.setdefault(st["label"], Line2D([], [], lw=1.3, ms=2.8, **st))
            if b is not None:
                ymax[i] = max(ymax[i], b[0].max())
                ax.plot(mid, b[0], ls=":", lw=0.9, color=st["color"], zorder=2)
                drawn.setdefault(IV_LABEL, Line2D([], [], ls=":", lw=0.9, color="black",
                                                  label=IV_LABEL))
            print(f"{label:>5} {run:<16} " + "  ".join(f"{v:5.2f}" for v in c)
                  + ("" if b is None else "   | " + "  ".join(f"{v:5.2f}" for v in b[0])))
        ax.set_title(label, fontsize=7.5, pad=2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(mid)
        ax.set_xticklabels([bucket_label(lo, hi) for lo, hi in lo_hi],
                           rotation=45, ha="right", fontsize=5.5)
        ax.minorticks_off()
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.tick_params(labelsize=6, length=2, pad=1.5)
    labelled = set()                        # the first panel of a y group within its row
    for group in GROUPS[a.yaxes]:
        hi = a.ymax or nice_top(max(ymax[i] for i in group))
        for i in group:
            panel[i].set_ylim(0, hi)
        for row in range(axes.shape[0]):
            in_row = [i for i in group if i // axes.shape[1] == row]
            if in_row:
                labelled.add(min(in_row))
    for i, ax in enumerate(panel):
        ax.tick_params(labelleft=i in labelled)
    for ax in axes[1]:
        ax.set_xlabel("distance (tokens)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel({"loss": "cross-entropy (nats)",
                       "benefit": "loss removed by the earlier\noccurrence (nats)"}[a.metric],
                      fontsize=7)
    # one legend for the whole figure, above the panels: inside a panel it would cover the
    # dotted baselines, which run across the top of every one of them. the intervention entry
    # is much the longest, so it takes the second row on its own with the two baselines.
    lema = [f"LEMA $d_h={h}$" for h in heads]
    base = [drawn[k] for k in drawn if k not in lema and k != IV_LABEL]       # rope, gdn
    head = [drawn[k] for k in lema if k in drawn]
    iv = [drawn[k] for k in (IV_LABEL,) if k in drawn]
    rows = ([base + head + iv] if len(base) + len(head) + len(iv) <= 4        # one row fits
            else [base + iv, head])                                          # else the heads
    rows = [r for r in rows if r]                                            # take their own
    for j, r in enumerate(rows):
        fig.legend(r, [h.get_label() for h in r], frameon=False, fontsize=7.5, ncol=len(r),
                   loc="upper center", bbox_to_anchor=(0.5, 1.015 - 0.037 * j),
                   handlelength=2.0, columnspacing=1.6, handletextpad=0.5)
    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.6,
                     rect=(0, 0, 1, 0.995 - 0.037 * len(rows)))
    out = a.out or (HERE / "out" /
                    f"sizes_T{a.ctx}{tag}{'' if a.metric == 'loss' else f'_{a.metric}'}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.02)
    if no_score:
        print("  no scores for: " + ", ".join(no_score))
    if no_iv:
        print(("  left out, no intervention file: " if a.metric == "benefit" else
               "  no intervention baseline for: ") + ", ".join(no_iv)
              + "  (run eval_intervention.py)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
