"""Figures from out/eval/*.json and retained attention panels.

out/recall.pdf        mean and min-max range over seeds for each architecture
out/recall_seeds.pdf  appendix: every seed separately
out/recall_main.pdf   main text: the accuracy panel next to the attention pictures of
                          the LEMA and RoPE models on one sample with n=4, read
                          from out/attention/panels.json (or checkpoints if available)
No GPU.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.style import COLOR, NAME, paper                       # noqa: E402

FIG_DIR = HERE / "out"
FIGSIZE = (3.3, 2.4)          # drawn at the size it is included at
TITLE_PAD = 5                 # points between a picture's title and the picture

# LEMA and softmax both sit at accuracy ~1 at every n, so whichever is drawn second
# would hide the other entirely. LEMA is solid underneath and RoPE dashed with hollow
# markers on top, which keeps both readable exactly where they coincide.
STYLE = {
    "gdn": dict(color=COLOR["gdn"], marker="^", ms=3.8, lw=1.4, ls="-", zorder=2),
    "lema": dict(color=COLOR["lema"], marker="o", ms=3.8, lw=1.9, ls="-", zorder=3),
    "rope": dict(color=COLOR["rope"], marker="s", ms=5.4, lw=1.2, ls=(0, (3.2, 2.4)),
                 zorder=4, markerfacecolor="none", markeredgewidth=1.0),
}
LABEL = NAME
DRAW = ("gdn", "lema", "rope")            # back to front
LEGEND = ("lema", "rope", "gdn")


def curves():
    """{arch: {seed: {n: acc}}} from every stored evaluation."""
    out = {}
    for f in sorted((HERE / "out" / "eval").glob("*.json")):
        arch, seed = f.stem.rsplit("_s", 1)
        r = json.loads(f.read_text())
        pts = r["curve"] if "curve" in r else r["points"]
        out.setdefault(arch, {})[int(seed)] = {int(n): a for n, a in pts.items()}
    return out


def common_ns(data):
    """The n every architecture was evaluated at. The curriculum baselines have a stage
    at n=2 and LEMA does not, and a curve starting one tick later than the others reads
    as a plotting error rather than as a protocol difference."""
    sets = [set(c) for arch in data.values() for c in arch.values()]
    return sorted(set.intersection(*sets)) if sets else []


def style(ax):
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of associations $n$")
    ax.set_ylabel("recall accuracy")
    ax.set_ylim(-0.02, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    # ticks on data points, including the first and last: with matplotlib's own choice
    # the curves run past the outermost labelled tick
    ax.set_xticks([4, 16, 64, 256, 1024, 4096])
    ax.set_xticklabels([f"$2^{{{e}}}$" for e in (2, 4, 6, 8, 10, 12)])
    ax.tick_params(axis="x", which="minor", length=0)
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    return ax


def legend(ax):
    """One compact box in the empty lower-left corner: the three architectures at the
    tick labels' size, opaque so the grid does not run through the text, with a hairline
    edge so that the opaque patch reads as a box rather than as a gap in the grid."""
    h = dict(zip(*(ax.get_legend_handles_labels()[::-1])))
    leg = ax.legend([h[LABEL[a]] for a in LEGEND if LABEL[a] in h],
                    [LABEL[a] for a in LEGEND if LABEL[a] in h],
                    loc="lower left", fontsize=plt.rcParams["xtick.labelsize"],
                    handlelength=2.4, handletextpad=0.6, markerscale=0.85,
                    borderpad=0.4, borderaxespad=0.4, labelspacing=0.25,
                    frameon=True, framealpha=1.0, fancybox=False,
                    edgecolor="0.8", facecolor="white")
    leg.get_frame().set_linewidth(0.5)


def accuracy_panel(ax, data, ns):
    """Mean and observed range across three seeds, into `ax`."""
    style(ax)
    for arch in DRAW:
        if arch not in data:
            continue
        if len(data[arch]) != 3:
            raise ValueError(f"{arch}: expected three complete seeds")
        values = np.array([[c[n] for n in ns] for c in data[arch].values()])
        mean = values.mean(axis=0)
        ax.fill_between(ns, values.min(axis=0), values.max(axis=0),
                        color=COLOR[arch], alpha=0.18, linewidth=0, zorder=1)
        ax.plot(ns, mean, label=LABEL[arch], **STYLE[arch])
        print(arch + "  " + "  ".join(f"{n}:{m:.5f}" for n, m in zip(ns, mean)))
    legend(ax)


def main_figure(data, ns, attention_data=None):
    """recall_main.pdf: the accuracy panel left, the two attention pictures right."""
    ck = {"lema": HERE / "out" / "lema" / "s0" / "model.pt",
          "rope": HERE / "out" / "rope" / "s0" / "n4" / "model_best.pt"}
    if attention_data is None and not all(c.exists() for c in ck.values()):
        print("(recall_main.pdf skipped: needs out/lema/s0/model.pt and out/rope/s0/n4)")
        return
    from experiments.main_recall.plot_attention import draw, weights, cached_panel
    n = 4
    if attention_data is None:
        import torch
        from experiments.main_recall.eval_recall import load
        from experiments.main_recall.train_recall import task_at
        x, y, _, _ = next(task_at(n).batches("val", 1, seed=0))
        q = torch.arange(2 * n + 1, 3 * n + 1)
    # One gridspec, so the pictures stay aligned with the accuracy panel's axes, and
    # the layout padding zeroed: constrained layout's default pad around every axes put
    # half a picture's height of empty space between the two of them, and what is left
    # here is what their titles need. That pays for most of the height.
    # The rest: the panel goes into a nested grid, whose decorations stay inside the
    # left column, so the panel's x label no longer reserves a band under the pictures
    # and the figure is only as tall as the pictures need. The panel takes the lower of
    # that grid's two rows, which leaves the upper one as the room the titles take, so
    # its axes still ends level with the top picture's. The width ratio then compares
    # the columns, y label included, and 1.02 leaves both panels their widths at 1.3.
    # 1.542 is where the pictures alone put the height; 2.04 adds two lines of text to
    # that, which go into the accuracy panel and the two pictures alike, and TITLE_PAD
    # sets a visible gap between a picture's title and the picture. The empty row's
    # share grows with that gap, so that the panel's axes still ends level with the top
    # picture's.
    fig = plt.figure(figsize=(5.5, 2.04), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.0, hspace=0.0)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.02])
    left = gs[:, 0].subgridspec(2, 1, height_ratios=[0.13, 1])
    accuracy_panel(fig.add_subplot(left[1]), data, ns)
    for row, arch in enumerate(("lema", "rope")):
        ax = fig.add_subplot(gs[row, 1])
        if attention_data is not None:
            cached_panel(ax, attention_data, arch, 0, power=0.5, fontsize=6.5)
        else:
            model = load(ck[arch], device="cpu")
            ws, logits = weights(model, x)
            draw(ax, x[0].tolist(), ws, torch.softmax(logits[q], -1), y[0, q], 0.5,
                 fontsize=6.5, short=True, color=COLOR[arch])
        ax.set_title(LABEL[arch], fontsize=8, pad=TITLE_PAD, color=COLOR[arch])
    fig.savefig(FIG_DIR / "recall_main.pdf")
    print(f"figure -> {FIG_DIR}/recall_main.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-data", type=Path,
                        default=HERE / "out" / "attention" / "panels.json",
                        help="saved n=4 attention panels; replot without checkpoints or torch")
    args = parser.parse_args()
    attention_data = (json.loads(args.attention_data.read_text())
                      if args.attention_data.exists() else None)
    paper()
    FIG_DIR.mkdir(exist_ok=True)
    data = curves()
    ns = common_ns(data)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    accuracy_panel(ax, data, ns)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG_DIR / "recall.pdf")

    # every seed, same colours: within an architecture the seeds are the message, so
    # they share a style and only the first one carries a legend entry
    fig, ax = plt.subplots(figsize=FIGSIZE)
    style(ax)
    for arch in DRAW:
        for i, (seed, c) in enumerate(sorted(data.get(arch, {}).items())):
            st = dict(STYLE[arch], lw=1.1, ms=3.2, alpha=0.85)
            ax.plot(ns, [c[n] for n in ns], label=LABEL[arch] if i == 0 else None, **st)
    legend(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG_DIR / "recall_seeds.pdf")
    print(f"figures -> {FIG_DIR}/recall.pdf, recall_seeds.pdf")
    main_figure(data, ns, attention_data)


if __name__ == "__main__":
    main()
