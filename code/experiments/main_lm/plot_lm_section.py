"""The LM section's figure: one row of three panels, 5.5 inches wide.

    python plot_lm_section.py                         # -> out/lm_section_834M.pdf
    python plot_lm_section.py --dim 1024 --out /tmp/lm_section.pdf

(a) validation cross-entropy against parameters for softmax, GDN and LEMA
    with d_h=64, read from ../main_lm/out/<run>/eval.json. Every point sits at the parameter
    count of its own run, counted from out/<run>/config.json (see model_params): the GDN
    models use d/128 heads of dimension 128 without value expansion and carry an output
    gate, which leaves them 1 to 7% heavier than the other two at the same width.
(b) the repeated-rare-bigram panel of one size: the mean loss on the second token of a
    repeated rare bigram against the distance to the earlier occurrence, from
    ../main_lm_bigrams/out/scores, against each model's own no-recall level (dotted, from
    out/intervention). The baselines are the context-extended <run>_16k checkpoints, the
    only ones that reach 16384 tokens.
(c) RULER S-NIAH-1 accuracy against context length, from
    ../app_retrieval/out/gen/niah_single_1-noise_<run>_T<ctx>.jsonl.

Two variants: --dim 1024 puts 309M models in (b) and (c), --dim 1536 puts 834M models
there.

Neither torch nor Python 3.10 is needed: SIZES is read out of train_lm.py as source.
No GPU.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]                                                # code/
sys.path.insert(0, str(CODE))
from experiments.app_retrieval.paper import ruler_cells as rc          # noqa: E402
from experiments.main_lm_bigrams.measurements import means
from experiments.style import COLOR, HEAD_COLOR, NAME, paper           # noqa: E402

BIG = CODE / "experiments" / "main_lm_bigrams" / "out"
FIGS = HERE / "out"
EDGES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
CTX, FREQ, HEAD = 16384, 100, 64
SNIAH1 = "niah_single_1-noise"
RULER_CTX = [2048, 4096, 8192, 16384]
IV_LABEL = "earlier bigram occurrences replaced"
# the x ticks of (a): labels at the decade and half-decade only, so the row stays short
MAJOR = {1e8: "100M", 5e8: "500M", 1e9: "1B"}
MINOR = [k * 1e7 for k in range(3, 10)] + [k * 1e8 for k in range(2, 10)]


# ---------------------------------------------------------------- (a) cross-entropy

def sizes():
    """train_lm.SIZES, read as source: importing the module would pull in torch."""
    tree = ast.parse((HERE / "train_lm.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "SIZES":
            return ast.literal_eval(node.value)
    raise SystemExit("no SIZES in train_lm.py")


SIZES = sizes()


def attn_params(m):
    """The attention block's parameters, from a saved model config."""
    d, H = m["dim"], m["num_heads"]
    if m["attn"] == "gated-deltanet":                 # fla.layers.GatedDeltaNet
        kdim = H * m["d_qk"]                          # num_heads * head_dim
        hv = int(m["d_v"] * m["gdn_expand_v"])        # head_v_dim
        vdim = H * hv
        return (2 * d * kdim + d * vdim               # q_proj, k_proj, v_proj
                + 2 * d * H + 2 * H                   # a_proj, b_proj, A_log, dt_bias
                + 4 * (2 * kdim + vdim)               # q/k/v_conv1d, depthwise, kernel 4
                + d * vdim                            # g_proj, the output gate
                + hv                                  # o_norm weight
                + vdim * d)                           # o_proj
    kv = m.get("num_kv_heads") or H
    return (d * H * m["d_qk"] + d * kv * m["d_qk"]    # q_proj, k_proj
            + d * kv * m["d_v"] + H * m["d_v"] * d)   # v_proj, o_proj


def model_params(m):
    """Untied embedding and head, per block two RMSNorms, the SwiGLU and the attention,
    the final norm. Reproduces the nominal 20N/32768 counts of train_lm.SIZES for softmax
    and LEMA to 0.1% and 892M for gdn1536."""
    d, L, V, ff = m["dim"], m["depth"], m["vocab_size"], m["dim_ff"]
    return 2 * V * d + L * (2 * d + 3 * d * ff + attn_params(m)) + d


def run_params(run):
    f = HERE / "out" / run / "config.json"
    return model_params(json.loads(f.read_text())["model"]) if f.exists() else None


def ce_of(run, ctx="2048"):
    f = HERE / "out" / run / "eval.json"
    return json.loads(f.read_text()).get(ctx, {}).get("ce") if f.exists() else None


def ce_panel(ax, dims):
    """The three curves, each point at the parameter count of its run."""
    runs = {"rope": {d: f"rope{d}" for d in dims}, "gdn": {d: f"gdn{d}" for d in dims},
            "lema": {d: f"lema{d}_h{HEAD}" for d in dims}}
    series = {k: [(d, run_params(r), ce_of(r)) for d, r in v.items()] for k, v in runs.items()}
    series = {k: [(d, n, c) for d, n, c in v if c is not None] for k, v in series.items()}
    st = {"rope": dict(color=COLOR["rope"], marker="s"),
          "gdn": dict(color=COLOR["gdn"], marker="^"),
          "lema": dict(color=HEAD_COLOR[HEAD], marker="o")}
    for key, pts in series.items():
        ax.plot([n for _, n, _ in pts], [c for _, _, c in pts], ls="-", lw=1.0, ms=3.2,
                zorder=3, **st[key])
    rope = {d: c for d, _, c in series["rope"]}
    lema = {d: c for d, _, c in series["lema"]}
    gap = [(d, rope[d], lema[d]) for d in dims if d in rope and d in lema]
    # returned for the log, not drawn: the text quotes parameter ratios, not gaps
    x = [n for v in series.values() for _, n, _ in v]
    ax.set_xscale("log")
    lo, hi = min(x) * 0.85, max(x) * 1.10
    ax.set_xlim(lo, hi)
    ax.set_xticks([t for t in MAJOR if lo <= t <= hi])
    ax.set_xticklabels([lab for t, lab in MAJOR.items() if lo <= t <= hi])
    ax.set_xticks([t for t in MINOR if lo <= t <= hi], minor=True)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=1.2)
    ax.set_xlabel("parameters")
    ax.set_ylabel("cross-entropy (nats)")
    ax.grid(True, lw=0.3, alpha=0.5)
    ylo = min(c for v in series.values() for _, _, c in v)
    yhi = max(c for v in series.values() for _, _, c in v)
    ax.set_ylim(ylo - 0.05 * (yhi - ylo), yhi + 0.05 * (yhi - ylo))
    return {"gap": [(d, round(b - r, 4)) for d, r, b in gap],
            "ce": {k: [(d, round(c, 4)) for d, _, c in v] for k, v in series.items()},
            "params": {k: [(d, n) for d, n, _ in v] for k, v in series.items()}}


# ---------------------------------------------------------------- (b) repeated bigrams

def bucket_label(lo, hi):
    if hi < 1024:
        return f"{lo}–{hi}"
    return f"{lo // 1024}k–{hi // 1024}k" if lo >= 1024 else f"{lo}–{hi // 1024}k"


def bigram_panel(ax, dim):
    """The panel of plot_sizes_grid.py at one size, with LEMA at d_h=64 only."""
    d = np.load(BIG / f"targets_T{CTX}_f{FREQ}.npz")
    dist = d["dist"]
    lo_hi = [(lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:]) if lo < CTX]
    cell = [(dist >= lo) & (dist < hi) for lo, hi in lo_hi]
    mid = [np.sqrt(lo * hi) for lo, hi in lo_hi]
    runs = [(f"rope{dim}_16k", dict(color=COLOR["rope"], marker="s")),
            (f"gdn{dim}_16k", dict(color=COLOR["gdn"], marker="^")),
            (f"lema{dim}_h{HEAD}", dict(color=HEAD_COLOR[HEAD], marker="o"))]
    got, missing, ymax = {}, [], 0.0
    for run, st in runs:
        c = means(run, CTX, FREQ, bounds=lo_hi)
        if c is None:
            missing.append(run)
            continue
        ax.plot(mid, c, ms=2.6, lw=1.2, zorder=3, **st)
        ymax = max(ymax, c.max())
        b = means(run, CTX, FREQ, kind="intervention", bounds=lo_hi)
        if b is not None:
            ax.plot(mid, b, ls=":", lw=0.9, color=st["color"], zorder=2)
            ymax = max(ymax, b.max())
        else:
            missing.append(f"{run} (intervention)")
        got[run] = (c, b)
    ax.set_xscale("log", base=2)
    # the bucket edges, every other one: the ranges themselves are too wide to stand
    # upright under a panel this narrow, and a point sits between the ticks it lies between
    edges = [lo for lo, _ in lo_hi[::3]]
    ax.set_xticks(edges)
    ax.set_xticklabels([f"{e // 1024}k" if e >= 1024 else str(e) for e in edges])
    ax.set_xlim(2 ** (np.log2(mid[0]) - 0.8), 2 ** (np.log2(mid[-1]) + 0.8))
    ax.minorticks_off()
    ax.set_ylim(0, np.ceil(ymax * 1.05 / 0.25) * 0.25)
    ax.set_xlabel("distance (tokens)")
    ax.set_ylabel("cross-entropy (nats)")
    ax.grid(True, lw=0.3, alpha=0.5)
    return {"buckets": [bucket_label(lo, hi) for lo, hi in lo_hi],
            "loss": {k: [round(x, 3) for x in v[0]] for k, v in got.items()},
            "intervention": {k: (None if v[1] is None else [round(x, 3) for x in v[1]])
                             for k, v in got.items()},
            "missing": missing}


# ---------------------------------------------------------------- (c) RULER S-NIAH-1

def ruler_panel(ax, runs):
    got, missing = rc.scan(SNIAH1, runs, RULER_CTX)
    for run in runs:
        pts = sorted(got.get(run, {}).items())
        if not pts:
            continue
        st = rc.style(run, COLOR, HEAD_COLOR)
        ax.plot([c for c, _ in pts], [a for _, (a, _) in pts], ms=2.6, lw=1.2, zorder=3, **st)
    ax.set_xscale("log", base=2)
    ax.set_xticks(RULER_CTX)
    ax.set_xticklabels([f"{c // 1024}k" for c in RULER_CTX])
    ax.minorticks_off()
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("context (tokens)")
    ax.set_ylabel("accuracy")
    ax.grid(True, lw=0.3, alpha=0.5)
    return {"acc": {r: {c: round(a, 3) for c, (a, _) in sorted(v.items())}
                    for r, v in got.items()},
            "n": {r: sorted({n for _, n in v.values()}) for r, v in got.items()},
            "missing": missing}


# ---------------------------------------------------------------- the figure

def decorations(fig, axes, titles):
    """Per axes, how far its labels reach beyond its frame, in inches (left, right,
    bottom, top), and the width of its title. With `titles` false the titles are hidden
    for the measurement: a title wider than its panel may reach over the gap, and letting
    it set the panel's width would starve the panel it belongs to."""
    W, H = fig.get_size_inches()
    inv = fig.dpi_scale_trans.inverted()
    for ax in axes:
        ax.title.set_visible(titles)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for ax in axes:
        t, p = ax.get_tightbbox(r).transformed(inv), ax.get_position()
        out.append((max(p.x0 * W - t.x0, 0.0), max(t.x1 - p.x1 * W, 0.0),
                    max(p.y0 * H - t.y0, 0.0), max(t.y1 - p.y1 * H, 0.0),
                    ax.title.get_window_extent(r).transformed(inv).width))
    for ax in axes:
        ax.title.set_visible(True)
    return out


def place(fig, axes, legs, pad=0.03, gap=0.11, leg_gap=0.02, title_gap=0.05):
    """Lay the row out by measurement: every panel the same axes width, whatever its own
    labels take, the legend rows on top and everything inside the page. tight_layout
    divides the figure into equal cells instead, which leaves the panel with the widest
    y axis the narrowest of the three."""
    W, H = fig.get_size_inches()
    inv = fig.dpi_scale_trans.inverted()
    g = gap
    for _ in range(4):          # width, the gap the titles need and the reach of a rotated
        side = decorations(fig, axes, False)            # label each depend on the others
        full = decorations(fig, axes, True)
        r = fig.canvas.get_renderer()
        dl, dr = [d[0] for d in side], [d[1] for d in side]
        tw = [d[4] for d in side]
        lh = sum(lg.get_window_extent(r).transformed(inv).height for lg in legs)
        w = (W - 2 * pad - sum(dl) - sum(dr) - g * (len(axes) - 1)) / len(axes)
        y0 = pad + max(d[2] for d in full)
        h = H - pad - lh - leg_gap - max(d[3] for d in full) - y0
        x = pad
        for i, ax in enumerate(axes):
            ax.set_position([(x + dl[i]) / W, y0 / H, w / W, h / H])
            x += dl[i] + w + dr[i] + g
        g = max([gap] + [(tw[i] + tw[i + 1]) / 2 + title_gap - w - dr[i] - dl[i + 1]
                         for i in range(len(axes) - 1)])
    # the three x labels on one line: a label sitting higher than its neighbours' reads as
    # a misalignment
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    top = min(ax.xaxis.label.get_window_extent(r).y1 for ax in axes)
    for ax in axes:
        ax.xaxis.set_label_coords(0.5, ax.transAxes.inverted().transform((0, top))[1])


def figure(dim, out, png):
    ruler_runs = [f"rope{dim}_16k", f"gdn{dim}_16k", f"lema{dim}_h{HEAD}"]
    paper()
    plt.rcParams.update({"axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.6))
    info = {"a": ce_panel(axes[0], sorted(d for d in SIZES if d <= 1536)),
            "b": bigram_panel(axes[1], dim),
            "c": ruler_panel(axes[2], ruler_runs)}
    for ax, t in zip(axes, ("Language modeling", "Repeated rare bigrams", "S-NIAH-1")):
        ax.set_title(t, pad=3)
        ax.tick_params(length=2, pad=1.5)

    # one legend above the three panels: the three architectures and the dotted no-recall
    # level of (b)
    leg_lw = 1.2                                      # line width of the legend samples, pt
    row1 = [Line2D([], [], color=COLOR["rope"], marker="s", lw=leg_lw, ms=2.6, label=NAME["rope"]),
            Line2D([], [], color=COLOR["gdn"], marker="^", lw=leg_lw, ms=2.6, label=NAME["gdn"]),
            Line2D([], [], color=HEAD_COLOR[HEAD], marker="o", lw=leg_lw, ms=2.6,
                   label=f"LEMA $d_h{{=}}{HEAD}$"),
            Line2D([], [], color="black", ls=":", lw=0.9, label=IV_LABEL)]
    rows = [row1]
    row_h, figh = 0.125, fig.get_figheight()          # inches per legend row
    legs = [fig.legend(r, [h.get_label() for h in r], frameon=False, fontsize=6.5,
                       ncol=len(r), loc="upper center",
                       bbox_to_anchor=(0.5, 1 - (0.01 + row_h * j) / figh),
                       handlelength=1.8, columnspacing=1.2, handletextpad=0.4,
                       borderpad=0.0)
            for j, r in enumerate(rows)]
    # 1.5 legend line widths of extra air between the legend and the titles
    place(fig, axes, legs, leg_gap=0.02 + 1.5 * leg_lw / 72)
    for f in (out, png):
        if f is None:
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f, dpi=200)
        print(f"-> {f}")
    plt.close(fig)
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=1536, choices=(1024, 1536),
                   help="which size fills (b) and (c); the paper shows 1536 (834M)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--png", type=Path, default=None, help="also write a PNG next to this path")
    a = p.parse_args()
    stem = "lm_section" if a.dim == 1024 else "lm_section_834M"
    out = a.out or FIGS / f"{stem}.pdf"
    info = figure(a.dim, out, a.png and a.png / f"{stem}.png")
    print(json.dumps(info, indent=1, default=str))


if __name__ == "__main__":
    main()
