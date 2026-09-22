"""Attention-flow picture of one recall sample, for a LEMA or softmax checkpoint.

    plot_attention.py <model_best.pt or run dir> --n 4 [--seed 0] [--width-power 0.5]

Top row: the input tokens. Bottom row: the model's prediction at every position after SEP, with its
probability. Dotted verticals: each position's residual stream. One band per layer:
position i attending to position j is a line from j at the top of the band to i at its
bottom, i.e. the information flow through that layer; width and opacity scale as
weight ** --width-power. LEMA weights are 0 or 1. `draw(ax, ...)` draws into any axes,
so the main text's figure can place the picture next to other panels.
Writes out/attention/<arch>_<seed>_n<n>.{pdf,png}. Runs on CPU.

    plot_attention.py --grid [--seeds 0 1 2] [--n 4] [--seed 0]

draws every seed of the LEMA (left) and softmax (right) model on the same sample,
one row per seed, in the main figure's compact style -> out/attention/seeds.pdf.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.style import COLOR, NAME, paper                       # noqa: E402

OUT_DIR = HERE / "out" / "attention"


def weights(model, x):
    """Per-layer `[T, T]` attention weights of the single head, and the logits."""
    import torch
    from lema.analysis import trace
    logits, recs = trace(model, x)
    ws = []
    for r in recs:
        if "w" in r:
            ws.append(r["w"][0, 0].cpu())
        else:
            src = r["src"][0, 0]
            w = torch.zeros(len(src), len(src))
            i = torch.arange(len(src))[src.cpu() >= 0]
            w[i, src.cpu()[src.cpu() >= 0]] = 1.0
            ws.append(w)
    return ws, logits[0].float().cpu()


def labeller(x, short=False):
    """Token -> label. `short` uses the paper's notation: the i-th pair is a<i> b<i> in
    order of appearance (the indices are positions, not vocabulary ids), repeated tokens
    keep their labels after SEP, as do the predicted tokens."""
    n = (len(x) - 1) // 3
    key_vocab = int(x[2 * n]) // 2
    idx = {int(x[2 * i]): i + 1 for i in range(n)} if short else {}
    idx.update({int(x[2 * i + 1]): i + 1 for i in range(n)} if short else {})

    def label(tok):
        if tok >= 2 * key_vocab:
            return "SEP"
        kind = "a" if tok < key_vocab else "b"
        if short:
            return f"{kind}{idx[tok]}" if tok in idx else f"{kind}?"
        return f"{kind}{tok if kind == 'a' else tok - key_vocab}"
    return label


def draw(ax, x, ws, probs, targets, power, fontsize=8, short=False, color="0.2"):
    """The picture for input tokens `x`, per-layer weights `ws`, the prediction
    probabilities `probs` at the positions after SEP and their `targets`, into `ax`.
    Every layer is a band; position i attending to j is a line from j at the top of the
    band to i at its bottom, with width and opacity by weight ** `power`. The output row
    shows the predicted token at each position after SEP, green when correct."""
    T = len(x)
    n = (T - 1) // 3
    L = len(ws)
    label = labeller(x, short)
    top, bot = 0.90, 0.10                        # the bands fill this range
    gap = 0.06
    band = (top - bot - gap * (L - 1)) / L
    for i in range(T):
        ax.plot([i, i], [bot - 0.02, top + 0.02], color="0.85", lw=0.5, zorder=1)
    for li, w in enumerate(ws):
        b_top = top - li * (band + gap)
        b_bot = b_top - band
        for i in range(T):
            for j in range(i + 1):
                a = float(w[i, j]) ** power
                if a > 0.02:
                    ax.plot([j, i], [b_top, b_bot], color=color, lw=1.3 * a, alpha=a,
                            solid_capstyle="round", zorder=2)
        ax.text(-0.7, (b_top + b_bot) / 2, f"layer {li + 1}", ha="right", va="center",
                fontsize=fontsize, color="0.3")
    ax.text(-0.7, 0.97, "input", ha="right", va="center", fontsize=fontsize, color="0.3")
    ax.text(-0.7, 0.03, "output", ha="right", va="center", fontsize=fontsize, color="0.3")
    for i, tok in enumerate(x):
        ax.text(i, 0.97, label(tok), ha="center", va="center", fontsize=fontsize,
                rotation=90 if T > 24 and not short else 0)
    for k in range(n):
        i = 2 * n + 1 + k
        pred = int(probs[k].argmax())
        ok = pred == int(targets[k])
        ax.text(i, 0.03, label(pred), ha="center", va="center", fontsize=fontsize,
                color="#2a9d5c" if ok else "#c0392b", fontweight="bold")
    ax.set_xlim(-1.6, T - 0.4)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")


def cached_panel(ax, data, arch, seed, power=0.5, fontsize=6.5):
    """Render recorded attention weights and predictions without importing torch."""
    matches = [p for p in data["panels"] if p["arch"] == arch and p["seed"] == seed]
    if len(matches) != 1:
        raise ValueError(f"expected one saved panel for {arch} seed {seed}")
    panel = matches[0]
    if "probs" in panel:
        probs = np.asarray(panel["probs"])
    else:
        predictions = np.asarray(panel["predictions"], dtype=int)
        probs = np.zeros((len(predictions), int(predictions.max()) + 1))
        probs[np.arange(len(predictions)), predictions] = 1
    draw(ax, panel["x"], np.asarray(panel["weights"]), probs,
         np.asarray(panel["targets"]), power, fontsize=fontsize,
         short=True, color=COLOR[arch])


def grid(seeds, n, sample_seed, power, out, data=None):
    """The appendix figure: all seeds of both architectures on one sample."""
    if data is None:
        import torch
        from experiments.main_recall.eval_recall import load
        from experiments.main_recall.train_recall import task_at
        x, y, _, _ = next(task_at(n).batches("val", 1, seed=sample_seed))
        q = torch.arange(2 * n + 1, 3 * n + 1)
    fig, axes = plt.subplots(len(seeds), 2, figsize=(5.5, 1.45 * len(seeds)),
                             layout="constrained", squeeze=False)
    for row, s in enumerate(seeds):
        for col, arch in enumerate(("lema", "rope")):
            ck = HERE / "out" / arch / f"s{s}" / ("" if arch == "lema" else f"n{n}")
            ax = axes[row, col]
            if data is not None:
                cached_panel(ax, data, arch, s, power, fontsize=6.5)
            else:
                model = load(ck, device="cpu")
                ws, logits = weights(model, x)
                draw(ax, x[0].tolist(), ws, torch.softmax(logits[q], -1), y[0, q], power,
                     fontsize=6.5, short=True, color=COLOR[arch])
            ax.set_title(f"{NAME[arch]}, seed {s}", fontsize=8, pad=1, color=COLOR[arch])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"-> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", nargs="?", help="model_best.pt or a run directory; bare names "
                                           "resolve under out/, e.g. lema/s0 or rope/s0/n4")
    p.add_argument("--attention-data", type=Path, default=OUT_DIR / "panels.json",
                   help="saved panels for --grid; no torch needed")
    p.add_argument("--grid", action="store_true", help="all seeds of both models, one figure")
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--out", type=Path, default=None, help="--grid: output file "
                                                          "(default out/attention/seeds.pdf)")
    p.add_argument("--n", type=int, default=4, help="number of associations")
    p.add_argument("--seed", type=int, default=0, help="which sample of the val stream")
    p.add_argument("--width-power", type=float, default=0.5,
                   help="arc width and opacity scale with weight ** this")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    paper()
    if a.grid:
        grid(a.seeds, a.n, a.seed, a.width_power, a.out or OUT_DIR / "seeds.pdf",
             json.loads(a.attention_data.read_text()) if a.attention_data.exists() else None)
        return
    if a.ckpt is None:
        p.error("a checkpoint is required unless --grid is given")

    import torch
    from experiments.main_recall.eval_recall import load
    from experiments.main_recall.train_recall import task_at
    a.device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(a.ckpt) if Path(a.ckpt).exists() else HERE / "out" / a.ckpt
    model = load(ckpt, device=a.device)
    assert model.cfg.num_heads == 1, "the picture assumes one head per layer"
    arch = "lema" if model.cfg.attn == "lema" else "rope"
    seed = next((s for s in ckpt.resolve().parts if re.fullmatch(r"s\d+", s)), "s")
    x, y, mask, _ = next(task_at(a.n).batches("val", 1, seed=a.seed))
    ws, logits = weights(model, x.to(a.device))
    q = torch.arange(2 * a.n + 1, 3 * a.n + 1)
    probs = torch.softmax(logits[q], -1)
    correct = int((probs.argmax(-1) == y[0, q]).sum())
    name = {"lema": "LEMA", "rope": "softmax"}[arch]
    title = f"{name}, {seed}, {a.n} pairs: {correct}/{a.n} predictions correct"
    T = x.shape[1]
    fig, ax = plt.subplots(figsize=(max(5.5, 0.42 * T), 1.0 + 0.9 * len(ws)))
    draw(ax, x[0].tolist(), ws, probs, y[0, q], a.width_power, short=True,
         color=COLOR[arch])
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{arch}_{seed}_n{a.n}"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"{title}\n-> {out}.pdf / .png")


if __name__ == "__main__":
    main()
