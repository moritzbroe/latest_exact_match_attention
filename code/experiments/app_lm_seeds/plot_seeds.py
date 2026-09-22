"""The appendix's seed figure: repeated-bigram recall of three seeds at d=256 and d=512.

    python plot_seeds.py --out out/lm_recall_seeds.pdf

Two panels, one per model dimension, with the buckets, colours and markers of
main_lm_bigrams/plot_sizes_grid.py (fig:lm-recall): the mean cross-entropy on the second
token of a repeated rare bigram against the distance to its previous occurrence. One colour
per architecture and one line style per seed, solid for seed 0, the run of the other figures.
Softmax and GDN are the context-extended _16k checkpoints, LEMA its own, as there. The
replacement baselines of fig:lm-recall are left out, nine curves per panel being enough.

Reads main_lm_bigrams/out/{targets,scores}. No GPU.
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
from experiments.style import COLOR, NAME, paper                       # noqa: E402
from experiments.main_lm_bigrams.plot_sizes_grid import bucket_label, nice_top  # noqa: E402

BIG = HERE.parent / "main_lm_bigrams" / "out"
EDGES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
SIZES = [(256, "$d=256$"), (512, "$d=512$")]
SEEDS = (0, 1, 2)
SEED_LS = {0: "-", 1: (0, (4, 1.6)), 2: (0, (1.2, 1.4))}       # solid, dashed, dotted
MARK = {"rope": "s", "gdn": "^", "lema": "o"}


def run_name(arch, dim, seed):
    """The run as main_lm/train_lm.py and finetune_ctx.py name it."""
    s = "" if seed == 0 else f"_s{seed}"
    return f"lema{dim}_h64{s}" if arch == "lema" else f"{arch}{dim}{s}_16k"


def per_bucket(path, cells, key="loss"):
    z = np.load(path)
    v = z[key].astype(np.float64)
    return np.array([v[m].mean() for m in cells])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--freq", type=int, default=100)
    p.add_argument("--out", type=Path, default=HERE / "out" / "lm_recall_seeds.pdf")
    a = p.parse_args()

    d = np.load(BIG / f"targets_T{a.ctx}_f{a.freq}.npz")
    dist, fp = d["dist"], int(d["fingerprint"])
    lo_hi = list(zip(EDGES[:-1], EDGES[1:]))
    mid = [np.sqrt(lo * hi) for lo, hi in lo_hi]
    cells = [(dist >= lo) & (dist < hi) for lo, hi in lo_hi]

    paper()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharex=True)
    top = 0.0
    for ax, (dim, label) in zip(axes, SIZES):
        for arch in ("rope", "gdn", "lema"):
            for seed in SEEDS:
                run = run_name(arch, dim, seed)
                c = means(run, a.ctx, a.freq, bounds=lo_hi)
                if c is None:
                    print(f"  (missing scores for {run})")
                    continue
                ax.plot(mid, c, color=COLOR[arch], ls=SEED_LS[seed], marker=MARK[arch],
                        ms=2.8, lw=1.3, zorder=3)
                top = max(top, c.max())
                print(f"{label:>4} {run:<20} " + "  ".join(f"{v:5.2f}" for v in c))
        ax.set_title(label, fontsize=7.5, pad=2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(mid)
        ax.set_xticklabels([bucket_label(lo, hi) for lo, hi in lo_hi],
                           rotation=45, ha="right", fontsize=5.5)
        ax.minorticks_off()
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.tick_params(labelsize=6, length=2, pad=1.5)
        ax.set_xlabel("distance (tokens)", fontsize=7)
    for ax in axes:
        ax.set_ylim(0, nice_top(top))
    axes[0].set_ylabel("cross-entropy (nats)", fontsize=7)
    axes[1].tick_params(labelleft=False)

    # one key per curve, the three seeds of an architecture under each other
    keys = [Line2D([], [], color=COLOR[k], ls=SEED_LS[s], marker=MARK[k], ms=2.8, lw=1.3,
                   label=f"{NAME[k]} s{s}") for k in ("rope", "gdn", "lema") for s in SEEDS]
    fig.legend(keys, [h.get_label() for h in keys], frameon=False, fontsize=6.5, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 1.015), handlelength=2.6,
               columnspacing=1.6, handletextpad=0.5)
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.86))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
