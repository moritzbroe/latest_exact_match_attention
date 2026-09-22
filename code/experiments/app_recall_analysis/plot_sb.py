"""The appendix figure on stick-breaking attention on the recall task.

    python plot_sb.py [--out out/recall_sb.pdf]

Left: accuracy against the number of pairs of the three plain stick-breaking runs trained
at n=8 (../main_recall/out/eval/sb_s*.json), with the three LEMA seeds for reference.
Right: the same checkpoints with every gate below tau zeroed at evaluation
(out/sb_threshold/sb_s*.json, sb_threshold.py), one curve per tau, averaged over the
seeds. No GPU.
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
from experiments.main_recall.plot_recall import STYLE, style           # noqa: E402

EVAL = HERE.parent / "main_recall" / "out" / "eval"
SB_COLOR = "#7570b3"          # as in the LM stick-breaking ablation
TAU_SHADES = ["#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#3f007d"]


def read(path):
    r = json.loads(Path(path).read_text())
    return {k: ({int(n): v for n, v in c.items()} if isinstance(c, dict) else c)
            for k, c in r.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=HERE / "out" / "recall_sb.pdf")
    a = p.parse_args()
    paper()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.4))

    sb = {int(f.stem.rsplit("_s", 1)[1]): read(f) for f in sorted(EVAL.glob("sb_s*.json"))}
    lema = {int(f.stem.rsplit("_s", 1)[1]): read(f) for f in sorted(EVAL.glob("lema_s*.json"))}
    ns = sorted(next(iter(sb.values()))["curve"]) if sb else []
    style(ax1)
    for i, (seed, r) in enumerate(sorted(lema.items())):
        ax1.plot(ns, [r["curve"][n] for n in ns], label=NAME["lema"] if i == 0 else None,
                 **dict(STYLE["lema"], lw=1.1, ms=3.0, alpha=0.85))
    for i, (seed, r) in enumerate(sorted(sb.items())):
        ax1.plot(ns, [r["curve"][n] for n in ns], color=SB_COLOR, marker="D", ms=3.0, lw=1.2,
                 label=NAME["sb"] if i == 0 else None)
        print(f"sb s{seed} " + " ".join(f"{n}:{r['curve'][n]:.3f}" for n in ns))
    ax1.legend(loc="lower left", fontsize=6, frameon=True, framealpha=1.0, edgecolor="0.8",
               fancybox=False)

    style(ax2)
    thr = [json.loads(f.read_text()) for f in sorted((HERE / "out" / "sb_threshold").glob("sb_s*.json"))]
    if thr:
        taus = sorted(float(t) for t in thr[0])
        for i, tau in enumerate(taus):
            curves = np.array([[r[str(tau)][str(n)] for n in ns] for r in thr])
            m = curves.mean(0)
            ax2.plot(ns, m, color=TAU_SHADES[i % len(TAU_SHADES)], marker="D", ms=3.0, lw=1.2,
                     label=r"$\tau=%g$" % tau)
            print(f"tau {tau:<5} " + " ".join(f"{n}:{v:.3f}" for n, v in zip(ns, m)))
        ax2.legend(loc="lower left", fontsize=6, frameon=True, framealpha=1.0, edgecolor="0.8",
                   fancybox=False, ncol=3, columnspacing=0.8, handlelength=1.4)
    else:
        ax2.text(0.5, 0.5, "run sb_threshold.py", ha="center", transform=ax2.transAxes)
    ax2.set_ylabel("")
    fig.tight_layout(pad=0.3, w_pad=1.0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
