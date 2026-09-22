"""Schedule and losses of a hardening run in one figure: top panel c and the forward /
backward alpha, bottom panel soft and hard CE, shared step axis. The bottom panel also
carries the hard CE of the matching run without hardening, when that run is on disk.
Usage: plot_trajectory.py [out/lema512_h64] [--out out/trajectory.pdf]
                          [--nohard out/lema512_h64_nohard | --nohard ""]"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.style import paper                                    # noqa: E402


def ce_axis(ax, lo=3.4, hi=12.0, ticks=(3.5, 4, 5, 6, 8, 10)):
    """Log CE axis that shows the whole curve, with plain-number ticks."""
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(ScalarFormatter())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", type=Path, default=HERE / "out" / "lema512_h64")
    p.add_argument("--out", type=Path, default=HERE / "out" / "trajectory.pdf")
    p.add_argument("--nohard", type=str, default=str(HERE / "out" / "lema512_h64_nohard"),
                   help="run trained without hardening, its hard CE joins the bottom "
                        "panel; pass an empty string to leave it out")
    a = p.parse_args()
    paper()
    recs = [json.loads(l) for l in (a.run / "probes.jsonl").read_text().splitlines() if l]
    d_qk = json.loads((a.run / "config.json").read_text())["model"]["d_qk"]
    step = [r["step"] / 1000 for r in recs]
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5.2, 3.9),
                                   gridspec_kw={"height_ratios": [1, 1.7]}, constrained_layout=True)
    blue, orange = "#08519c", "#d95f02"
    ax1.plot(step, [r["alpha"] for r in recs], color=blue, lw=1.2, label=r"$\alpha$ (forward)")
    ax1.plot(step, [r["alpha_bwd"] for r in recs], color=blue, lw=1.2, ls="--", label=r"$\alpha$ (backward)")
    ax1.set_ylabel(r"$\alpha$", color=blue)
    ax1.set_ylim(0, 10.8)
    ax1.set_yticks([0, 2, 5, 10])
    ax1c = ax1.twinx()
    ax1c.plot(step, [r["c"] for r in recs], color=orange, lw=1.2, label="$c$")
    ax1c.set_ylabel("$c$", color=orange)
    ax1c.set_ylim(0, d_qk * 1.08)
    ax1c.set_yticks([0, d_qk // 2, d_qk - 1])
    ax1c.spines["top"].set_visible(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1c.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.6)
    ax2.plot(step, [r["soft_ce"] for r in recs], color=orange, lw=1.2, label="soft: surrogate at the current $\\alpha$")
    ax2.plot(step, [r["hard_ce"] for r in recs], color=blue, lw=1.2, label="hard: exact latest match")
    nh = Path(a.nohard) / "probes.jsonl" if a.nohard else None
    if nh is not None and nh.exists():
        nrec = [json.loads(l) for l in nh.read_text().splitlines() if l]
        ax2.plot([r["step"] / 1000 for r in nrec], [r["hard_ce"] for r in nrec],
                 color="#737373", lw=1.2, label="without hardening")
    elif nh is not None:
        print(f"  (no probes at {nh}, drawing without that curve)")
    ce_axis(ax2, hi=8.0)
    ax2.set_ylabel("cross-entropy (nats)")
    ax2.set_xlabel("step (thousands)")
    ax2.set_xlim(0, step[-1] * 1.01)
    ax2.legend(frameon=False, loc="upper right", handlelength=2.2)
    for ax in (ax1, ax2):
        ax.grid(True, lw=0.3, alpha=0.5)
    fig.savefig(a.out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
