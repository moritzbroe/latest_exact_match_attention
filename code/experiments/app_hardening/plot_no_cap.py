"""Uncapped vs capped backward alpha: the alphas, the q/k gradient norm (log scale) and
the soft CE over training, from log.jsonl / probes.jsonl of out/lema512_h64_nocap and
the control out/lema512_h64 (the trajectory run).
Usage: plot_no_cap.py [--out out/no_cap.pdf]"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.style import paper                                    # noqa: E402


def jsonl(f: Path):
    return [json.loads(l) for l in f.read_text().splitlines() if l] if f.exists() else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=HERE / "out" / "no_cap.pdf")
    a = p.parse_args()
    paper()
    blue, red = "#08519c", "#c0392b"
    runs = [("backward cap $2$ (recipe)", HERE / "out" / "lema512_h64", blue),     # the trajectory run
            ("no backward cap", HERE / "out" / "lema512_h64_nocap", red)]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(5.2, 5.2),
                                        gridspec_kw={"height_ratios": [1, 1.4, 1.4]}, constrained_layout=True)
    last = 0
    for i, (label, run, col) in enumerate(runs):
        log, probes = jsonl(run / "log.jsonl"), jsonl(run / "probes.jsonl")
        if not log:
            continue
        st = [r["step"] / 1000 for r in log]
        last = max(last, st[-1])
        if i == 0:
            ax1.plot(st, [r["alpha"] for r in log], color="0.3", lw=1.2, label=r"forward $\alpha$")
        ax1.plot(st, [r["alpha_bwd"] for r in log], color=col, lw=1.2, ls="--",
                 label=r"backward $\alpha$, " + ("cap $2$" if i == 0 else "no cap"))
        gq = [(r["step"] / 1000, r["grad_norm_groups"]["qk"]) for r in log
              if r.get("grad_norm_groups") and "qk" in r["grad_norm_groups"]]
        ax2.plot([s for s, _ in gq], [g for _, g in gq], color=col, lw=0.9, label=label)
        if probes:
            ps = [r["step"] / 1000 for r in probes]
            ax3.plot(ps, [r["soft_ce"] for r in probes], color=col, lw=1.2, label=label)
    ax1.set_ylabel(r"$\alpha$")
    ax1.set_ylim(0, 10.8)
    ax1.set_yticks([0, 2, 5, 10])
    ax1.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               handlelength=2.2, columnspacing=1.6)
    ax2.set_yscale("log")
    ax2.set_ylabel("gradient norm,\nquery/key projections")
    ax2.legend(frameon=False, loc="lower left", handlelength=2.2)
    ax3.set_ylabel("soft cross-entropy (nats)")
    ax3.set_ylim(3.5, 4.5)
    ax3.set_xlabel("step (thousands)")
    ax3.set_xlim(0, last * 1.01)
    for ax in (ax1, ax2, ax3):
        ax.grid(True, lw=0.3, alpha=0.5)
    fig.savefig(a.out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
