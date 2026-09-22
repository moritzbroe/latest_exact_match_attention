"""Probe CE over training steps (log axis) of the direct-at-n runs, one panel per
architecture, one curve per n. Usage: plot_discovery.py [--out out/discovery.pdf]   (no GPU)"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.style import NAME, paper                              # noqa: E402


def runs():
    """{arch: {n: [(seed, probes)]}} from every out/<arch>/n<n>/s<seed>/probes.jsonl."""
    out = {}
    for f in sorted((HERE / "out").glob("*/n*/s*/probes.jsonl")):
        arch, n, seed = f.parts[-4], int(f.parts[-3][1:]), int(f.parts[-2][1:])
        recs = [json.loads(l) for l in f.read_text().splitlines() if l]
        out.setdefault(arch, {}).setdefault(n, []).append((seed, recs))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=HERE / "out" / "discovery.pdf")
    a = p.parse_args()
    data = runs()
    if not data:
        raise SystemExit("no runs under out/")
    paper()
    fig, axes = plt.subplots(1, len(data), figsize=(2.75 * len(data), 2.3), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ax, (arch, by_n) in zip(axes[0], sorted(data.items())):
        ns = sorted(by_n)
        for i, n in enumerate(ns):
            for seed, recs in by_n[n]:
                pts = [(r["step"], r["ce"]) for r in recs if r["step"] > 0]
                ax.plot([s for s, _ in pts], [c for _, c in pts],
                        color=cmap(i / max(len(ns) - 1, 1)), lw=1.0,
                        label=f"$n={n}$" if seed == by_n[n][0][0] else None)
        ax.set_title(NAME[arch])
        ax.set_xscale("log")
        ax.set_xlabel("step")
        ax.set_ylim(bottom=0)
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.legend(frameon=False)
    axes[0, 0].set_ylabel("held-out CE")
    fig.tight_layout(pad=0.3)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
