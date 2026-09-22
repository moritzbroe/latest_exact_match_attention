"""The trained 0.8B model against its worst-case curves (appendix): time per generated
token against context length, and prefill time against prompt length, from --checkpoint
runs and the random-weight runs of the same geometry.

    python plot_trained.py [--worst out/gen_08B_ram.jsonl] [--trained out/gen_08B_ram_trained.jsonl]
                           [--worst-prefill out/prefill_08B_ram.jsonl]
                           [--trained-prefill out/prefill_08B_ram_trained.jsonl]
                           [--out out/trained_vs_worst.pdf]

Left: ms per token, both curves (the trained model samples its own text at temperature 1,
its codes are whatever it emits). Right: prefill time, both curves (the trained model on
held-out text, the worst case on random tokens with random codes). The random-code curve
ends in a small 'x' where its table reaches MAX_LOAD occupancy, the trained curve has room
to spare and runs to the end of the axis. No GPU.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.style import COLOR, paper                             # noqa: E402


MAX_LOAD = 0.95         # the occupancy the runs of this figure fill the table to
END_TOL = 0.01          # a prefill sweep stops one step short of it, still its end


def curve(path, ykey):
    """The (context, y) points and whether the run ended with a full table."""
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    pts = sorted((r["context"], r[ykey]) for r in recs if r.get(ykey) and not r.get("oom"))
    loads = [r.get("load") or 0 for r in recs]
    return pts, (max(loads, default=0) >= MAX_LOAD - END_TOL
                 or any(r.get("oom") for r in recs))


def _tokens(x, _):
    return f"{x/1e6:g}M" if x >= 1e6 else f"{x/1e3:g}k" if x >= 1e3 else f"{x:g}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worst", type=Path, default=HERE / "out" / "gen_08B_ram.jsonl")
    p.add_argument("--trained", type=Path, default=HERE / "out" / "gen_08B_ram_trained.jsonl")
    p.add_argument("--worst-prefill", type=Path, default=HERE / "out" / "prefill_08B_ram.jsonl")
    p.add_argument("--trained-prefill", type=Path,
                   default=HERE / "out" / "prefill_08B_ram_trained.jsonl")
    p.add_argument("--out", type=Path, default=HERE / "out" / "trained_vs_worst.pdf")
    a = p.parse_args()
    paper()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.1))
    panels = [(ax1, "ms_per_tok", a.worst, a.trained),
              (ax2, "seconds", a.worst_prefill, a.trained_prefill)]
    for ax, ykey, worst, trained in panels:
        last = {}
        for path, name, ls in ((worst, "random codes", (0, (3.2, 1.8))),
                               (trained, "trained model", "-")):
            if not path.exists():
                print(f"  (missing {path})")
                continue
            pts, ended = curve(path, ykey)
            if not pts:
                print(f"  (no usable records in {path})")
                continue
            ax.plot([q[0] for q in pts], [q[1] for q in pts], ls=ls, lw=1.3,
                    color=COLOR["lema"], label=name)
            last[name] = pts[-1][0]
            if ended:
                ax.plot([pts[-1][0]], [pts[-1][1]], marker="x", ms=4, mew=1.1, zorder=3,
                        color=COLOR["lema"], clip_on=False)
        # the axis ends where the trained curve does, with room for the mark as long as the
        # random-code run stops at the same context as the trained one
        xhi = max(last.values(), default=1)
        ax.set_xlim(0, xhi * (1.04 if last.get("random codes", 0) >= 0.98 * xhi else 1.0))
    ax1.set_ylabel("time per token (ms)")
    ax2.set_ylabel("prefill time (s)")
    for ax in (ax1, ax2):
        ax.set_xlabel("context (tokens)")
        ax.xaxis.set_major_formatter(FuncFormatter(_tokens))
        ax.set_ylim(bottom=0)
        ax.grid(True, lw=0.3, alpha=0.5)
    # the worst case ends on a steep rise, and the autoscale margin is a fraction of the
    # data range, a sliver of an axis that starts at 0: its last point would sit on the
    # panel's top edge and the curve would read as cut off there
    ax1.set_ylim(0, 4.5)
    ax1.legend(frameon=False)
    fig.tight_layout(pad=0.4)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
