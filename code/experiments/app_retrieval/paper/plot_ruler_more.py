"""The further-RULER figure of the appendix (fig:ruler-more): five tasks in five columns,
the 834M models only, RULER accuracy on top and the cross-entropy of the correct answer
below, against the context length, from out/answer_ce/summary.csv (answer_ce.py).

    python plot_ruler_more.py            # -> ../out/ruler_more.pdf
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]                                                # code/
sys.path.insert(0, str(CODE))
from experiments.app_retrieval.paper import ruler_cells as rc          # noqa: E402
from experiments.style import COLOR, HEAD_COLOR, NAME, paper           # noqa: E402

FIGS = HERE.parent / "out"
SUMMARY = HERE.parent / "out" / "answer_ce" / "summary.csv"

RUNS = ["rope1536_16k", "gdn1536_16k", "lema1536_h64"]                 # legend order
TASKS = [("niah_single_1-noise", "S-NIAH-1"),
         ("niah_single_2-essay", "S-NIAH-2"),
         ("niah_single_3-essay", "S-NIAH-3"),
         ("niah_multikey_1-essay", "MK-NIAH-1"),
         ("niah_multikey_2-needle", "MK-NIAH-2")]


def read(path=SUMMARY):
    """{cell: {run: {ctx: (accuracy or None, cross-entropy)}}}. An accuracy the csv does
    not carry -- a cell still generating when it was written -- is taken from the cell
    file if that has landed since, and is otherwise left out of the top row."""
    got, filled = {}, []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            run = r["model"]
            if run not in RUNS:
                continue
            ctx, acc = int(r["ctx"]), r["gen_match"].strip()
            if acc:
                acc = float(acc)
            else:
                f = rc.path_of(r["cell"], run, ctx)
                m = rc.meta_of(f) if f.exists() else None
                acc = float(m["match"]) if m and "match" in m else None
                filled.append(f"{r['cell']} {run}@{ctx}: "
                              + ("%.3f from the cell file" % acc if acc is not None
                                 else "no cell file, accuracy left out"))
            got.setdefault(r["cell"], {}).setdefault(run, {})[ctx] = (
                acc, float(r["answer_ce"]))
    return got, filled


def st(run):
    return rc.style(run, COLOR, HEAD_COLOR)


def main():
    paper()
    got, filled = read()
    fig, axes = plt.subplots(2, len(TASKS), figsize=(5.5, 3.0), sharex="col",
                             squeeze=False)          # the bottom row is not shared: one
                                                     # y range per task, see below
    hs, ls = [], []
    for j, (cell, title) in enumerate(TASKS):
        top, bot = axes[0][j], axes[1][j]
        ctxs = sorted({c for v in got.get(cell, {}).values() for c in v})
        hi = 0.0
        for run in RUNS:
            pts = sorted(got.get(cell, {}).get(run, {}).items())
            if not pts:
                continue
            acc = [(c, a) for c, (a, _) in pts if a is not None]
            line, = top.plot([c for c, _ in acc], [a for _, a in acc], ms=2.6, lw=1.1,
                             **st(run))
            bot.plot([c for c, _ in pts], [n for _, (_, n) in pts], ms=2.6, lw=1.1,
                     **st(run))
            hi = max(hi, max(n for _, (_, n) in pts))
            if j == 0:
                hs.append(line)
                ls.append(NAME[rc.MODELS[run][0]])
        for ax in (top, bot):
            ax.set_xscale("log", base=2)
            ax.set_xticks(ctxs)
            ax.set_xticklabels([f"{c // 1024}k" for c in ctxs], rotation=45, ha="right")
            ax.minorticks_off()
            ax.grid(True, lw=0.3, alpha=0.5)
            ax.tick_params(labelsize=6, length=2, pad=1.5)
        top.set_title(title, pad=3, fontsize=7.5)
        top.set_ylim(-0.02, 1.02)                    # the accuracy row shares one range
        top.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        if j:
            top.set_yticklabels([])
        bot.set_ylim(0, hi * 1.08)                   # the cross-entropy row does not
        bot.yaxis.set_major_locator(plt.MaxNLocator(4))
    axes[0][0].set_ylabel("accuracy", fontsize=7.5)
    axes[1][0].set_ylabel("answer cross-entropy (nats)", fontsize=7.5)

    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.7, rect=(0, 0.035, 1, 0.935))
    fig.supxlabel("context (tokens)", fontsize=7.5, y=0.005)
    fig.legend(hs, ls, frameon=False, fontsize=7, ncol=len(hs), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.4,
               handletextpad=0.4)

    FIGS.mkdir(parents=True, exist_ok=True)
    for f in (FIGS / "ruler_more.pdf",):
        fig.savefig(f, dpi=200)
        print(f"-> {f}")
    plt.close(fig)
    for m in filled:
        print("   " + m)
    for cell, title in TASKS:
        for run in RUNS:
            v = got.get(cell, {}).get(run, {}).get(16384)
            if v:
                a = "n/a" if v[0] is None else f"{v[0]:.3f}"
                print(f"   16k {title:10s} {run:14s} acc {a:>5s}  ce {v[1]:.2f}")


if __name__ == "__main__":
    main()
