"""The paper's inference figure: columns are model sizes, rows are generation and prefill.

    python plot_paper.py --out out/inference.pdf [--curves-dir out]

Reads gen_<tag>_{softmax,gdn,ram}.jsonl and prefill_<tag>_{softmax,gdn,ram}.jsonl for the tags
in SIZES (softmax and gdn from measure_vllm.py, ram from measure_generation.py /
measure_prefill.py); a method whose files are absent is left out of the figure and of its
legend, so the two-curve figure is exactly the figure without any gdn file.
The gated DeltaNet curve is drawn first, so that the LEMA line lies on top where the two
coincide (--gdn-lw-factor widens the gated DeltaNet line if wanted).
Each column has its own context range (the sizes span very different ones: the 0.8B softmax
cache holds 150k tokens, the 8B one 12k), shared between its two rows so the
prefill curve below ends where the generation curve above does. Linear axes: a softmax
generation curve is a rising line and a LEMA one flat; a softmax prefill curve is a
parabola and a LEMA one a line. A curve that ran out of memory ends in a small 'x': a softmax
curve where the next token no longer fits in VRAM, a LEMA curve where its hash table reaches
--max-load (default 0.95, the last occupancy the measurements cover). A gated DeltaNet has no
limit of its own, so it carries no mark: its runs go --gdn-past times as far as the LEMA curve
of that size, which is also where the column's axis ends, so its curve runs past the LEMA mark
and stops exactly at the frame.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # code/
from experiments.style import COLOR, NAME, paper                       # noqa: E402

SIZES = [("08B", "0.8B"), ("3B", "3B"), ("8B", "8B")]
ROWS = [("gen", "ms_per_tok", "time per token (ms)"),
        ("prefill", "seconds", "prefill time (s)")]
LW = 1.3                              # the softmax and the LEMA line
GDN_LW_FACTOR = 1.0              # the gdn band, in those line widths (--gdn-lw-factor)
DRAW = ["gdn", "ram", "softmax"]      # back to front: the gdn band under the LEMA line it
                                      # coincides with, softmax on top
END_TOL = 0.01          # a prefill sweep stops one step short of --max-load, still its end
GDN_PAST = 1.15         # gdn curve and axis end, in LEMA ends (--gdn-past)


def methods(gdn_lw):
    """(file tag, method as the records name it, legend label, colour, line width) per
    method, with the gated DeltaNet line at `gdn_lw`."""
    return [("softmax", "softmax", NAME["rope"], COLOR["rope"], LW),
            ("gdn", "gdn", NAME["gdn"], COLOR["gdn"], gdn_lw),
            ("ram", "lema-ram", NAME["lema"], COLOR["lema"], LW)]


def read_curve(path, ykey, max_load):
    """The curve's (context, y) points and whether it ran out of memory (VRAM full for
    softmax, `max_load` occupancy for LEMA); LEMA records beyond `max_load` are dropped."""
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    lema = bool(recs) and recs[0]["method"].startswith("lema")
    keep = [r for r in recs if not r.get("oom") and r.get(ykey)
            and not (lema and (r.get("load") or 0) > max_load)]
    pts = sorted((r["context"], r[ykey]) for r in keep)
    ended = any(r.get("oom") for r in recs) or (
        lema and max((r.get("load") or 0) for r in recs) >= max_load - END_TOL)
    return pts, ended


def _tokens(x, _):
    return f"{x/1e6:g}M" if x >= 1e6 else f"{x/1e3:g}k" if x >= 1e3 else f"{x:g}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--curves-dir", default="out")
    p.add_argument("--out", default="out/inference.pdf")
    p.add_argument("--size", type=float, nargs=2, default=(5.5, 3.1), metavar=("W", "H"))
    p.add_argument("--max-load", type=float, default=0.95,
                   help="draw LEMA curves up to this hash-table occupancy")
    p.add_argument("--gdn-past", type=float, default=GDN_PAST,
                   help="where the gated DeltaNet curve and the axis end, as a multiple of "
                        "the LEMA curve's end (default %(default).4g; the runs reach 1.2)")
    p.add_argument("--gdn-lw-factor", type=float, default=GDN_LW_FACTOR,
                   help="width of the gated DeltaNet line as a multiple of the other lines' "
                        "(default %(default).4g, the band of the paper's figure)")
    a = p.parse_args()

    meths = methods(LW * a.gdn_lw_factor)
    by_tag = {m[0]: m for m in meths}
    paper()
    # the two rows are shorter than the axis labels are long: "time per token (ms)" at 8pt
    # runs into "prefill time (s)" below it, so the axis labels go down to the tick size
    plt.rcParams["axes.labelsize"] = 7
    fig, axes = plt.subplots(len(ROWS), len(SIZES), figsize=tuple(a.size))
    drawn = set()
    for j, (tag, label) in enumerate(SIZES):
        axes[1, j].sharex(axes[0, j])
        # both rows are read before either is drawn: the column's axis ends at --gdn-past
        # times the LEMA end, the gated DeltaNet curves are cut there, and the two rows
        # share that limit, so the prefill curve below ends where the generation curve above
        # does and every gated DeltaNet curve ends at the frame
        col = {}
        for i, (kind, ykey, _) in enumerate(ROWS):
            for mtag, _, _, colour, lw in (by_tag[t] for t in DRAW):
                path = Path(a.curves_dir) / f"{kind}_{tag}_{mtag}.jsonl"
                if not path.exists():
                    print(f"  (missing {path})")
                    continue
                pts, ended = read_curve(path, ykey, a.max_load)
                if pts:
                    col[i, mtag] = (pts, ended, colour, lw)
        lema_end = max((v[0][-1][0] for (i, m), v in col.items() if m == "ram"), default=0)
        xhi = (a.gdn_past * lema_end
               or max((v[0][-1][0] for v in col.values()), default=0) * 1.10 or 1)
        for i, (kind, ykey, ylabel) in enumerate(ROWS):
            ax = axes[i, j]
            ends = []
            for mtag in DRAW:
                if (i, mtag) not in col:
                    continue
                pts, ended, colour, lw = col[i, mtag]
                if mtag == "gdn":
                    pts = [q for q in pts if q[0] <= xhi]
                xs, ys = zip(*pts)
                ax.plot(xs, ys, lw=lw, color=colour)
                drawn.add(mtag)
                if ended and mtag != "gdn":
                    ends.append((xs[-1], ys[-1], colour))
            for x, y, colour in ends:       # after the lines, so no line runs over a mark
                ax.plot([x], [y], marker="x", ms=4, mew=1.1, color=colour, clip_on=False)
            ax.set_ylim(bottom=0)
            ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(True, alpha=0.25, lw=0.5)
            ax.tick_params(length=2.5, pad=2)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(label, pad=4)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("context (tokens)")
        axes[0, j].set_xlim(0, xhi)
        axes[1, j].xaxis.set_major_formatter(FuncFormatter(_tokens))
        axes[1, j].xaxis.set_major_locator(MaxNLocator(4))

    handles = [Line2D([], [], color=c, lw=max(lw, 1.6), label=name)
               for mtag, _, name, c, lw in meths if mtag in drawn or not drawn]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.0), columnspacing=2.5 if len(handles) < 3 else 1.2,
               handlelength=2.4)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.8, w_pad=1.0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
