"""Batched generation (appendix): throughput against context for the 0.8B geometry, one
curve per batch size, softmax with 8-fold GQA against the trained LEMA model. Reads out/genB<b>_08B_trained_ram.jsonl for
LEMA, out/gen_08B_gqa8_softmax.jsonl and out/genB<b>_08B_gqa8_softmax.jsonl for softmax. LEMA curves drawn to --max-load, softmax to the
last token that fits; both axes logarithmic since the ranges span two orders of magnitude.
A gated DeltaNet of the same geometry (out/gen_08B_gdn.jsonl, out/genB<b>_08B_gdn.jsonl from
measure_vllm.py) is drawn first, so that the LEMA line lies on top where the two coincide
(--gdn-lw-factor widens it if wanted); the curves of a method whose files are absent are
left out; the gdn curves have no limit of their own and run to the end of the axis, while a
softmax or LEMA curve that ran out of memory ends in a small 'x'.

    python plot_batched.py --out out/batched.pdf
"""
import argparse, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.style import COLOR, paper

BATCHES = [1, 4, 16, 64, 256]
SHADE = {1: 0.4, 4: 0.53, 16: 0.68, 64: 0.83, 256: 1.0}   # 1 - 0.85 * (1 - old shade)
XMIN = 50
XMAX = 250_000      # where the runs of this figure stop; the gdn curve at batch 1 is the one
                    # of the paper's figure and runs further
LW = 1.1                      # the softmax and the LEMA lines
END_TOL = 0.01                # a sweep stops one step short of --max-load, still its end
GDN_LW_FACTOR = 1.0      # the gdn band, in those line widths (--gdn-lw-factor)
KEY_LW = 2.1                  # legend keys, a touch wider so the shades read at that size

def read(path, max_load):
    if not Path(path).exists(): return [], False
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    lema = bool(recs) and recs[0]["method"] == "lema-ram"
    b = recs[0].get("batch", 1) if recs else 1
    keep = [r for r in recs if not r.get("oom") and r.get("ms_per_tok") and not (lema and (r.get("load") or 0) > max_load)]
    pts = sorted((r["context"], b / r["ms_per_tok"] * 1e3) for r in keep)
    ended = any(r.get("oom") for r in recs) or (lema and max((r.get("load") or 0) for r in recs) >= max_load - END_TOL)
    return pts, ended

def blend(hexcol, f):
    r, g, b = int(hexcol[1:3], 16), int(hexcol[3:5], 16), int(hexcol[5:7], 16)
    return tuple(1 - f * (1 - c / 255) for c in (r, g, b))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--curves-dir", default="out")
    p.add_argument("--out", default="out/batched.pdf")
    p.add_argument("--max-load", type=float, default=0.95)
    p.add_argument("--gdn-lw-factor", type=float, default=GDN_LW_FACTOR,
                   help="width of the gated DeltaNet lines as a multiple of the other lines' "
                        "(default %(default).4g)")
    a = p.parse_args()
    gdn_lw = LW * a.gdn_lw_factor
    paper()
    fig, ax = plt.subplots(figsize=(5.5, 3.67))   # the printed width, so fonts are not scaled up
    gdn = False
    xlo = XMAX                    # the axis spans exactly the range the curves cover
    for b in BATCHES:
        curves = {}
        for method, lw, col, name in (("gdn", gdn_lw, COLOR["gdn"], f"gen_08B_gdn.jsonl" if b == 1 else f"genB{b}_08B_gdn.jsonl"),
                                      ("softmax", LW, COLOR["rope"], f"gen_08B_gqa8_softmax.jsonl" if b == 1 else f"genB{b}_08B_gqa8_softmax.jsonl"),
                                      ("lema", LW, COLOR["lema"], f"genB{b}_08B_trained_ram.jsonl")):
            pts, ended = read(Path(a.curves_dir) / name, a.max_load)
            pts = [q for q in pts if XMIN <= q[0] <= XMAX]
            if not pts: print("  (missing or empty:", name, ")")
            curves[method] = [pts, ended, lw, col]
        # a gdn run has no limit of its own: its curve runs to XMAX, the end of the runs
        for method in ("gdn", "softmax", "lema"):    # the gdn band first, under the lines
            pts, ended, lw, col = curves[method]     # it coincides with
            if not pts: continue
            xs, ys = zip(*pts)
            xlo = min(xlo, xs[0])
            c = blend(col, SHADE[b])
            ax.plot(xs, ys, lw=lw, color=c)
            gdn = gdn or method == "gdn"
            # the mark belongs at the end of the run, not at an end the axis range made
            if ended and method != "gdn" and xs[-1] < XMAX:
                ax.plot([xs[-1]], [ys[-1]], marker="x", ms=4.8, mew=1.2, color=c, clip_on=False)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xlo, XMAX)        # the curves that ran to the end stop at the frame
    ax.set_xlabel("context (tokens)"); ax.set_ylabel("tokens per second")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e3:g}k" if x >= 1e3 else f"{x:g}"))
    ax.grid(True, alpha=0.25, lw=0.75, which="both"); ax.tick_params(length=3.5, pad=2)
    # Two rows of keys above the axes, so that no key sits on a curve. Row 1 names the
    # methods at full shade; row 2 gives the batch size of each shade, one key per batch
    # size showing that shade of every method's colour -- the shades are what the plot
    # uses, no colour appears in the legend that does not appear in the plot.
    cols = [COLOR["rope"]] + ([COLOR["gdn"]] if gdn else []) + [COLOR["lema"]]
    names = ["softmax with GQA"] + (["GDN"] if gdn else []) + ["LEMA"]
    lws = [KEY_LW] + ([max(gdn_lw, KEY_LW)] if gdn else []) + [KEY_LW]
    meth = [Line2D([], [], color=c, lw=w, label=n) for c, w, n in zip(cols, lws, names)]
    keys = [tuple(Line2D([], [], color=blend(c, SHADE[b]), lw=KEY_LW) for c in cols)
            for b in BATCHES]
    labels = [f"batch {b}" if i == 0 else str(b) for i, b in enumerate(BATCHES)]
    fig.legend(handles=meth, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(meth),
               frameon=False, fontsize=7.5, handlelength=1.8, columnspacing=1.4)
    fig.legend(handles=keys, labels=labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=len(BATCHES), frameon=False, fontsize=7.5, columnspacing=1.0,
               handlelength=0.9 * len(cols) + 0.3,      # every method's colour in one key
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.89))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out); print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
