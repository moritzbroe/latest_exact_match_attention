"""The examples of examples.py as a figure: one line per bigram, the trigger and the token
after it highlighted, the hidden stretch between occurrences as a token count.

    python examples_figure.py [--src out/examples/bucket_8192_16384_f100.txt] [--out out/recall_examples.pdf]

Reads the text file examples.py writes, so no corpus or tokenizer is needed. The targets of
the analysis, where the trigger does not recur between the two occurrences, so the token
after it can only come from the earlier occurrence; with --distracted also the targets that
Zoology's definition keeps and ours drops, where the trigger recurs in between with another
continuation, its last occurrence shown as the middle span. Highlights: the trigger
in blue wherever it appears, the token after the scored occurrence and the same token after
the earlier occurrence in orange. Monospace, so a run's width is its character count.
"""
import argparse
import ast
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams["pdf.fonttype"] = 42          # embed TrueType, not Type 3

HERE = Path(__file__).resolve().parent
TRIGGER, TARGET, GAP, TEXT = "#cfe3f7", "#fbd3a6", "0.55", "0.15"
SEP = re.compile(r"\]\.\.\.\((\d+)\)\.\.\.\[")


def parse(path):
    """[(section title, [(trigger, target, [span text ...], [gap ...])])]"""
    sections, cur = [], None
    for line in Path(path).read_text().splitlines():
        if line.startswith("### "):
            cur = (line[4:].strip(), [])
            sections.append(cur)
        elif line and not line.startswith("#") and cur is not None:
            head, chain = line.split(" :: ", 1)
            trig, targ = (ast.literal_eval(t) for t in head.split(" -> ", 1))
            gaps = [int(g) for g in SEP.findall(chain)]
            spans = [s for s in SEP.split(chain) if not s.isdigit()]
            spans[0] = spans[0][1:]                              # the outer brackets
            spans[-1] = spans[-1][:-1]
            # a span ends at a token boundary, which can cut a character in half; the
            # tokenizer decodes the leftover bytes to U+FFFD, which every font draws as
            # its replacement mark. There is no character to show, so drop the mark.
            spans = [sp.strip(" ").replace("�", "") for sp in spans]
            cur[1].append((trig, targ, spans, gaps))
    return sections


def runs_of(span, trig, targ, scored):
    """Split a span into (text, kind) runs: the trigger, the trigger followed by the
    target, and plain text. In the earlier and the scored occurrence the pair trigger+target
    is what to mark; a distractor span only carries the trigger."""
    out, i = [], 0
    pair = trig + targ
    while i < len(span):
        j = span.find(pair, i) if scored else -1
        k = span.find(trig, i)
        if scored and j >= 0 and (k < 0 or j <= k):
            out += [(span[i:j], "text"), (trig, "trigger"), (targ, "target")]
            i = j + len(pair)
        elif k >= 0:
            out += [(span[i:k], "text"), (trig, "trigger")]
            i = k + len(trig)
        else:
            out.append((span[i:], "text"))
            break
    return [r for r in out if r[0]]


def draw(ax, sections, fs):
    cw = fs * 0.602 / 72                        # DejaVu Sans Mono advance, inches
    lh = fs * 1.55 / 72
    y = 0.0
    for title, rows in sections:
        ax.text(0, -y, title, fontsize=fs + 0.5, family="sans-serif", weight="bold",
                color=TEXT, va="top", ha="left")
        y += lh * 1.35
        for trig, targ, spans, gaps in rows:
            x = 0.0
            items = []
            for n, span in enumerate(spans):
                last = n == len(spans) - 1
                # the earlier occurrence (first span) and the scored one (last span) carry
                # trigger+target; a merged final span carries both a distractor and the target
                items += runs_of(span, trig, targ, scored=(n == 0 or last))
                if not last:
                    items.append((f" …({gaps[n]})… ", "gap"))
            for text, kind in items:
                w = len(text) * cw
                if kind in ("trigger", "target"):
                    ax.add_patch(Rectangle((x, -y - lh * 0.82), w, lh * 0.95, lw=0,
                                           facecolor=TRIGGER if kind == "trigger" else TARGET,
                                           zorder=1))
                ax.text(x, -y, text, fontsize=fs, family="DejaVu Sans Mono", va="top",
                        ha="left", color=GAP if kind == "gap" else TEXT, zorder=2)
                x += w
            y += lh
        y += lh * 0.6
    ax.set_xlim(0, 5.5)
    ax.set_ylim(-y, 0.0)
    ax.axis("off")
    return y


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=HERE / "out" / "examples" / "bucket_8192_16384_f100.txt")
    p.add_argument("--out", type=Path, default=HERE / "out" / "recall_examples.pdf")
    p.add_argument("--distracted", action="store_true",
                   help="also draw the block of targets that Zoology's definition keeps and ours drops")
    p.add_argument("--fontsize", type=float, default=5.6,
                   help="upper bound; lowered so that the longest line fits the width")
    a = p.parse_args()
    sections = parse(a.src)
    if not a.distracted:
        sections = [sec for sec in sections if not sec[0].startswith("DISTRACTED")]
    longest = max(sum(len(sp) for sp in spans) + sum(len(f" …({g})… ") for g in gaps)
                  for _, rows in sections for _, _, spans, gaps in rows)
    fs = min(a.fontsize, 5.5 * 72 / (0.602 * longest) * 0.995)
    titles = {"UNDISTRACTED": "targets of the analysis: the trigger does not recur in between",
              "DISTRACTED": "dropped by our definition, kept by Arora et al. (2024): "
                            "the trigger recurs in between"}
    sections = [(titles.get(t.split(" ")[0], t) + f"  ({t.split('(')[1]}", r) for t, r in sections]
    fig = plt.figure(figsize=(5.5, 1.0))
    ax = fig.add_axes([0, 0, 1, 1])
    h = draw(ax, sections, fs)
    fig.set_size_inches(5.5, h + 0.02)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print(f"-> {a.out}  ({h:.2f} in tall, {fs:.1f} pt, longest line {longest} characters)")


if __name__ == "__main__":
    main()
