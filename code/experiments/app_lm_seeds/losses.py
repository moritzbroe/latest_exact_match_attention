"""The appendix's seed table: validation cross-entropy of the three seeds at d=256 and 512.

    python losses.py            # the numbers
    python losses.py --latex    # the rows of tab:lm-seeds

Reads main_lm/out/<run>/eval.json, the same protocol as every other loss in the paper, at
context length 2048. The runs are <arch><dim>[_s<seed>], seed 0 being the run of
\\Cref{tab:lm-ce}. No GPU.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.style import NAME                                     # noqa: E402

LM = HERE.parent / "main_lm" / "out"
SIZES = [(256, "29M"), (512, "77M")]
ARCHS = ("rope", "gdn", "lema")
SEEDS = (0, 1, 2)


def run_name(arch, dim, seed):
    s = "" if seed == 0 else f"_s{seed}"
    return f"lema{dim}_h64{s}" if arch == "lema" else f"{arch}{dim}{s}"


def ce(run, ctx=2048):
    f = LM / run / "eval.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text()).get(str(ctx))
    return None if d is None else float(d["ce"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latex", action="store_true")
    p.add_argument("--ctx", type=int, default=2048)
    a = p.parse_args()
    cell = "{:.3f}".format
    for arch in ARCHS:
        row = []
        for dim, _ in SIZES:
            v = [ce(run_name(arch, dim, s), a.ctx) for s in SEEDS]
            row.append(v)
        flat = [x for v in row for x in v]
        if a.latex:
            vals = " & ".join("$-$" if x is None else f"${cell(x)}$" for x in flat)
            print(f"{NAME[arch]} & {vals} \\\\")
        else:
            txt = "   ".join("  ".join("  -  " if x is None else cell(x) for x in v)
                             for v in row)
            spread = [None if None in v else max(v) - min(v) for v in row]
            print(f"{NAME[arch]:9s} {txt}   spread "
                  + "  ".join("-" if s is None else f"{s:.3f}" for s in spread))
    if not a.latex:
        print(f"{'':9s} " + "   ".join(" ".join(f"   s{s}" for s in SEEDS)
                                       for _, _ in SIZES) + "      29M / 77M")


if __name__ == "__main__":
    main()
