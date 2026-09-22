"""tab:lm-lr: validation CE at half, rule and double peak learning rate for the three
architectures at d = 256, 512, 768. The rule column is the main run under ../main_lm/out,
the half and double columns are out/<run>_{half,double}, all evaluated by eval_lm.py at
context 2048 on the 2^26 held-out tokens.

Usage: collect.py [--latex]    (no GPU, no torch; reads eval.json only). Missing runs print as "--".
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABLATION = HERE / "out"
MAIN = HERE.parent / "main_lm" / "out"
DIMS = ((256, "29M"), (512, "77M"), (768, "162M"))
CTX = "2048"


def ce(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text()).get(CTX, {}).get("ce")


ARCHS = (("softmax", "rope{d}"), ("GDN", "gdn{d}"), ("LEMA", "lema{d}_h64"))


def row(run: str):
    return (ce(ABLATION / f"{run}_half" / "eval.json"),
            ce(MAIN / run / "eval.json"),
            ce(ABLATION / f"{run}_double" / "eval.json"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latex", action="store_true")
    a = p.parse_args()
    fmt = lambda v: "--" if v is None else f"{v:.3f}"
    if not a.latex:
        print(f"{'model':<6} {'params':>7} {'d':>5} {'half':>7} {'rule':>7} {'double':>7}")
    for label, pattern in ARCHS:
      for dim, params in DIMS:
        if label == "GDN":
            params = {256: "29M", 512: "79M", 768: "170M"}[dim]
        half, rule, double = row(pattern.format(d=dim))
        if a.latex:
            print(f"{label} & ${params[:-1]}$M & ${dim}$ & {fmt(half)} & {fmt(rule)} & "
                  f"{fmt(double)} \\\\")
        else:
            vals = (half, rule, double)
            # only a complete row says anything about the rule; a missing run is not a loss
            mark = ("" if any(v is None for v in vals) else
                    "   (rule best)" if rule == min(vals) else
                    f"   (rule NOT best: {min(vals):.3f})")
            print(f"{label:<12} {params:>7} {dim:>5} {fmt(half):>7} {fmt(rule):>7} "
                  f"{fmt(double):>7}{mark}")


if __name__ == "__main__":
    main()
