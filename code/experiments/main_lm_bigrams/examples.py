"""Bigrams of the longest distance bucket with their context, as plain text.

    python examples.py [--n 10]

Writes out/examples/bucket_{lo}_{hi}_f{freq}.txt with two sections. UNDISTRACTED holds
targets from the analysis proper, where the trigger does not recur between the two
occurrences. DISTRACTED holds targets from the complement, where it does, and each of those
also shows the LAST such occurrence, which is the one whose continuation a lookup keyed on
the trigger would return.

One line per bigram: the trigger and the token that follows it, then the earlier occurrence
with `--pad` tokens either side, the number of tokens hidden in brackets, the last distractor
if there is one, and the scored occurrence. `--n` takes a pseudo-random sample of that
many per section, seeded with --seed; 0 writes every target in the bucket.

Usage: examples.py [--ctx 16384] [--freq 100] [--n 10] [--pad 2] [--seed 0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.main_lm_bigrams.targets import load                   # noqa: E402
from experiments.main_lm_bigrams.eval_pertoken import windows          # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--freq", type=int, default=100)
    p.add_argument("--n", type=int, default=10, help="sample this many per section; 0 = all")
    p.add_argument("--pad", type=int, default=2, help="tokens shown either side of an occurrence")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    w, task = windows(a.ctx)
    x = w[:, :-1]
    lo, hi = a.ctx // 2, a.ctx                       # the longest distance bucket

    def seg(r, lo, hi):
        s = tok.decode([int(t) for t in w[r, max(lo, 0):hi]])
        return "[" + s.replace("\n", "\\n").replace("\r", "\\r") + "]"

    def chain(r, pts):
        """the occurrences at `pts`, each with --pad tokens either side; spans that would
        overlap because two occurrences are close are merged into one"""
        spans = [[q - a.pad, q + 2 + a.pad] for q in pts]
        merged = [spans[0]]
        for lo, hi in spans[1:]:
            if lo <= merged[-1][1]:          # touching or overlapping: one continuous span
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        out = []
        for i, (lo, hi) in enumerate(merged):
            if i:
                out.append(f"...({lo - merged[i - 1][1]})...")
            out.append(seg(r, lo, hi))
        return "".join(out)

    out = a.out or HERE / "out" / "examples" / f"bucket_{lo}_{hi}_f{a.freq}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write(f"# bigrams occurring at most {a.freq} times per 10B training tokens whose two\n"
                 f"# occurrences are {lo}-{hi} tokens apart, in {a.ctx}-token held-out windows.\n"
                 f"# trigger -> token :: [earlier occurrence]...(tokens hidden)...[scored occurrence]\n"
                 f"# sampled with seed {a.seed}, {a.pad} tokens of context either side.\n")
        for distracted in (False, True):
            d = load(a.ctx, a.freq, distracted)
            win, snd, src = (d["win"].astype(np.int64), d["second"].astype(np.int64),
                             d["source"].astype(np.int64))
            m = np.flatnonzero((d["dist"] >= lo) & (d["dist"] < hi))
            rng = np.random.default_rng(a.seed)
            pick = np.sort(rng.choice(m, a.n, replace=False)) if 0 < a.n < len(m) else m
            fh.write(f"\n### {'DISTRACTED' if distracted else 'UNDISTRACTED'} "
                     f"({len(pick)} of {len(m)})\n")
            if distracted:
                fh.write("# the middle span is the LAST occurrence of the trigger before the\n"
                         "# scored one, whose continuation differs\n")
            for i in pick:
                r, q, s0 = int(win[i]), int(snd[i]), int(src[i])
                pts = [s0, q]
                if distracted:
                    between = np.arange(s0 + 1, q)
                    same = between[x[r][between] == x[r][q]]
                    if len(same):
                        pts = [s0, int(same[-1]), q]
                fh.write(f"{tok.decode([int(x[r][q])])!r} -> {tok.decode([int(w[r][q + 1])])!r}"
                         f" :: {chain(r, pts)}\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
