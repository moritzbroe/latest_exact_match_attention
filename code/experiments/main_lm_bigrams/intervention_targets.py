"""Corrupted copies of the target windows: the no-recall baseline by intervention. No GPU.

The figure's dotted line is each model's loss on the SAME token in the SAME place, with
every earlier copy of its bigram destroyed, so recall has nothing left to find and what the
model still scores is what it knows without it. Scoring the first occurrence of the bigram
instead would compare a different token in a different place, and part of that gap would
just be that the two tokens are not equally predictable on their own (in practice the two
baselines agree to within 0.3 nats).

Per distance bucket of plot_sizes_grid.py, up to --per-bucket targets are sampled (all of them where
the bucket holds fewer, as in the longest). For a sampled target the bigram is
(a, b) = (w[second], w[second+1]) -- `second` indexes x = w[:, :-1], so the trigger sits at
`second` and the scored token at `second+1`, and the loss on it is entry `second` of a
per-token row. Every j < second with (w[j], w[j+1]) == (a, b) is an earlier copy, and both
of its tokens are replaced by tokens drawn from the empirical unigram distribution of the
validation shard: random positions of the shard, so a replacement is as likely as it is
frequent and the corrupted text keeps the frequency profile of the original. Positions
`second` and later are never touched, so the trigger and the scored token stay exactly what
they were -- which for a == b means the copy at second-1 loses only its first token. A draw
can in principle rebuild the bigram it destroyed; the build counts how often any copy
survives, and it is 0.

One target per window copy, so corruptions never interfere, and the same copies are used by
every model.

Writes out/intervention_T{ctx}_f{freq}[_distracted].npz: target (the index into the targets
file), win, second, bucket, the corruption as pos/tok/off (flat arrays plus one slice per
target), the seed, and the fingerprints of the window set and of the targets file.

Usage: intervention_targets.py [--ctx 16384] [--freq 100] [--distracted]
                               [--per-bucket 1024] [--seed 0] [--force]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.main_lm.eval_lm import windows_fingerprint           # noqa: E402
from experiments.main_lm_bigrams import targets                       # noqa: E402
from experiments.main_lm_bigrams.eval_pertoken import windows         # noqa: E402

EDGES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def buckets_of(ctx: int):
    """The distance bins of plot_sizes_grid.py, the ones a --ctx window can hold."""
    return [(lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:]) if lo < ctx]


def unigram_draws(task, n: int, rng, split: str = "val") -> np.ndarray:
    """`n` tokens drawn from the empirical unigram distribution of a split: uniformly
    random positions of its shards, which is that distribution without ever building a
    table of it."""
    arrs = task.arrs[split]
    off = np.r_[0, np.cumsum([len(a) for a in arrs])]
    pos = rng.integers(0, int(off[-1]), size=n)
    which = np.searchsorted(off, pos, side="right") - 1
    out = np.empty(n, np.int32)
    for i, a in enumerate(arrs):                     # fancy-indexed straight off the memmap
        m = np.flatnonzero(which == i)
        if len(m):
            out[m] = np.asarray(a[pos[m] - off[i]], dtype=np.int32)
    return out


def path(ctx: int, freq: int, distracted: bool = False) -> Path:
    tag = targets.tag_of(argparse.Namespace(distracted=distracted))
    return HERE / "out" / f"intervention_T{ctx}_f{freq}{tag}.npz"


def load(ctx: int, freq: int, distracted: bool = False):
    f = path(ctx, freq, distracted)
    if not f.exists():
        raise SystemExit(f"no {f.name}; run intervention_targets.py first")
    return np.load(f)


def build(ctx: int, freq: int, distracted: bool, per_bucket: int, seed: int):
    d = targets.load(ctx, freq, distracted)
    w, task = windows(ctx)
    assert windows_fingerprint(w) == int(d["fingerprint"]), "targets file is for other windows"
    win, snd = d["win"].astype(np.int64), d["second"].astype(np.int64)
    src, dist = d["source"].astype(np.int64), d["dist"]
    lo_hi = buckets_of(ctx)

    rng = np.random.default_rng([seed, 0])           # which targets
    sel, bkt = [], []
    for b, (lo, hi) in enumerate(lo_hi):
        m = np.flatnonzero((dist >= lo) & (dist < hi))
        take = np.sort(rng.choice(m, per_bucket, replace=False)) if len(m) > per_bucket else m
        sel.append(take)
        bkt.append(np.full(len(take), b, np.int32))
    sel, bkt = np.concatenate(sel), np.concatenate(bkt)

    POS, off, nocc = [], [0], np.empty(len(sel), np.int32)
    for k, i in enumerate(sel):
        row, q = w[int(win[i])], int(snd[i])
        a, b = row[q], row[q + 1]
        j = np.flatnonzero((row[:q] == a) & (row[1:q + 1] == b))   # every copy before q
        assert (j == src[i]).any(), "the labelled earlier occurrence is not a copy of the bigram"
        pos = np.unique(np.concatenate([j, j + 1]))
        pos = pos[pos < q]                           # the trigger and the scored token stay
        POS.append(pos.astype(np.int32))
        off.append(off[-1] + len(pos))
        nocc[k] = len(j)
    pos = np.concatenate(POS)
    off = np.array(off, np.int64)
    tok = unigram_draws(task, len(pos), np.random.default_rng([seed, 1]))

    survive = 0
    for k, i in enumerate(sel):                      # nothing of the bigram is left to find
        row, q = w[int(win[i])].copy(), int(snd[i])
        s = slice(off[k], off[k + 1])
        row[pos[s]] = tok[s]
        survive += int(((row[:q] == row[q]) & (row[1:q + 1] == row[q + 1])).any())

    print(f"context {ctx}, threshold {freq}, "
          f"{'distracted' if distracted else 'undistracted'}, "
          f"up to {per_bucket} targets per bucket, seed {seed}")
    for b, (lo, hi) in enumerate(lo_hi):
        m = bkt == b
        print(f"    distance {lo:>6}-{hi:<6}: {int(m.sum()):>6,} targets"
              + (f", {nocc[m].mean():5.2f} copies each, "
                 f"{np.diff(off)[m].mean():5.2f} tokens replaced" if m.any() else ""))
    print(f"  {len(sel):,} targets, {nocc.mean():.2f} copies each, "
          f"{len(pos):,} tokens replaced, {survive} targets with a surviving copy")
    return dict(target=sel.astype(np.int64), win=win[sel].astype(np.int32),
                second=snd[sel].astype(np.int32), bucket=bkt, pos=pos, tok=tok, off=off,
                nocc=nocc, per_bucket=per_bucket, seed=seed, ctx=ctx, freq=freq,
                distracted=bool(distracted), fingerprint=int(d["fingerprint"]),
                targets=targets.ident(d))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--freq", type=int, default=100)
    p.add_argument("--distracted", action="store_true",
                   help="the distracted target set instead of the undistracted one")
    p.add_argument("--per-bucket", type=int, default=1024,
                   help="targets sampled per distance bucket, all of them if it holds fewer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    f = path(a.ctx, a.freq, a.distracted)
    if f.exists() and not a.force:
        raise SystemExit(f"{f.name} exists; --force to rebuild")
    d = build(a.ctx, a.freq, a.distracted, a.per_bucket, a.seed)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **d)
    print(f"-> {f}")


if __name__ == "__main__":
    main()
