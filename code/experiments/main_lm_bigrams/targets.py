"""The scored tokens of the analysis, with their first occurrences.

A TARGET is a token whose bigram (trigger, token) occurred earlier in the same document of a
held-out window, where the bigram occurs at most --freq times in the first 10B training
tokens -- count 0 included, and those are the rarest and most demanding cases -- so that
predicting it needs the earlier occurrence rather than knowledge from training. Its DISTANCE
is the gap to that earlier occurrence.

By default a target must be UNDISTRACTED: the trigger token does not recur between the two
occurrences, so the earlier occurrence of the bigram is also the most recent occurrence of
the trigger. This is the associative-recall hit of Arora et al. (2024) with the ambiguous
cases removed, and it is what makes the distance axis mean what it says. Measured in the
8192-16384 bucket: for undistracted targets the labelled distance IS the distance to the
nearest earlier occurrence of the trigger, for every one of them; for the distracted ones
the trigger reappears after a median of 519 tokens, so the label would be fiction.

`--distracted` builds the complement instead: bigrams that do recur but whose trigger appears
in between followed by something else. Those ask a different question -- choosing between
competing recent continuations -- and are analysed separately.

Each target also records the FIRST occurrence of the same bigram in the document.
Its loss is an auxiliary diagnostic, not the paper's baseline: available context differs.
The paper uses matched interventions from eval_intervention.py instead.

Writes out/targets_T{ctx}_f{freq}[_distracted].npz: win, second (the trigger whose next token
is scored), source, first, dist, and the window fingerprint.

Usage: targets.py [--ctx 16384] [--freq 100] [--distracted] [--force]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.main_lm.eval_lm import windows_fingerprint           # noqa: E402
from experiments.main_lm_bigrams.count_bigrams import counts_of       # noqa: E402
from experiments.main_lm_bigrams.eval_pertoken import windows         # noqa: E402


def build(ctx: int, freq: int, distracted: bool = False):
    w, task = windows(ctx)
    V, eot = task.vocab_size, task.meta["eot"]
    x, y = w[:, :-1], w[:, 1:]
    T = x.shape[1]
    big = x.astype(np.int64) * (V + 1) + y.astype(np.int64)
    rare = counts_of(big) <= freq          # count 0 -- absent from training -- included

    WIN, SND, SRC, FST = [], [], [], []
    for r in range(len(w)):
        hits = np.flatnonzero(rare[r])
        if not len(hits):
            continue
        doc = np.cumsum(x[r] == eot)
        # the previous position carrying the same trigger token
        order = np.argsort(x[r], kind="stable")
        xs = x[r][order]
        prev = np.full(T, -1, np.int64)
        same = xs[1:] == xs[:-1]
        prev[1:][same] = order[:-1][same]
        prev_x = np.full(T, -1, np.int64)
        prev_x[order] = prev
        # occurrences of each rare bigram inside one document, in position order
        key = big[r][hits] * (int(doc.max()) + 2) + doc[hits]
        o = np.argsort(key, kind="stable")
        ks, ps = key[o], hits[o]
        new = np.r_[True, ks[1:] != ks[:-1]]
        gid = np.cumsum(new) - 1
        first_of = ps[new]                                 # positions ascend inside a group
        later = ~new                                       # every occurrence after the first
        second = ps[later]
        src = np.r_[-1, ps[:-1]][later]                    # the previous one of the same bigram
        fst = first_of[gid[later]]
        # undistracted: nothing carrying the trigger stands between the two occurrences
        clean = prev_x[second] == src
        keep = ~clean if distracted else clean
        second, src, fst = second[keep], src[keep], fst[keep]
        if not len(second):
            continue
        WIN.append(np.full(len(second), r, np.int32))
        SND.append(second.astype(np.int32))
        SRC.append(src.astype(np.int32))
        FST.append(fst.astype(np.int32))
    win = np.concatenate(WIN); snd = np.concatenate(SND)
    src = np.concatenate(SRC); fst = np.concatenate(FST)
    assert (fst <= src).all() and (src < snd).all(), "first occurrence must precede the source"
    return dict(win=win, second=snd, source=src, first=fst,
                dist=(snd.astype(np.int64) - src), ctx=ctx, freq=freq,
                distracted=bool(distracted),
                fingerprint=windows_fingerprint(w), shape=np.array(w.shape))


def tag_of(a) -> str:
    """The filename suffix identifying which of the two target sets is meant."""
    return "_distracted" if getattr(a, "distracted", False) else ""


def ident(d) -> int:
    """Identifies the exact target LIST a result was produced against, so a score and
    anything derived from it can prove they line up target for target."""
    return int(np.int64(d["win"]).sum() * 1_000_003 + np.int64(d["second"]).sum())


class _Sel:
    def __init__(self, distracted):
        self.distracted = distracted


def path(ctx: int, freq: int, distracted: bool = False) -> Path:
    return HERE / "out" / f"targets_T{ctx}_f{freq}{tag_of(_Sel(distracted))}.npz"


def load(ctx: int, freq: int, distracted: bool = False):
    """The target set, built and cached on first use -- it is shared by every model, so it
    is computed once rather than inside each scoring run, and its file is what lets a score
    assert it was produced against this exact list."""
    f = path(ctx, freq, distracted)
    if not f.exists():
        print(f"building {f.name} (once, ~2 min)", flush=True)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, **build(ctx, freq, distracted))
    return np.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--freq", type=int, default=100)
    p.add_argument("--distracted", action="store_true",
                   help="the complement: the trigger recurs between the two occurrences")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    f = path(a.ctx, a.freq, a.distracted)
    if f.exists() and not a.force:
        raise SystemExit(f"{f.name} exists; --force to rebuild")
    d = build(a.ctx, a.freq, a.distracted)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **d)
    print(f"context {a.ctx}, threshold {a.freq} per 10B training tokens, "
          f"{'distracted' if a.distracted else 'undistracted'}")
    print(f"  targets: {len(d['win']):,} "
          f"({100 * len(d['win']) / (d['shape'][0] * (d['shape'][1] - 1)):.3f}% of scored tokens)")
    EDGES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        if lo < a.ctx:
            print(f"    distance {lo:>6}-{hi:<6}: "
                  f"{int(((d['dist'] >= lo) & (d['dist'] < hi)).sum()):>8,}")


if __name__ == "__main__":
    main()
