"""One loss per scored token, alongside its first-occurrence loss. No GPU.

    python score.py <model> ... [--distracted]

targets.py fixes which token of which window is scored, and both numbers come out of the
same per-token evaluation: the loss on the scored token, and the loss on the first occurrence
of the same bigram (the auxiliary `baseline` field, not the paper's intervention). Nothing is bucketed or averaged
here -- the output is one loss per target, in the order of the targets file -- so the
analysis can group them any way later, and adding a model to a figure costs no forward pass.

Writes out/scores/<model>_T{ctx}_f{freq}[_distracted].npz: loss, baseline, and the
fingerprints of the window set and of the targets file, which every consumer checks.

Usage: score.py <run|dir> ... [--ctx 16384] [--freq 100] [--distracted] [--force]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.main_lm_bigrams import targets                        # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="model names with a per-token file")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--freq", type=int, default=100)
    p.add_argument("--distracted", action="store_true",
                   help="score the distracted target set instead of the undistracted one")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    d = targets.load(a.ctx, a.freq, a.distracted)
    win, second = d["win"].astype(np.int64), d["second"].astype(np.int64)
    first = d["first"].astype(np.int64)
    tid = targets.ident(d)                               # identifies this target set
    out_dir = HERE / "out" / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = targets.tag_of(a)
    print(f"{len(win):,} targets, context {a.ctx}, threshold {a.freq}, "
          f"{'distracted' if a.distracted else 'undistracted'}", flush=True)
    for name in a.runs:
        short = Path(name).name
        out = out_dir / f"{short}_T{a.ctx}_f{a.freq}{tag}.npz"
        if out.exists() and not a.force:
            print(f"{short}: done"); continue
        f = HERE / "out" / "pertoken" / f"{short}_T{a.ctx}.npz"
        if not f.exists():
            print(f"{short}: no per-token file, run eval_pertoken.py first"); continue
        pt = np.load(f)
        assert int(pt["fingerprint"]) == int(d["fingerprint"]), f"{short}: window mismatch"
        ce = np.asarray(pt["ce"])
        loss = ce[win, second].astype(np.float32)
        base = ce[win, first].astype(np.float32)         # the same model where no copy existed
        np.savez(out, loss=loss, baseline=base, fingerprint=int(d["fingerprint"]), targets=tid,
                 ctx=a.ctx, freq=a.freq, distracted=bool(a.distracted))
        print(f"{short}: mean {loss.mean():.4f}, baseline {base.mean():.4f} "
              f"over {len(loss):,} targets -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
