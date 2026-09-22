"""The hardening trajectory of one LM run, measured densely -- and the control of the
backward-cap ablation.

The main recipe (`../main_lm/train_lm.py`, imported) for the d=512, d_h=64 LEMA model
with one change: a probe set of `--probe-batches` held-out batches is evaluated every
`--probe-every` steps -- soft CE (the surrogate at the current forward alpha) and hard CE
(the exact op), per-layer match and code-flip rates -- so out/lema512_h64/probes.jsonl
holds the trajectory at >= 200 points; log.jsonl has the gradient norms per parameter
group every 100 steps. plot_trajectory.py draws it; train_no_cap.py is the same run with
the cap removed.

Usage: [torchrun ...] train_trajectory.py [--dim 512] [--head 64] [--probe-every 200]
           [--probe-batches 8] [--grad-accum G]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from lema import train
from experiments.main_lm.train_lm import make


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--head", type=int, default=64)
    p.add_argument("--probe-every", type=int, default=200)     # 47100 steps -> 236 points
    p.add_argument("--probe-batches", type=int, default=8)      # 8 x 16 x 2048 = 262k tokens
    p.add_argument("--grad-accum", type=int, default=1)
    a = p.parse_args()
    train(*make("lema", a.dim, a.head, grad_accum=a.grad_accum, out_root=HERE / "out",
                probe_every=a.probe_every, probe_batches=a.probe_batches))


if __name__ == "__main__":
    main()
