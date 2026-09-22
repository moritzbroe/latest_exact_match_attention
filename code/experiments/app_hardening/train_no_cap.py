"""Without the backward alpha cap: the gate's gradient at the forward temperature.

The main recipe (`../main_lm/train_lm.py`, imported) for the d=512, d_h=64 LEMA model,
with `alpha_bwd_cap` raised to the alpha target: the backward temperature follows the
forward one all the way to 10 instead of stopping at 2. The claim being tested is that
the cap keeps gradient flowing through the gate once the surrogate is nearly exact --
at alpha 10 the sigmoid gate is saturated for matches and mismatches alike, so its
derivative (and with it the gradient reaching the query/key projections) should collapse
as alpha rises.

Tracked exactly like the control, which is the trajectory run out/lema512_h64
(train_trajectory.py): log.jsonl carries the pre-clip gradient norm and its split over
parameter groups (`qk` is the code path) every 100 steps, probes.jsonl soft/hard CE and
the per-layer match and code-flip rates every --probe-every steps on --probe-batches
held-out batches. Writes out/lema512_h64_nocap; plot_no_cap.py draws both.

Usage: [torchrun ...] train_no_cap.py [--dim 512] [--head 64] [--probe-every 200]
           [--probe-batches 8] [--grad-accum G]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from lema import train
from experiments.main_lm.train_lm import ALPHA_TARGET, make, run_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--head", type=int, default=64)
    p.add_argument("--probe-every", type=int, default=200)
    p.add_argument("--probe-batches", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    a = p.parse_args()
    name = run_name("lema", a.dim, a.head) + "_nocap"
    train(*make("lema", a.dim, a.head, grad_accum=a.grad_accum, out_root=HERE / "out",
                out=str(HERE / "out" / name), probe_every=a.probe_every,
                probe_batches=a.probe_batches, alpha_bwd_cap=ALPHA_TARGET))


if __name__ == "__main__":
    main()
