"""Training hard from the start: is the annealing schedule needed at all?

The main recipe (`../main_lm/train_lm.py`, imported) for the d=512 LEMA model, but with
the hardening removed: c = d_qk - 1 from step 0 and the forward pass is
the EXACT latest-match op (`exact_forward`), so the model is "LEMA" from the first step
and its CE is the hard CE throughout. Only the backward uses the stick-breaking surrogate,
at alpha 2 -- the recipe's backward cap -- constant. Everything else (lr, data, tokens,
beta, batch) is the main run's, so out/lema512_h{w}_nohard compares to the main run
../main_lm/out/lema512_h{w} (and, at d_h=64, to the trajectory run out/lema512_h64).

What is tracked (probes.jsonl every --probe-every steps on --probe-batches held-out
batches): hard CE, and the MATCH RATE -- the fraction of queries whose code occurred
before, per layer and averaged -- which is near zero at initialisation and is exactly
what the annealed schedule lets grow before it starts to matter. Finish with the main
evaluation protocol for the number that compares to the main table:
    python ../main_lm/eval_lm.py --out out

Usage: [torchrun ...] train_no_hardening.py --head {8,16,32,64} [--dim 512]
           [--probe-every 200] [--probe-batches 8] [--grad-accum G]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from lema import train
from experiments.main_lm.train_lm import ALPHA_BWD_CAP, make, run_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", type=int, required=True, choices=[8, 16, 32, 64])
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--probe-every", type=int, default=200)
    p.add_argument("--probe-batches", type=int, default=8)     # 8 x 16 x 2048 = 262k tokens
    p.add_argument("--grad-accum", type=int, default=1)
    a = p.parse_args()
    name = run_name("lema", a.dim, a.head) + "_nohard"
    train(*make("lema", a.dim, a.head, grad_accum=a.grad_accum, out_root=HERE / "out",
                out=str(HERE / "out" / name),
                probe_every=a.probe_every, probe_batches=a.probe_batches,
                # no schedule: full threshold and the exact op from step 0, the
                # surrogate only in the backward at the (constant) capped alpha
                c_start=0, c_steps=0, alpha_start=0,
                alpha_init=ALPHA_BWD_CAP, alpha_target=ALPHA_BWD_CAP,
                alpha_bwd_cap=ALPHA_BWD_CAP, exact_forward=True))


if __name__ == "__main__":
    main()
