"""The LM recipe at five times the tokens, at one size.

Everything is `main_lm/train_lm.py`, imported: same corpus, same 32768 tokens/step
(global batch 16 x seq 2048) at any world size, same peak lr = 1e-3 * 1024/dim, same
seed 0, same bf16 autocast, same LEMA hardening. The one deviation is the step budget,
`--factor` x the recipe's: 5 x 47100 = 235500 steps at dim 512, i.e. 7.72B tokens
instead of 1.54B (100 tokens per parameter instead of 20).

The recipe derives every schedule from `steps` -- 2% warmup, whole-run cosine to lr/100,
the c ramp over 2%-10%, the alpha ramp from the 10% mark to the LAST step, the
checkpoint marks -- so multiplying the budget stretches all of them by the same factor
and each stays at its fraction of the run rather than ending early.

Run names carry the factor, rope512_5x, gdn512_5x, lema512_h64_5x, and live under
../main_lm/out so that eval_lm.py, the context extension and the bigram scripts find them
like any other run.

Usage: [torchrun --standalone --nproc_per_node=N] train_tokens.py <lema|rope|gdn> <dim>
           [--head W] [--factor 5] [--grad-accum G]
Evaluate with the main protocol:
    python ../main_lm/eval_lm.py <run> [--ctx 16384]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from lema import train
from experiments.main_lm import train_lm
from experiments.main_lm.train_lm import add_args, make, run_name


def main():
    p = argparse.ArgumentParser()
    add_args(p)
    p.add_argument("--factor", type=float, default=5,
                   help="token-budget multiplier over the main recipe's step count")
    a = p.parse_args()
    depth, steps = train_lm.SIZES[a.dim]
    # The single deviation, stated where the recipe reads it: `make()` derives the
    # schedules and the checkpoint marks from this step count, so they scale with it.
    train_lm.SIZES[a.dim] = (depth, round(a.factor * steps))
    out = (HERE.parent / "main_lm" / "out"
           / f"{run_name(a.arch, a.dim, a.head, kv_heads=a.kv_heads, seed=a.seed)}_{a.factor:g}x")
    train(*make(a.arch, a.dim, a.head, grad_accum=a.grad_accum, kv_heads=a.kv_heads, seed=a.seed, out=str(out)))


if __name__ == "__main__":
    main()
