"""Learning-rate ablation of the LM recipe: the main run at half and at double its lr.

Everything else is `main_lm/train_lm.py`, imported -- same data, sizes, schedule and
hardening; the one deviation is `lr_mult`. Run names carry it: lema256_h64_half,
rope256_double, ... Evaluate with the main protocol, pointed at this folder:
    python ../main_lm/eval_lm.py --out out

Usage: [torchrun ...] train_lr_ablation.py <lema|rope|gdn> <dim> [--head W] {--half|--double}
           [--grad-accum G]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from lema import train
from experiments.main_lm.train_lm import add_args, make


def main():
    p = argparse.ArgumentParser()
    add_args(p)
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--half", action="store_const", dest="mult", const=0.5)
    m.add_argument("--double", action="store_const", dest="mult", const=2.0)
    a = p.parse_args()
    train(*make(a.arch, a.dim, a.head, lr_mult=a.mult, grad_accum=a.grad_accum,
                kv_heads=a.kv_heads, seed=a.seed, out_root=HERE / "out"))


if __name__ == "__main__":
    main()
