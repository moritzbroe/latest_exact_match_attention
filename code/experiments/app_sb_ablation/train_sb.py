"""The stick-breaking ablation of the appendix: plain stick-breaking attention (real-valued
queries and keys, no hardening, no positional encoding) at d=512 with the main_lm recipe
otherwise unchanged (8 heads of 64, 47100 steps, lr 2e-3, 32768 tokens/step, seed 0). Run
name sb512 under ../main_lm/out so that eval_lm.py and eval_pertoken.py find it like any
other run.

    python train_sb.py            (resumes from ../main_lm/out/sb512/checkpoint.pt when rerun)
"""
import dataclasses
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                 # code/
sys.path.insert(0, str(HERE.parent / "main_lm"))          # train_lm.make
from lema import train                                     # noqa: E402
from train_lm import make                                  # noqa: E402

OUT = HERE.parent / "main_lm" / "out" / "sb512"


def main():
    mcfg, task, cfg = make("rope", 512, out=str(OUT))
    mcfg = dataclasses.replace(mcfg, attn="sb", pos="nope")
    train(mcfg, task, cfg)


if __name__ == "__main__":
    main()
