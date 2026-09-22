"""Can a baseline discover the recall mechanism at a large n directly?

    train_discovery.py rope --n 256 [--seed 0] [--steps 100000]
    train_discovery.py gdn  --n 256

One run from a fresh initialisation at a fixed n, with the configuration of a curriculum
stage in ../main_recall/train_recall.py (lr 1e-3, warmup 1000, 32768/n sequences per
step, a held-out probe every 250 steps, early stop at accuracy 1) -- without the earlier
stages, and with a CONSTANT learning rate instead of the stage's cosine, so that a late
discovery is not ruled out by a decayed lr. probes.jsonl holds accuracy and CE over steps.
Writes out/<arch>/n<n>/s<seed>/; a finished run (model.pt present) is skipped.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from lema import TrainConfig, train                                    # noqa: E402
from experiments.main_recall.train_recall import batch_at, model_cfg, task_at   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("arch", choices=["rope", "gdn"])
    p.add_argument("--n", type=int, required=True, help="number of associations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=None,
                   help="sequences per step (default: the stage recipe's 32768 / n)")
    a = p.parse_args()
    out = HERE / "out" / a.arch / f"n{a.n}" / f"s{a.seed}"
    cfg = TrainConfig(steps=a.steps, out=str(out), lr=a.lr, lr_final=None, warmup=1000,
                      batch=a.batch or batch_at(a.n),
                      c_start=10 ** 9, alpha_start=10 ** 9, seed=a.seed, probe_every=250)
    train(model_cfg(a.arch), task_at(a.n), cfg)


if __name__ == "__main__":
    main()
