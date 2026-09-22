"""Checkpoint generation for the recall experiment (accuracy vs number of pairs).

    train_recall.py lema [--seed K]   one seed of the single-count LEMA recipe (n=8)
    train_recall.py sb   [--seed K]   plain stick-breaking attention at n=8, no hardening
    train_recall.py rope [--seed K]   softmax curriculum, n = 2 -> 4096 in one call
    train_recall.py gdn  [--seed K]   gated-deltanet, same curriculum

LEMA trains once at 8 pairs and is evaluated everywhere (eval_recall.py); the baselines
cannot discover the task at larger counts from scratch, so each is trained by a staged
curriculum: every stage starts from the previous stage's weights with a fresh optimizer.
Every 250 steps one held-out batch is probed: its accuracy ends a curriculum stage at
1.000 and picks the stage's checkpoint the next stage starts from. The LEMA recipe: 50000
steps, c from 0 to d_h - 1 over steps 1000 to --c-end (15000), alpha from 1/sqrt(d_h) at
--c-end linearly to 10 at the last step, lr 3e-4 with a drop to --lr-harden once alpha
starts to rise, never stopped early and evaluated from its final checkpoint. The
stick-breaking run is the appendix ablation: the same model and lr without any ramp, also
run to the end. All defaults are the paper protocol; a unit whose model.pt exists is
skipped.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lema import ModelConfig, Recall, TrainConfig, Transformer, train

KEY_VOCAB = 4096
DIM, DEPTH, HEADS, D_HEAD = 64, 2, 1, 64
LADDER = [2 ** k for k in range(1, 13)]                    # 2 .. 4096 pairs
OUT = Path(__file__).parent / "out"


def task_at(pairs: int) -> Recall:
    return Recall(pairs, key_vocab=KEY_VOCAB)


def model_cfg(arch: str) -> ModelConfig:
    attn = {"lema": "lema", "sb": "sb", "rope": "softmax", "gdn": "gated-deltanet"}[arch]
    return ModelConfig(vocab_size=task_at(2).vocab_size, depth=DEPTH, dim=DIM,
                       num_heads=HEADS, d_qk=D_HEAD, d_v=D_HEAD, attn=attn,
                       pos="rope" if arch == "rope" else "nope")


def batch_at(pairs: int) -> int:
    return 32768 // pairs


def train_lema(seed: int, a):
    out = OUT / "lema" / f"s{seed}"
    if (out / "model.pt").exists():
        print(f"skip {out} (exists)")
        return
    cfg = TrainConfig(steps=a.steps, out=str(out), lr=a.lr, warmup=1000,
                      lr_alpha_phase=a.lr_harden, batch=4096,
                      c_start=1000, c_steps=a.c_end - 1000,                # c over 1000 .. c_end
                      alpha_speed=round((a.steps - a.c_end) / (10.0 - D_HEAD ** -0.5)),
                      alpha_target=10.0,                                  # 10 at the last step
                      alpha_bwd_cap=2.0, beta=a.beta,
                      seed=seed, probe_every=250, early_stop_acc=2.0)   # the whole schedule
    train(model_cfg("lema"), task_at(8), cfg)


def train_sb(seed: int, a):
    """Plain stick-breaking attention at n=8: the LEMA run without any hardening (c stays
    0, alpha at 1/sqrt(d_h)), the lr held at its pre-hardening value, run to the end."""
    out = OUT / "sb" / f"s{seed}"
    if (out / "model.pt").exists():
        print(f"skip {out} (exists)")
        return
    cfg = TrainConfig(steps=a.steps, out=str(out), lr=a.lr, warmup=1000, batch=4096,
                      c_start=10 ** 9, alpha_start=10 ** 9, seed=seed, probe_every=250,
                      early_stop_acc=2.0)
    train(model_cfg("sb"), task_at(8), cfg)


def train_ladder(arch: str, seed: int, a):
    import torch
    for i, pairs in enumerate(LADDER):
        out = OUT / arch / f"s{seed}" / f"n{pairs}"
        if (out / "model.pt").exists():
            print(f"skip {out} (exists)")
            continue
        model = None
        if i > 0:                          # every stage starts from its parent's best
            parent = OUT / arch / f"s{seed}" / f"n{LADDER[i - 1]}" / "model_best.pt"
            ck = torch.load(parent, map_location="cpu", weights_only=False)
            model = Transformer(model_cfg(arch))
            model.load_state_dict(ck["state_dict"])
        # a stage ends at probe accuracy 1.0 or, conclusively, after the full cosine cap
        cfg = TrainConfig(steps=a.stage_steps, out=str(out), lr=a.lr_baseline,
                          lr_final=0.0, warmup=1000, batch=batch_at(pairs),
                          c_start=10 ** 9, alpha_start=10 ** 9, seed=seed + i,
                          probe_every=250)
        train(model_cfg(arch), task_at(pairs), cfg, model=model)


def main():
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument("arch", choices=["lema", "sb", "rope", "gdn"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50000, help="lema, sb: total steps")
    p.add_argument("--c-end", type=int, default=15000,
                   help="lema: the step at which c reaches d_h - 1 and alpha starts to rise")
    p.add_argument("--lr", type=float, default=3e-4, help="lema, sb: pre-hardening lr")
    p.add_argument("--lr-harden", type=float, default=5e-5, help="lema: lr from c-end on")
    p.add_argument("--beta", type=float, default=4.0, help="lema: STE temperature")
    p.add_argument("--stage-steps", type=int, default=50000,
                   help="baselines: step cap per stage (stages stop early at accuracy 1)")
    p.add_argument("--lr-baseline", type=float, default=1e-3)
    p.add_argument("--out-root", type=Path, default=OUT)
    a = p.parse_args()
    OUT = a.out_root
    if a.arch == "lema" and not 1000 < a.c_end < a.steps:
        p.error("LEMA requires 1000 < --c-end < --steps for its hardening schedule")
    if a.arch == "lema":
        train_lema(a.seed, a)
    elif a.arch == "sb":
        train_sb(a.seed, a)
    else:
        train_ladder(a.arch, a.seed, a)


if __name__ == "__main__":
    main()
