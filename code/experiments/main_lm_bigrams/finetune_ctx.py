"""Context extension of the finished LM runs to 16k tokens by continued pretraining.

The recipe, every run of it named out/<run>_16k (or _<seq>k for another --seq):
  * the finished ../main_lm run's weights and a fresh AdamW with the pretraining's betas
    (0.9, 0.999), weight decay 0.1 and clip 1.0; --seq-token windows from the SAME training
    shards; 32768 tokens per step as in pretraining (2 windows at 16k); 30518 steps = 1.0B tokens,
    16% of the 309M models' pretraining; seed 1 (a different window stream);
  * lr 1/10 of the run's pretraining peak, 2% linear warmup, cosine to 1/10 of that, which
    is the pretraining's own final lr. The long-context literature fine-tunes at 2e-5 from
    3e-4 peaks with short warmup and cosine (Chen et al. 2023, Xiong et al. 2023,
    Roziere et al. 2023), budgets from 0.1% to 20% of pretraining tokens;
  * softmax+RoPE: base 1e4 -> 1e6, the Code Llama recipe for 16k; the adjusted base beats
    position interpolation in Xiong et al. 2023. The base is stored in the model config,
    so every later evaluation of the run uses it. Only when --seq exceeds the pretraining
    context: `--seq 2048` is the control that repeats the whole procedure WITHOUT extending
    anything, which separates the effect of the longer context from that of continued
    training at a raised learning rate;
  * LEMA: the final phase of the pretraining schedule, continued -- c at its maximum,
    alpha 10 in the forward, backward cap 2, beta 4, surrogate forward -- and evaluation in
    hard mode as everywhere else; GDN: nothing changes.
Rolling model+optimizer checkpoint every 1000 steps: rerunning the same command resumes,
a finished run (model.pt) is never re-entered. `--checkpoint-blocks` recomputes each
block's activations in the backward: the same computation, for a fraction of the memory.

Usage: [torchrun --standalone --nproc_per_node=2] finetune_ctx.py <run> [--seq 16384]
           [--grad-accum G] [--checkpoint-blocks] [--steps N --out DIR]
`--steps`/`--out` exist for smoke tests only (they write elsewhere and stop early).
"""
import argparse
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # code/
from lema import ModelConfig, TokenCorpus, TrainConfig, Transformer, train   # noqa: E402
from experiments.paths import DATA, LM_OUT                             # noqa: E402

HERE = Path(__file__).resolve().parent
SEQ = 16384
PRETRAIN_SEQ = 2048                        # the context every ../main_lm run was trained at
TOKENS_PER_STEP = 32768                    # as in pretraining
TOKENS = 1_000_000_000
STEPS = math.ceil(TOKENS / TOKENS_PER_STEP)   # 30518
LR_FRAC = 0.1                              # of the run's pretraining peak
WARMUP_F = 0.02
ROPE_BASE = 1e6


def make(run: str, *, seq: int = SEQ, grad_accum: int = 1, steps: int = STEPS,
         out_root: Path = HERE / "out", checkpoint_blocks: bool = False, seed: int = 1):
    """(model config, task, train config, pretrained state dict) for one run.

    `run` is a name under ../main_lm/out, or a path to any finished run -- including one of
    the continuations here, so a continuation can itself be continued."""
    src = Path(run) if (Path(run) / "model.pt").exists() else LM_OUT / run
    if not (src / "model.pt").exists():
        raise SystemExit(f"no finished run at {src}")
    ck = torch.load(src / "model.pt", map_location="cpu", weights_only=False)
    mcfg = ModelConfig.from_dict(ck["config"])
    if mcfg.pos == "rope" and seq > PRETRAIN_SEQ:
        mcfg = dataclasses.replace(mcfg, rope_base=ROPE_BASE)
    pre = json.loads((src / "config.json").read_text())["train"]
    batch = TOKENS_PER_STEP // seq
    if batch % grad_accum:
        raise SystemExit(f"{batch} windows per step do not split into {grad_accum} micro-steps")
    lr = pre["lr"] * LR_FRAC
    warmup = round(WARMUP_F * steps)
    cfg = TrainConfig(
        # a pretrained run gets the context in its name, a continuation of a continuation
        # keeps the name it already has
        steps=steps, out=str(Path(out_root) / (
            (f"{src.name}_{seq // 1024}k" if src.parent == LM_OUT else src.name)
            + ("" if seed == 1 else f"_s{seed}"))),
        lr=lr, lr_final=lr / 10, warmup=warmup,
        weight_decay=pre["weight_decay"], adam_betas=tuple(pre["adam_betas"]),
        grad_clip=pre["grad_clip"], batch=batch // grad_accum, grad_accum=grad_accum, seed=seed,
        # LEMA: the pretraining schedule's final state from step 0 (c_at -> c_max at
        # c_steps 0, alpha_at -> alpha_init = alpha_target flat)
        c_start=0, c_steps=0, alpha_start=0, alpha_init=pre["alpha_target"],
        alpha_target=pre["alpha_target"], alpha_speed=1, alpha_bwd_cap=pre["alpha_bwd_cap"],
        beta=pre["beta"], exact_forward=False, checkpoint_blocks=checkpoint_blocks,
        log_every=100, probe_every=0, resume_every=1000)
    task = TokenCorpus(DATA, seq_len=seq)
    return mcfg, task, cfg, ck["state_dict"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="a finished run under ../main_lm/out")
    p.add_argument("--seq", type=int, default=SEQ,
                   help=f"training context; {PRETRAIN_SEQ} repeats the recipe without extending")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=1,
                   help="window stream of the continued training; anything but 1 goes to "
                        "<run>_<seq>k_s<seed>, so seeds do not overwrite each other")
    p.add_argument("--checkpoint-blocks", action="store_true",
                   help="activation checkpointing per block, for smaller GPUs")
    p.add_argument("--steps", type=int, default=STEPS, help="smoke tests only")
    p.add_argument("--out", type=Path, default=None, help="smoke tests only")
    a = p.parse_args()
    mcfg, task, cfg, state = make(
        a.run, seq=a.seq, grad_accum=a.grad_accum, steps=a.steps,
        out_root=a.out or HERE / "out", checkpoint_blocks=a.checkpoint_blocks, seed=a.seed)
    model = Transformer(mcfg)
    keep = model.state_dict().keys()
    missing, unexpected = model.load_state_dict({k: v for k, v in state.items() if k in keep},
                                                strict=False)
    if missing:
        raise SystemExit(f"pretrained weights lack {sorted(missing)[:5]} ...")
    main_rank = int(os.environ.get("RANK", "0")) == 0
    if main_rank:
        if unexpected:
            print(f"dropped {len(unexpected)} unknown parameters: {sorted(unexpected)[:5]}")
        print(f"{a.run} -> {cfg.out}: {mcfg.summary()} rope_base {mcfg.rope_base:g}; "
              f"seq {a.seq}, {cfg.steps} steps x {TOKENS_PER_STEP} tokens, "
              f"lr {cfg.lr:g} -> {cfg.lr_final:g}, warmup {cfg.warmup}, "
              f"checkpoint_blocks {cfg.checkpoint_blocks}", flush=True)
    train(mcfg, task, cfg, model=model)
    if main_rank and torch.cuda.is_available():
        print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB on rank 0",
              flush=True)


if __name__ == "__main__":
    main()
