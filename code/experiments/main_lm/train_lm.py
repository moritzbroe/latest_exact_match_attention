"""LM suite: lema / rope / gdn at dim 256-1536, ctx 2048, fineweb-edu 100BT.

Protocol (all constants below): depth = dim/64; softmax at head size 64, GDN at
head_dim 128 without value expansion; LEMA at whatever `--head` says -- the head size is
part of the LEMA recipe, not a default, so it is required and always in the run name.
20 tokens per transformer-parameter at each size;
lr = 1e-3 * 1024/dim with 2% warmup and cosine decay to lr/100; 32768 tokens/step
(global batch 16 x seq 2048) at ANY world size -- run under torchrun for multi-gpu,
the trainer splits the global batch across ranks. LEMA hardening: c over steps
2%-10%, alpha 1/sqrt(d_qk) -> 10 linearly from the 10% mark to the last step,
backward alpha capped at 2, beta 4. Rolling model+optimizer checkpoints every 5000
steps; rerunning the same command resumes bit-exactly (same world size). Seed 0 unless
--seed says otherwise, which names the run <arch><dim>_s<seed>.

Usage: [torchrun --standalone --nproc_per_node=N] train_lm.py <lema|rope|gdn> <dim>
           [--head W] [--grad-accum G] [--seed S]
--head: LEMA head size (required for lema, rejected otherwise). --grad-accum: split each
step into G micro-batches (tokens/step unchanged) for smaller gpus. --kv-heads: key-value
heads of a softmax run (grouped-query attention, default all heads), a drop-in change.

Run names: lema{dim}_h{head}, rope{dim} (rope{dim}_kv{K} with --kv-heads K), gdn{dim}, with
_s{seed} appended for a seed other than 0. The appendix experiments import `make()` from
here and override single TrainConfig fields, so the recipe lives in one place.
"""
import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # code/
from lema import ModelConfig, TokenCorpus, TrainConfig, train
from experiments.paths import DATA

HERE = Path(__file__).resolve().parent

SIZES = {256: (4, 17800), 512: (8, 47100), 768: (12, 99000), 1024: (16, 188300),
         1280: (20, 320600), 1536: (24, 509100)}   # dim: (depth, steps = 20N/32768)
SEQ, BATCH = 2048, 16
WIDTH = {"rope": 64, "gdn": 128}            # baseline head sizes; lema: --head
ARCH = {"lema": dict(attn="lema", pos="nope"),
        "rope": dict(attn="softmax", pos="rope"),
        "gdn": dict(attn="gated-deltanet", pos="nope", gdn_expand_v=1)}
WARMUP_F, CSTART_F, CSTEPS_F, ASTART_F = 0.02, 0.02, 0.08, 0.10
ALPHA_TARGET, ALPHA_BWD_CAP, BETA = 10.0, 2.0, 4.0
LR_MULT_TAG = {0.5: "_half", 2.0: "_double"}     # the lr ablation's names


def head_width(arch: str, head: int | None) -> int:
    """The head size a run gets: `--head` for lema (required), the fixed baseline size
    otherwise (and `--head` is an error there -- the baselines have no head ablation)."""
    if arch == "lema":
        if head is None:
            raise SystemExit("lema needs --head (8, 16, 32 or 64): the head size is part "
                             "of the run, there is no default")
        return head
    if head is not None:
        raise SystemExit(f"--head only applies to lema; {arch} always uses {WIDTH[arch]}")
    return WIDTH[arch]


def run_name(arch: str, dim: int, head: int | None = None, lr_mult: float = 1.0,
             kv_heads: int | None = None, seed: int = 0) -> str:
    width = head_width(arch, head)             # validates the pair for every arch
    name = f"{arch}{dim}" + (f"_h{width}" if arch == "lema" else "")
    if kv_heads is not None:
        if arch != "rope":
            raise SystemExit("--kv-heads only applies to rope (softmax attention)")
        name += f"_kv{kv_heads}"
    if lr_mult != 1.0:
        name += LR_MULT_TAG[lr_mult]
    if seed:
        name += f"_s{seed}"
    return name


def make(arch: str, dim: int, head: int | None = None, *, lr_mult: float = 1.0,
         grad_accum: int = 1, kv_heads: int | None = None, seed: int = 0,
         out_root: Path = HERE / "out", **overrides):
    """(model config, task, train config) for one run of the recipe.

    `overrides` replace TrainConfig fields after the recipe has derived them -- this is how
    the appendix experiments (lr ablation, hardening plot, no-hardening, no backward cap)
    state their one deviation each. Pass `out=` there to name the run differently.
    """
    width = head_width(arch, head)
    if dim % width:
        raise SystemExit(f"head size {width} does not divide dim {dim}")
    depth, steps = SIZES[dim]
    assert BATCH % grad_accum == 0    # micro-batch * accum * seq stays 32768 tok/step
    micro = BATCH // grad_accum
    assert micro * grad_accum * SEQ == 32768
    lr = 1e-3 * 1024 / dim * lr_mult
    task = TokenCorpus(DATA, seq_len=SEQ)
    mcfg = ModelConfig(vocab_size=task.vocab_size, depth=depth, dim=dim,
                       num_heads=dim // width, d_qk=width, d_v=width,
                       num_kv_heads=kv_heads or 0, **ARCH[arch])
    c_end = round(CSTART_F * steps) + round(CSTEPS_F * steps)
    marks = {c_end, round(ASTART_F * steps), round(0.5 * steps), round(0.9 * steps)}
    cfg = TrainConfig(
        steps=steps, out=str(Path(out_root) / run_name(arch, dim, head, lr_mult, kv_heads, seed)),
        lr=lr, lr_final=lr / 100, warmup=round(WARMUP_F * steps),
        batch=micro, grad_accum=grad_accum, seed=seed,
        c_start=round(CSTART_F * steps), c_steps=round(CSTEPS_F * steps),
        alpha_start=round(ASTART_F * steps),
        # +1 alpha per alpha_speed steps: 1/sqrt(d_qk) at the 10% mark -> exactly
        # alpha_target at the LAST step
        alpha_speed=round((1 - ASTART_F) * steps / (ALPHA_TARGET - width ** -0.5)),
        alpha_target=ALPHA_TARGET,
        alpha_bwd_cap=ALPHA_BWD_CAP, beta=BETA,
        log_every=100, probe_every=0,
        checkpoint_at=tuple(sorted(marks | set(range(20000, steps, 20000)))))
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return mcfg, task, cfg


def add_args(p: argparse.ArgumentParser):
    """The recipe's command line, shared with the appendix experiments."""
    p.add_argument("arch", choices=list(ARCH))
    p.add_argument("dim", type=int, choices=list(SIZES))
    p.add_argument("--head", type=int, default=None, choices=[8, 16, 32, 64],
                   help="lema head size (required for lema)")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--kv-heads", type=int, default=None,
                   help="key-value heads of a softmax run (grouped-query attention)")
    p.add_argument("--seed", type=int, default=0, help="0 is the main run, others get _s<seed>")


def main():
    p = argparse.ArgumentParser()
    add_args(p)
    a = p.parse_args()
    train(*make(a.arch, a.dim, a.head, grad_accum=a.grad_accum, kv_heads=a.kv_heads,
                seed=a.seed))


if __name__ == "__main__":
    main()
