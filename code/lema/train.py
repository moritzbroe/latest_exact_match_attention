"""The trainer: one function, one schedule.

The hardening recipe is exactly this:
  * lr: linear warmup, then constant, or a whole-run cosine to `lr_final`.
    Optionally a fixed `lr_alpha_phase` from the step hardening starts (the recall recipe).
  * c:  linear from 0 to d_qk - 1 over [c_start, c_start + c_steps], constant after.
  * alpha: `alpha_init` (default 1/sqrt(d_qk)) until `alpha_start` (default: the step c
    finishes), then +1 per `alpha_speed` steps until `alpha_target`, then constant. The
    target only sets where the ramp stops, never its rate.
  * backward alpha = min(alpha, alpha_bwd_cap): the forward uses alpha, the backward the
    capped value (see lema.attention.lema).
  * `exact_forward`: the exact latest-match op in the forward from step 0 and the
    surrogate (at the capped alpha) in the backward only. Off in the recipe.

Two output streams, both jsonl:
  * `log.jsonl` every `log_every` steps: mean train loss over the window, lr, c, forward
    and backward alpha, pre-clip gradient norm and its split over parameter groups.
  * `probes.jsonl` every `probe_every` steps: evaluation passes on a fixed set of
    `probe_batches` held-out batches (see probes.py). Off unless an experiment asks.
Checkpoints: `model.pt` at the end, `model_step<k>.pt` at `checkpoint_at`, a rolling
`checkpoint.pt` (model + optimizer) every `resume_every` steps from which a rerun of the
same command resumes. The accuracy tasks additionally keep `model_best.pt` (best probe
accuracy) and stop early at `early_stop_acc`.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .model import ModelConfig, Transformer, best_dtype
from .probes import probe


def _ddp_setup():
    """Rank and world size under torchrun; (0, 1) as a plain process. `cfg.batch` is
    always the GLOBAL batch per micro-step, split over the ranks."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        import torch.distributed as dist
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        return dist.get_rank(), world
    return 0, 1


@dataclass
class TrainConfig:
    steps: int
    out: str
    # optimisation
    lr: float = 1e-3
    lr_final: float | None = None          # None -> constant lr; else whole-run cosine
    warmup: int = 500
    weight_decay: float = 0.1
    adam_betas: tuple = (0.9, 0.999)
    grad_clip: float = 1.0
    grad_accum: int = 1
    batch: int = 64
    # hardening schedule (steps)
    c_start: int = 1000
    c_steps: int = 10000
    alpha_start: int | None = None         # None -> c_start + c_steps
    alpha_speed: int = 1000                # steps per +1.0 of alpha
    alpha_target: float = 10.0
    alpha_init: float | None = None        # None -> 1 / sqrt(d_qk)
    alpha_bwd_cap: float = 2.0
    exact_forward: bool = False
    checkpoint_blocks: bool = False        # activation checkpointing per block (memory)
    lr_alpha_phase: float = 0.0            # > 0: the lr from max(alpha_start, c end) on
    beta: float = 2.0                      # STE temperature of the binarization
    # bookkeeping
    log_every: int = 100                   # log.jsonl cadence; 0 = off
    probe_every: int = 0                   # probes.jsonl cadence; 0 = no probe set
    probe_batches: int = 1                 # held-out batches (of `batch` sequences) probed
    early_stop_acc: float = 1.0            # stop once the probe accuracy reaches this
    checkpoint_at: tuple = ()              # extra steps at which to save model_step*.pt
    resume_every: int = 5000
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self):
        if self.alpha_start is None:
            self.alpha_start = self.c_start + self.c_steps
        if self.alpha_speed < 1:
            raise ValueError(f"alpha_speed must be >= 1 step per unit, got {self.alpha_speed}")
        if self.probe_every and self.probe_batches < 1:
            raise ValueError("probe_batches must be >= 1 when probing")


def lr_at(cfg: TrainConfig, step: int) -> float:
    warm = min(1.0, (step + 1) / cfg.warmup) if cfg.warmup else 1.0
    if cfg.lr_alpha_phase > 0 and step >= max(cfg.alpha_start, cfg.c_start + cfg.c_steps):
        return cfg.lr_alpha_phase
    if cfg.lr_final is None:
        return warm * cfg.lr
    t = min(step / max(cfg.steps - 1, 1), 1.0)
    return warm * (cfg.lr_final + (cfg.lr - cfg.lr_final) * 0.5 * (1 + math.cos(math.pi * t)))


def c_at(cfg: TrainConfig, step: int, c_max: float) -> float:
    if cfg.c_steps <= 0:
        return c_max if step >= cfg.c_start else 0.0
    u = min(max((step - cfg.c_start) / cfg.c_steps, 0.0), 1.0)
    return c_max * u


def alpha_at(cfg: TrainConfig, step: int, a0: float) -> float:
    """Flat at a0 until alpha_start, then +1 per alpha_speed steps up to alpha_target."""
    if step <= cfg.alpha_start:
        return a0
    return min(cfg.alpha_target, a0 + (step - cfg.alpha_start) / cfg.alpha_speed)


def _loss(model, x, y, mask):
    if mask is None and x.shape[1] > 4096:
        return _loss_chunked(model, x, y)
    logits = model(x)
    if mask is None:
        return torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), y.flatten())
    sel = mask.flatten()
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1)[sel].float(),
                                             y.flatten()[sel])


def _loss_chunked(model, x, y, chunk: int = 1024):
    """The mean cross-entropy of a long window without ever holding its full logits: the
    output head and the loss run chunk by chunk, recomputed in the backward, which is what
    fits the 16k-window context extension of the 834M model into 24 GB. Same value as
    _loss up to the summation order."""
    from torch.utils.checkpoint import checkpoint
    net = getattr(model, "module", model)
    head, net.lm_head = net.lm_head, torch.nn.Identity()
    try:
        h = model(x)
    finally:
        net.lm_head = head

    def ce_sum(hc, yc):
        return torch.nn.functional.cross_entropy(head(hc).float().flatten(0, 1), yc.flatten(),
                                                 reduction="sum")
    total = 0.0
    for a in range(0, x.shape[1], chunk):
        total = total + checkpoint(ce_sum, h[:, a:a + chunk], y[:, a:a + chunk],
                                   use_reentrant=False)
    return total / y.numel()


# Parameter groups the gradient norm is split over in log.jsonl. "qk" is the code path,
# the projections the STE and the gate surrogate reach.
_GROUPS = ("qk", "vo", "mlp", "emb", "other")


def _group_of(name: str) -> str:
    if "q_proj" in name or "k_proj" in name:
        return "qk"
    if "v_proj" in name or "o_proj" in name:
        return "vo"
    if ".mlp." in name:
        return "mlp"
    if "tok_emb" in name or "lm_head" in name:
        return "emb"
    return "other"


@torch.no_grad()
def grad_norms(model) -> dict:
    """L2 norm of the current gradient per parameter group (call BEFORE clipping)."""
    grads = {g: [] for g in _GROUPS}
    for n, p in model.named_parameters():
        if p.grad is not None:
            grads[_group_of(n)].append(p.grad)
    return {g: float(torch.stack(torch._foreach_norm(gs)).norm())
            for g, gs in grads.items() if gs}


def train(mcfg: ModelConfig, task, cfg: TrainConfig, model: Transformer | None = None):
    """Train `model` (a fresh one from `mcfg` unless given) on `task` per `cfg`."""
    rank, world = _ddp_setup()
    main = rank == 0
    if cfg.batch % world:
        raise ValueError(f"global batch {cfg.batch} not divisible by {world} ranks")
    out = Path(cfg.out)
    if (out / "model.pt").exists():
        # model.pt is written once, at the end: this run is finished, and rerunning the
        # tail from the rolling checkpoint must not overwrite the evaluated weights
        if main:
            print(f"{out} already holds model.pt (finished run); nothing to do", flush=True)
        if world > 1:
            import torch.distributed as dist
            dist.destroy_process_group()
        return None
    if main:
        out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    device = f"cuda:{torch.cuda.current_device()}" if world > 1 else cfg.device
    dtype = best_dtype(device)
    ac_dev = "cuda" if str(device).startswith("cuda") else str(device)
    model = (model or Transformer(mcfg)).to(device)
    for att in model.lema_layers():
        att.beta = float(cfg.beta)
    model.set_exact_forward(cfg.exact_forward)
    model.set_checkpoint_blocks(cfg.checkpoint_blocks)
    a0 = cfg.alpha_init if cfg.alpha_init is not None else 1.0 / math.sqrt(mcfg.d_qk)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": other, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=cfg.adam_betas, eps=1e-8)

    # auto-resume: model + optimizer + step from the rolling checkpoint; the data streams
    # are replayed to the same position, so a resumed run sees the same batches
    start_step, best_acc = 0, -1.0
    ck_path = out / "checkpoint.pt"
    if ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step, best_acc = ck["step"] + 1, ck.get("best_acc", -1.0)
        if main:
            print(f"resumed from {ck_path} at step {ck['step']}", flush=True)

    run = model
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # gated-deltanet leaves some fla-internal parameters out of the graph, which
        # plain DDP rejects; static_graph handles the (constant) unused set
        run = DDP(model, broadcast_buffers=False,
                  static_graph=(mcfg.attn == "gated-deltanet"))

    if main:
        (out / "config.json").write_text(json.dumps(
            {"model": asdict(mcfg), "train": asdict(cfg), "task": task.describe()},
            indent=1))
    mode = "a" if start_step else "w"
    logf = open(out / "log.jsonl", mode) if main and cfg.log_every else None
    probef = open(out / "probes.jsonl", mode) if main and cfg.probe_every else None
    # ranks draw independent window streams that together form the global batch; `skip`
    # puts a stream at the resume point without reading what it passes over
    batches = task.batches("train", cfg.batch // world, seed=cfg.seed + 7919 * rank,
                           skip=start_step * cfg.grad_accum, prefetch=4)
    # the probe set is identical on every rank, so the best-checkpoint and early-stop
    # decisions agree everywhere without communication
    probe_set, probe_state = None, {}
    if cfg.probe_every:
        held = task.batches("val", cfg.batch, seed=cfg.seed)
        probe_set = [next(held) for _ in range(cfg.probe_batches)]
    c_max = float(mcfg.d_qk - 1)
    loss_acc = torch.zeros((), device=device)      # window accumulators for log.jsonl
    gnorm_acc = torch.zeros((), device=device)
    n_acc = 0
    t0 = time.time()

    for step in range(start_step, cfg.steps):
        c_now, alpha_now = c_at(cfg, step, c_max), alpha_at(cfg, step, a0)
        model.set_hardening(c=c_now, cap=cfg.alpha_bwd_cap, alpha=alpha_now)
        for g in opt.param_groups:
            g["lr"] = lr_at(cfg, step)

        opt.zero_grad(set_to_none=True)
        for micro in range(cfg.grad_accum):
            x, y, mask, _info = next(batches)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            mask = mask.to(device) if mask is not None else None
            sync = (run.no_sync() if world > 1 and micro < cfg.grad_accum - 1
                    else contextlib.nullcontext())
            with sync:
                with torch.autocast(ac_dev, dtype=dtype):
                    loss = _loss(run, x, y, mask) / cfg.grad_accum
                loss.backward()
            loss_acc += loss.detach()
        log_now = cfg.log_every and (step % cfg.log_every == 0 or step == cfg.steps - 1)
        groups = grad_norms(model) if log_now and main else None    # pre-clip
        gnorm_acc += torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        n_acc += 1
        opt.step()
        if step == 0 and main:
            dead = [n for n, p in model.named_parameters() if p.grad is None]
            if dead:
                print(f"UNUSED PARAMETERS ({len(dead)}): {dead[:20]}", flush=True)

        if log_now:
            if main:
                logf.write(json.dumps({
                    "step": step, "loss": float(loss_acc) / n_acc, "lr": lr_at(cfg, step),
                    "c": c_now, "alpha": alpha_now,
                    "alpha_bwd": min(alpha_now, cfg.alpha_bwd_cap),
                    "grad_norm": float(gnorm_acc) / n_acc, "grad_norm_groups": groups,
                }) + "\n")
                logf.flush()
            loss_acc.zero_()
            gnorm_acc.zero_()
            n_acc = 0

        if probe_set is not None and (step % cfg.probe_every == 0 or step == cfg.steps - 1):
            rec = probe(model, probe_set, device, dtype, probe_state)
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()     # the probe's shapes fragment the allocator
            rec.update(step=step, loss=float(loss.detach()) * cfg.grad_accum,
                       lr=lr_at(cfg, step), c=c_now, alpha=alpha_now,
                       alpha_bwd=min(alpha_now, cfg.alpha_bwd_cap))
            if main:
                probef.write(json.dumps(rec) + "\n")
                probef.flush()
            # accuracy tasks: keep the single best checkpoint and stop once good enough
            acc = rec.get("acc", rec.get("hard_acc"))
            stop = False
            if acc is not None:
                if acc > best_acc:
                    best_acc = acc
                    if main:
                        torch.save({"config": asdict(mcfg),
                                    "state_dict": model.state_dict(),
                                    "step": step, "acc": acc}, out / "model_best.pt")
                stop = best_acc >= cfg.early_stop_acc
            if world > 1:                    # every rank must take the same branch
                import torch.distributed as dist
                flag = torch.tensor([float(stop)], device=device)
                dist.broadcast(flag, 0)
                stop = bool(flag.item())
            if stop:
                break
        if step in cfg.checkpoint_at and main:
            torch.save({"config": asdict(mcfg), "state_dict": model.state_dict(),
                        "step": step}, out / f"model_step{step}.pt")
        if (cfg.resume_every and main and step > start_step
                and step % cfg.resume_every == 0):
            tmp = out / "checkpoint.pt.tmp"          # atomic: never a torn checkpoint
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "best_acc": best_acc,
                        "config": asdict(cfg)}, tmp)
            tmp.replace(ck_path)

    if main:
        torch.save({"config": asdict(mcfg), "state_dict": model.state_dict(),
                    "step": cfg.steps}, out / "model.pt")
        for f in (logf, probef):
            if f is not None:
                f.close()
        print(f"done: {cfg.steps} steps in {(time.time() - t0) / 60:.1f} min -> {out}")
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    return model
