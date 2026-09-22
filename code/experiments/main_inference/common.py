"""Shared machinery for the generation and prefill benchmarks (measure_generation.py,
measure_prefill.py).

The section this feeds (paper Sec. "Efficient inference") makes one claim: a LEMA
transformer's per-token cost is independent of context length, because attention is a
dictionary lookup whose table lives OFF the GPU, in host RAM. So we measure two decode
methods that differ only in where the state lives:

    softmax    the baseline: a preallocated key/value cache in VRAM (KVCache), read in
               full every step; OOMs when the kv cache fills the card.
    lema-ram   the LEMA store: one C++ hash table in host RAM shared by every layer and
               head (HashStore).

Everything here is model-agnostic: NOTHING fixes a model size. Geometry comes entirely
from the command line (--depth --dim --num-heads --head-dim ...), so the paper's size
sweep is a handful of shell commands, not a table baked into the code.

Both methods are PROVISIONED, once, from a byte budget the caller states (`--store-gb`):
VRAM for the softmax kv cache, host RAM for the LEMA hash table. Nothing is ever
reallocated mid-curve. This is what the deployed thing does -- you know your memory, not
your final context length -- and it is symmetric across the two. So the curves end
differently: softmax OOMs (its budget converts to a fixed number of tokens and the next
one does not fit), LEMA fills (linear probing lengthens as the table does: ~6 probes on a
miss at 70% occupancy, ~50 at 90%, ~200 at 95%, so its curve turns up at the end). Both
are real ceilings on the same budget; only one of them is an allocation failure.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # .../code on sys.path

import torch

from lema import ModelConfig, Transformer, load_trained

HERE = Path(__file__).resolve().parent
CARRIER_BYTES = 2          # the store carries bf16 values as uint16 (numpy has no bf16)
DTYPE_BYTES = 2            # bf16 everywhere on the GPU side


# --------------------------------------------------------------------------- geometry
def add_geometry_args(p: argparse.ArgumentParser):
    """The knobs that define a model. `--depth` and `--num-heads` are always required; the
    residual width is `--dim` (alias `--width`) OR `--head-dim x --num-heads`. Head widths
    default to dim // num_heads and can be split with --d-qk/--d-v; LEMA needs d_qk <= 64."""
    g = p.add_argument_group("model geometry (random model; nothing is fixed in code)")
    g.add_argument("--checkpoint", type=str, default=None,
                   help="a trained run dir from experiments/main_lm (holding model.pt); "
                        "its geometry and attention kind override the flags below")
    g.add_argument("--depth", type=int, help="number of transformer blocks")
    g.add_argument("--dim", "--width", type=int, dest="dim", help="residual width")
    g.add_argument("--num-heads", type=int, help="attention heads per layer")
    g.add_argument("--num-kv-heads", type=int, default=0,
                   help="key/value heads per layer (grouped-query attention, softmax "
                        "only); 0 = one per query head")
    g.add_argument("--head-dim", type=int, default=None,
                   help="sets d_qk=d_v (and dim=num_heads*head_dim if --dim is omitted)")
    g.add_argument("--d-qk", type=int, default=0, help="override query/key width")
    g.add_argument("--d-v", type=int, default=0, help="override value width")
    g.add_argument("--dim-ff", type=int, default=0, help="MLP width (default 8*dim/3)")
    g.add_argument("--vocab", type=int, default=50257, help="vocabulary size")
    g.add_argument("--seed", type=int, default=0)


def _resolve_geometry(a) -> dict:
    if a.head_dim is not None:
        dim = a.dim if a.dim else a.num_heads * a.head_dim
        d_qk = a.d_qk or a.head_dim
        d_v = a.d_v or a.head_dim
    else:
        dim, d_qk, d_v = a.dim, a.d_qk, a.d_v
    if not a.depth or not a.num_heads or not dim:
        raise SystemExit("geometry underspecified: need --depth, --num-heads and "
                         "(--dim or --head-dim). See the README for example commands.")
    return dict(depth=a.depth, dim=dim, num_heads=a.num_heads, num_kv_heads=a.num_kv_heads,
                d_qk=d_qk, d_v=d_v, dim_ff=a.dim_ff, vocab_size=a.vocab)


def _load_checkpoint(run_dir, device: str = "cuda"):
    """Load a model trained by experiments/main_lm: a run dir holding `model.pt` with `config`
    + `state_dict`. Geometry (and the attention kind) come from the checkpoint, so the
    --depth/--dim/... flags are ignored, and the kind fixes which --method applies (a 'lema'
    checkpoint -> lema-ram, a 'softmax' checkpoint -> softmax)."""
    model, cfg = load_trained(run_dir, device)
    return model.to(torch.bfloat16).requires_grad_(False), cfg


def make_model(a, attn: str, device: str = "cuda"):
    """A random model of the requested geometry, built directly on the GPU in bf16.

    Built on-device and in bf16 from the start on purpose: an 8B model in fp32 is 32 GB
    on the host and would not fit on the card at all. sign(Wx) is scale-invariant, so bf16 init changes nothing the exact op sees, and
    inference runs bf16 regardless -- this is the dtype the numbers should be in.
    """
    if a.checkpoint is not None:
        return _load_checkpoint(a.checkpoint, device)
    geo = _resolve_geometry(a)
    pos = "rope" if attn == "softmax" else "nope"
    cfg = ModelConfig(attn=attn, pos=pos, **geo)
    torch.manual_seed(a.seed)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = Transformer(cfg)
    finally:
        torch.set_default_dtype(prev)
    # requires_grad_(False) is NOT cosmetic here (and _load_checkpoint already does it):
    # without it every decode builds an autograd graph, the compiled segments save their
    # activations for a backward that never comes, and VRAM climbs ~13 hidden-sized tensors
    # per layer per step -- 482 MB per grid point at 8B, which killed that curve outright.
    # It also put autograd bookkeeping inside every timed step.
    model.eval().requires_grad_(False)
    if attn == "lema":
        model.set_hard(True)
    return model, cfg


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def weight_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


# ------------------------------------------------------------------- capacity accounting
# Everything here converts BETWEEN a memory budget and a context length. That direction is
# the point: both cache kinds are now provisioned once from a budget the caller states
# (--store-gb) rather than grown to fit a context, so the question is always "how far does
# this many bytes get me", never "how many bytes will this context need".
def partitions(cfg, batch: int = 1) -> int:
    """Hash partitions sharing the one table: one per (layer, batch item, head). Under
    worst-case codes every token adds one entry to each of them."""
    return cfg.depth * batch * cfg.num_heads


def slot_bytes(cfg) -> int:
    """One store slot: a 16-byte (code, group) header + one value row, interleaved so a
    probe fetches both."""
    return 16 + cfg.d_v * CARRIER_BYTES


def store_capacity(cfg, budget: int) -> int:
    """Slots that fit in `budget` bytes. Exact: the store indexes with a multiply, not a
    mask, so capacity is not rounded to a power of two and the whole budget is usable."""
    return max(16, budget // slot_bytes(cfg))


def kv_bytes(cfg, maxlen: int, batch: int = 1) -> int:
    """Bytes the softmax kv cache reserves for `maxlen` tokens (keys+values, all layers)."""
    return cfg.depth * batch * cfg.num_kv_heads * (cfg.d_qk + cfg.d_v) * DTYPE_BYTES * maxlen


def kv_capacity(cfg, budget: int, batch: int = 1) -> int:
    """Tokens the softmax kv cache can hold in `budget` bytes -- where its curve ends."""
    per_tok = cfg.depth * batch * cfg.num_kv_heads * (cfg.d_qk + cfg.d_v) * DTYPE_BYTES
    return budget // per_tok


# --------------------------------------------------------------------- worst-case codes
def worst_case_codes(depth: int):
    """A `LemaCache` codes hook that replaces every layer's codes -- queries AND keys --
    with fresh uniformly random 64-bit codes that never repeat: every lookup lands on a
    random slot and probes to the end of its run (a miss), every insert claims a fresh
    random slot. A real step does exactly one of each per head, so this is the worst
    access pattern any model could produce, with no locality for the memory system to
    exploit. Randomising only the queries would let the inserts land on the same warm
    line every step -- half of the true worst case.

    The codes of all `depth` layers are drawn at layer 0 with ONE kernel into a buffer
    that the later layers only view, so a decode step carries a single extra kernel in
    its first CUDA graph and nothing in the others (48 tiny per-layer draws cost a
    measurable 0.1 ms per token; the numpy draw on the host that preceded them about as
    much). torch registers its generator with the graph, so every replay draws afresh.
    """
    bufs = {}

    def hook(layer, qc, kc):
        key = (qc.device, tuple(qc.shape))
        if layer == 0 or key not in bufs:
            if key not in bufs:
                bufs[key] = torch.empty((depth, 2) + tuple(qc.shape), dtype=qc.dtype,
                                        device=qc.device)
            bufs[key].random_()
        b = bufs[key]
        return b[layer, 0], b[layer, 1]
    return hook


def meminfo_available() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def vmswap_kb() -> int:
    """Pages of this process currently swapped out, in kB (VmSwap of /proc/self/status),
    recorded with every curve point. A LEMA curve is only valid while this stays 0: a
    swapped table turns probes into disk reads. Run the benchmarks with swap off."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmSwap:"):
                return int(line.split()[1])
    except Exception:
        pass
    return -1


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                              capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def machine_meta() -> dict:
    p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "gpu": p.name if p else None,
        "vram_gb": round(p.total_memory / 1e9, 1) if p else None,
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": round(meminfo_available() / 1e9, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "host": platform.node(),
    }


def reset_jsonl(path):
    """Truncate/create the output file at the start of a run: one run = one curve = one
    file, so re-running with the same --out overwrites rather than accumulating."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def append_jsonl(path, record: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
