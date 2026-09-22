"""The transformer. Deliberately boring outside the attention op.

Pre-RMSNorm, SwiGLU MLP, untied head, no dropout, no biases. The only choices are the
attention kind and the positional scheme, so that LEMA and its baselines differ in
exactly one component and share everything else.

Geometry: `d_qk` and `d_v` default to dim // num_heads, `dim_ff` to 8 * dim / 3.
`vocab_size` and `dim_ff` are rounded up to a multiple of 128 so every bf16 GEMM stays
on the tensor-core fast path; the padded vocabulary rows are never targets.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn

from .attention import binarize, lema, lema_hard, lema_soft, pack_codes

ATTN_KINDS = ("lema", "sb", "softmax", "gated-deltanet")
POS_KINDS = ("nope", "rope")


def _pad128(n) -> int:
    return -(-int(n) // 128) * 128


@dataclass
class ModelConfig:
    vocab_size: int
    depth: int
    dim: int
    num_heads: int
    d_qk: int = 0            # 0 -> dim // num_heads
    d_v: int = 0             # 0 -> dim // num_heads
    dim_ff: int = 0          # 0 -> 8 * dim / 3
    num_kv_heads: int = 0    # 0 -> num_heads; fewer = grouped-query attention (softmax only)
    attn: str = "lema"       # lema | sb (plain stick-breaking) | softmax | gated-deltanet
    pos: str = "nope"        # nope | rope (softmax only)
    rope_base: float = 10000.0   # RoPE base; raised for context extension (Code Llama: 1e6)
    gdn_expand_v: int = 2    # gated-deltanet value expansion (the published default is 2)
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.attn not in ATTN_KINDS:
            raise ValueError(f"attn must be one of {ATTN_KINDS}, got {self.attn!r}")
        if self.pos not in POS_KINDS:
            raise ValueError(f"pos must be one of {POS_KINDS}, got {self.pos!r}")
        if (not self.d_qk or not self.d_v) and self.dim % self.num_heads:
            raise ValueError(f"dim {self.dim} is not divisible by num_heads {self.num_heads}")
        head = self.dim // self.num_heads
        self.d_qk = self.d_qk or head
        self.d_v = self.d_v or head
        self.dim_ff = _pad128(self.dim_ff or 8 * self.dim / 3)
        self.vocab_size = _pad128(self.vocab_size)
        self.num_kv_heads = self.num_kv_heads or self.num_heads
        if self.num_heads % self.num_kv_heads:
            raise ValueError(f"num_heads {self.num_heads} is not a multiple of num_kv_heads "
                             f"{self.num_kv_heads}")
        if self.num_kv_heads != self.num_heads and self.attn != "softmax":
            raise ValueError("grouped-query attention (num_kv_heads < num_heads) is "
                             "implemented for softmax attention only")
        if self.attn == "lema" and self.d_qk > 64:
            raise ValueError(f"d_qk {self.d_qk} > 64: codes must pack into one uint64")
        if self.pos == "rope" and self.d_qk % 2:
            raise ValueError("rope needs an even d_qk")

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """A config from a saved json, ignoring unknown fields."""
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def summary(self) -> str:
        kv = f" kv{self.num_kv_heads}" if self.num_kv_heads != self.num_heads else ""
        return (f"{self.depth}L d{self.dim} h{self.num_heads}{kv} d_qk {self.d_qk} "
                f"d_v {self.d_v} dim_ff {self.dim_ff} attn {self.attn} pos {self.pos}")


class SwiGLU(nn.Module):
    def __init__(self, dim: int, dim_ff: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim_ff, bias=False)
        self.up = nn.Linear(dim, dim_ff, bias=False)
        self.down = nn.Linear(dim_ff, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def best_dtype(device="cuda") -> torch.dtype:
    """bf16 wherever the hardware has it (training runs under bf16 autocast, and a
    bf16-trained model can hold values outside fp16's range), fp16 on older cards,
    fp32 on the CPU."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


_ROPE_TABLES: dict = {}


def _rope_tables(half: int, upto: int, device, base: float = 10000.0):
    """cos/sin for positions [0, upto), cached per (half, base, device) and grown
    geometrically."""
    key = (half, float(base), str(device))
    hit = _ROPE_TABLES.get(key)
    if hit is None or hit[0].shape[0] < upto:
        n = max(upto, 2 * (hit[0].shape[0] if hit else 0), 4096)
        freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32)
                          * (-math.log(float(base)) / half))
        ang = torch.outer(torch.arange(n, device=device, dtype=torch.float32), freqs)
        _ROPE_TABLES[key] = (ang.cos(), ang.sin())
    return _ROPE_TABLES[key]


def rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary positions, rotate-half, for `[B, H, T, D]`. Tables are built for whatever
    length is asked, so evaluating beyond the training length is defined."""
    T, D = x.shape[2], x.shape[3]
    half = D // 2
    cos, sin = _rope_tables(half, T, x.device, base)
    cos, sin = cos[:T], sin[:T]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


def _gated_deltanet(cfg: ModelConfig):
    try:
        from fla.layers import GatedDeltaNet
    except ImportError as e:                    # optional dependency
        raise ImportError("attn='gated-deltanet' needs flash-linear-attention "
                          "(`pip install flash-linear-attention`)") from e
    return GatedDeltaNet(hidden_size=cfg.dim, num_heads=cfg.num_heads, head_dim=cfg.d_v,
                         expand_v=cfg.gdn_expand_v, mode="chunk")


class Attention(nn.Module):
    """One attention block. `kind` fixes the op; the projections are shared.

    The hardening state of a LEMA block -- `alpha` (surrogate temperature), `cap` (the
    backward temperature is min(alpha, cap)), `c` (gate threshold) -- is set by `Transformer.set_hardening` every training
    step; `beta` (STE temperature of the binarization) is set once by the trainer. `hard` switches to the exact op, `exact_fwd` to the exact op in the
    forward pass with the surrogate in the backward only.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.kind = cfg.attn
        if cfg.attn == "gated-deltanet":       # owns its own projections
            self.inner = _gated_deltanet(cfg)
            return
        self.inner = None
        self.q_proj = nn.Linear(cfg.dim, cfg.num_heads * cfg.d_qk, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.num_kv_heads * cfg.d_qk, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.num_kv_heads * cfg.d_v, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.d_v, cfg.dim, bias=False)
        self.alpha, self.cap, self.c, self.beta = 1.0, 2.0, 0.0, 2.0
        self.hard = False
        self.exact_fwd = False

    def project(self, x):
        """`[B, T, dim]` -> query and key pre-activations and values, `[B, H, T, d]`."""
        B, T, _ = x.shape
        c = self.cfg
        zq = self.q_proj(x).view(B, T, c.num_heads, c.d_qk).transpose(1, 2)
        zk = self.k_proj(x).view(B, T, c.num_kv_heads, c.d_qk).transpose(1, 2)
        v = self.v_proj(x).view(B, T, c.num_kv_heads, c.d_v).transpose(1, 2)
        return zq, zk, v

    def merge(self, y):
        B, _, T, _ = y.shape
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))

    def forward(self, x):
        if self.inner is not None:
            T = x.shape[1]
            if self.training and T <= 64:
                # fla forces inference mode at q_len <= 64, which asserts in training;
                # causal, so padding the tail changes nothing for real positions
                out = self.inner(F.pad(x, (0, 0, 0, 65 - T)))
                out = out[0] if isinstance(out, tuple) else out
                return out[:, :T]
            out = self.inner(x)
            return out[0] if isinstance(out, tuple) else out
        zq, zk, v = self.project(x)
        if self.kind == "lema":
            if self.hard:
                # the exact op reads the codes through sign(z) only: no STE, no +-1 tensors
                y = lema_hard(zq, zk, v)
            else:
                qb, kb = binarize(zq, self.beta), binarize(zk, self.beta)
                y = lema(qb, kb, v, alpha_fwd=self.alpha, alpha_bwd=min(self.alpha, self.cap),
                         c=self.c, exact_fwd=self.exact_fwd)
        elif self.kind == "sb":                    # plain stick-breaking: no codes, no gate
            y, _ = lema_soft(zq, zk, v, 1.0 / math.sqrt(self.cfg.d_qk), 0.0)
        else:                                      # softmax
            if self.cfg.pos == "rope":
                zq, zk = rope(zq, self.cfg.rope_base), rope(zk, self.cfg.rope_base)
            y = F.scaled_dot_product_attention(
                zq, zk, v, is_causal=True, scale=1.0 / math.sqrt(self.cfg.d_qk),
                enable_gqa=self.cfg.num_kv_heads != self.cfg.num_heads)
        return self.merge(y)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg.dim, cfg.dim_ff)
        self.seg = None            # compiled (pre, post) segments; see compile_inference

    # At inference a layer is two GPU segments around one call torch.compile cannot trace:
    # the host round trip to the hash table (LEMA) or the fused kv-cache kernel (softmax).
    # Naming the boundary lets each side be compiled and captured into a CUDA graph.
    def gpu_pre(self, x):
        """LEMA segment 1: normalise, project, pack the codes the host store needs."""
        zq, zk, v = self.attn.project(self.norm1(x))
        return pack_codes(zq), pack_codes(zk), v

    def gpu_post(self, x, got):
        """LEMA segment 2: merge the retrieved values and finish the layer."""
        x = x + self.attn.merge(got)
        return x + self.mlp(self.norm2(x))

    def kv_pre(self, x):
        """Softmax segment 1: normalise and project as `[B, T, H, d]` for the kernel."""
        B, T, _ = x.shape
        c, at = self.attn.cfg, self.attn
        h = self.norm1(x)
        return (at.q_proj(h).view(B, T, c.num_heads, c.d_qk),
                at.k_proj(h).view(B, T, c.num_kv_heads, c.d_qk),
                at.v_proj(h).view(B, T, c.num_kv_heads, c.d_v))

    def kv_post(self, x, y):
        """Softmax segment 2: merge the heads and finish the layer."""
        B, T, _ = x.shape
        x = x + self.attn.o_proj(y.reshape(B, T, -1))
        return x + self.mlp(self.norm2(x))

    def forward(self, x, store=None):
        if store is not None:
            pre, post = self.seg or (self.gpu_pre, self.gpu_post)
            qc, kc, v = pre(x)
            return post(x, store.exchange(qc, kc, v))
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.norm_f = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.checkpoint_blocks = False       # recompute each block in the backward (memory)
        self._init_weights()

    def _init_weights(self):
        """Gaussian, std 1/sqrt(fan_in) for every Linear and 1/sqrt(dim) for the
        embedding; the two residual-branch outputs of each block additionally scaled by
        1/sqrt(2 * depth) so the residual stream's scale is depth-invariant at init."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.weight.shape[1]))
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(self.cfg.dim))
        for b in self.blocks:
            attn_out = b.attn.inner.o_proj if b.attn.inner is not None else b.attn.o_proj
            for lin in (attn_out, b.mlp.down):
                lin.weight.data.mul_(1.0 / math.sqrt(2 * self.cfg.depth))

    # ---------------------------------------------------------------- hardening
    def lema_layers(self):
        return [b.attn for b in self.blocks if b.attn.kind == "lema"]

    def set_hardening(self, c: float, cap: float, alpha: float):
        """This step's gate threshold, backward cap and surrogate temperature."""
        for att in self.lema_layers():
            att.c, att.cap, att.alpha = float(c), float(cap), float(alpha)

    def set_hard(self, hard: bool = True):
        for att in self.lema_layers():
            att.hard = hard

    def set_exact_forward(self, exact: bool = True):
        for att in self.lema_layers():
            att.exact_fwd = exact

    def set_checkpoint_blocks(self, on: bool = True):
        """Activation checkpointing per block: the same computation, ~1/3 more of it,
        for a fraction of the activation memory. Training only, plain forward only."""
        self.checkpoint_blocks = on

    # ---------------------------------------------------------------- inference
    def new_cache(self, place: str = "ram", **kw):
        """A retrieval cache for this model: a LEMA cache lives in host RAM (`place`
        must be "ram"), the softmax baseline's key/value cache in VRAM. Provisioned once
        and never resized: pass `max_cache_len=` (the longest sequence to serve) or, for a
        LEMA cache, `capacity=` slots to spend a memory budget directly; `batch=` for
        batched streams."""
        from .cache import KVCache, LemaCache
        if not self.lema_layers():
            if place not in ("vram", "ram"):
                raise ValueError(f"attn={self.cfg.attn!r} decodes through a VRAM key/value "
                                 f"cache; place={place!r} is not an implementation of it")
            return KVCache(self, max_len=kw.get("max_cache_len", 8192),
                           batch=kw.get("batch", 1), num_splits=kw.get("num_splits", 0))
        if place != "ram":
            raise ValueError(f"a LEMA cache lives in host RAM; place={place!r} is not an "
                             f"implementation of it")
        p = next(self.parameters())
        kw.setdefault("dtype", p.dtype)          # store exactly what the model emits
        return LemaCache(self.cfg, pin=p.is_cuda, **kw)

    def compile_inference(self, mode: str = "default"):
        """Compile each layer's two GPU segments (kernel fusion only; launch overhead is
        removed by the CUDA graphs in lema/decode.py). Weights are baked in: do not
        mutate parameters afterwards."""
        for b in self.blocks:
            pre, post = ((b.gpu_pre, b.gpu_post) if b.attn.kind == "lema"
                         else (b.kv_pre, b.kv_post))
            b.seg = (torch.compile(pre, mode=mode, fullgraph=True),
                     torch.compile(post, mode=mode, fullgraph=True))
        return self

    def forward(self, idx, *, cache=None, return_logits=True):
        """Run `idx` through the model.

            forward(x)                                batched, on device, no state
            forward(x, cache=c, return_logits=False)  prefill: fills c, no logits
            forward(next, cache=c)                    one stream, extending c

        With a `cache`, queries match keys from before this call, so a long sequence
        can be fed in pieces by a loop in the caller. `return_logits=False` skips the
        `T x vocab` logits of a prefill.
        """
        from .cache import KVCache
        if isinstance(cache, KVCache):
            return self._run_kv(idx, cache, return_logits)
        if cache is not None and not (self.lema_layers()
                                      and all(a.hard for a in self.lema_layers())):
            raise ValueError("a cache needs an all-LEMA model in hard mode")
        x = self.tok_emb(idx)
        ckpt = self.checkpoint_blocks and cache is None and self.training and torch.is_grad_enabled()
        for li, blk in enumerate(self.blocks):
            if ckpt:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x, store=(cache.layer(li) if cache is not None else None))
        if cache is not None:
            cache.t += idx.shape[-1]
        return self.lm_head(self.norm_f(x)) if return_logits else None

    @torch.no_grad()
    def _run_kv(self, idx, cache, return_logits=True):
        """The softmax baseline's cached forward: `idx` (1 or many tokens) appended to the
        preallocated key/value buffers and attended to by `flash_attn_with_kvcache`,
        which is told the fill level as a device tensor, applies the rotary tables at
        those positions and reads only the live prefix. Nothing depends on the context
        length from python's point of view, which lets `lema.decode.KVDecoder` capture
        a step into one CUDA graph. `num_splits=32` for single-token steps: the kernel's
        own heuristic sizes the split for the buffer length, not the live one."""
        from flash_attn import flash_attn_with_kvcache
        if self.cfg.attn != "softmax":
            raise NotImplementedError(f"kv decode for attn={self.cfg.attn!r}")
        T = idx.shape[-1]
        x = self.tok_emb(idx.reshape(-1, T))
        # the kernel sees the live range: for a single-token step the cache sliced to the
        # tokens in it plus this one, split into one wave of blocks (`decode_splits`), for a
        # prefill chunk the whole buffer, where the queries supply the parallelism
        # (the kernel folds the query heads of a group into its query length and wants the
        # cache at least that long, hence the floor)
        floor = max(2, self.cfg.num_heads // self.cfg.num_kv_heads)
        L = min(cache.max_len, max(floor, cache.n + T)) if T == 1 else cache.max_len
        splits = (cache.num_splits or decode_splits(cache.batch, self.cfg.num_kv_heads)) \
            if T == 1 else 1
        for li, blk in enumerate(self.blocks):
            pre, post = blk.seg or (blk.kv_pre, blk.kv_post)
            q, k, v = pre(x)
            y = flash_attn_with_kvcache(
                q, cache.k[li][:, :L], cache.v[li][:, :L], k, v, rotary_cos=cache.cos,
                rotary_sin=cache.sin, cache_seqlens=cache.pos, causal=True,
                rotary_interleaved=False, num_splits=splits)
            x = post(x, y)
        cache.pos.add_(T)
        cache.n += T
        return self.lm_head(self.norm_f(x)) if return_logits else None

    def n_params(self, embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if not embedding:
            n -= self.tok_emb.weight.numel() + self.lm_head.weight.numel()
        return n


def decode_splits(batch: int, kv_heads: int) -> int:
    """Split-KV count for a single-token step. The decode kernel runs one thread block per
    key/value head, sequence and split, each streaming its share of the live range, so with
    few of them (small batch, grouped queries) the range is split into as many pieces as
    fill the SMs once: one full wave of long blocks, never a second wave, since blocks
    carry a fixed cost of a few hundred tokens' worth of streaming each and a wave that is
    nearly empty costs as much as a full one. This is how serving kernels partition the
    live range per step (one chunk per SM). At most the kernel's 128 splits."""
    base = batch * kv_heads
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    return max(1, min(128, sms // base))


def load_trained(path, device="cuda") -> tuple[Transformer, ModelConfig]:
    """A finished training run, ready to evaluate: `path` is the run directory or a
    checkpoint file in it. A LEMA model comes back in hard mode -- the exact op is the
    trained model; the surrogate is a training device. Unknown parameters in the
    checkpoint are dropped."""
    path = Path(path)
    if path.is_dir():
        path = path / "model.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ck["config"])
    model = Transformer(cfg).to(device).eval()
    keep = model.state_dict().keys()
    model.load_state_dict({k: v for k, v in ck["state_dict"].items() if k in keep})
    if cfg.attn == "lema":
        model.set_hardening(c=float(cfg.d_qk - 1), cap=2.0, alpha=10.0)
        model.set_hard(True)
    return model, cfg
