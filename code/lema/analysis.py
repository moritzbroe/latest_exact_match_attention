"""Looking inside a model: the attention state of every layer on one forward pass.

`trace(model, x)` runs `x` (`[B, T]` tokens) through the model and returns the logits
and one record per layer, computed from the same pre-activations the forward uses:

    LEMA layer     qc, kc  [B, H, T] int64   query and key codes
                   src     [B, H, T] int64   the position each query reads, -1 on no match
                   (exactly what the exact op does, whether or not the model is in hard mode)
    softmax layer  w       [B, H, T, T]      attention weights (T x T per head: short
                                             sequences only)
    other layers   {}                        (gated-deltanet has no attention state)

`reduce(layer, rec) -> value`, if given, is applied inside the hook and its result kept
instead of the record, so a long sequence's codes and sources never pile up across layers.
The probes, the attention pictures and the code census all go through here, so they look
at the same quantities in the same way.
"""
from __future__ import annotations

import math

import torch

from .attention import latest_prev_match, pack_codes
from .model import rope


@torch.no_grad()
def trace(model, x, reduce=None, return_logits=True):
    cfg = model.cfg
    recs = [None] * cfg.depth

    def hook(li):
        def fn(blk, inp, _out):
            att = blk.attn
            if att.inner is not None or att.kind == "sb":
                rec = {}
            else:
                zq, zk, _ = att.project(blk.norm1(inp[0]))
                if att.kind == "lema":
                    qc, kc = pack_codes(zq), pack_codes(zk)
                    B, H, T = qc.shape
                    src = latest_prev_match(qc.reshape(B * H, T), kc.reshape(B * H, T))
                    rec = {"qc": qc, "kc": kc, "src": src.view(B, H, T)}
                else:
                    if cfg.pos == "rope":
                        zq, zk = rope(zq, cfg.rope_base), rope(zk, cfg.rope_base)
                    s = zq.float() @ zk.float().transpose(-1, -2) / math.sqrt(cfg.d_qk)
                    T = s.shape[-1]
                    causal = torch.ones(T, T, dtype=torch.bool, device=s.device).triu(1)
                    rec = {"w": torch.softmax(s.masked_fill(causal, float("-inf")), -1)}
            recs[li] = reduce(li, rec) if reduce is not None else rec
        return fn

    handles = [blk.register_forward_hook(hook(li)) for li, blk in enumerate(model.blocks)]
    try:
        logits = model(x, return_logits=return_logits)
    finally:
        for h in handles:
            h.remove()
    return logits, recs
