"""Held-out probes, written to probes.jsonl every `probe_every` training steps.

Baselines get CE and accuracy on the probe set. LEMA models get soft CE and accuracy
(the surrogate at the current forward alpha; omitted when the forward is already the
exact op) and hard CE and accuracy (the exact op), plus per layer, averaged over the
probe set:
    match    the fraction of queries that find an earlier key under the exact op;
    prevhit  recall only: the fraction of value tokens whose query reads the position
             directly before it, its own key -- the first-layer half of the circuit;
    kflip    the fraction of key codes that changed since the previous probe.
"""
from __future__ import annotations

import torch

from .analysis import trace


def _score(logits, y, mask):
    if mask is not None:
        sel = mask.flatten()
        logits, y = logits.flatten(0, 1)[sel], y.flatten()[sel]
    else:
        logits, y = logits.flatten(0, 1), y.flatten()
    ce = torch.nn.functional.cross_entropy(logits.float(), y)
    return float(ce), float((logits.argmax(-1) == y).float().mean())


def _mean(vals):
    return float(sum(vals) / len(vals))


@torch.no_grad()
def probe(model, batches, device, dtype, state: dict) -> dict:
    """`batches`: the fixed probe set, `(x, y, mask, info)` tuples of equal size; `state`
    persists between calls (it holds the previous probe's key codes for `kflip`)."""
    model.eval()
    is_lema = model.cfg.attn == "lema"
    was_hard = [a.hard for a in model.lema_layers()]
    exact = is_lema and all(a.exact_fwd for a in model.lema_layers())
    soft, hard, layers = [], [], []
    for bi, (x, y, mask, info) in enumerate(batches):
        x, y = x.to(device), y.to(device)
        mask = mask.to(device) if mask is not None else None
        with torch.autocast(device, dtype=dtype):
            if is_lema and not exact:
                model.set_hard(False)
                soft.append(_score(model(x), y, mask))
            model.set_hard(True)
            logits, recs = trace(model, x)
        hard.append(_score(logits, y, mask))
        if is_lema:
            layers.append(_code_stats(recs, info, state, bi))
    for att, h in zip(model.lema_layers(), was_hard):
        att.hard = h
    model.train()
    if not is_lema:
        return {"ce": _mean([h[0] for h in hard]), "acc": _mean([h[1] for h in hard])}
    rec = {"hard_ce": _mean([h[0] for h in hard]), "hard_acc": _mean([h[1] for h in hard])}
    if soft:
        rec["soft_ce"], rec["soft_acc"] = _mean([s[0] for s in soft]), _mean([s[1] for s in soft])
    rec["layers"] = [{k: round(_mean([ls[li][k] for ls in layers]), 4) for k in layers[0][li]}
                     for li in range(len(layers[0]))]
    rec["match_rate"] = round(_mean([l["match"] for l in rec["layers"]]), 4)
    return rec


def _code_stats(recs, info, state, bi):
    vpos = info.get("value_pos") if info else None
    stats = []
    for li, r in enumerate(recs):
        src, kc = r["src"], r["kc"]
        prev = state.get((bi, li))
        kflip = float((kc != prev).float().mean()) if prev is not None else 0.0
        state[(bi, li)] = kc
        s = {"match": round(float((src >= 0).float().mean()), 4), "kflip": round(kflip, 4)}
        if vpos is not None:
            v = torch.as_tensor(vpos, device=src.device)
            s["prevhit"] = round(float((src[:, :, v] == (v - 1)).float().mean()), 4)
        stats.append(s)
    return stats
