"""The LEMA attention op: exact latest-match, its stick-breaking surrogate, and the code STE.

Conventions, fixed everywhere:
  * codes are +-1 vectors of width d_qk, with sign(0) := +1;
  * attention is STRICTLY causal (j < i) -- a query never matches its own key;
  * an exact match has <q,k> = d_qk and a one-bit mismatch has d_qk - 2, so the hard op
    thresholds at d_qk - 1 and the surrogate's gate threshold c ramps to d_qk - 1;
  * no match => output 0.

Two non-differentiable operations, each with a surrogate used only in the backward pass:

  binarization   q = sign(W_q x)          surrogate  tanh(beta_eff * z)
  matching       latest j with k_j = q_i  surrogate  stick-breaking with gate
                                                     sigmoid(alpha * (<q,k> - c))

Both use the additive-detach identity  y = surrogate + (hard - surrogate).detach()  so the
forward is exactly the hard op and the backward is exactly the surrogate's.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["hard_sign", "binarize", "lema_soft", "lema_hard",
           "lema", "pack_codes", "latest_prev_match"]

_SB_ATTN = None


def _sb_attn():
    global _SB_ATTN
    if _SB_ATTN is None:  # lazy: importing the kernel pulls in triton (GPU only)
        from .kernel.sb_attn import sb_attn
        _SB_ATTN = sb_attn
    return _SB_ATTN


def hard_sign(z: torch.Tensor) -> torch.Tensor:
    """sign(z) with sign(0) := +1, and no gradient path to z."""
    return torch.where(z >= 0, z.new_ones(()), z.new_full((), -1.0))


def binarize(z: torch.Tensor, beta: float) -> torch.Tensor:
    """q = sign(z) forward; backward is d/dz tanh(beta * z / rms(z)).

    The per-position RMS normalization (over the code dimension, DIFFERENTIATED) is the
    only place the pre-activation's scale enters. Note what it is not: the forward is
    `sign(W x)` and sign is scale-invariant, so this changes nothing at inference -- it is
    a training-time device. It gives every position the same effective STE temperature
    regardless of how loud its pre-activations are, and its gradient kills the radial
    component (growing all of a position's bits at once does nothing), which is exactly
    the direction that carries no code information.
    """
    rms = z.float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6).to(z.dtype)
    surr = torch.tanh(beta * (z / rms))
    return surr + (hard_sign(z) - surr).detach()


def lema_soft(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              alpha: float, c: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Stick-breaking surrogate: gate sigmoid(alpha*(<q,k> - c)), latest match first,
    strictly causal. Returns (output, remainder); the remainder is the unclaimed stick,
    i.e. the surrogate's "no match" mass, and is returned detached for diagnostics.

    alpha and c go into the kernel as runtime scalars (see lema/kernel): no fold, so d_qk
    is unconstrained. q, k, v are zero-padded to one common head dim because the kernel
    reads a single dim for all three; tl.dot needs at least 16.
    """
    D, Dv = q.shape[-1], v.shape[-1]
    P = max(D, Dv, 16)
    qp = F.pad(q, (0, P - D)) if P > D else q
    kp = F.pad(k, (0, P - D)) if P > D else k
    vp = F.pad(v, (0, P - Dv)) if P > Dv else v
    o, rem = _sb_attn()(qp.to(torch.bfloat16).contiguous(),
                        kp.to(torch.bfloat16).contiguous(),
                        vp.to(torch.bfloat16).contiguous(),
                        inv_temp=float(alpha), bias=-float(alpha) * float(c),
                        attend_current=False)
    return o[..., :Dv].to(v.dtype), rem.detach().float()


def pack_codes(z: torch.Tensor) -> torch.Tensor:
    """[..., D] -> one int64 code per row, on the input device.

    Thresholds at `z >= 0`, matching sign(0) := +1, so this may be given either binarized
    +-1 codes or the RAW pre-activations -- both yield the same code. That equivalence is
    what lets the exact op skip materialising +-1 tensors at all.

    d_qk <= 64 (enforced by ModelConfig). At exactly 64 the top bit makes the value
    negative; harmless, since only equality and grouping are ever used.
    """
    D = z.shape[-1]
    if D > 64:
        raise ValueError(f"d_qk {D} > 64 cannot pack into a uint64")
    pw = (torch.ones((), dtype=torch.int64, device=z.device)
          << torch.arange(D, device=z.device, dtype=torch.int64))
    return ((z >= 0).to(torch.int64) * pw).sum(-1)


def latest_prev_match(qc: torch.Tensor, kc: torch.Tensor) -> torch.Tensor:
    """For each i, the largest j < i with kc[j] == qc[i], else -1.

    qc, kc: [N, T] integer codes, N independent streams (batch x heads). Sort each row's 2T
    events by (code, position, query-before-key), then take a segmented running maximum of
    key positions inside each code run: O(T log T) time, O(T) memory, and no T x T matrix
    ever exists. Batched, on whatever device the codes live on.
    """
    N, T = qc.shape
    dev = qc.device
    code = torch.cat([qc, kc], dim=1)                              # [N, 2T]
    pos = torch.arange(T, device=dev).repeat(2)
    is_key = torch.cat([torch.zeros(T, dtype=torch.long, device=dev),
                        torch.ones(T, dtype=torch.long, device=dev)])
    o1 = (pos * 2 + is_key).expand(N, -1).argsort(dim=1, stable=True)
    o2 = code.gather(1, o1).argsort(dim=1, stable=True)             # two stable passes
    order = o1.gather(1, o2)
    cs = code.gather(1, order)
    ks = is_key.expand(N, -1).gather(1, order).bool()
    ps = pos.expand(N, -1).gather(1, order)
    new_seg = torch.ones_like(cs, dtype=torch.bool)
    new_seg[:, 1:] = cs[:, 1:] != cs[:, :-1]
    segid = new_seg.long().cumsum(1) - 1
    BIG = T + 2                                # runs cannot bleed into one another
    x = torch.where(ks, ps + 1, torch.zeros_like(ps)) + segid * BIG
    cm = x.cummax(dim=1).values - segid * BIG
    # Scatter every event, sending KEY events to a dump column instead of masking them out.
    # Boolean-mask indexing (`ps[~ks]`) would be the obvious way to keep only the queries,
    # but its output shape depends on the data, which makes the whole function impossible
    # to capture in one graph -- and this runs inside a compiled prefill segment.
    # Each query position is written exactly once; the dump column is discarded.
    idx = torch.where(ks, torch.full_like(ps, T), ps)
    return ps.new_full((N, T + 1), -1).scatter_(1, idx, cm - 1)[:, :T]


def lema_hard(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, store=None) -> torch.Tensor:
    """Exact latest match: o_i = v_j for the largest j < i with k_j == q_i, else 0.

    `q` and `k` may be binarized codes OR raw pre-activations: pack_codes thresholds at
    zero either way, and nothing else here looks at their values.

    With a `store` (inference) the whole rule is one exchange against it: every query is
    looked up and every key inserted, in position order, so a query sees the keys before it
    -- in this window and in everything fed before -- and not its own. Without one this is
    a self-contained forward over the window, batched over B and entirely on device, in
    O(T log T) time and O(T) memory: there is no T x T matrix here, which is what lets the
    same code serve a 512-token training eval and a 10^9-token stream. The match is
    piecewise constant in q and k, so it carries no gradient; the gather keeps the value
    path differentiable, the honest almost-everywhere gradient of the hard op.
    """
    B, H, T, _ = q.shape
    with torch.no_grad():
        qc, kc = pack_codes(q), pack_codes(k)                       # [B, H, T]
    if store is not None:
        return store.exchange(qc, kc, v)
    with torch.no_grad():
        src = latest_prev_match(qc.reshape(B * H, T), kc.reshape(B * H, T)).view(B, H, T)
    y = torch.gather(v, 2, src.clamp_min(0).unsqueeze(-1).expand_as(v))
    return y * (src >= 0).unsqueeze(-1).to(v.dtype)


def lema(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
         alpha_fwd: float, alpha_bwd: float, c: float, exact_fwd: bool = False) -> torch.Tensor:
    """The training-time LEMA op: the surrogate with a SPLIT temperature. The forward
    uses alpha_fwd and the backward alpha_bwd <= alpha_fwd, so a large alpha_fwd brings
    the forward arbitrarily close to the exact op while the capped alpha_bwd keeps the
    gate's gradient from collapsing onto the exactly-matching pair. `exact_fwd` is the
    limit alpha_fwd = infinity: the forward IS the exact op and only the backward is the
    surrogate at alpha_bwd -- the straight-through treatment of the matching op itself.
    The exact op alone is `lema_hard`.
    """
    y, _ = lema_soft(q, k, v, alpha_bwd, c)               # carries the gradient
    if exact_fwd:
        return y + (lema_hard(q, k, v) - y).detach()
    if alpha_fwd > alpha_bwd:
        with torch.no_grad():
            y_fwd, _ = lema_soft(q, k, v, alpha_fwd, c)
        y = y + (y_fwd - y).detach()                      # value at alpha_fwd, grads at alpha_bwd
    return y
