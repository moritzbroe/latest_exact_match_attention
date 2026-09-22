import math

import torch
import triton.language as tl

from .sb_bwd import _bwd
from .sb_fwd import _fwd


FWD_BLOCK_M: tl.constexpr = 64
FWD_BLOCK_N: tl.constexpr = 32
BWD_BLOCK_M: tl.constexpr = 64
BWD_BLOCK_N: tl.constexpr = 32


class StickBreakingAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, inv_temp: float,
                bias=0.0, attend_current: bool = False):
        # bias: float (broadcast to all heads) or per-head (num_heads,) tensor.
        # lema patch: per-head tensor bias is DIFFERENTIABLE (learnable-alpha).
        no_grad = not ctx.needs_input_grad[0]
        if not torch.is_tensor(bias):
            bias_t = torch.full((q.shape[1],), float(bias),
                                dtype=torch.float32, device=q.device)
        else:
            bias_t = bias.detach().to(device=q.device, dtype=torch.float32).contiguous()
        o, rem, neg_log_acc = _fwd(
            q, k, v, logit_scale=inv_temp, logit_bias=bias_t, no_grad=no_grad,
            return_attention=False, BLOCK_M=FWD_BLOCK_M, BLOCK_N=FWD_BLOCK_N,
            attend_current=attend_current,
        )
        ctx.save_for_backward(q, k, v, neg_log_acc, bias_t)
        ctx.logit_scale = inv_temp
        ctx.attend_current = attend_current
        return o, rem

    @staticmethod
    def backward(ctx, do: torch.Tensor, drem: torch.Tensor):
        q, k, v, neg_log_acc, bias_t = ctx.saved_tensors
        dq, dk, dv, dbias = _bwd(
            do,
            drem,
            q,
            k,
            v,
            neg_log_acc,
            ctx.logit_scale,
            bias_t,
            attend_current=ctx.attend_current,
            BLOCK_M=BWD_BLOCK_M,
            BLOCK_N=BWD_BLOCK_N,
        )
        # inv_temp comes from the schedule -> no gradient; bias gets a real one
        # (lema patch v2) when a differentiable tensor was passed.
        dbias_out = dbias if ctx.needs_input_grad[4] else None
        return dq, dk, dv, None, dbias_out, None


def sb_attn(q, k, v, inv_temp=None, bias=0.0, zero_start=True, attend_current=False):
    """Stick-breaking attention with gate sigmoid(inv_temp * <q,k> + bias).

    `bias` is a lema addition: it lets the LEMA surrogate gate sigmoid(alpha*(<q,k> - c))
    be expressed directly as inv_temp=alpha, bias=-alpha*c, instead of folding alpha and c
    into an extra q/k dimension (which forced d_qk to be a multiple of 8 minus 1). Both are
    RUNTIME scalars, so a schedule that changes them every step never re-JITs the kernel.
    """
    if inv_temp is None:
        inv_temp = 1 / math.sqrt(q.size(-1))
    return StickBreakingAttention.apply(q, k, v, inv_temp, bias, attend_current)
