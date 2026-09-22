"""Single-token decoding with the GPU work captured into CUDA graphs.

At batch 1 a decode step is a few hundred small kernels, and launching them one by one
from python costs more than running them: the GPU idles between launches. A CUDA graph
records the kernels once and replays them as a single launch. That needs every tensor
address and shape in the step to be fixed, which both caches arrange for -- the kv cache by
keeping its fill level on the device (`KVCache.pos`), the LEMA cache by staging through
fixed pinned buffers -- so a token costs:

    KVDecoder     depth+1 graph replays with the attention kernel launched between them
                  on the live context (see the class)
    LemaDecoder   depth+1 graph replays with the host hash-table exchange between them.
                  Graph i finishes layer i-1 and starts layer i: it copies layer i-1's
                  retrieved values up, runs the rest of that layer, then the projection
                  and codes of layer i and the copy of those codes down. Only the C++
                  exchange itself runs in python, between replays.

The captured work is exactly `Transformer.forward(tok, cache=...)` -- the same segment
functions (`Block.kv_pre/kv_post`, `Block.gpu_pre/gpu_post`) the eager path calls, so a
replay reproduces an eager step. Compile the model first (`model.compile_inference()`) for
fused kernels inside the graphs.

The graph ends by writing the chosen token into `tok`, which the next replay consumes.
`tok` starts as whatever the caller passes (the last prompt token, after an eager prefill
of the rest), and holds the newest generated token after each `step()`. `temperature=0`
picks the argmax; anything larger samples (see `_chooser`).
"""
from __future__ import annotations

import torch

from .cache import KVCache, LemaCache


def _as_tuple(r):
    return r if isinstance(r, tuple) else (r,)


def _warm_then_capture(fns, between=None):
    """Run `fns` as a chain (each fed the previous one's outputs) twice on a side stream
    -- first calls compile and allocate, which must not happen during capture -- then
    capture each into its own graph. The graphs share one memory pool: they only ever
    replay in this order, so one's intermediates may reuse the previous one's. Returns
    (graphs, outputs); the outputs are the fixed tensors each replay writes. `between(i,
    out)`, if given, runs eagerly after function i, in the warmup and again during the
    capture (outside the graph): the work that stays outside the graphs."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            out = ()
            for i, fn in enumerate(fns):
                out = _as_tuple(fn(*out))
                if between is not None:
                    between(i, out)
    torch.cuda.current_stream().wait_stream(s)
    graphs, outs, out, pool = [], [], (), None
    for i, fn in enumerate(fns):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            res = fn(*out)
        pool = pool or g.pool()
        graphs.append(g)
        outs.append(res)
        out = _as_tuple(res)
        if between is not None:
            between(i, out)
    return graphs, outs


def _chooser(temperature: float):
    """How a step turns logits into the next token: argmax, or a sample at `temperature`.

    Sampling is the Gumbel-max trick -- argmax(logits / T + g) with g ~ Gumbel(0, 1) is
    exactly a draw from softmax(logits / T) -- rather than softmax + multinomial, because
    it is one elementwise pass and an argmax, the same shapes the greedy path already has,
    and it needs no normalisation. Gumbel noise comes from an Exponential(1) draw, since
    -log(E) IS Gumbel(0, 1); that avoids a log(-log(u)) that blows up when u rounds to 0
    or 1 in low precision. In float32 because bf16 logits differing by <0.03 would
    otherwise compare equal and quietly flatten the tail of the distribution.

    Captured into a CUDA graph, the draw stays random: torch registers the generator with
    the graph, so each replay advances the philox offset and produces fresh noise.
    """
    if temperature <= 0:
        return lambda logits: logits.argmax(-1)

    def pick(logits):
        z = logits.float() / temperature
        g = torch.empty_like(z).exponential_().log().neg_()      # Gumbel(0, 1)
        return (z + g).argmax(-1)
    return pick


class KVDecoder:
    """The softmax baseline: the dense parts of a step as CUDA graphs, the attention kernel
    launched between them for exactly the live context.

    A step is depth+1 graph replays with one kernel launch between consecutive replays:
    graph i finishes layer i-1 (output projection, residual, MLP) and starts layer i
    (normalization, the query, key and value projections), and the kv-cache kernel of
    layer i then runs on the cache sliced to the live length, with the split count that
    fills the SMs in one wave (`model.decode_splits`). This is how serving engines decode
    (vLLM's piecewise graphs, SGLang's per-step plan): the kernel must not be captured on
    the whole provisioned buffer, since it partitions the range it is given among its
    thread blocks and most of them would idle at a live context far shorter than the
    buffer. The launches are asynchronous, so the host stays ahead of the GPU and the
    structure costs no synchronization."""

    def __init__(self, model, cache: KVCache, tok: torch.Tensor | None = None,
                 temperature: float = 0.0):
        from flash_attn import flash_attn_with_kvcache
        from .model import decode_splits
        self.model, self.cache = model, cache
        cfg = model.cfg
        B, L = cache.batch, cfg.depth
        dev = cache.pos.device
        self.tok = torch.zeros(B, 1, dtype=torch.long, device=dev)
        if tok is not None:
            self.tok.copy_(tok)
        choose = _chooser(temperature)
        segs = [b.seg or (b.kv_pre, b.kv_post) for b in model.blocks]
        p = next(model.parameters())
        y = torch.empty(B, 1, cfg.num_heads, cfg.d_v, dtype=p.dtype, device=dev)
        self.splits = decode_splits(B, cfg.num_kv_heads)
        self._flash = flash_attn_with_kvcache
        floor = max(2, cfg.num_heads // cfg.num_kv_heads)   # the kernel folds a group's
        # query heads into its query length and wants the cache slice at least that long

        def head():
            x = model.tok_emb(self.tok)
            return (x,) + tuple(segs[0][0](x))

        def mid(i):                     # finish layer i-1, start layer i
            def fn(x, q, k, v):
                x = segs[i - 1][1](x, y)
                return (x,) + tuple(segs[i][0](x))
            return fn

        def tail(x, q, k, v):
            x = segs[L - 1][1](x, y)
            logits = model.lm_head(model.norm_f(x))
            self.tok.copy_(choose(logits))
            cache.pos.add_(1)
            return logits

        def attend(i, out):             # eager, between graph i and i+1
            _, q, k, v = out
            n = min(cache.max_len, max(floor, cache.n + 1))   # the live length incl. this token
            y.copy_(self._flash(q, cache.k[i][:, :n], cache.v[i][:, :n], k, v,
                                rotary_cos=cache.cos, rotary_sin=cache.sin,
                                cache_seqlens=cache.pos, causal=True,
                                rotary_interleaved=False, num_splits=self.splits))

        self._attend = attend
        # warmup and capture run the step three times, which appends to the cache and
        # overwrites tok: put both back (the garbage entries are overwritten by real steps)
        tok0, n0 = self.tok.clone(), cache.n
        self.graphs, outs = _warm_then_capture([head] + [mid(i) for i in range(1, L)] + [tail],
                                               between=self._between)
        self.outs = outs
        self.logits = outs[-1]
        self.tok.copy_(tok0)
        cache.pos.fill_(n0)
        cache.n = n0

    def _between(self, i, out):
        if i < self.model.cfg.depth:
            self._attend(i, out)
            self.cache.n += 1 if i == self.model.cfg.depth - 1 else 0

    def step(self) -> torch.Tensor:
        L = self.model.cfg.depth
        for i in range(L):
            self.graphs[i].replay()
            self._attend(i, self.outs[i])
        self.graphs[L].replay()
        self.cache.n += 1
        torch.cuda.current_stream().synchronize()
        return self.tok


class LemaDecoder:
    """A LEMA model: depth+1 CUDA graphs per token, the hash-table exchange between."""

    def __init__(self, model, cache: LemaCache, tok: torch.Tensor | None = None,
                 temperature: float = 0.0):
        cfg = model.cfg
        if not model.lema_layers() or not all(a.hard for a in model.lema_layers()):
            raise ValueError("LemaDecoder needs an all-LEMA model in hard mode")
        self.model, self.cache = model, cache
        B, L = cache.batch, cfg.depth
        dev = next(model.parameters()).device
        self.tok = torch.zeros(B, 1, dtype=torch.long, device=dev)
        if tok is not None:
            self.tok.copy_(tok)
        segs = [b.seg or (b.gpu_pre, b.gpu_post) for b in model.blocks]
        self.views = [cache.layer(i) for i in range(L)]
        self.st = cache.staging(B * cfg.num_heads, cfg.d_v)
        got = torch.empty(B, cfg.num_heads, 1, cfg.d_v, dtype=self.st.out.dtype,
                          device=dev)

        def head():
            x = model.tok_emb(self.tok)
            self.views[0].stage(*segs[0][0](x))
            return x

        def mid(i):                     # finish layer i-1, start layer i
            def fn(x):
                self.views[i - 1].fetch(self.st, got)
                x = segs[i - 1][1](x, got)
                self.views[i].stage(*segs[i][0](x))
                return x
            return fn

        choose = _chooser(temperature)

        def tail(x):
            self.views[L - 1].fetch(self.st, got)
            x = segs[L - 1][1](x, got)
            logits = model.lm_head(model.norm_f(x))
            self.tok.copy_(choose(logits))
            return logits

        tok0 = self.tok.clone()          # warmup and capture overwrite it (nothing else:
        self.graphs, outs = _warm_then_capture(   # the store is untouched without host())
            [head] + [mid(i) for i in range(1, L)] + [tail])
        self.logits = outs[-1]
        self.tok.copy_(tok0)

    def step(self) -> torch.Tensor:
        sync = torch.cuda.current_stream().synchronize
        self.graphs[0].replay()
        for i, view in enumerate(self.views):
            sync()                          # this layer's codes and values have landed
            view.host(self.st)
            self.graphs[i + 1].replay()
        sync()
        self.cache.t += 1
        return self.tok


def decoder(model, cache, tok=None, temperature: float = 0.0):
    """The decoder for a cache: `KVDecoder` for a kv cache, `LemaDecoder` for a LEMA one.
    `temperature=0` decodes greedily, anything larger samples."""
    if isinstance(cache, KVCache):
        return KVDecoder(model, cache, tok, temperature)
    return LemaDecoder(model, cache, tok, temperature)
