"""RULER's metric: greedy generation plus `string_match_all`, for all three architectures.

RULER generates greedily and asks whether the gold value(s) appear anywhere in the generated
text, with partial credit when several values are wanted (`string_match_all`, copied into
`ruler.py`). That is the protocol behind the RULER columns of the GDN, Titans,
MesaNet and Log-Linear Attention papers.

Two generation paths, and the second is only ever used after it has reproduced the first:

  `--path full`   one forward over the whole buffer per generated token, with no state of
                  any kind. Architecture-neutral by construction and the definition of
                  correct here. Costs `budget x prompt` tokens of forward pass per prompt.
  `--path cache`  one prefill, then one single-token forward per generated token against
                  the state the prefill left behind, ~30x cheaper. The state is the
                  architecture's own: rotated keys and values for softmax+RoPE, the
                  code/value store for LEMA (whose rule, "the value of the latest earlier
                  key with this code", is a lookup table and nothing else), and `fla`'s
                  recurrent plus short-convolution state for gated DeltaNet. The prefill of
                  every path runs the SAME op the model runs in evaluation
                  (`F.scaled_dot_product_attention`, `lema_hard`, `GatedDeltaNet`), so only
                  the single-token steps are new code.
  `--validate`    runs both paths on the same prompts and reports the fraction whose
                  generated token ids agree exactly, plus where they first diverge.

One file per (task, haystack, model, length):
out/gen/<task>-<haystack>_<model>_T<ctx>.jsonl, first line a `_meta` object, then one line
per prompt with the needle depth and distance, the prompt length, the document boundaries
removed from the haystack, the gold value(s) and which of them were hit, the generated
token ids and text, RULER's match, the match on the first generated line only, whether the
first generated token equals the first gold token, and whether the first |gold| generated
tokens are the gold tokens.

Generation runs to the full budget with no early stop; the text is cut at the first
<|endoftext|> only, which is the end-of-sequence stop a serving engine applies. See
`ruler.py` for the haystacks and for everything that deviates from RULER.

    python gen_eval.py lema1536_h64 --task niah_single_1 --hay noise --ctx 2048 4096 --n 500
    python gen_eval.py rope1536_16k gdn1536_16k lema1536_h64 --validate --n 50 \
        --task niah_single_2 --hay essay --ctx 2048
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from lema import load_trained                                          # noqa: E402
from experiments.paths import LM_OUT                                   # noqa: E402
from lema.attention import lema_hard, pack_codes                       # noqa: E402
from lema.model import _rope_tables, rope                              # noqa: E402
from tokens import EOT, dec, enc                                       # noqa: E402
import ruler                                                           # noqa: E402

DTYPE = torch.bfloat16


BIGRAMS = LM_OUT.parent.parent / "main_lm_bigrams" / "out"


def resolve(name: str) -> Path:
    """A run name or directory -> the directory holding model.pt. The context-extended
    baselines live under main_lm_bigrams/out, the pretrained runs under main_lm/out."""
    p = Path(name)
    for cand in (p, HERE / "out" / name, LM_OUT / name, BIGRAMS / name):
        if (cand / "model.pt").exists():
            return cand
    raise SystemExit(f"no finished run for {name!r}")


# ---------------------------------------------------------------- forward

@torch.no_grad()
def hidden(model, x):
    """The trunk of Transformer.forward, stopping before the lm_head.

    Copied rather than imported because `forward` always pays for the [T, vocab] logits,
    which at 16k-64k tokens is most of the memory while all we need is a handful of
    positions. Same ops in the same order, so the numbers are the model's.
    """
    h = model.tok_emb(x)
    for blk in model.blocks:
        h = blk(h)
    return model.norm_f(h)


def kind_of(cfg) -> str:
    return {"lema": "lema", "gated-deltanet": "gdn", "softmax": "rope",
            "sb": "sb"}[cfg.attn]


# ---------------------------------------------------------------- path 1: full reforward


@torch.no_grad()
def gen_full(model, ids, plen: int, budget: int, device="cuda") -> np.ndarray:
    """Greedy continuation by repeated full forward. `ids` [B, plen] are the prompts, all
    of the same length. Returns [B, budget] int64 on the cpu.

    No state, no cache, no architecture-specific code: at step s the whole buffer
    `prompt + generated[:s]` goes through the trunk and the argmax at the last position is
    appended. This is the reference both for the sweep and for `--validate`.
    """
    B = ids.shape[0]
    buf = torch.full((B, plen + budget), EOT, dtype=torch.long, device=device)
    buf[:, :plen] = ids
    out = torch.zeros((B, budget), dtype=torch.long, device=device)
    for s in range(budget):
        end = plen + s
        with torch.autocast(device, dtype=DTYPE):
            h = hidden(model, buf[:, :end])
        tok = model.lm_head(h[:, -1]).float().argmax(-1)
        out[:, s] = tok
        buf[:, end] = tok
    return out.cpu().numpy()


# ---------------------------------------------------------------- path 2: cached state


def _rope_at(x, base: float, pos: int):
    """`lema.model.rope` at one absolute position: the same formula and the same cached
    tables, evaluated at row `pos` instead of rows 0..T-1."""
    half = x.shape[3] // 2
    cos, sin = _rope_tables(half, pos + 1, x.device, base)
    c, s = cos[pos:pos + 1], sin[pos:pos + 1]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)


class KVState:
    """softmax+RoPE: the rotated keys and the values of every position, per layer.

    The step attends a single query over the live range with the mask off, which for the
    last position of a causal sequence is the causal result; the prefill keeps
    `F.scaled_dot_product_attention(is_causal=True)`, the op `Attention.forward` runs.
    """
    bytes_per_row = staticmethod(
        lambda cfg, T: cfg.depth * T * cfg.num_kv_heads * (cfg.d_qk + cfg.d_v) * 2)

    def __init__(self, model, B: int, maxlen: int, device="cuda"):
        c = model.cfg
        self.model, self.n = model, 0
        self.k = [torch.zeros(B, c.num_kv_heads, maxlen, c.d_qk, dtype=DTYPE,
                              device=device) for _ in model.blocks]
        self.v = [torch.zeros(B, c.num_kv_heads, maxlen, c.d_v, dtype=DTYPE,
                              device=device) for _ in model.blocks]

    def _scale(self):
        return 1.0 / float(self.model.cfg.d_qk) ** 0.5

    def prefill(self, x):
        m, c = self.model, self.model.cfg
        gqa = c.num_kv_heads != c.num_heads
        h = m.tok_emb(x)
        T = x.shape[1]
        for li, blk in enumerate(m.blocks):
            zq, zk, v = blk.attn.project(blk.norm1(h))
            zq, zk = rope(zq, c.rope_base), rope(zk, c.rope_base)
            self.k[li][:, :, :T] = zk.to(DTYPE)
            self.v[li][:, :, :T] = v.to(DTYPE)
            y = F.scaled_dot_product_attention(zq, zk, v, is_causal=True,
                                               scale=self._scale(), enable_gqa=gqa)
            h = h + blk.attn.merge(y)
            h = h + blk.mlp(blk.norm2(h))
        self.n = T
        return m.norm_f(h[:, -1:])

    def step(self, tok):
        m, c = self.model, self.model.cfg
        gqa = c.num_kv_heads != c.num_heads
        p = self.n
        h = m.tok_emb(tok)
        for li, blk in enumerate(m.blocks):
            zq, zk, v = blk.attn.project(blk.norm1(h))
            zq = _rope_at(zq, c.rope_base, p)
            zk = _rope_at(zk, c.rope_base, p)
            self.k[li][:, :, p:p + 1] = zk.to(DTYPE)
            self.v[li][:, :, p:p + 1] = v.to(DTYPE)
            y = F.scaled_dot_product_attention(zq, self.k[li][:, :, :p + 1],
                                               self.v[li][:, :, :p + 1], is_causal=False,
                                               scale=self._scale(), enable_gqa=gqa)
            h = h + blk.attn.merge(y)
            h = h + blk.mlp(blk.norm2(h))
        self.n = p + 1
        return m.norm_f(h)


def _code_dtype(d_qk: int):
    """`pack_codes` returns int64; a d_qk-bit code fits a narrower integer, which matters
    because the store holds one code per head and position. Equality is all that is ever
    asked of a code, so the narrowing is exact."""
    if d_qk <= 15:
        return torch.int16
    if d_qk <= 31:
        return torch.int32
    return torch.int64


class LemaState:
    """LEMA in hard mode: the code and the value of every position, per layer.

    The op is `o_i = v_j` for the largest `j < i` with `code(k_j) == code(q_i)`, else 0
    (`lema.attention.lema_hard`). After a prefill that stored every (code, value), a step
    is that rule read off the store, which is exactly what the paper's inference-time hash
    table does -- here as a scan, so it needs no extension module and no host round trip.
    The prefill itself calls `lema_hard`, the scorer's op, unchanged.
    """
    bytes_per_row = staticmethod(
        lambda cfg, T: cfg.depth * T * cfg.num_heads * (cfg.d_v * 2 + 8))

    def __init__(self, model, B: int, maxlen: int, device="cuda"):
        c = model.cfg
        self.model, self.n = model, 0
        self.cdt = _code_dtype(c.d_qk)
        self.kc = [torch.zeros(B, c.num_heads, maxlen, dtype=self.cdt, device=device)
                   for _ in model.blocks]
        self.v = [torch.zeros(B, c.num_heads, maxlen, c.d_v, dtype=DTYPE, device=device)
                  for _ in model.blocks]

    def prefill(self, x):
        m = self.model
        h = m.tok_emb(x)
        T = x.shape[1]
        for li, blk in enumerate(m.blocks):
            zq, zk, v = blk.attn.project(blk.norm1(h))
            y = lema_hard(zq, zk, v)                  # the scorer's op, unchanged
            self.kc[li][:, :, :T] = pack_codes(zk).to(self.cdt)
            self.v[li][:, :, :T] = v.to(DTYPE)
            h = h + blk.attn.merge(y)
            h = h + blk.mlp(blk.norm2(h))
        self.n = T
        return m.norm_f(h[:, -1:])

    @staticmethod
    def _latest(kc, qc, p: int, chunk: int = 16384):
        """[B, H, 1] int64: the largest j < p with kc[..., j] == qc, else -1. Scanned in
        chunks so the comparison never materialises more than `chunk` positions at once."""
        best = torch.full_like(qc, -1, dtype=torch.int64)
        for lo in range(0, p, chunk):
            hi = min(p, lo + chunk)
            ar = torch.arange(lo, hi, device=kc.device, dtype=torch.int64)
            eq = kc[:, :, lo:hi] == qc
            cand = torch.where(eq, ar, torch.full_like(ar, -1)).amax(-1, keepdim=True)
            best = torch.maximum(best, cand)
        return best

    def step(self, tok):
        m = self.model
        p = self.n
        h = m.tok_emb(tok)
        for li, blk in enumerate(m.blocks):
            zq, zk, v = blk.attn.project(blk.norm1(h))
            qc = pack_codes(zq).to(self.cdt)                        # [B, H, 1]
            j = self._latest(self.kc[li], qc, p)                    # [B, H, 1]
            got = self.v[li].gather(
                2, j.clamp_min(0).unsqueeze(-1).expand(-1, -1, -1, self.v[li].shape[-1]))
            y = got * (j >= 0).unsqueeze(-1).to(got.dtype)
            self.kc[li][:, :, p:p + 1] = pack_codes(zk).to(self.cdt)
            self.v[li][:, :, p:p + 1] = v.to(DTYPE)
            h = h + blk.attn.merge(y.to(v.dtype))
            h = h + blk.mlp(blk.norm2(h))
        self.n = p + 1
        return m.norm_f(h)


class GDNCache:
    """Everything `fla.layers.GatedDeltaNet.forward` asks of a cache, and nothing else.

    Both fla versions used here (0.2.2 and 0.5.2) use exactly three
    operations -- `len(cache)`, `cache[layer_idx]` and
    `cache.update(layer_idx=, recurrent_state=, conv_state=, offset=)`, the latter through
    `fla.layers.utils.update_layer_cache` in 0.5.2. fla's own `fla.models.utils.Cache`
    implements them by subclassing `transformers.Cache`, whose constructor now demands a
    layer specification, so it cannot be instantiated under 0.2.2. This holds
    the same per-layer recurrent and short-convolution states in a list and is identical
    under both versions.
    """

    def __init__(self):
        self.states: list[dict] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        return self.states[i]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen_tokens

    def update(self, recurrent_state=None, attn_state=None, conv_state=None,
               ffn_state=None, layer_idx: int = 0, offset: int = 1, cache_kwargs=None,
               **kw):
        if layer_idx == 0:
            self._seen_tokens += offset
        st = dict(recurrent_state=recurrent_state, attn_state=attn_state,
                  conv_state=conv_state, ffn_state=ffn_state)
        if len(self.states) <= layer_idx:
            self.states.append(st)
        else:
            self.states[layer_idx] = st
        return st


class GDNState:
    """gated DeltaNet: `fla`'s own cache. The recurrent state and the short-convolution
    state are fixed size, so there is nothing to budget.

    `fla.layers.GatedDeltaNet.forward` switches to `fused_recurrent_gated_delta_rule` at
    q_len <= 64 and uses `chunk_gated_delta_rule` otherwise, so the prefill is the chunked
    kernel the scorer uses and a step is the recurrent form of the same rule. The layer
    must know its index to address the cache; `GatedDeltaNet` is constructed without one
    in `lema/model.py`, so the runner sets it on the loaded model.
    """
    bytes_per_row = staticmethod(lambda cfg, T: 0)

    def __init__(self, model, B: int, maxlen: int, device="cuda"):
        self.model, self.n = model, 0
        for li, blk in enumerate(model.blocks):
            blk.attn.inner.layer_idx = li
        self.cache = GDNCache()

    def _run(self, x):
        m = self.model
        h = m.tok_emb(x)
        for blk in m.blocks:
            out = blk.attn.inner(hidden_states=blk.norm1(h), past_key_values=self.cache,
                                 use_cache=True)
            h = h + (out[0] if isinstance(out, tuple) else out)
            h = h + blk.mlp(blk.norm2(h))
        return h

    def prefill(self, x):
        h = self._run(x)
        self.n = x.shape[1]
        return self.model.norm_f(h[:, -1:])

    def step(self, tok):
        h = self._run(tok)
        self.n += 1
        return self.model.norm_f(h)


STATES = {"rope": KVState, "lema": LemaState, "gdn": GDNState}


@torch.no_grad()
def gen_cache(model, kind: str, ids, plen: int, budget: int, device="cuda") -> np.ndarray:
    B = ids.shape[0]
    with torch.autocast(device, dtype=DTYPE):
        st = STATES[kind](model, B, plen + budget, device)
        h = st.prefill(ids)
    out = torch.zeros((B, budget), dtype=torch.long, device=device)
    tok = model.lm_head(h[:, -1]).float().argmax(-1)
    out[:, 0] = tok
    for s in range(1, budget):
        with torch.autocast(device, dtype=DTYPE):
            h = st.step(tok[:, None])
        tok = model.lm_head(h[:, -1]).float().argmax(-1)
        out[:, s] = tok
    del st
    return out.cpu().numpy()


# ---------------------------------------------------------------- driving


def batches(plen: np.ndarray, B: int):
    """Row groups of at most B rows, each group of ONE prompt length. Both paths need it:
    the cached path because a state is per position, the full path because it keeps the
    comparison prompt-for-prompt identical."""
    for L in sorted(set(int(x) for x in plen)):
        idx = np.flatnonzero(plen == L)
        for lo in range(0, len(idx), B):
            yield int(L), idx[lo:lo + B]


def batch_size(kind: str, cfg, T: int, budget: int, batch_tokens: int,
               cache_bytes: float, forced: int = 0) -> int:
    if forced:
        return forced
    B = max(1, batch_tokens // max(T, 1))
    per = STATES[kind].bytes_per_row(cfg, T + budget)
    if per:
        B = max(1, min(B, int(cache_bytes // per)))
    return B


def evaluate(model, kind: str, b: dict, budget: int, path: str, B: int,
             device="cuda") -> dict:
    ids_np, plen_np = b["ids"], b["plen"]
    gen = np.zeros((len(ids_np), budget), np.int64)
    for L, rows in batches(plen_np, B):
        x = torch.from_numpy(ids_np[rows, :L].astype(np.int64)).to(device)
        gen[rows] = (gen_full(model, x, L, budget, device) if path == "full"
                     else gen_cache(model, kind, x, L, budget, device))
    return score_gen(b, gen, budget)


def score_gen(b: dict, gen: np.ndarray, budget: int) -> dict:
    """RULER's metric on a finished batch of generations.

    The generation ends at the first <|endoftext|>, which is what a serving engine's
    end-of-sequence stop does (RULER passes no stop words for the synthetic tasks and
    relies on the model's eos), and the text after it would be a second, unrelated
    document. `gen_ids` keeps the full budget so that decision can be undone.
    """
    rec = []
    for i in range(len(gen)):
        g = gen[i]
        eot = np.flatnonzero(g == EOT)
        cut = int(eot[0]) if len(eot) else len(g)
        txt = dec(g[:cut])
        refs = b["refs"][i]
        gold = b["ans"][i, :b["alen"][i]].astype(np.int64)
        hit = [1.0 if r.lower() in txt.strip().lower() else 0.0 for r in refs]
        r = dict(i=i, depth=float(b["depth"][i]), dist=int(b["dist"][i]),
                 plen=int(b["plen"][i]),
                 ref_depth=[int(x) for x in b["ref_depth"][i]],
                 ref_dist=[int(x) for x in b["ref_dist"][i]],
                 refs=refs, ref_hit=hit, n_gen=cut,
                 match=float(np.mean(hit)),
                 match_firstline=ruler.string_match_all(txt.split("\n")[0], refs),
                 first_gold=int(gold[0]), first_gen=int(g[0]),
                 first_ok=bool(int(gold[0]) == int(g[0])),
                 gold_prefix_ok=bool(np.array_equal(g[:len(gold)], gold)),
                 gen_ids=[int(t) for t in g], gen=txt)
        rec.append(r)
    return dict(rec=rec, gen=gen)


def summarize(rec) -> dict:
    f = lambda k: float(np.mean([r[k] for r in rec]))                  # noqa: E731
    s = dict(n=len(rec), match=f("match"), match_firstline=f("match_firstline"),
             first_ok=f("first_ok"), gold_prefix_ok=f("gold_prefix_ok"),
             solved=float(np.mean([r["match"] == 1.0 for r in rec])))
    for k in rec[0]:
        if k.startswith("match_") and k not in s:
            s[k] = f(k)
    return s


def gold_step(g: np.ndarray, refs) -> int:
    """The smallest number of generated tokens whose text already contains every gold
    value, or -1 if the full generation never does. A divergence after this step cannot
    change `string_match_all`."""
    for k in range(1, len(g) + 1):
        t = dec(g[:k]).lower()
        if all(r.lower() in t for r in refs):
            return k
    return -1


@torch.no_grad()
def validate(model, kind: str, b: dict, budget: int, B: int, meta: dict) -> dict:
    """Both paths on the same prompts.

    Token equality is the strict test. It is not reachable for LEMA and the reason is
    worth stating precisely: the prefill of the cached path is BIT-IDENTICAL to the
    scorer's trunk (checked separately), but a single-token step multiplies a [B, dim]
    activation by the projections while the re-forward multiplies a [B*T, dim] one, and
    cuBLAS does not promise the same accumulation order for the two shapes. For softmax
    and gated DeltaNet that is a 1e-3 difference in a continuous quantity and changes
    nothing; LEMA reads `sign(W x)` off the same product, so a pre-activation within
    rounding distance of zero flips a code bit and the retrieved value changes
    discontinuously. The full re-forward has the same exposure -- it recomputes every
    earlier position at every step, with a different T each time -- so neither path is the
    privileged one. What matters is whether the metric moves, hence the three numbers
    below: the mean match on each path, how many prompts score differently, and how many
    prompts diverge BEFORE the gold value is complete (a divergence after it cannot change
    `string_match_all`).
    """
    gf = np.zeros((len(b["ids"]), budget), np.int64)
    gc = np.zeros_like(gf)
    for L, rows in batches(b["plen"], B):
        x = torch.from_numpy(b["ids"][rows, :L].astype(np.int64)).cuda()
        gf[rows] = gen_full(model, x, L, budget)
        gc[rows] = gen_cache(model, kind, x, L, budget)
    same = (gf == gc).all(1)
    rf = score_gen(b, gf, budget)["rec"]
    rc = score_gen(b, gc, budget)["rec"]
    first, gstep, before, noans = [], [], 0, 0
    for i in range(len(gf)):
        d = -1 if same[i] else int(np.flatnonzero(gf[i] != gc[i])[0])
        k = gold_step(gf[i], b["refs"][i])
        first.append(d)
        gstep.append(k)
        if k < 0:
            noans += 1                 # the full path never produced the answer
        elif d >= 0 and d < k:
            before += 1                # ... and the paths parted before it did
    return dict(**meta, n_no_answer=int(noans),
                agree=float(same.mean()), n_agree=int(same.sum()),
                n_cmp=int(len(same)), tokens_equal=float((gf == gc).mean()),
                match_full=summarize(rf)["match"], match_cache=summarize(rc)["match"],
                first_ok_full=summarize(rf)["first_ok"],
                first_ok_cache=summarize(rc)["first_ok"],
                n_match_differs=int(sum(1 for x, y in zip(rf, rc)
                                        if x["match"] != y["match"])),
                n_divergence_before_answer=int(before),
                first_divergence=[x for x in first if x >= 0],
                gold_step=gstep)


def env_meta() -> dict:
    m = dict(host=socket.gethostname(), torch=torch.__version__,
             python=platform.python_version(),
             gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
             cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    try:
        import fla
        m["fla"] = getattr(fla, "__version__", "?")
    except Exception:
        pass
    try:
        import triton
        m["triton"] = triton.__version__
    except Exception:
        pass
    return m


# ---------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--task", nargs="+", default=None)
    p.add_argument("--hay", nargs="+", default=None,
                   help="essay (RULER's Paul Graham essays), noise (S-NIAH-1's repeated "
                        "sentence), needle (MK-NIAH-2's key-value lines)")
    p.add_argument("--ctx", nargs="+", type=int, default=[2048])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--path", choices=["full", "cache"], default="cache")
    p.add_argument("--budget", type=int, default=0, help="override ruler.BUDGET")
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--batch-tokens", type=int, default=65536)
    p.add_argument("--cache-bytes", type=float, default=8e9)
    p.add_argument("--out", type=Path, default=HERE / "out" / "gen")
    p.add_argument("--force", action="store_true")
    p.add_argument("--validate", action="store_true",
                   help="run both paths on the same prompts and compare the token ids")
    a = p.parse_args()

    for name in a.models:
        run = resolve(name)
        model, cfg = load_trained(run)
        kind = kind_of(cfg)
        print(f"# {name}: {run} | {cfg.summary()} | kind={kind}", flush=True)
        for task, hay in ruler.cells(a.task, a.hay):
            budget = a.budget or ruler.BUDGET[task]
            for ctx in a.ctx:
                tag = f"{ruler.label(task, hay)}_{name}_T{ctx}"
                dst = (a.out / "validate" / f"{tag}.json") if a.validate else \
                    (a.out / f"{tag}.jsonl")
                if dst.exists() and not a.force:
                    print(f"{tag}: done", flush=True)
                    continue
                t0 = time.time()
                try:
                    b = ruler.bank(task, ctx, a.n, a.seed, hay=hay)
                except SystemExit as e:
                    print(f"{tag}: SKIPPED ({e})", flush=True)
                    continue
                B = batch_size(kind, cfg, ctx, budget, a.batch_tokens, a.cache_bytes,
                               a.batch)
                meta = dict(model=name, run=str(run), kind=kind, task=task, hay=hay,
                            cell=ruler.label(task, hay), ctx=ctx,
                            n=a.n, seed=a.seed, budget=budget, batch=B,
                            fp=int(b["fp"]), resampled=int(b["n_resampled"]),
                            unchecked=int(b["n_failed"]),
                            plen=[int(b["plen"].min()), int(b["plen"].max())],
                            **env_meta())
                dst.parent.mkdir(parents=True, exist_ok=True)

                if a.validate:
                    r = validate(model, kind, b, budget, B, meta)
                    r["secs"] = time.time() - t0
                    dst.write_text(json.dumps(r, indent=1) + "\n")
                    print(f"VALIDATE {tag}: tokens {r['tokens_equal']:.4f}  prompts "
                          f"{r['agree']:.3f} ({r['n_agree']}/{r['n_cmp']})  "
                          f"match full {r['match_full']:.4f} cache "
                          f"{r['match_cache']:.4f}  per-prompt match differs "
                          f"{r['n_match_differs']}  divergence before the answer "
                          f"{r['n_divergence_before_answer']}  ({r['secs']:.0f}s)",
                          flush=True)
                    continue

                out = evaluate(model, kind, b, budget, a.path, B)
                s = summarize(out["rec"])
                meta.update(path=a.path, secs=time.time() - t0, **s)
                # written through a temporary and renamed: two processes may be given the
                # same cell (two gpus sharing a model), and a rename leaves one complete
                # file rather than two interleaved halves
                tmp = dst.with_suffix(f".tmp{os.getpid()}")
                with tmp.open("w") as fh:
                    fh.write(json.dumps(dict(_meta=meta)) + "\n")
                    for r in out["rec"]:
                        fh.write(json.dumps(r) + "\n")
                os.replace(tmp, dst)
                print(f"{tag}: match {s['match']:.3f} solved {s['solved']:.3f} "
                      f"first_ok {s['first_ok']:.3f} prefix {s['gold_prefix_ok']:.3f} "
                      f"(B={B}, {time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
