"""The softmax baseline through vLLM: generation speed vs context and time to first token vs
prompt length, with the engine core in the benchmark process and every engine step timed.

    python measure_vllm.py gen     --model out/hf_08B --out out/gen_08B_softmax.jsonl --backend TRITON_ATTN
    python measure_vllm.py gen     --model out/hf_08B_gqa8 --batch 64 --max-context 250000 --out out/genB64_08B_gqa8_softmax.jsonl
    python measure_vllm.py prefill --model out/hf_08B --step 1024 --out out/prefill_08B_softmax.jsonl
    python measure_vllm.py probe   --model out/hf_08B            # the kv-cache capacity in tokens
    python measure_vllm.py gen     --model out/hf_08B_gdn --method gdn --max-context 8192 --out out/gen_08B_gdn.jsonl

`--model` is a HuggingFace Llama directory from export_llama.py, or a Qwen3-Next directory of
gated DeltaNet layers from export_gdn.py. Records are written like those of
measure_generation.py / measure_prefill.py (`method` from `--method`, one record per bin resp.
per prompt length, an `oom` record where the cache is full), so the plot scripts read them
unchanged. `--method` only labels the records and says whether the state grows with the
context: a softmax model has a kv-cache whose capacity ends the run, while the state of a
gated DeltaNet is one tensor per layer and constant, so such a run has no end of its own and
`--max-context` says where to stop it (no `probe`, no `oom` record).

Why vLLM, and why these settings. A hand-written decoder invites the question whether the
baseline was optimized; vLLM is what transformers are served with. It runs here with its
defaults (paged kv-cache provisioned from 95% of VRAM in blocks of 16 tokens for softmax and
80 for the gated DeltaNet, where vLLM sizes a block to hold one state, chunked prefill of 2048
tokens, CUDA graphs for decode, async scheduling), with prefix caching switched off so that no
run is served from another's blocks, and with the faster of its two attention kernels for this
GPU in each scenario. On Ampere, vLLM's FlashAttention-2 path
cannot split the keys of a head over thread blocks (its interface raises on num_splits > 1),
so at batch 1 with one kv head per query head most SMs idle and the time per token grows at
about a third of the bandwidth-bound slope; vLLM's Triton kernel splits and runs at ~75% of
peak there, but it re-reads the cache once per query head under grouped-query attention and
its prefill is 2x slower. Hence TRITON_ATTN for batch-1 generation of the multi-head models,
FLASH_ATTN (the default) for prefill and for batched generation. FlashInfer needs nvcc for its
JIT and FlexAttention was 3x slower than either; block size (16..128), graph mode and eager
execution change nothing measurable.

Runs the engine in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so that `LLMEngine.step()` can be
timed per token; the cache capacity is read from a separate probe process, because an engine
cannot be torn down cleanly enough in-process to build a second one with the full VRAM.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

LOG: list[str] = []                         # vllm's own log lines, for the record


def _allow_dense_qwen3next(model: str):
    """vLLM's Qwen3-Next is a mixture of experts and its constructor raises when no layer has
    one ("No Qwen3Next layer found in the model.layers."); it is the only model of vLLM 0.17
    with gated DeltaNet layers that a checkpoint can select (Qwen3.5's dense
    `Qwen3_5ForCausalLM` exists but is not in its registry, and the registered Qwen3.5
    architectures carry a vision tower). Our export is dense at every layer, so the expert
    bookkeeping, which only serves expert parallelism, is empty. Nothing else is changed."""
    hf = json.load(open(os.path.join(model, "config.json")))
    if hf.get("model_type") != "qwen3_next" or hf.get("num_experts"):
        return
    from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM

    def set_moe_parameters(self):
        self.expert_weights, self.moe_layers = [], []
        self.num_moe_layers = self.num_logical_experts = self.num_physical_experts = 0
        self.num_local_physical_experts = self.num_routed_experts = 0
        self.num_redundant_experts = self.num_shared_experts = 0
        self.num_expert_groups = 1
    Qwen3NextForCausalLM.set_moe_parameters = set_moe_parameters
    print("   (dense Qwen3-Next: its empty expert bookkeeping is skipped)", flush=True)


def _engine(a, max_len: int, batch: int):
    import torch                                    # noqa: F401  (vllm needs the cuda context)
    import vllm
    from vllm import EngineArgs, LLMEngine

    class Grab(logging.Handler):
        def emit(self, r):
            LOG.append(r.getMessage())
    logging.getLogger("vllm").addHandler(Grab())
    _allow_dense_qwen3next(a.model)
    extra = {"attention_backend": a.backend} if a.backend else {}
    args = EngineArgs(model=a.model, dtype="bfloat16", max_model_len=max_len,
                      gpu_memory_utilization=a.util, skip_tokenizer_init=True,
                      max_num_seqs=batch, enable_prefix_caching=False, seed=a.seed,
                      disable_log_stats=True, **extra)
    t = time.perf_counter()
    e = LLMEngine.from_engine_args(args)
    print(f"   engine up in {time.perf_counter() - t:.0f} s  (vllm {vllm.__version__}, "
          f"backend {a.backend or 'default'})", flush=True)
    return e


def kv_tokens(e) -> int:
    cc = e.vllm_config.cache_config
    if cc.num_gpu_blocks:
        return cc.num_gpu_blocks * cc.block_size
    for line in LOG:
        m = re.search(r"KV cache size:\s*([\d,]+)\s*tokens", line)
        if m:
            return int(m.group(1).replace(",", ""))
    raise SystemExit("kv-cache capacity not found")


def probe_capacity(a) -> int:
    """The kv-cache capacity in tokens, from a fresh process."""
    cmd = [sys.executable, os.path.abspath(__file__), "probe", "--model", a.model,
           "--batch", str(a.batch), "--util", str(a.util)] + (
        ["--backend", a.backend] if a.backend else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"KV_TOKENS=(\d+)", r.stdout)
    if not m:
        raise SystemExit("probe failed:\n" + r.stderr[-3000:] + r.stdout[-1000:])
    return int(m.group(1))


def grows_with_context(a) -> int:
    """The context a run reaches: the kv-cache capacity for softmax, 0 (no limit of its own,
    stop at --max-context) for a model whose state is constant."""
    if a.method != "softmax":
        if not a.max_context:
            raise SystemExit(f"--method {a.method} has a constant state and thus no cache "
                             "limit to stop at: --max-context says where to stop")
        return 0
    return a.kv_tokens or probe_capacity(a)


def geometry(hf: dict) -> dict:
    """The geometry of the export, from its HuggingFace config."""
    if "linear_key_head_dim" in hf:                       # Qwen3-Next: gated DeltaNet
        return dict(depth=hf["num_hidden_layers"], dim=hf["hidden_size"],
                    num_heads=hf["linear_num_key_heads"],
                    num_kv_heads=hf["linear_num_value_heads"],
                    d_qk=hf["linear_key_head_dim"], d_v=hf["linear_value_head_dim"],
                    dim_ff=hf["intermediate_size"], vocab=hf["vocab_size"],
                    conv=hf["linear_conv_kernel_dim"], attn="gated-deltanet", pos="nope")
    d_h = hf["hidden_size"] // hf["num_attention_heads"]
    return dict(depth=hf["num_hidden_layers"], dim=hf["hidden_size"],
                num_heads=hf["num_attention_heads"], num_kv_heads=hf["num_key_value_heads"],
                d_qk=d_h, d_v=d_h, dim_ff=hf["intermediate_size"], vocab=hf["vocab_size"],
                attn="softmax", pos="rope")


def base_record(e, a, batch: int, kind: str) -> dict:
    import torch
    import vllm
    sc, cc = e.vllm_config.scheduler_config, e.vllm_config.cache_config
    hf = json.load(open(os.path.join(a.model, "config.json")))
    return dict(kind=kind, method=a.method, engine=f"vllm {vllm.__version__}", batch=batch,
                attention_backend=a.backend or "default",
                gpu_memory_utilization=a.util, block_size=cc.block_size, kv_tokens=kv_tokens(e),
                max_num_batched_tokens=sc.max_num_batched_tokens,
                async_scheduling=getattr(sc, "async_scheduling", None),
                geometry=geometry(hf),
                weights="random", worst_case=False,
                machine=dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                             cuda=torch.version.cuda, python=sys.version.split()[0]),
                time=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()))


def append(path, rec):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def cmd_probe(a):
    e = _engine(a, 8192, a.batch)
    print(f"KV_TOKENS={kv_tokens(e)}", flush=True)


def cmd_gen(a):
    """`--batch` sequences from a one-token prompt, each generating as many tokens as the cache
    holds for the batch (a block of slack per sequence), or `--max-context`, whichever is
    smaller; the wall-clock time of every engine step, averaged over `--points` bins plus a
    few early records at powers of two."""
    from vllm import SamplingParams
    K = grows_with_context(a)                  # 0 = constant state, stop at --max-context
    B = a.batch
    n_fit = (K - 64 * B) // B // 16 * 16 if K else a.max_context
    full = bool(K) and (not a.max_context or n_fit <= a.max_context)
    N = min(n_fit, a.max_context) if a.max_context else n_fit
    e = _engine(a, N + 64, B)
    if K:
        assert kv_tokens(e) >= K - 4096, (kv_tokens(e), K)
    base = dict(base_record(e, a, B, "generation"), n_tokens=N, cache_full_at_end=full)
    print(f"== {a.method}/vllm batch {B}: {N} tokens per sequence, the cache holds "
          f"{base['kv_tokens']}"
          f" ({'to the cache limit' if full else f'capped at {a.max_context}'})", flush=True)
    sp = SamplingParams(max_tokens=N, temperature=1.0, ignore_eos=True, detokenize=False)
    for r in range(B):
        e.add_request(str(r), {"prompt_token_ids": [a.bos]}, sp)
    if os.path.exists(a.out):
        os.remove(a.out)
    binw = max(1, -(-N // a.points))
    ms, last, steps, t0, nxt = [], 0, 0, time.perf_counter(), 0.05
    while e.has_unfinished_requests():
        t = time.perf_counter()
        e.step()
        ms.append((time.perf_counter() - t) * 1e3)
        steps += 1
        c = steps + 1                          # context per sequence after this step
        if steps <= 3:                         # first-replay effects belong to no context
            last = steps
            continue
        early = c < binw and c >= 16 and (c & (c - 1)) == 0
        if c % binw == 0 or steps == N or early:
            append(a.out, dict(base, context=c, ms_per_tok=sum(ms[last:]) / len(ms[last:]),
                               oom=False, load=None))
            last = steps
        if steps / N >= nxt:
            el = time.perf_counter() - t0
            print(f"   ctx {c:>8}: {sum(ms[-binw:]) / min(binw, len(ms)):7.3f} ms/step  "
                  f"[{el/60:.1f} min, ~{el/steps*(N-steps)/60:.1f} left]", flush=True)
            nxt += 0.05
    if full:
        append(a.out, dict(base, context=N + 1, ms_per_tok=None, oom=True, load=None))
    print(f"   done: {steps} steps for {N} tokens (a preempted sequence would add steps), "
          f"mean {sum(ms)/len(ms):.3f} ms/step", flush=True)


def cmd_prefill(a):
    """Time to first token for a random prompt of every `--step` tokens, up to what the cache
    holds: each prompt is one request to the running engine, after warm-up requests."""
    import torch
    from vllm import SamplingParams
    K = grows_with_context(a)                  # 0 = constant state, stop at --max-context
    if K:
        N = (min(K, a.max_context) if a.max_context else K)
        N = (N - 1024) // a.step * a.step      # slack, so that the last prompt still fits
    else:
        N = a.max_context // a.step * a.step
    e = _engine(a, N + 64, 1)
    base = dict(base_record(e, a, 1, "prefill"), step=a.step)
    print(f"== {a.method}/vllm prefill: prompts of every {a.step} tokens up to {N}, chunks of "
          f"{base['max_num_batched_tokens']}", flush=True)
    sp = SamplingParams(max_tokens=1, temperature=1.0, ignore_eos=True, detokenize=False)
    g = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(0, base["geometry"]["vocab"], (N,), generator=g).tolist()

    def ttft(n, rid):
        t = time.perf_counter()
        e.add_request(rid, {"prompt_token_ids": ids[:n]}, sp)
        while e.has_unfinished_requests():
            e.step()
        return time.perf_counter() - t

    for n in (a.step, min(N, 8192), a.step):   # warm: compile paths and the block allocator
        ttft(n, f"warm{n}")
    if os.path.exists(a.out):
        os.remove(a.out)
    t0 = time.perf_counter()
    for i, n in enumerate(range(a.step, N + 1, a.step)):
        s = ttft(n, f"p{n}")
        append(a.out, dict(base, context=n, seconds=s, tok_per_s=n / s, oom=False, load=None))
        if i % 8 == 0 or n == N:
            print(f"   prompt {n:>8}: {s:8.3f} s  [{(time.perf_counter()-t0)/60:.1f} min]",
                  flush=True)
    if K:
        append(a.out, dict(base, context=N + a.step, seconds=None, tok_per_s=None, oom=True,
                           load=None))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["gen", "prefill", "probe"])
    p.add_argument("--model", required=True,
                   help="HuggingFace model dir (export_llama.py, export_gdn.py)")
    p.add_argument("--method", default="softmax",
                   help="the label of the records, and whether the state grows with the "
                        "context: 'softmax' (kv-cache, the run ends where it is full) or "
                        "e.g. 'gdn' (constant state, the run ends at --max-context)")
    p.add_argument("--out", help="jsonl to write (gen, prefill)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--max-context", type=int, default=0,
                   help="stop at this context; 0 = where the cache is full")
    p.add_argument("--points", type=int, default=400, help="gen: records per curve")
    p.add_argument("--step", type=int, default=1024, help="prefill: a prompt every this many tokens")
    p.add_argument("--backend", default=None,
                   help="vLLM attention backend, e.g. FLASH_ATTN (default) or TRITON_ATTN")
    p.add_argument("--util", type=float, default=0.95, help="gpu_memory_utilization")
    p.add_argument("--kv-tokens", type=int, default=0,
                   help="the cache capacity, if already known (else probed in a subprocess)")
    p.add_argument("--bos", type=int, default=50256, help="the one-token prompt of generation")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.what != "probe" and not a.out:
        p.error("--out is required for gen and prefill")
    {"gen": cmd_gen, "prefill": cmd_prefill, "probe": cmd_probe}[a.what](a)


if __name__ == "__main__":
    main()
