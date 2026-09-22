"""Prefill time vs prompt length, for one model and one placement of its state.

One run = one curve = one file. A random prompt is fed through the model in chunks of
--chunk tokens into a fresh, provisioned cache, and after every --step tokens the elapsed
wall time is recorded: the time it takes to ingest a prompt of that length. --out gets
one JSON line per record.

    python measure_prefill.py --method lema-ram --out out/prefill_08B_ram.jsonl \\
        --depth 24 --dim 1536 --num-heads 24 --head-dim 64 --store-gb 50 --chunk 2048
    python measure_prefill.py --method softmax --out out/prefill_08B_softmax.jsonl \\
        --depth 24 --dim 1536 --num-heads 24 --head-dim 64 --store-gb 0

Chunked prefill is what serving systems do (vLLM, SGLang and TensorRT-LLM all default to
it): it bounds the activation memory at one chunk's worth, so the only memory that grows
with the prompt is the cache itself -- the kv cache in VRAM for softmax, which ends the
curve with an OOM, the hash table in RAM for LEMA, which ends it at --max-load (a trained
checkpoint fills the table far more slowly, so with --checkpoint the prompt runs to
--max-context instead, see the branch below). The work
is the same as a one-shot prefill (every chunk attends to the whole prefix; for softmax
that is quadratic in the prompt, for LEMA one exchange per token). And because a chunked
prefill of n tokens IS the first n tokens of a longer one, a single run yields the whole
curve: the record at n is the cumulative time.

Both caches are provisioned ONCE from a byte budget (--store-gb) and never reallocated.
As in measure_generation.py, a random-weight LEMA model's store is hit with fresh
uniform-random codes on both sides (the worst access pattern) and a --checkpoint uses the
model's own codes, on a prompt of held-out validation text instead of random ids.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import torch

import common as C

METHODS = ("softmax", "lema-ram")


@torch.no_grad()
def run(a):
    method = a.method
    attn = "softmax" if method == "softmax" else "lema"
    model, cfg = C.make_model(a, attn, "cuda")
    if cfg.attn != attn:      # a checkpoint fixes its own attention kind
        raise SystemExit(f"--method {method} expects a {attn!r} model, but the checkpoint "
                         f"was trained with attn={cfg.attn!r}")
    if a.compile:
        model.compile_inference()
    chunk = a.chunk or a.step
    if a.step % chunk and chunk % a.step:
        raise SystemExit(f"--chunk {chunk} and --step {a.step}: one must divide the other")
    unit = max(chunk, a.step)      # a record lands at the end of a chunk, never inside one

    # ---- the budget -> how far this cache reaches ---------------------------------------
    if a.store_gb:
        budget = int(a.store_gb * 1e9)
    elif method == "softmax":
        budget = torch.cuda.mem_get_info()[0] - int(1e9)   # a chunk's activations
    else:
        budget = C.meminfo_available() - int(8e9)   # the kernel must not have to swap
    if method == "softmax":
        capacity = None
        reach = C.kv_capacity(cfg, budget, a.batch)
    else:
        capacity = C.store_capacity(cfg, budget)
        reach = int(capacity * a.max_load) // C.partitions(cfg, a.batch)
    # `reach` assumes the worst case, one new entry per partition and token, as in
    # measure_generation.py. A trained checkpoint repeats codes and fills the table far
    # later, so with --max-context it may prefill past `reach`; every record carries the
    # table's actual occupancy, and the run stops at --max-context either way.
    if a.max_context and a.checkpoint and method != "softmax":
        n_max = a.max_context // unit * unit
    else:
        n_max = (min(a.max_context, reach) if a.max_context else reach) // unit * unit
    if n_max < unit:
        raise SystemExit(f"budget {budget/1e9:.1f} GB holds fewer than {unit} tokens")
    oom_at = n_max + a.step if (method == "softmax" and n_max + a.step > reach) else None

    def new_cache(n, slots):
        if method == "softmax":
            return model.new_cache("vram", max_cache_len=n, batch=a.batch)
        hook = None if a.checkpoint else C.worst_case_codes(cfg.depth)
        return model.new_cache("ram", capacity=slots, batch=a.batch, codes_hook=hook)

    base = dict(kind="prefill", method=method, batch=a.batch, step=a.step, chunk=chunk,
                geometry=dict(depth=cfg.depth, dim=cfg.dim, num_heads=cfg.num_heads,
                              num_kv_heads=cfg.num_kv_heads,
                              d_qk=cfg.d_qk, d_v=cfg.d_v, dim_ff=cfg.dim_ff,
                              vocab=cfg.vocab_size, attn=cfg.attn, pos=cfg.pos),
                n_params=C.n_params(model), weight_bytes=C.weight_bytes(model),
                weights="trained" if a.checkpoint else "random", checkpoint=a.checkpoint,
                worst_case=(attn == "lema" and a.checkpoint is None), compiled=a.compile,
                store_budget_bytes=budget, capacity=capacity,
                machine=C.machine_meta(), git=C.git_sha(), time=C.now_iso())
    print(f"== {method}  {cfg.summary()}  ({base['n_params']/1e6:.0f}M params, "
          f"{base['weight_bytes']/1e9:.2f} GB weights, {base['weights']} weights)",
          flush=True)
    print(f"   budget {budget/1e9:.1f} GB -> {reach} tokens; prompt of {n_max} in chunks "
          f"of {chunk}, a record every {a.step}", flush=True)

    # one chunk through a throwaway cache first (sized for that chunk, not the budget):
    # compilation and kernel loading are not part of any prompt's time
    torch.manual_seed(a.seed)
    if a.checkpoint:
        # a trained model gets held-out text, so its codes are the ones it produces on
        # real prompts; random ids would give a random model's kind of codes
        from lema import TokenCorpus
        from experiments.paths import DATA
        val = TokenCorpus(str(DATA), seq_len=n_max).arrs["val"][0]
        ids = torch.from_numpy(val[:a.batch * n_max].astype("int64")).view(a.batch, n_max).cuda()
    else:
        ids = torch.randint(0, cfg.vocab_size, (a.batch, n_max), device="cuda")
    # the throwaway table only has to hold one chunk's worth of entries (never more than
    # the real one's slots, which a large --chunk would otherwise ask for many times over)
    warm_slots = min(2 * chunk * C.partitions(cfg, a.batch), capacity or 1 << 62)
    warm = new_cache(chunk, warm_slots)
    model(ids[:, :chunk], cache=warm, return_logits=False)
    torch.cuda.synchronize()
    del warm                  # the real table is allocated next and both are tens of GB:
    gc.collect()              # hand this one back before asking for that one

    cache = new_cache(n_max, capacity)
    t_start = time.perf_counter()
    next_report = 0.05
    for pos in range(0, n_max, chunk):
        model(ids[:, pos:pos + chunk], cache=cache, return_logits=False)
        n = pos + chunk
        if n % a.step == 0:
            torch.cuda.synchronize()
            el = time.perf_counter() - t_start
            C.append_jsonl(a.out, dict(base, context=n, seconds=el, tok_per_s=n / el,
                                       oom=False, load=(cache.load() if capacity else None),
                                       swap_kb=C.vmswap_kb()))
            if n / n_max >= next_report:
                print(f"   ctx {n:>8}: {el:8.2f} s  ({n/el:9,.0f} tok/s so far)"
                      + (f"  load {cache.load()*100:5.1f}%" if capacity else ""), flush=True)
                next_report += 0.05
    if oom_at is not None:
        need = C.kv_bytes(cfg, oom_at, a.batch)
        C.append_jsonl(a.out, dict(base, context=oom_at, seconds=None, tok_per_s=None,
                                   oom=True, load=None, kv_bytes=need))
        print(f"   ctx {oom_at:>8}: OOM  (kv needs {need/1e9:.1f} GB > "
              f"{budget/1e9:.1f} GB)", flush=True)
    print(f"   done: {n_max} tokens in {time.perf_counter()-t_start:.1f} s", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="prefill time vs prompt length for one (model, placement)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--out", required=True, help="JSONL curve file (one line per record)")
    p.add_argument("--store-gb", type=float, default=0,
                   help="GB to provision for the cache: VRAM (softmax) or host RAM "
                        "(lema-ram). 0 = take what is free.")
    p.add_argument("--max-context", type=int, default=0,
                   help="longest prompt to measure; 0 = as far as --store-gb reaches")
    p.add_argument("--step", type=int, default=1024,
                   help="record the prefill time at every multiple of this many tokens")
    p.add_argument("--chunk", type=int, default=0,
                   help="tokens per forward pass (chunked prefill); 0 = --step. One of "
                        "--chunk and --step must divide the other; records land at the "
                        "end of a chunk.")
    p.add_argument("--max-load", type=float, default=0.95,
                   help="occupancy the LEMA store is filled up to")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--no-compile", dest="compile", action="store_false",
                   help="skip torch.compile of the per-layer segments")
    p.add_argument("--reset", action="store_true", help="truncate --out first")
    C.add_geometry_args(p)
    a = p.parse_args()
    if a.reset:
        C.reset_jsonl(a.out)
    run(a)
    sys.stdout.flush()      # see measure_generation.py: skip the interpreter teardown
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
