"""Generation speed vs context length, for one model and one placement of its state.

One run = one curve = one file. The model generates tokens one at a time, from an empty
context until its cache is used up, and every token's wall-clock time is recorded; --out
gets one JSON line per bin of consecutive tokens with the mean ms/token in that bin. There
is no grid and no synthetic fill: the state at context c is whatever generating c tokens
left behind, and the curve's integral is the time the run actually took.

    # 0.8B geometry, LEMA hash table in 50 GB of host RAM
    python measure_generation.py --method lema-ram --out out/gen_08B_ram.jsonl \\
        --depth 24 --dim 1536 --num-heads 24 --head-dim 64 --store-gb 50

    # the softmax baseline: same geometry, kv cache in whatever VRAM is free
    python measure_generation.py --method softmax --out out/gen_08B_softmax.jsonl \\
        --depth 24 --dim 1536 --num-heads 24 --head-dim 64 --store-gb 0

    # a trained checkpoint from experiments/main_lm (geometry comes from the checkpoint)
    python measure_generation.py --method lema-ram --out out/gen_08B_ram.jsonl \\
        --checkpoint ../main_lm/out/lema1536_h64 --store-gb 50

Both caches are provisioned ONCE from a byte budget (--store-gb) and never reallocated, so
the two curves end differently: the softmax kv cache is full at the last token (the next
one would not fit -- recorded as an OOM point), while the LEMA table FILLS: linear probing
lengthens as occupancy rises, so its curve turns up before --max-load and stops there. A
trained checkpoint repeats codes and fills the table far more slowly, so with --checkpoint
the run goes to --max-context and the occupancy stays low.

Each token is a `lema.decode` step: the whole GPU side captured into CUDA graphs, the
next token sampled at --temperature (0 = greedy) and fed back, starting from the GPT-2
end-of-text token. Without a checkpoint the weights are random and a LEMA model would emit
the same code every step, so the store is hit with fresh uniform-random codes on both sides
(every lookup a full unsuccessful probe, every insert a fresh random slot): the worst
access pattern any model can produce. With --checkpoint the model's own codes are used.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

import common as C
from lema.decode import decoder

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

    # ---- the budget -> how far this cache reaches ---------------------------------------
    if a.store_gb:
        budget = int(a.store_gb * 1e9)
    elif method == "softmax":
        budget = torch.cuda.mem_get_info()[0] - int(1e9)   # graph pool, workspace, logits
    else:
        budget = C.meminfo_available() - int(8e9)   # the kernel must not have to swap
    if method == "softmax":
        capacity = None
        reach = C.kv_capacity(cfg, budget, a.batch)
    else:
        capacity = C.store_capacity(cfg, budget)
        # tokens at the target occupancy: every token adds one entry per partition
        reach = int(capacity * a.max_load) // C.partitions(cfg, a.batch)
    # `reach` assumes the worst case, one new entry per partition and token. A trained
    # checkpoint repeats codes and fills the table much later, so with --max-context it may
    # run past `reach`; the loop then stops at --max-load from the table's actual occupancy.
    if a.max_context and a.checkpoint and method != "softmax":
        n_tok = a.max_context
    else:
        n_tok = min(a.max_context, reach) if a.max_context else reach
    if n_tok < 1 + a.prefill:
        raise SystemExit(f"budget {budget/1e9:.1f} GB holds no context for this geometry")
    oom_at = n_tok + 1 if (method == "softmax" and n_tok == reach) else None
    binw = max(1, -(-n_tok // a.points))

    base = dict(kind="generation", method=method, batch=a.batch,
                geometry=dict(depth=cfg.depth, dim=cfg.dim, num_heads=cfg.num_heads,
                              num_kv_heads=cfg.num_kv_heads,
                              d_qk=cfg.d_qk, d_v=cfg.d_v, dim_ff=cfg.dim_ff,
                              vocab=cfg.vocab_size, attn=cfg.attn, pos=cfg.pos),
                n_params=C.n_params(model), weight_bytes=C.weight_bytes(model),
                weights="trained" if a.checkpoint else "random", checkpoint=a.checkpoint,
                worst_case=(attn == "lema" and a.checkpoint is None), compiled=a.compile,
                kv_splits=(a.kv_splits if method == "softmax" else None),
                store_budget_bytes=budget, capacity=capacity, bin_tokens=binw,
                machine=C.machine_meta(), git=C.git_sha(), time=C.now_iso())
    print(f"== {method}  {cfg.summary()}  ({base['n_params']/1e6:.0f}M params, "
          f"{base['weight_bytes']/1e9:.2f} GB weights, {base['weights']} weights)",
          flush=True)
    print(f"   budget {budget/1e9:.1f} GB -> {reach} tokens"
          + (f" ({capacity} slots, {a.max_load*100:.0f}% full at the end)"
             if capacity else "")
          + f"; generating {n_tok}, {binw} tokens per record", flush=True)

    # ---- the cache and the decoder --------------------------------------------------------
    if method == "softmax":
        cache = model.new_cache("vram", max_cache_len=n_tok, batch=a.batch,
                                num_splits=a.kv_splits)
    else:
        cache = model.new_cache("ram", capacity=capacity, batch=a.batch,
                                codes_hook=None if a.checkpoint else C.worst_case_codes(cfg.depth))
    # Start where a document starts: the GPT-2 separator the training data joins on,
    # rather than token 0 ('!'). Only the token stream cares -- with random weights, or
    # with the worst-case codes hook, the text is meaningless either way.
    bos = torch.full((a.batch, 1), min(50256, cfg.vocab_size - 1), dtype=torch.long,
                     device="cuda")
    if a.prefill:
        # start from a context of --prefill random tokens fed through the cache untimed,
        # for probing a curve segment without generating everything before it
        torch.manual_seed(a.seed)
        ids = torch.randint(0, cfg.vocab_size, (a.batch, a.prefill), device="cuda")
        for lo in range(0, a.prefill, 1024):
            model(ids[:, lo:lo + 1024], cache=cache, return_logits=False)
        torch.cuda.synchronize()
        bos = ids[:, -1:]
    dec = decoder(model, cache, tok=bos, temperature=a.temperature)
    n0 = a.prefill + 3
    for _ in range(3):                    # first-replay effects belong to no context
        dec.step()

    # ---- generate --------------------------------------------------------------------------
    ms = np.empty(n_tok - n0, dtype=np.float64)
    t_start = time.perf_counter()
    next_report = 0.05
    last = 0                              # index after the last recorded token
    for i in range(n_tok - n0):
        t0 = time.perf_counter()
        dec.step()
        ms[i] = (time.perf_counter() - t0) * 1e3
        c = n0 + i + 1                    # context after this token
        # a record per bin, plus one at every power of two before the first bin, so that
        # short and long curves alike start at 16 tokens rather than at their bin width
        early = c < binw and c >= 16 and (c & (c - 1)) == 0
        if c % binw == 0 or c == n_tok or early:
            lo = last
            last = i + 1
            load = cache.load() if capacity else None
            C.append_jsonl(a.out, dict(base, context=c, ms_per_tok=float(ms[lo:i + 1].mean()),
                                       oom=False, load=load, swap_kb=C.vmswap_kb()))
            if load is not None and load >= a.max_load:
                print(f"   ctx {c:>8}: table at {load*100:.1f}% (--max-load), stopping", flush=True)
                ms = ms[:i + 1]
                break
        if (i + 1) / (n_tok - n0) >= next_report:
            el = time.perf_counter() - t_start
            print(f"   ctx {c:>8}: {ms[max(0, i + 1 - binw):i + 1].mean():7.3f} ms/tok"
                  + (f"  load {cache.load()*100:5.1f}%" if capacity else "")
                  + f"  [{el/60:.1f} min, ~{el/(i+1)*(n_tok-n0-i-1)/60:.1f} left]",
                  flush=True)
            next_report += 0.05
    if oom_at is not None:
        need = C.kv_bytes(cfg, oom_at, a.batch)
        C.append_jsonl(a.out, dict(base, context=oom_at, ms_per_tok=None, oom=True,
                                   load=None, kv_bytes=need))
        print(f"   ctx {oom_at:>8}: OOM  (kv needs {need/1e9:.1f} GB > "
              f"{budget/1e9:.1f} GB)", flush=True)
    print(f"   done: {n_tok} tokens, mean {ms.mean():.3f} ms/tok, "
          f"{(time.perf_counter()-t_start)/60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="generation speed vs context for one (model, placement)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--out", required=True, help="JSONL curve file (one line per bin)")
    p.add_argument("--store-gb", type=float, default=0,
                   help="GB to provision for the cache: VRAM (softmax) or host RAM "
                        "(lema-ram). 0 = take what is free.")
    p.add_argument("--max-context", type=int, default=0,
                   help="stop after this many tokens; 0 = as far as --store-gb reaches")
    p.add_argument("--points", type=int, default=400,
                   help="records per curve (tokens are binned to about this many)")
    p.add_argument("--max-load", type=float, default=0.95,
                   help="occupancy the LEMA store is generated up to. Measured cost per "
                        "lookup: 46ns at 0.5, 218ns at 0.9, 572ns at 0.95 -- so this "
                        "decides how much of the tail turns up")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--prefill", type=int, default=0,
                   help="start from this many random tokens of context (untimed prefill)")
    p.add_argument("--kv-splits", type=int, default=0,
                   help="softmax: fixed split-KV count of the decode kernel; 0 = the "
                        "count that fills the SMs in whole waves for the live range")
    p.add_argument("--no-compile", dest="compile", action="store_false",
                   help="skip torch.compile of the per-layer segments (it is ON by "
                        "default; the numbers are not comparable without it)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="sampling temperature of the generated tokens (0 = greedy); only "
                        "matters with a checkpoint, where the text decides the codes")
    p.add_argument("--reset", action="store_true", help="truncate --out first")
    C.add_geometry_args(p)
    a = p.parse_args()
    if a.reset:
        C.reset_jsonl(a.out)
    run(a)
    # Exit without interpreter teardown: tearing down CUDA graphs, a compiled model AND a
    # multi-gigabyte store segfaults in torch's static destructors -- after every record is
    # already on disk, so the crash would cost nothing but a non-zero exit code.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
