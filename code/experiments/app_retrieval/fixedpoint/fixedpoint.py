#!/usr/bin/env python
"""The fixed point of S-NIAH-1's filler: after how many repetitions of the filler sentence
the dictionaries of a LEMA model stop changing, before the needle and after it.

    python fixedpoint.py lema1024_h64 lema1536_h64

The prompt is RULER's S-NIAH-1 prompt: header, the filler sentence repeated, the needle
among the repetitions, the question. It is fed through the inference path (host-RAM hash
table, bf16) one repetition at a time, and the complete dictionary of every head is read
out of the table after each. A LEMA model has no positional encoding, so a repetition that
leaves every dictionary unchanged is a fixed point: the next repetition starts from the same
dictionaries with the same tokens and writes the same entries. Reports, per model, the last
repetition that changed any entry before the needle and after it, and the number of entries
at each stage. Writes ../out/fixedpoint/<model>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                 # tokens.py, ruler.py, gen_eval.py
sys.path.insert(0, str(HERE.parents[2]))             # code/
import ruler                                                            # noqa: E402
from lema import load_trained                                           # noqa: E402
from tokens import dec, enc                                             # noqa: E402
from gen_eval import resolve                                            # noqa: E402

OUT = HERE.parent / "out" / "fixedpoint"
DTYPE = torch.bfloat16          # the dtype policy of every evaluation in this repository
DEV = "cuda"


def frame() -> tuple[str, str]:
    """(header, question) of a single-needle prompt, from `ruler`'s own strings through
    RULER's singularization. The question carries `\\x01` where the key goes."""
    t = ruler.TASK_TEMPLATE + ruler.ANSWER_PREFIX
    t = t.replace("Some", "A").replace("are all", "is")
    t = t.replace("are", "is").replace("answers", "answer")
    full = t.format(type_needle_v="number", context="\x00", query="\x01")
    head, tail = full.split("\n\x00\n")
    return head, tail


HEADER, _ = frame()
NOISE = ruler.NOISE


def parse_prompt(text: str) -> tuple[list[str], int, str]:
    """A bank prompt into (haystack lines, needle line index, question)."""
    parts = text.split("\n")
    if parts[0] != HEADER:
        raise ValueError("first line is not RULER's header")
    lines, question = parts[1:-1], parts[-1]
    where = [j for j, ln in enumerate(lines) if ln != NOISE]
    if len(where) != 1:
        raise ValueError(f"expected exactly one non-filler line, found {len(where)}")
    return lines, where[0], question


# ------------------------------------------------------------ reading the table out
# `lema/csrc/store.cpp`: a slot is [code u64 | state u64 | value bytes], and the state
# word is `group + 2` for a published entry. Nothing here writes to the table.


def dump(cache) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(group, code, value bits) of every entry, sorted by (group, code); group is
    layer * heads + head at batch 1."""
    arr = cache.store.table.numpy()
    w = arr.view(np.uint64)
    sel = np.flatnonzero(w[:, 1] >= 2)
    g = (w[sel, 1] - 2).astype(np.int64)
    c = np.ascontiguousarray(w[sel, 0])
    v = np.ascontiguousarray(arr[sel, 16:]).view(np.uint16)
    o = np.lexsort((c, g))
    return g[o], c[o], v[o]


def differs(a, b) -> tuple[bool, bool]:
    """(any entry differs, any key set differs) between two dumps."""
    ga, ca, va = a
    gb, cb, vb = b
    keys = not (len(ca) == len(cb) and np.array_equal(ga, gb) and np.array_equal(ca, cb))
    return keys or not np.array_equal(va, vb), keys


class Runner:
    def __init__(self, model, cfg, capacity: int):
        self.model, self.cfg = model, cfg
        self.cache = model.new_cache("ram", capacity=capacity, batch=1, dtype=DTYPE)

    def reset(self):
        self.cache.store.table.fill_(0)
        self.cache.store.counts.zero_()
        self.cache.t = 0

    @torch.no_grad()
    def feed(self, ids: np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(ids, np.int64))[None].to(DEV)
        with torch.autocast("cuda", dtype=DTYPE):
            self.model(t, cache=self.cache, return_logits=False)

    def entries(self) -> dict:
        c = self.cache.counts().flatten().numpy()
        return dict(total=int(c.sum()), per_head_mean=round(float(c.mean()), 1),
                    per_head_max=int(c.max()))


def until_fixed(r: Runner, ids_rep: np.ndarray, max_reps: int, patience: int, log) -> dict:
    """Feed the filler one repetition at a time until `patience` consecutive repetitions
    leave every dictionary unchanged. `fixed_after` is the last repetition that changed any
    entry (0 if none did), `keys_fixed_after` the last that changed a key set."""
    prev = dump(r.cache)
    last_any = last_keys = 0
    quiet = 0
    for k in range(1, max_reps + 1):
        r.feed(ids_rep)
        cur = dump(r.cache)
        any_, keys = differs(prev, cur)
        if any_:
            last_any, quiet = k, 0
            if keys:
                last_keys = k
        else:
            quiet += 1
        log(f"   rep {k:3d}  entries {len(cur[1]):7,}  {'changed' if any_ else 'unchanged'}")
        prev = cur
        if quiet >= patience:
            break
    return dict(fixed_after=last_any, keys_fixed_after=last_keys, reps_fed=k,
                unchanged_tail=quiet, entries=r.entries())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--ctx", type=int, default=16384, help="the bank prompt the needle comes from")
    p.add_argument("--max-reps", type=int, default=200)
    p.add_argument("--patience", type=int, default=16,
                   help="stop after this many consecutive unchanged repetitions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--store-mb", type=float, default=512.0)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bank = ruler.bank("niah_single_1", a.ctx, 1, seed=a.seed, hay="noise")
    lines, j, question = parse_prompt(dec(bank["ids"][0, :int(bank["plen"][0])]))
    needle = lines[j]
    ids_head, ids_rep = enc(HEADER), enc("\n" + NOISE)
    ids_needle, ids_q = enc("\n" + needle), enc("\n" + question)
    rebuilt = np.concatenate([ids_head] + [ids_rep] * j + [ids_needle]
                             + [ids_rep] * (len(lines) - j - 1) + [ids_q])
    if not np.array_equal(rebuilt, bank["ids"][0, :int(bank["plen"][0])]):
        raise RuntimeError("the prompt is not a token-boundary concatenation of its lines")
    for name in a.models:
        model, cfg = load_trained(resolve(name), device=DEV)
        slot = 16 + cfg.d_v * torch.finfo(DTYPE).bits // 8
        r = Runner(model, cfg, max(1 << 14, int(a.store_mb * 1e6) // slot))
        t0 = time.perf_counter()
        print(f"== {name}: {cfg.depth} layers x {cfg.num_heads} heads, needle after "
              f"{j} of {len(lines)} filler lines in the {a.ctx}-token prompt", flush=True)
        r.reset()
        r.feed(ids_head)
        print("   before the needle", flush=True)
        before = until_fixed(r, ids_rep, a.max_reps, a.patience, print)
        r.feed(ids_needle)
        after_needle = r.entries()
        print("   after the needle", flush=True)
        after = until_fixed(r, ids_rep, a.max_reps, a.patience, print)
        r.feed(ids_q)
        final = r.entries()
        res = dict(model=name, ctx=a.ctx, seed=a.seed, needle=needle, question=question,
                   filler_lines=len(lines), needle_after=j, before=before,
                   needle_entries=after_needle, after=after, final_entries=final,
                   secs=round(time.perf_counter() - t0, 1))
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
        print(f"   fixed after {before['fixed_after']} repetitions before the needle "
              f"(key sets after {before['keys_fixed_after']}), {before['entries']['total']} "
              f"entries; after {after['fixed_after']} repetitions after it (key sets after "
              f"{after['keys_fixed_after']}), {after['entries']['total']} entries; "
              f"{final['total']} after the question ({res['secs']}s)", flush=True)
        del model, r
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
