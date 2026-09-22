"""Which occurrence does a model read when every key is repeated?

    repeated_keys.py <model_best.pt or run dir under ../main_recall/out> --n-queries 64
                     --key-repeats 4 [--min-queries 8192] [--seed 0]

n-queries distinct keys, each appearing key-repeats times with values that are all
distinct within a sample, pair order shuffled, SEP, then every key once as a query.
Reports the mean probability mass on the value of the 1st, 2nd, ..., last occurrence of
the query's key, over at least --min-queries query tokens -- a LEMA model can only read
the LAST one (the earlier entries were overwritten), a softmax model may spread its mass.
Writes out/repeated_keys/<arch>_<seed>_q<Q>r<R>.json.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from lema.tasks import Recall                                # noqa: E402
from experiments.main_recall.eval_recall import OUT as RECALL_OUT, load   # noqa: E402
from experiments.main_recall.train_recall import KEY_VOCAB   # noqa: E402

SEP = 2 * KEY_VOCAB
OUT_DIR = HERE / "out" / "repeated_keys"


def build(rng, B, Q, R):
    """x [B, T] and occ_val [B, Q, R]: the value at the r-th occurrence of query i's key."""
    N = Q * R
    keys = Recall._distinct(rng, B, Q, KEY_VOCAB)
    vals = Recall._distinct(rng, B, N, KEY_VOCAB) + KEY_VOCAB
    kk = np.repeat(keys, R, axis=1)
    perm = np.argsort(rng.random((B, N)), axis=1)
    kk, vals = np.take_along_axis(kk, perm, 1), np.take_along_axis(vals, perm, 1)
    order = np.argsort(rng.random((B, Q)), axis=1)
    x = np.full((B, 2 * N + 1 + Q), SEP, dtype=np.int64)
    x[:, 0:2 * N:2], x[:, 1:2 * N:2] = kk, vals
    x[:, 2 * N + 1:] = np.take_along_axis(keys, order, 1)
    by_key = np.argsort(kk * N + np.arange(N)[None], axis=1).reshape(B, Q, R)
    rank = np.argsort(np.argsort(keys, axis=1), axis=1)
    pair = np.take_along_axis(by_key, np.take_along_axis(rank, order, 1)[:, :, None], 1)
    occ_val = np.take_along_axis(vals, pair.reshape(B, -1), 1).reshape(B, Q, R)
    return x, occ_val


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="model_best.pt or a run dir (bare names resolve under "
                                "../main_recall/out, e.g. lema/s0)")
    p.add_argument("--n-queries", type=int, required=True)
    p.add_argument("--key-repeats", type=int, required=True)
    p.add_argument("--min-queries", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    Q, R = a.n_queries, a.key_repeats
    if Q * R > KEY_VOCAB:
        raise SystemExit(f"n-queries * key-repeats must be <= {KEY_VOCAB} (distinct values)")

    ckpt = Path(a.ckpt) if Path(a.ckpt).exists() else RECALL_OUT / a.ckpt
    model = load(ckpt, device=a.device)
    B = -(-a.min_queries // Q)
    x, occ_val = build(np.random.default_rng(a.seed), B, Q, R)
    T = x.shape[1]
    micro = max(1, int(2e9 // (T * model.cfg.vocab_size * 4)))
    mass = np.zeros(R)
    for lo in range(0, B, micro):
        xb = torch.from_numpy(x[lo:lo + micro]).to(a.device)
        with torch.autocast(a.device, dtype=torch.bfloat16, enabled=model.cfg.attn == "gated-deltanet"):
            logits = model(xb)[:, 2 * Q * R + 1:].float()
        pr = torch.softmax(logits, -1)
        ov = torch.from_numpy(occ_val[lo:lo + micro]).to(a.device)
        mass += torch.gather(pr, 2, ov).sum((0, 1)).cpu().numpy()
    mass /= B * Q
    print(f"{ckpt.resolve().relative_to(RECALL_OUT.parent)}  {Q} queries x {R} repeats, "
          f"{T} tokens, {B * Q} query tokens")
    for r in range(R):
        print(f"  occurrence {r + 1:>2}{' (last)' if r == R - 1 else '':<7} mass {mass[r]:.4f}")
    print(f"  elsewhere{'':<12} mass {1 - mass.sum():.4f}")
    arch = {"lema": "lema", "softmax": "rope", "gated-deltanet": "gdn"}[model.cfg.attn]
    seed = next((s for s in ckpt.resolve().parts if re.fullmatch(r"s\d+", s)), "s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{arch}_{seed}_q{Q}r{R}.json"
    out.write_text(json.dumps({"ckpt": str(ckpt), "arch": arch, "seed": seed,
                               "n_queries": Q, "key_repeats": R, "query_tokens": B * Q,
                               "tokens": T, "sample_seed": a.seed,
                               "mass_by_occurrence": [round(float(m), 5) for m in mass],
                               "mass_elsewhere": round(float(1 - mass.sum()), 5)},
                              indent=1) + "\n")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
