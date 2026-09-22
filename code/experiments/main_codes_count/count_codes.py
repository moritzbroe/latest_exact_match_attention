"""Distinct-code census of a trained LEMA model: how many entries does each head's
kv table hold after t tokens of held-out text, for every t up to --length?

A LEMA head's state after t tokens is one entry per DISTINCT key code among its first t
keys, so the curve "entries vs context length" is the head's state-size growth: flat where
a head reuses a few codes, linear where every token gets a new one, and capped at 2^d_qk.
This is the number the paper's state-size figure plots, and the quantity the inference
section's worst-case (never-repeating) codes are the upper bound of.

Method: `--count` random windows of `--length` tokens from the validation split, each run
through the model's exact forward in ONE pass (the dense latest-match op is O(T log T),
so a 256k-token window needs no chunking and no cache); `lema.analysis.trace` hands over
every layer's key codes, and a stable sort per head marks each code's first occurrence --
its cumulative sum over positions is the entry count after every prefix. Averaged over
the windows.

Output: out/<run>.json with the model geometry, a `lengths` grid (log-spaced, --points of
them; 0 = every position) and `heads["l{layer}h{head}"]` = mean entry count at each grid
length, plus `mean` over heads. table.py prints the tables of the appendix.

Usage: count_codes.py <run dir or name under ../main_lm/out> [--length 262144] [--count 8]
           [--points 1000] [--seed 0]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
import numpy as np
import torch

from lema import TokenCorpus, load_trained
from lema.analysis import trace
from experiments.paths import DATA, LM_OUT


def prefix_distinct(codes: torch.Tensor) -> torch.Tensor:
    """[H, T] codes -> [H, T] int64: the number of distinct codes among positions
    <= t, for every t. Stable sort puts equal codes adjacent in position order, so the
    first element of every run is the earliest occurrence of its code."""
    order = codes.argsort(dim=1, stable=True)
    s = codes.gather(1, order)
    first = torch.ones_like(s, dtype=torch.bool)
    first[:, 1:] = s[:, 1:] != s[:, :-1]
    is_first = torch.zeros_like(first).scatter_(1, order, first)
    return is_first.long().cumsum(1)


@torch.no_grad()
def census(model, x: torch.Tensor, device="cuda"):
    """[depth, heads, T] entry counts after each prefix of `x` ([1, T] tokens)."""
    with torch.autocast(device, dtype=torch.bfloat16):
        _, counts = trace(model, x.to(device), return_logits=False,
                          reduce=lambda li, r: prefix_distinct(r["kc"][0]))
    return torch.stack(counts)                         # [L, H, T]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="run dir, or a name under ../main_lm/out")
    p.add_argument("--length", type=int, default=262144)
    p.add_argument("--count", type=int, default=8, help="validation windows averaged")
    p.add_argument("--points", type=int, default=1000,
                   help="log-spaced context lengths kept in the json (0 = all of them)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="default: out/<run name>.json")
    a = p.parse_args()
    run = Path(a.run) if Path(a.run).exists() else LM_OUT / a.run
    model, cfg = load_trained(run)
    if cfg.attn != "lema":
        raise SystemExit(f"{run.name} is not a LEMA model")
    task = TokenCorpus(DATA, seq_len=a.length)
    windows = task.batches("val", 1, seed=a.seed)
    acc = torch.zeros(cfg.depth, cfg.num_heads, a.length, dtype=torch.float64)
    for i in range(a.count):
        x = next(windows)[0]
        acc += census(model, x).double().cpu()
        torch.cuda.empty_cache()
        last = acc[:, :, -1] / (i + 1)
        print(f"window {i + 1}/{a.count}: entries/head at T={a.length}: "
              f"mean {last.mean():.0f} min {last.min():.0f} max {last.max():.0f}", flush=True)
    acc /= a.count
    if a.points:
        # log-spaced grid, always including the powers of two up to --length so tables
        # can quote exact context lengths (2k, 4k, ...) without interpolating
        lengths = np.unique(np.concatenate([
            np.round(np.geomspace(1, a.length, a.points)).astype(int),
            2 ** np.arange(0, int(np.log2(a.length)) + 1)]))
        lengths = lengths[lengths <= a.length]
    else:
        lengths = np.arange(1, a.length + 1)
    sel = acc[:, :, torch.as_tensor(lengths - 1)]     # count after `length` tokens
    rec = {"run": run.name,
           "model": {k: getattr(cfg, k) for k in ("dim", "depth", "num_heads", "d_qk")},
           "max_codes": 2 ** cfg.d_qk, "length": a.length, "count": a.count,
           "seed": a.seed, "lengths": lengths.tolist(),
           "mean": [round(v, 3) for v in sel.mean((0, 1)).tolist()],
           "heads": {f"l{li}h{h}": [round(v, 3) for v in sel[li, h].tolist()]
                     for li in range(cfg.depth) for h in range(cfg.num_heads)}}
    out = a.out or HERE / "out" / f"{run.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
