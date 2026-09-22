"""What the heads of a trained LEMA language model read: hit rate and attention distance.

    python analyze_heads.py lema1536_h64 [--length 16384] [--count 8] [--seed 0]

For `--count` held-out windows of `--length` tokens, `lema.analysis.trace` gives every
head's source position for every query (-1: no match). Recorded per layer and head:
  hit        the fraction of queries that find a key (over all positions);
  dist       histogram of the distance i - src over hits, in bins 1, 2, 3-4, 5-8, ...
Averaged over the windows. Writes out/<run>.json; plot_distances.py draws the distance figure from it.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="run dir, or a name under ../main_lm/out")
    p.add_argument("--length", type=int, default=16384)
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    run = Path(a.run) if Path(a.run).exists() else LM_OUT / a.run
    model, cfg = load_trained(run)
    if cfg.attn != "lema":
        raise SystemExit(f"{run.name} is not a LEMA model")
    T = a.length
    nbins = int(np.ceil(np.log2(T))) + 1
    hit = torch.zeros(cfg.depth, cfg.num_heads, dtype=torch.float64)
    dist = torch.zeros(cfg.depth, cfg.num_heads, nbins, dtype=torch.float64)
    pos = torch.arange(T, device="cuda")
    windows = TokenCorpus(DATA, seq_len=T).batches("val", 1, seed=a.seed)

    def reduce(li, r):
        src = r["src"][0]                                        # [H, T]
        found = src >= 0
        hit[li] += found.double().mean(1).cpu()
        d = (pos - src).clamp_min(1)
        b = torch.where(d <= 2, d - 1, torch.ceil(torch.log2(d.double())).long())  # 1,2,3-4,5-8..
        for h in range(cfg.num_heads):
            dist[li, h] += torch.bincount(b[h][found[h]], minlength=nbins).double().cpu()[:nbins]
        return None

    for i in range(a.count):
        x = next(windows)[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            trace(model, x.cuda(), reduce=reduce, return_logits=False)
        torch.cuda.empty_cache()
        print(f"window {i + 1}/{a.count}: mean hit rate so far {hit.mean() / (i + 1):.3f}", flush=True)
    hit /= a.count
    dist = dist / dist.sum(-1, keepdim=True).clamp_min(1)
    rec = {"run": run.name,
           "model": {k: getattr(cfg, k) for k in ("dim", "depth", "num_heads", "d_qk")},
           "length": T, "count": a.count, "seed": a.seed,
           "dist_bins": ["1", "2"] + [f"{2 ** (k - 1) + 1}-{2 ** k}" for k in range(2, nbins)],
           "hit": [[round(v, 4) for v in row] for row in hit.tolist()],
           "dist": [[[round(v, 4) for v in h] for h in layer] for layer in dist.tolist()]}
    out = a.out or HERE / "out" / f"{run.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
