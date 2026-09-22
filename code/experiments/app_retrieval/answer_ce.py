"""Cross-entropy of the correct answer on a RULER cell (appendix, further RULER tasks).

For every prompt of a cell the prompt is prefilled, the ":" that every model emits first is
fed, and the gold answer tokens are fed one at a time; the cross-entropy of the answer is
the sum of the negative log-probabilities of its tokens. Same banks, cached-state path and
bf16 autocast as gen_eval.py.

    python answer_ce.py lema1536_h64 gdn1536_16k rope1536_16k \
        --cell niah_single_2-essay --ctx 2048 4096 8192 16384
    python answer_ce.py --summary          # out/answer_ce/summary.csv from the cells on disk

Writes out/answer_ce/<cell>_<model>_T<ctx>.json: a _meta line with the mean answer
cross-entropy and the RULER accuracy of the matching out/gen cell, then one record per
prompt. The summary joins every cell into one csv for paper/plot_ruler_more.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))                               # code/
import gen_eval as ge                                                  # noqa: E402
from tokens import enc                                                 # noqa: E402
import ruler                                                           # noqa: E402
from lema import load_trained                                          # noqa: E402
from gen_eval import resolve                                           # noqa: E402

OUT = HERE / "out" / "answer_ce"
GEN = HERE / "out" / "gen"
COLON = enc(":")
assert len(COLON) == 1, COLON
COLON = int(COLON[0])


@torch.no_grad()
def run_batch(model, kind, ids, ans, alen, device="cuda"):
    """The answer cross-entropy of one same-length batch: [B] sums over the gold tokens."""
    B, L = ids.shape
    A = int(alen.max())
    ar = torch.arange(B, device=device)
    st = ge.STATES[kind](model, B, L + A + 2, device)
    with torch.autocast(device, dtype=ge.DTYPE):
        h = st.prefill(ids)
        h = st.step(torch.full((B, 1), COLON, dtype=torch.long, device=device))
    nll = torch.zeros((B, A), device=device)
    for j in range(A):
        lp = F.log_softmax(model.lm_head(h[:, -1]).float(), -1)
        nll[:, j] = -lp[ar, ans[:, j]]
        with torch.autocast(device, dtype=ge.DTYPE):
            h = st.step(ans[:, j:j + 1])
    del st
    mask = torch.arange(A, device=device)[None, :] < torch.as_tensor(alen, device=device)[:, None]
    return (nll * mask).sum(1).cpu().numpy()


def gen_match(cell: str, model: str, ctx: int):
    """The RULER accuracy of the matching free-generation cell, or None."""
    f = GEN / f"{cell}_{model}_T{ctx}.jsonl"
    if not f.exists():
        return None
    with f.open() as fh:
        return float(json.loads(fh.readline())["_meta"]["match"])


def summary():
    rows = []
    for f in sorted(OUT.glob("*.json")):
        with f.open() as fh:
            m = json.loads(fh.readline())["_meta"]
        rows.append(dict(cell=m["cell"], model=m["model"], ctx=m["ctx"], n=m["n"],
                         gen_match=m["gen_match"], answer_ce=m["answer_ce"]))
    with (OUT / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["cell", "model", "ctx", "n", "gen_match", "answer_ce"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cells -> {OUT / 'summary.csv'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*")
    p.add_argument("--cell", nargs="+", default=["niah_single_1-noise"],
                   help="<task>-<haystack>, as the out/gen file names spell it")
    p.add_argument("--ctx", nargs="+", type=int, default=[2048])
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-tokens", type=int, default=65536)
    p.add_argument("--cache-bytes", type=float, default=8e9)
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    if a.summary:
        return summary()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in a.models:
        run = resolve(name)
        model, cfg = load_trained(run)
        kind = ge.kind_of(cfg)
        for cell in a.cell:
            task, hay = cell.split("-", 1)
            for ctx in a.ctx:
                dst = OUT / f"{cell}_{name}_T{ctx}.json"
                if dst.exists():
                    print(f"{dst.name}: done", flush=True)
                    continue
                t0 = time.time()
                b = ruler.bank(task, ctx, a.n, a.seed, hay=hay)
                B = ge.batch_size(kind, cfg, ctx, 2, a.batch_tokens, a.cache_bytes)
                ce = np.zeros(len(b["ids"]))
                for L, rows in ge.batches(b["plen"], B):
                    ids = torch.from_numpy(b["ids"][rows, :L].astype(np.int64)).cuda()
                    ans = torch.from_numpy(b["ans"][rows].astype(np.int64)).cuda()
                    ce[rows] = run_batch(model, kind, ids, ans, b["alen"][rows])
                meta = dict(model=name, cell=cell, task=task, hay=hay, ctx=ctx, n=len(ce),
                            seed=a.seed, fp=int(b["fp"]), answer_ce=float(ce.mean()),
                            gen_match=gen_match(cell, name, ctx), secs=time.time() - t0)
                with dst.open("w") as fh:
                    fh.write(json.dumps({"_meta": meta}) + "\n")
                    for i, v in enumerate(ce):
                        fh.write(json.dumps(dict(i=i, answer_ce=round(float(v), 4),
                                                 refs=b["refs"][i])) + "\n")
                print(f"{dst.name}: answer_ce {ce.mean():.2f} gen_match {meta['gen_match']} "
                      f"({meta['secs']:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
