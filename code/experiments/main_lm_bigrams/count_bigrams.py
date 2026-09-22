"""How often each bigram of the held-out split occurs in the training data.

Counts every bigram over the FIRST 10B training tokens on the GPU, then keeps the counts of
only those bigrams that actually occur in the held-out split, which is a few tens of millions
rather than the (V+1)^2 of the full table. Writes out/bigram_counts.npz: `ids` sorted, and
`count` beside it. A bigram of the split that is absent from the table has count 0 and is
kept: those are the rarest ones.

Once per corpus. Usage: count_bigrams.py [--ref-tokens 10000000000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from lema import TokenCorpus                                          # noqa: E402
from experiments.paths import DATA                                    # noqa: E402


def path() -> Path:
    return HERE / "out" / "bigram_counts.npz"


def load():
    f = path()
    if not f.exists():
        raise SystemExit(f"no {f.name}; run count_bigrams.py first")
    d = np.load(f)
    return d["ids"], d["count"]


def counts_of(bigrams: np.ndarray) -> np.ndarray:
    """Training count of every bigram id given, 0 where the bigram never occurred."""
    ids, cnt = load()
    j = np.searchsorted(ids, bigrams.ravel())
    j[j == len(ids)] = 0
    out = np.where(ids[j] == bigrams.ravel(), cnt[j], 0)
    return out.reshape(bigrams.shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-tokens", type=int, default=10_000_000_000)
    p.add_argument("--split", default="val")
    a = p.parse_args()
    task = TokenCorpus(DATA, seq_len=2048)
    if task.tokens("train") < a.ref_tokens:
        raise SystemExit(f"need {a.ref_tokens/1e9:.0f}B training tokens, "
                         f"have {task.tokens('train')/1e9:.2f}B")
    V = task.vocab_size
    flat = np.concatenate([np.asarray(arr, dtype=np.int64) for arr in task.arrs[a.split]])
    want = np.unique(flat[:-1] * (V + 1) + flat[1:])
    del flat
    print(f"{a.split}: {len(want):,} distinct bigrams", flush=True)

    counts = torch.zeros((V + 1) * (V + 1), dtype=torch.int32, device="cuda")
    left = a.ref_tokens
    for arr in task.arrs["train"]:
        n = min(len(arr), left + 1)
        for lo in range(0, n - 1, 50_000_000):
            t = torch.from_numpy(np.asarray(arr[lo:min(lo + 50_000_001, n)],
                                            dtype=np.int64)).cuda()
            ids = t[:-1] * (V + 1) + t[1:]
            counts.index_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
        left -= n - 1
        if left <= 0:
            break
    got = counts[torch.from_numpy(want).cuda()].cpu().numpy()
    (HERE / "out").mkdir(exist_ok=True)
    np.savez(path(), ids=want, count=got, ref_tokens=a.ref_tokens, split=a.split)
    print(f"first {a.ref_tokens/1e9:.0f}B train tokens: of the split's bigrams, "
          f"{int((got == 0).sum()):,} never occur, "
          + ", ".join(f"{int(((got > lo) & (got <= hi)).sum()):,} occur {lo+1}-{hi}"
                      for lo, hi in ((0, 10), (10, 100), (100, 1000))))


if __name__ == "__main__":
    main()
