"""The shared hash store under contention: many partitions on many threads, small code
space so the same code lives in many groups, small capacity so runs are long and claims
collide. Every result is checked against a per-partition python dict (latest write wins),
across chunked calls and decode-sized calls. CPU only: python tests/test_store.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from lema.cache import HashStore

D_V = 8                                  # value row: 8 x uint16 = 16 bytes


def check(seed: int, P: int, T: int, chunk: int, n_codes: int, capacity: int):
    rng = np.random.default_rng(seed)
    store = HashStore(D_V, groups=P, dtype=np.uint16, capacity=capacity)
    ref = [dict() for _ in range(P)]
    # codes from a small space: heavy cross-partition sharing of codes (distinct keys since
    # the group differs), including the codes 0 and 1, which are ordinary keys like any other
    codes = rng.integers(0, n_codes, size=(2, P, T), dtype=np.int64) * 7919 % (n_codes + 2)
    vals = rng.integers(0, 1 << 16, size=(P, T, D_V), dtype=np.uint16)
    for lo in range(0, T, chunk):
        hi = min(T, lo + chunk)
        qc = torch.from_numpy(np.ascontiguousarray(codes[0, :, lo:hi]).reshape(-1))
        kc = torch.from_numpy(np.ascontiguousarray(codes[1, :, lo:hi]).reshape(-1))
        v = torch.from_numpy(np.ascontiguousarray(vals[:, lo:hi]).reshape(-1, D_V))
        out = torch.empty_like(v)
        store.exchange(qc, kc, v, out, hi - lo, 0)
        out = out.numpy().reshape(P, hi - lo, D_V)
        for p in range(P):                       # the same rule, in python
            for t in range(lo, hi):
                q, k = int(codes[0, p, t]), int(codes[1, p, t])
                want = ref[p].get(q, np.zeros(D_V, np.uint16))
                assert (out[p, t - lo] == want).all(), (seed, p, t, q)
                ref[p][k] = vals[p, t]
    counts = store.counts.numpy()
    assert (counts == [len(r) for r in ref]).all(), "entries per partition"
    return store.load()


load = check(seed=0, P=200, T=2000, chunk=500, n_codes=400, capacity=90_000)   # threaded
print(f"test_store: 200 partitions x 2000 positions, chunks of 500, threaded, load {load:.2f} OK")
load = check(seed=1, P=64, T=300, chunk=1, n_codes=150, capacity=11_000)        # decode-sized
print(f"test_store: 64 partitions, one position per call (decode), load {load:.2f} OK")
load = check(seed=2, P=400, T=1000, chunk=1000, n_codes=300, capacity=130_000)  # one shot
print(f"test_store: 400 partitions in one call, load {load:.2f} OK")
try:
    check(seed=3, P=50, T=100, chunk=100, n_codes=100, capacity=1000)           # overfull
    raise AssertionError("an overfull table must raise")
except RuntimeError as e:
    assert "full" in str(e)
    print("test_store: overfull table raises OK")
