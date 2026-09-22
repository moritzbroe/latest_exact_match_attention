"""CPU invariants of the Recall task. Run: python tests/test_task.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from lema import Recall

t = Recall(16, key_vocab=1024)
assert t.seq_len == 49 and t.vocab_size == 2049 and t.sep == 2048
x, y, m, info = next(t.batches("train", 8, seed=0))
assert x.shape == (8, 49) and info["pairs"] == 16
keys, vals = x[:, 0:32:2], x[:, 1:32:2]
assert (keys < 1024).all() and (vals >= 1024).all() and (vals < 2048).all()
assert (x[:, 32] == t.sep).all()                       # the single separator
q = x[:, 33:]
assert (np.sort(q.numpy()) == np.sort(keys.numpy())).all()   # every key queried once
assert m[:, 33:].all() and m.sum() == 8 * 16           # exactly the query positions scored
for r in range(8):
    assert len(set(keys[r].tolist())) == 16            # keys distinct within a sequence
    lut = dict(zip(keys[r].tolist(), vals[r].tolist()))
    assert [lut[k] for k in q[r].tolist()] == y[r][m[r]].tolist()

t = Recall(4096, key_vocab=4096)                       # the full-key-space sequence
x, y, m, _ = next(t.batches("val", 1, seed=1))
assert x.shape == (1, 3 * 4096 + 1)
assert len(set(x[0, 0:8192:2].tolist())) == 4096       # every key exactly once
print("test_task: OK")


# --- stream identity: skip and prefetch must not change WHICH batches come out -------
# The resume path relies on this: a run continued from a checkpoint has to see exactly
# the batches the uninterrupted run would have seen. So `skip` must consume the rng in
# the same order the producing path does, and `prefetch` must not reorder anything.
# Both references below are the generators as they read before either existed.
import torch

from lema import TokenCorpus


def _ref_recall(task, batch_size, seed):
    rng = np.random.default_rng([seed, 0])
    p, B = task.p, batch_size
    while True:
        keys = task._distinct(rng, B, p, task.K)
        vals = rng.integers(task.K, 2 * task.K, size=(B, p))
        order = np.argsort(rng.random((B, p)), axis=1)
        x = np.full((B, task.seq_len), task.sep, dtype=np.int64)
        x[:, 0:2 * p:2], x[:, 1:2 * p:2] = keys, vals
        x[:, 2 * p + 1:] = np.take_along_axis(keys, order, axis=1)
        yield torch.from_numpy(x),


def _ref_windows(arrays, batch_size, seq_len, rng):
    lens = np.array([len(a) for a in arrays], dtype=np.float64)
    probs = lens / lens.sum()
    while True:
        which = rng.choice(len(arrays), size=batch_size, p=probs)
        rows = []
        for w in which:
            a = arrays[w]
            s = int(rng.integers(0, len(a) - seq_len - 1))
            rows.append(np.asarray(a[s:s + seq_len + 1], dtype=np.int64))
        t = torch.from_numpy(np.stack(rows))
        yield t[:, :-1],


def _check_stream(make, reference, n=6, k=3):
    """`make(skip, prefetch)` must agree with `reference` batch for batch."""
    want = [next(reference)[0].numpy() for _ in range(n)]
    for skip in (0, k):
        for prefetch in (0, 4):
            gen = make(skip, prefetch)
            for i in range(skip, n):
                got = next(gen)[0].numpy()
                assert np.array_equal(got, want[i]), \
                    f"batch {i} differs at skip={skip} prefetch={prefetch}"


rec = Recall(8, key_vocab=256)
_check_stream(lambda s, p: rec.batches("train", 4, seed=1, skip=s, prefetch=p),
              _ref_recall(rec, 4, seed=1))
print("test_task: recall stream identity under skip/prefetch OK")

_corpus = Path(__file__).resolve().parents[1] / "data" / "fineweb_edu100_gpt2"
if (_corpus / "meta.json").exists():
    corp = TokenCorpus(str(_corpus), seq_len=64)
    _check_stream(lambda s, p: corp.batches("val", 4, seed=1, skip=s, prefetch=p),
                  _ref_windows(corp.arrs["val"], 4, 64, np.random.default_rng([1, 1])))
    print("test_task: token-corpus stream identity under skip/prefetch OK")
else:
    print("test_task: no corpus on disk, skipped the token-corpus stream check")
