"""Tasks. Every task yields `(x, y, mask)` batches of int64 CPU tensors:

    x     [B, T]  inputs
    y     [B, T]  target at each position (position i predicts y[i])
    mask  [B, T] bool, or None meaning "every position is scored"

Language modelling uses the shifted-input convention with mask=None. Associative recall
scores ONLY the query positions, so it carries an explicit mask and its targets are not a
shift of the input.
"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import numpy as np
import torch

_SPLIT = {"train": 0, "val": 1, "test": 2}


# ------------------------------------------------------------------ recall
class Recall:
    """Multi-query associative recall at a fixed pair count:

        k1 v1 k2 v2 ... kp vp  SEP  q1 q2 ... qp        (3p + 1 tokens)

    Keys are drawn distinct from [0, key_vocab), values from [key_vocab, 2 * key_vocab),
    SEP is the single dedicated separator (id 2 * key_vocab). The queries are a random
    permutation of the keys; each query position is scored against its key's value and
    only those positions are scored.
    """

    def __init__(self, pairs: int, key_vocab: int = 4096):
        if not 1 <= pairs <= key_vocab:
            raise ValueError(f"need 1 <= pairs <= key_vocab, got {pairs}")
        self.p, self.K = pairs, key_vocab
        self.sep = 2 * key_vocab
        self.vocab_size = 2 * key_vocab + 1
        self.seq_len = 3 * pairs + 1

    @staticmethod
    def _distinct(rng, rows, n, N):
        """[rows, n] uniform ordered samples of n DISTINCT values from [0, N).
        Floyd's algorithm, vectorised; independent of N; rows shuffled afterwards so the
        order is unbiased (Floyd's is uniform as a SET only)."""
        if n > N:
            raise ValueError(f"cannot draw {n} distinct values from {N}")
        j = np.arange(N - n, N)
        t = rng.integers(0, j + 1, size=(rows, n))
        out = np.empty((rows, n), dtype=np.int64)
        out[:, 0] = t[:, 0]
        for i in range(1, n):
            out[:, i] = np.where((out[:, :i] == t[:, i, None]).any(axis=1), j[i], t[:, i])
        return np.take_along_axis(out, np.argsort(rng.random((rows, n)), axis=1), axis=1)

    def batches(self, split, batch_size, seed=0, skip=0, prefetch=0):
        rng = np.random.default_rng([seed, _SPLIT[split]])
        p, B = self.p, batch_size

        def draw():
            """One sample's worth of rng, in the order the assembling path consumes it."""
            return (self._distinct(rng, B, p, self.K),
                    rng.integers(self.K, 2 * self.K, size=(B, p)),
                    np.argsort(rng.random((B, p)), axis=1))

        def gen():
            for _ in range(skip):
                draw()
            while True:
                keys, vals, order = draw()
                x = np.full((B, self.seq_len), self.sep, dtype=np.int64)
                y = np.zeros((B, self.seq_len), dtype=np.int64)
                mask = np.zeros((B, self.seq_len), dtype=bool)
                x[:, 0:2 * p:2], x[:, 1:2 * p:2] = keys, vals
                x[:, 2 * p + 1:] = np.take_along_axis(keys, order, axis=1)
                y[:, 2 * p + 1:] = np.take_along_axis(vals, order, axis=1)
                mask[:, 2 * p + 1:] = True
                info = {"pairs": p, "value_pos": np.arange(p) * 2 + 1}
                yield (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask),
                       info)
        return _prefetch(gen(), prefetch) if prefetch else gen()

    def describe(self) -> str:
        return f"recall: {self.p} pairs, key_vocab {self.K}, vocab {self.vocab_size}"


# --------------------------------------------------------------------- LM
def _prefetch(gen, depth):
    """Yield from `gen`, filling it one thread ahead. Order is untouched -- this only
    moves WHEN each batch is produced.

    A window is a random few-kB read scattered over a corpus far larger than page cache,
    so on a slow filesystem a batch of them costs ~0.3s of pure latency: comparable to
    a whole training step, and paid synchronously in the middle of one. Overlapping it
    with compute saves hours on a 185k-step run.
    """
    q = queue.Queue(maxsize=depth)
    threading.Thread(target=lambda: [q.put(b) for b in gen], daemon=True).start()
    while True:
        yield q.get()


def _windows(arrays, batch_size, seq_len, rng, skip=0):
    """Random (seq_len+1)-windows, sampling a shard in proportion to its length.

    `skip` advances the stream by that many batches without reading them. That is exactly
    what a resume needs -- the skipped windows are discarded, only the rng position
    carries over -- and it is the difference between a resume costing seconds and costing
    hours, since the draws are pure arithmetic and only the reads are slow.
    """
    lens = np.array([len(a) for a in arrays], dtype=np.float64)
    probs = lens / lens.sum()

    def draw():
        """One batch's worth of rng, in the exact order the reading path consumes it."""
        which = rng.choice(len(arrays), size=batch_size, p=probs)
        return [(w, int(rng.integers(0, len(arrays[w]) - seq_len - 1))) for w in which]

    for _ in range(skip):
        draw()
    while True:
        rows = [np.asarray(arrays[w][s:s + seq_len + 1], dtype=np.int64)
                for w, s in draw()]
        t = torch.from_numpy(np.stack(rows))
        yield t[:, :-1], t[:, 1:], None, None


class TokenCorpus:
    """Pre-tokenized corpus: one or more flat uint16 shards per split, plus meta.json."""

    def __init__(self, root, seq_len=512, dtype=np.uint16):
        root = Path(root)
        self.seq_len = seq_len
        meta = json.loads((root / "meta.json").read_text())
        self.vocab_size = meta["vocab_size"]
        self.meta = meta
        # a corpus may hold only some splits (e.g. the held-out shard alone, for
        # evaluation): asking for a missing split fails in `batches`
        self.arrs = {}
        for split in ("train", "val", "test"):
            shards = sorted(root.glob(f"{split}_*.bin")) or sorted(root.glob(f"{split}.bin"))
            if shards:
                self.arrs[split] = [np.memmap(s, dtype=dtype, mode="r") for s in shards]
        if not self.arrs:
            raise FileNotFoundError(f"no token shards under {root}")

    def batches(self, split, batch_size, seed=0, skip=0, prefetch=0):
        if split not in self.arrs:
            raise FileNotFoundError(f"no {split} shards in this corpus")
        gen = _windows(self.arrs[split], batch_size, self.seq_len,
                       np.random.default_rng([seed, _SPLIT[split]]), skip=skip)
        return _prefetch(gen, prefetch) if prefetch else gen

    def tokens(self, split="train") -> int:
        return int(sum(len(a) for a in self.arrs[split]))

    def describe(self) -> str:
        return (f"tokens: vocab {self.vocab_size}, "
                + ", ".join(f"{s} {self.tokens(s) / 1e9:.3f}B" for s in self.arrs))
