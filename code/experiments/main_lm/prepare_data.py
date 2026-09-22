#!/usr/bin/env python3
"""Tokenize a pretraining corpus into flat uint16 shards.

    python prepare_data.py                              # all of sample/100BT (~104B tokens)
    python prepare_data.py --shards 41 --out /some/dir    # a prefix of it

Writes one `.bin` per source shard plus `meta.json`, so the job is incremental and
restartable: shards already on disk are skipped, and enlarging the corpus later is just
a bigger `--shards`. The LAST source shard is never trained on: its first half of
documents becomes val, its second half test.

Tokenizing is ~99% of the work and threads inside `encode_batch`, so one process with
many cores is already efficient; `--part i --of n` splits the TRAIN shards over n such
processes so that one part's download overlaps another's tokenization. Every part writes
disjoint files, val/test are always part 0's, and `meta.json` is written by whichever
part finishes last (it is a pure function of the files on disk).

Documents are joined by the tokenizer's end-of-text token. uint16 is enough for the GPT-2
vocabulary (50257 < 65536).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # code/
from experiments.paths import DATA as DEFAULT_OUT


def tokenizer(name="gpt2"):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from tokenizers import Tokenizer
    return Tokenizer.from_pretrained(name)


def tokenize_shard(path, tok, out_path, eot, rows=2000, max_docs=None):
    """Stream one parquet shard through the tokenizer into `out_path`. Atomic."""
    tmp = Path(str(out_path) + ".tmp")
    written = docs = 0
    with open(tmp, "wb") as f:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=rows, columns=["text"]):
            texts = batch.column("text").to_pylist()
            if max_docs is not None and docs + len(texts) > max_docs:
                texts = texts[: max_docs - docs]
            docs += len(texts)
            chunks = []
            for e in tok.encode_batch(texts, add_special_tokens=False):
                chunks.append(np.asarray(e.ids, dtype=np.uint16))
                chunks.append(np.array([eot], dtype=np.uint16))
            if chunks:
                arr = np.concatenate(chunks)
                f.write(arr.tobytes())
                written += arr.size
            if max_docs is not None and docs >= max_docs:
                break
    tmp.replace(out_path)
    return written


def drop_parquet(path):
    """Delete a downloaded parquet: the snapshot entry and the blob it points at.
    The corpus is ~3x smaller than its parquet source, so keeping the whole download
    around triples the disk this job needs for nothing."""
    p = Path(path)
    for f in (p, p.resolve()):
        try:
            f.unlink()
        except FileNotFoundError:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--subset", default="sample/100BT")
    p.add_argument("--shards", type=int, default=None,
                   help="train shards (~0.73B tokens each); default: all but the held-out one")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--eot", type=int, default=50256)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--val-docs", type=int, default=None,
                   help="documents from the held-out shard (default: all, split half val, half test)")
    p.add_argument("--part", type=int, default=0, help="this process's index in [0, --of)")
    p.add_argument("--of", type=int, default=1, help="number of cooperating processes")
    p.add_argument("--rm-parquet", action="store_true",
                   help="delete each downloaded parquet once it is tokenized")
    p.add_argument("--smoke", action="store_true", help="1000 docs per shard")
    a = p.parse_args()
    if not 0 <= a.part < a.of:
        raise SystemExit(f"--part must be in [0, {a.of}), got {a.part}")

    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(f for f in HfApi().list_repo_files(a.repo, repo_type="dataset")
                   if f.startswith(a.subset) and f.endswith(".parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {a.repo}:{a.subset}")
    if a.shards is None:
        a.shards = len(files) - 1            # everything except the held-out shard
    if a.shards + 1 > len(files):
        raise SystemExit(f"{a.subset} has {len(files)} shards; asked for {a.shards} + 1 held out")
    a.out.mkdir(parents=True, exist_ok=True)
    held, train_files = files[-1], files[: a.shards]
    cap = 1000 if a.smoke else None

    plan = [("train", f, i, cap) for i, f in enumerate(train_files)]
    half = 250 if a.smoke else (a.val_docs // 2 if a.val_docs else None)
    # val and test come from ONE held-out shard, so they stay in the same part
    held_plan = [("val", held, 0, half), ("test", held, 1, half)]   # None -> half the shard
    expected = [a.out / f"{s}_{i:03d}.bin" for s, _, i, _ in plan + held_plan]
    mine = plan[a.part::a.of] + (held_plan if a.part == 0 else [])
    print(f"part {a.part}/{a.of}: {len(mine)} of {len(expected)} shards", flush=True)

    tok = tokenizer(a.tokenizer)
    for split, src, i, docs in mine:
        dest = a.out / f"{split}_{i:03d}.bin"
        if dest.exists():
            print(f"[{split}] {dest.name} exists, skipping "
                  f"({dest.stat().st_size / 2e6:.1f}M tokens)", flush=True)
            continue
        t0 = time.perf_counter()
        local = hf_hub_download(a.repo, src, repo_type="dataset")
        t1 = time.perf_counter()
        # val and test come from the SAME held-out shard: take disjoint document ranges
        n = (tokenize_shard(local, tok, dest, a.eot, max_docs=docs) if split == "train"
             else _held(local, tok, dest, a, i, docs))
        t2 = time.perf_counter()
        if a.rm_parquet and split != "val":         # "test" reuses the val download
            drop_parquet(local)
        print(f"[{split}] {dest.name}: {n / 1e6:.1f}M tokens "
              f"(download {t1 - t0:.0f}s, tokenize {t2 - t1:.0f}s)", flush=True)

    missing = [d.name for d in expected if not d.exists()]
    if missing:
        print(f"\npart {a.part}/{a.of} done; {len(missing)} shard(s) still missing, so "
              f"meta.json is left to the last part ({', '.join(missing[:3])} ...)")
        return
    counts = {d.name: d.stat().st_size // 2 for d in expected}
    meta = {"repo": a.repo, "subset": a.subset, "tokenizer": a.tokenizer,
            "vocab_size": a.vocab_size, "eot": a.eot, "dtype": "uint16",
            "train_shards": train_files, "held_out_shard": held, "smoke": a.smoke,
            "tokens": counts,
            "train_tokens": sum(v for k, v in counts.items() if k.startswith("train"))}
    tmp = a.out / f"meta.json.{a.part}.tmp"          # parts may finish together; the
    tmp.write_text(json.dumps(meta, indent=2) + "\n")   # content is identical either way
    tmp.replace(a.out / "meta.json")
    print(f"\ntrain {meta['train_tokens'] / 1e9:.3f}B tokens -> {a.out}")


def _held(local, tok, dest, a, part, docs):
    """val = the first half of the held-out shard's documents, test = the second half
    (or the first `docs` / next `docs` when --val-docs is given)."""
    tmp = Path(str(dest) + ".tmp")
    written = seen = 0
    if docs is None:
        docs = pq.ParquetFile(local).metadata.num_rows // 2
    lo, hi = part * docs, (part + 1) * docs
    with open(tmp, "wb") as f:
        for batch in pq.ParquetFile(local).iter_batches(batch_size=2000, columns=["text"]):
            texts = batch.column("text").to_pylist()
            sel = [t for j, t in enumerate(texts) if lo <= seen + j < hi]
            seen += len(texts)
            if sel:
                chunks = []
                for e in tok.encode_batch(sel, add_special_tokens=False):
                    chunks.append(np.asarray(e.ids, dtype=np.uint16))
                    chunks.append(np.array([a.eot], dtype=np.uint16))
                arr = np.concatenate(chunks)
                f.write(arr.tobytes())
                written += arr.size
            if seen >= hi:
                break
    tmp.replace(dest)
    return written


if __name__ == "__main__":
    main()
