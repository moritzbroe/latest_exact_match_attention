"""Does the pictured recall circuit hold at large n? Measured on a trained checkpoint:

    mechanism.py <model_best.pt or run dir> --n 4096 [--min-queries 8192] [--seed 0]

At every VALUE position, does the head read the position directly before it (its key),
and at every QUERY position, does it read the correct value? For a LEMA model these are
fractions of positions whose exact-op source is that position; for a softmax model the
mean attention weight on it. Both are reported for both layers (layer 0 should carry the
former, layer 1 the latter). For LEMA the layer-0 reads at query positions are also
classified (separator / nothing / key token / value token / earlier query), and the codes
themselves are counted: the number of distinct layer-0 insert codes over the key tokens
and lookup codes over the value tokens (the circuit needs one each, equal), the number
of distinct layer-1 insert codes over the value positions (one per key for a perfect
recall), and whether a key's layer-1 insert code is the same in every sample it appears
in, i.e. a function of the key alone. Writes out/mechanism/<arch>_<seed>_n<n>.json.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from lema.analysis import trace                                        # noqa: E402
from experiments.main_recall.eval_recall import OUT as RECALL_OUT, load   # noqa: E402
from experiments.main_recall.train_recall import task_at               # noqa: E402

OUT_DIR = HERE / "out" / "mechanism"


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="model_best.pt or a run dir (bare names resolve under "
                                "../main_recall/out, e.g. lema/s0 or rope/s0/n4096)")
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--min-queries", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = p.parse_args()
    ckpt = Path(a.ckpt) if Path(a.ckpt).exists() else RECALL_OUT / a.ckpt
    model = load(ckpt, device=a.device)
    if model.cfg.attn not in ("lema", "softmax"):
        raise SystemExit(f"{model.cfg.attn} has no attention state to trace")
    lema = model.cfg.attn == "lema"
    n = a.n
    B = -(-a.min_queries // n)
    x, y, _, _ = next(task_at(n).batches("val", B, seed=a.seed))
    T = x.shape[1]
    v = torch.arange(1, 2 * n, 2, device=a.device)            # value positions
    q = torch.arange(2 * n + 1, 3 * n + 1, device=a.device)   # query positions
    acc = {"prev": [0.0] * model.cfg.depth, "value": [0.0] * model.cfg.depth,
           "q0_sep": 0.0, "q0_none": 0.0, "q0_key": 0.0, "q0_value": 0.0, "q0_query": 0.0}
    codes = {"l0_key_insert": set(), "l0_value_lookup": set(), "l1_value_insert": set()}
    key_code = {}                                             # key token -> {layer-1 insert codes}
    correct = 0
    for b in range(B):                                        # one sample at a time: a
        xb = x[b:b + 1].to(a.device)                          # softmax layer is T x T
        keys = xb[0, 0:2 * n:2]
        key_index = torch.full((model.cfg.vocab_size,), -1, device=a.device)
        key_index[keys] = torch.arange(n, device=a.device)
        corr = 2 * key_index[xb[0, q]] + 1                    # the queried key's value
        logits, recs = trace(model, xb)
        correct += int((logits[0, q].argmax(-1).cpu() == y[b, q.cpu()]).sum())
        for li, r in enumerate(recs):
            if lema:
                src = r["src"][0, 0]
                acc["prev"][li] += float((src[v] == v - 1).float().mean())
                acc["value"][li] += float((src[q] == corr).float().mean())
                qc, kc = r["qc"][0, 0], r["kc"][0, 0]
                if li == 0:
                    sq = src[q]
                    acc["q0_sep"] += float((sq == 2 * n).float().mean())
                    acc["q0_none"] += float((sq == -1).float().mean())
                    acc["q0_key"] += float(((sq >= 0) & (sq < 2 * n) & (sq % 2 == 0)).float().mean())
                    acc["q0_value"] += float(((sq >= 0) & (sq < 2 * n) & (sq % 2 == 1)).float().mean())
                    acc["q0_query"] += float((sq > 2 * n).float().mean())
                    codes["l0_key_insert"].update(kc[0:2 * n:2].tolist())
                    codes["l0_value_lookup"].update(qc[v].tolist())
                if li == 1:
                    vc = kc[v].tolist()
                    codes["l1_value_insert"].update(vc)
                    for k, c in zip(keys.tolist(), vc):
                        key_code.setdefault(k, set()).add(c)
            else:
                w = r["w"][0, 0]
                acc["prev"][li] += float(w[v, v - 1].mean())
                acc["value"][li] += float(w[q, corr].mean())
    for k in ("prev", "value"):
        acc[k] = [round(s / B, 4) for s in acc[k]]
    for k in ("q0_sep", "q0_none", "q0_key", "q0_value", "q0_query"):
        acc[k] = round(acc[k] / B, 4) if lema else None
    if lema:
        acc["l0_distinct_key_insert_codes"] = len(codes["l0_key_insert"])
        acc["l0_distinct_value_lookup_codes"] = len(codes["l0_value_lookup"])
        acc["l0_codes_equal"] = codes["l0_key_insert"] == codes["l0_value_lookup"]
        acc["l1_distinct_value_insert_codes"] = len(codes["l1_value_insert"])
        acc["l1_keys_seen"] = len(key_code)
        acc["l1_insert_code_constant_per_key"] = round(
            sum(len(c) == 1 for c in key_code.values()) / len(key_code), 4)
    arch = "lema" if lema else "rope"
    seed = next((s for s in ckpt.resolve().parts if re.fullmatch(r"s\d+", s)), "s")
    rec = {"ckpt": str(ckpt), "arch": arch, "seed": seed, "n": n, "tokens": T,
           "samples": B, "queries": B * n, "accuracy": round(correct / (B * n), 4),
           "sample_seed": a.seed, **acc}
    print(json.dumps(rec, indent=1))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"{arch}_{seed}_n{n}.json"
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
