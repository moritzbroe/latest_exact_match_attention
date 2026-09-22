"""Evaluation for the recall experiment: accuracy-vs-n curves from stored checkpoints.

With no arguments, evaluates every (arch, seed) under out/ that does not yet have its
out/eval/<arch>_s<seed>.json: lema's single run from its final checkpoint through the
exact hard path at every n, the stick-breaking run the same way, the curriculum baselines
stage-by-stage, each stage at its own n from the checkpoint the curriculum advanced from.
Name archs to restrict, --force to recompute. Figures are a separate no-GPU step:
plot_recall.py.

Usage: eval_recall.py [lema|sb|rope|gdn ...] [--force]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from lema import ModelConfig, Transformer
from experiments.main_recall.train_recall import LADDER, OUT, task_at

EVAL_NS = [2 ** k for k in range(2, 13)]                   # 4 .. 4096
EVAL_DIR = OUT / "eval"


def load(ckpt: Path, arch: str | None = None, device: str = "cuda") -> Transformer:
    """`ckpt` is a model file or a run directory (its model_best.pt); `arch` defaults to
    the checkpoint's own attention kind."""
    ckpt = Path(ckpt)
    if ckpt.is_dir():
        ckpt = ckpt / "model_best.pt"
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ck["config"])
    # the weights are authoritative for the vocabulary size
    cfg.vocab_size = ck["state_dict"]["tok_emb.weight"].shape[0]
    cfg.dim_ff = ck["state_dict"]["blocks.0.mlp.gate.weight"].shape[0]
    model = Transformer(cfg).to(device).eval()
    keep = model.state_dict().keys()
    model.load_state_dict({k: v for k, v in ck["state_dict"].items() if k in keep})
    if (arch or cfg.attn) == "lema":
        model.set_hard(True)
    return model


def accuracy_at(model: Transformer, pairs: int, iters: int = 4) -> float:
    gen = task_at(pairs).batches("val", max(1, 8192 // pairs), seed=7)
    hit = total = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(iters):
            x, y, m, _ = next(gen)
            pred = model(x.cuda())[m.cuda()].argmax(-1).cpu()
            hit += int((pred == y[m]).sum())
            total += int(m.sum())
    return hit / total


def eval_single(arch: str, seed: int) -> dict:
    """A run trained once at n=8 (lema, sb): its final checkpoint at every n."""
    model = load(OUT / arch / f"s{seed}" / "model.pt", arch)
    curve = {n: accuracy_at(model, n) for n in EVAL_NS}
    print("  " + " ".join(f"{n}:{v:.4f}" for n, v in curve.items()))
    return {"curve": curve}


def eval_ladder(arch: str, seed: int) -> dict:
    points = {}
    for pairs in LADDER:
        ck = OUT / arch / f"s{seed}" / f"n{pairs}" / "model_best.pt"
        if not ck.exists():
            print(f"  missing {ck}, stopping ladder eval")
            break
        points[pairs] = accuracy_at(load(ck, arch), pairs)
        print(f"  n={pairs}: {points[pairs]:.5f}")
    return {"points": points}


def main():
    global OUT, EVAL_DIR
    p = argparse.ArgumentParser()
    p.add_argument("archs", nargs="*", help="lema/sb/rope/gdn, default: all")
    p.add_argument("--force", action="store_true", help="re-evaluate existing results")
    p.add_argument("--out-root", type=Path, default=OUT)
    a = p.parse_args()
    OUT, EVAL_DIR = a.out_root, a.out_root / "eval"
    if bad := set(a.archs) - {"lema", "sb", "rope", "gdn"}:
        raise SystemExit(f"unknown arch: {', '.join(bad)}")
    archs = a.archs or ["lema", "sb", "rope", "gdn"]
    checkpoints = [f for arch in archs for f in (OUT / arch).glob("s*/**/model*.pt")]
    if not checkpoints:
        p.error(f"no checkpoints found under {OUT} for {', '.join(archs)}")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for arch in a.archs or ["lema", "sb", "rope", "gdn"]:
        for d in sorted((OUT / arch).glob("s*")) if (OUT / arch).is_dir() else []:
            if not re.fullmatch(r"s\d+", d.name):
                continue
            seed = int(d.name[1:])
            out = EVAL_DIR / f"{arch}_s{seed}.json"
            if out.exists() and not a.force:
                print(f"{arch} s{seed}: up to date")
                continue
            print(f"{arch} s{seed}:")
            result = (eval_single(arch, seed) if arch in ("lema", "sb")
                      else eval_ladder(arch, seed))
            out.write_text(json.dumps(result, indent=1))
            print(f"  -> {out}")


if __name__ == "__main__":
    main()
