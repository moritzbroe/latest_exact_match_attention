"""Per-token CE of a model on the FULL validation split, in consecutive --ctx windows.

Writes out/pertoken/<name>_T{ctx}.npz: the CE of every predicted token (float16) plus the
window fingerprint, which is what score.py reads. Models are ../main_lm run names (the
pretrained models) or run directories (out/<run>_16k, the context-extended ones), and
<name> is the directory name. A RoPE model at base 1e4 is skipped beyond 2048 -- beyond its
training length its numbers mean nothing, and --rope-extrapolate asks for them anyway,
which is the "before" column of the context-extension table -- while the extended ones
(base 1e6) are evaluated at any --ctx. The windows are ALL complete (ctx+1)-blocks of the validation split in order,
so every model at a given --ctx scores exactly the same tokens. The split has 123M tokens,
7500 windows of 16384.

A checkpoint file (out/<run>/model_step<k>.pt) is taken as well and goes to
<run>_step<k>_T{ctx}.npz.

Usage: eval_pertoken.py <run|dir|checkpoint.pt> ... [--ctx 16384] [--rope-extrapolate] [--force]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # code/
from lema import TokenCorpus, load_trained                             # noqa: E402
from experiments.main_lm.eval_lm import run_pass, windows_fingerprint  # noqa: E402
from experiments.paths import DATA, LM_OUT                             # noqa: E402

HERE = Path(__file__).resolve().parent


def windows(ctx: int, split: str = "val"):
    """[n, ctx+1] int64: all complete consecutive blocks of the split, and the corpus."""
    task = TokenCorpus(DATA, seq_len=ctx)
    flat = np.concatenate([np.asarray(a, dtype=np.int64) for a in task.arrs[split]])
    n = (len(flat) - 1) // ctx
    idx = np.arange(n)[:, None] * ctx + np.arange(ctx + 1)[None, :]
    return flat[idx], task


def resolve(name: str) -> Path:
    p = Path(name)
    if p.is_file() and p.suffix == ".pt":                 # one checkpoint of a run
        return p
    for cand in (p, HERE / "out" / name, LM_OUT / name):
        if (cand / "model.pt").exists():
            return cand
    raise SystemExit(f"no finished run for {name!r}")


def stem_of(d: Path) -> str:
    """What the output is named after: the run for a directory, run plus step for a
    checkpoint file, so <run>_step<k> scores next to <run> without colliding."""
    return d.name if d.is_dir() else f"{d.parent.name}_{d.stem.replace('model_', '')}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--rope-extrapolate", action="store_true",
                   help="evaluate a RoPE model beyond its training context instead of skipping it")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    dirs = [resolve(r) for r in a.runs]
    (HERE / "out" / "pertoken").mkdir(parents=True, exist_ok=True)
    w, _ = windows(a.ctx)
    fp = windows_fingerprint(w)
    print(f"{len(w)} windows of {a.ctx} = {w[:, 1:].size / 1e6:.1f}M scored tokens, "
          f"fingerprint {fp}", flush=True)
    for d in dirs:
        name = stem_of(d)
        out = HERE / "out" / "pertoken" / f"{name}_T{a.ctx}.npz"
        if out.exists() and not a.force:
            print(f"{name}: done"); continue
        model, mcfg = load_trained(d)
        if (mcfg.pos == "rope" and mcfg.rope_base == 10000.0 and a.ctx > 2048
                and not a.rope_extrapolate):
            print(f"{name}: skipped at T={a.ctx} (RoPE beyond its training context)")
            continue
        ce = run_pass(model, w, a.ctx)
        np.savez(out, ce=ce.astype(np.float16), fingerprint=fp, ctx=a.ctx, split="val")
        print(f"{name} T={a.ctx}: ce {ce.mean():.4f} -> {out.name}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
