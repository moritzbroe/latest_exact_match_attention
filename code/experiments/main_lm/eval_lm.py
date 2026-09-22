"""Validation CE of LM runs on the canonical window set.

For each out/<run> with a finished model.pt: writes the overall CE into <run>/eval.json,
keyed by context length. lema evaluates in hard mode, the exact latest-match operator --
that IS the trained model, and the stickbreaking surrogate is a training device with no
meaning once the schedule has ended, so there is no soft option here. rope models are
skipped at contexts beyond their 2048 training window.

The default is "evaluate whatever is new": every run without a result at this context is
evaluated, ablation variants included. Name runs to restrict the set, --force to recompute.
Per-token losses over the full held-out split are a different protocol and live in
../main_lm_bigrams/eval_pertoken.py.

Usage: eval_lm.py [run ...] [--ctx 2048] [--samples N] [--force] [--out DIR]
--out: the run directory to sweep (default: this experiment's out/); the appendix
experiments point it at theirs so every LM number in the paper comes from this one
protocol.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # code/
import numpy as np
import torch
import torch.nn.functional as F

from lema import TokenCorpus, load_trained
from experiments.paths import DATA

HERE = Path(__file__).resolve().parent


def val_windows(ctx: int, samples: int):
    """[samples, ctx+1] int64 window array (x = [:, :-1], y = [:, 1:]) plus the corpus.

    The windows are the FIRST `samples` consecutive, non-overlapping blocks of the
    validation stream -- not random draws. Two consequences: the evaluated tokens are
    distinct (`samples * ctx` really is the token count), and the set depends only on
    the leading tokens of the validation split, so any corpus built from the same
    held-out shard yields identical windows however many documents it kept. With
    samples = 2**26 / ctx, every context length covers the same 67.1M tokens and the
    resulting losses are comparable across contexts as well as across models. Raises if
    the split is too short rather than sampling anything twice.
    """
    task = TokenCorpus(DATA, seq_len=ctx)
    need = samples * ctx + 1
    have = sum(len(a) for a in task.arrs["val"])
    if have < need:
        raise SystemExit(f"validation split has {have/1e6:.1f}M tokens, "
                         f"{samples} blocks of {ctx} need {need/1e6:.1f}M")
    flat = np.concatenate([np.asarray(a, dtype=np.int64) for a in task.arrs["val"]]) \
        if len(task.arrs["val"]) > 1 else np.asarray(task.arrs["val"][0][:need],
                                                     dtype=np.int64)
    idx = np.arange(samples)[:, None] * ctx + np.arange(ctx + 1)[None, :]
    return flat[idx], task


def windows_fingerprint(w) -> int:
    """Stamped into every result so a number can prove which window set produced it."""
    return int(np.int64(w[:, 0]).sum() + np.int64(w[:, -1]).sum())


def find_runs(names, out=HERE / "out"):
    """The run directories to evaluate.

    `names` may be paths or bare names under `out`; empty means every finished run --
    model.pt is written only at the end of training, so an in-flight run is skipped.
    A named run that is not finished is an error, an unfinished one found by the sweep
    is simply not returned.
    """
    out = Path(out)
    if names:
        runs = [p if (p := Path(n)).exists() else out / n for n in names]
        if missing := [str(r) for r in runs if not (r / "model.pt").exists()]:
            raise SystemExit(f"no model.pt in: {', '.join(missing)}")
        return runs
    runs = sorted(r for r in out.iterdir() if r.is_dir() and (r / "model.pt").exists()) \
        if out.is_dir() else []
    if not runs:
        raise SystemExit(f"no finished run under {out}")
    return runs


@torch.no_grad()
def run_pass(model, w, ctx, device="cuda"):
    B = max(1, 65536 // ctx)
    ces = []
    for lo in range(0, len(w), B):
        x = torch.from_numpy(w[lo:lo + B, :-1]).to(device)
        y = torch.from_numpy(w[lo:lo + B, 1:]).to(device)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(x)
        flat, yf = logits.flatten(0, 1), y.flatten()
        ce = torch.empty(len(yf), dtype=torch.float32, device=device)
        for i in range(0, len(yf), 8192):
            ce[i:i + 8192] = F.cross_entropy(flat[i:i + 8192].float(), yf[i:i + 8192],
                                             reduction="none")
        ces.append(ce.view_as(y).cpu().numpy())
    return np.concatenate(ces)


def model_cfg(run):
    """The run's model config, from the json the trainer writes before step 0 -- so the
    cheap skip decisions below never have to open a checkpoint."""
    return json.loads((run / "config.json").read_text())["model"]


def up_to_date(run, ctx) -> bool:
    """A result for this context is already on disk."""
    ev = run / "eval.json"
    return ev.exists() and str(ctx) in json.loads(ev.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*", help="run dirs or names under out/ (default: all)")
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--samples", type=int, default=None)   # None -> 2**26 / ctx (~67M tokens)
    p.add_argument("--force", action="store_true", help="re-evaluate runs already done")
    p.add_argument("--out", type=Path, default=HERE / "out", help="run directory to sweep")
    a = p.parse_args()
    if a.samples is None:
        a.samples = max(1024, 2 ** 26 // a.ctx)
    runs = find_runs(a.runs, a.out)
    w, _ = val_windows(a.ctx, a.samples)
    for run in runs:
        mcfg = model_cfg(run)
        if mcfg["pos"] == "rope" and a.ctx > 2048:
            print(f"{run.name}: skipped at T={a.ctx} (beyond training context)")
            continue
        if up_to_date(run, a.ctx) and not a.force:
            print(f"{run.name}: up to date at T={a.ctx}")
            continue
        model, _ = load_trained(run)
        ce = run_pass(model, w, a.ctx)
        rec = {"ce": float(ce.mean()), "samples": a.samples, "tokens": int(ce.size),
               "fingerprint": windows_fingerprint(w)}
        ev = run / "eval.json"
        d = json.loads(ev.read_text()) if ev.exists() else {}
        d[str(a.ctx)] = rec
        ev.write_text(json.dumps(d, indent=1) + "\n")
        print(f"{run.name} T={a.ctx}: ce {rec['ce']:.4f}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
