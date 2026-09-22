"""What a model still scores on a target once every earlier copy of its bigram is gone.

    python eval_intervention.py rope1024_16k --ctx 16384

intervention_targets.py fixes the sample and the corruption, so the windows evaluated here
are the same text for every model. Each of them is one window of the canonical set with the
earlier copies of one target's bigram overwritten, and the number taken from it is the
cross-entropy at that one target: with x = w[:, :-1] and y = w[:, 1:], the trigger sits at
x[second] and the scored token at y[second], so the loss is the logit row at `second`
against y[second] -- the same entry score.py reads out of a per-token row.

Writes out/intervention/<name>_T{ctx}_f{freq}[_distracted].npz: loss (the corrupted window)
and clean (the same targets in the untouched window, read from out/pertoken/<name>_T{ctx}.npz)
in the order of the intervention file, for a paired comparison. Names, model loading and the
batch size are eval_pertoken.py's, so a RoPE model at base 1e4 is skipped beyond 2048 here
too.

Every run first scores 32 of the windows UNTOUCHED and refuses to go on unless they
reproduce the per-token file, which is what validates the window and position conventions;
it costs seconds and no result can come out of a misread convention. --check does that for
every target instead of 32 of them. --limit N takes N targets spread over the whole file,
every bucket represented, and writes them to a _limit{N} file that can never pass for the
real thing.

Usage: eval_intervention.py <run|dir> ... [--ctx 16384] [--freq 100] [--distracted]
                            [--batch N] [--limit N] [--check] [--rope-extrapolate] [--force]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from lema import load_trained                                          # noqa: E402
from experiments.main_lm.eval_lm import windows_fingerprint            # noqa: E402
from experiments.main_lm_bigrams import intervention_targets, targets  # noqa: E402
from experiments.main_lm_bigrams.eval_pertoken import resolve, stem_of, windows  # noqa: E402


@torch.no_grad()
def target_ce(model, w, win, second, off=None, pos=None, tok=None, B=4, device="cuda"):
    """The cross-entropy at one position of each window, the windows corrupted if `off` is
    given and untouched if it is not.

    Only the logit row at the target is kept, so the T x vocab logits of a 16k window are
    the whole memory cost and the batch size is eval_pertoken.py's.
    """
    out = np.empty(len(win), np.float32)
    for lo in range(0, len(win), B):
        hi = min(lo + B, len(win))
        rows = w[win[lo:hi]].copy()
        if off is not None:
            for i in range(lo, hi):
                s = slice(off[i], off[i + 1])
                rows[i - lo, pos[s]] = tok[s]
        x = torch.from_numpy(rows[:, :-1]).to(device)
        y = torch.from_numpy(rows[:, 1:]).to(device)
        q = torch.from_numpy(second[lo:hi]).to(device)
        r = torch.arange(hi - lo, device=device)
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(x)
        ce = F.cross_entropy(logits[r, q].float(), y[r, q], reduction="none")
        out[lo:hi] = ce.float().cpu().numpy()
        del logits
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run dirs or names, as eval_pertoken.py takes them")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--freq", type=int, default=100)
    p.add_argument("--distracted", action="store_true",
                   help="the distracted target set instead of the undistracted one")
    p.add_argument("--batch", type=int, default=None, help="windows per forward; eval_pertoken's")
    p.add_argument("--limit", type=int, default=0,
                   help="a smoke test: this many targets spread over the whole file")
    p.add_argument("--check", action="store_true",
                   help="compare every untouched window to the per-token file, not just 32")
    p.add_argument("--rope-extrapolate", action="store_true",
                   help="evaluate a RoPE model beyond its training context instead of skipping it")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    B = a.batch or max(1, 65536 // a.ctx)
    dirs = [resolve(r) for r in a.runs]
    d = intervention_targets.load(a.ctx, a.freq, a.distracted)
    t = targets.load(a.ctx, a.freq, a.distracted)
    assert targets.ident(t) == int(d["targets"]), "intervention file is for another target set"
    n = len(d["target"])
    sel = (np.unique(np.linspace(0, n - 1, a.limit).round().astype(np.int64))
           if 0 < a.limit < n else np.arange(n))
    tgt = d["target"].astype(np.int64)[sel]
    win, snd = d["win"].astype(np.int64)[sel], d["second"].astype(np.int64)[sel]
    o = d["off"].astype(np.int64)
    pos, tok = d["pos"].astype(np.int64), d["tok"].astype(np.int64)
    off = np.r_[0, np.cumsum(o[sel + 1] - o[sel])]    # the selected slices, packed
    pos = np.concatenate([pos[o[i]:o[i + 1]] for i in sel]) if len(sel) else pos[:0]
    tok = np.concatenate([tok[o[i]:o[i + 1]] for i in sel]) if len(sel) else tok[:0]

    w, _ = windows(a.ctx)
    fp = windows_fingerprint(w)
    assert fp == int(d["fingerprint"]), "the corpus does not give the windows this file was built on"
    tag = targets.tag_of(a) + (f"_limit{a.limit}" if 0 < a.limit < n else "")
    (HERE / "out" / "intervention").mkdir(parents=True, exist_ok=True)
    print(f"{len(sel):,} of {n:,} corrupted windows of {a.ctx}, batch {B}, "
          f"fingerprint {fp}", flush=True)
    for dd in dirs:
        name = stem_of(dd)
        out = HERE / "out" / "intervention" / f"{name}_T{a.ctx}_f{a.freq}{tag}.npz"
        if out.exists() and not a.force:
            print(f"{name}: done"); continue
        pt = HERE / "out" / "pertoken" / f"{name}_T{a.ctx}.npz"
        if not pt.exists():
            print(f"{name}: no per-token file, run eval_pertoken.py first"); continue
        z = np.load(pt)
        assert int(z["fingerprint"]) == fp, f"{name}: window mismatch"
        clean = np.asarray(z["ce"])[win, snd].astype(np.float32)
        model, mcfg = load_trained(dd)
        if (mcfg.pos == "rope" and mcfg.rope_base == 10000.0 and a.ctx > 2048
                and not a.rope_extrapolate):
            print(f"{name}: skipped at T={a.ctx} (RoPE beyond its training context)")
            del model; torch.cuda.empty_cache(); continue
        g = (np.arange(len(sel)) if a.check else
             np.unique(np.linspace(0, len(sel) - 1, min(32, len(sel))).round().astype(np.int64)))
        rec = target_ce(model, w, win[g], snd[g], B=B)
        e = np.abs(rec - clean[g])
        print(f"{name} check: the untouched windows score {rec.mean():.4f} against "
              f"{clean[g].mean():.4f} in the per-token file over {len(g)} targets, "
              f"mean |diff| {e.mean():.2e}, max {e.max():.2e} (that file stores float16, "
              f"so ~1e-3 is agreement)", flush=True)
        assert e.mean() < 0.01, (f"{name}: the untouched windows do not reproduce the "
                                 "per-token file, so the window or the position convention "
                                 "is wrong")
        loss = target_ce(model, w, win, snd, off, pos, tok, B)
        np.savez(out, loss=loss, clean=clean, target=tgt, bucket=d["bucket"][sel],
                 nocc=d["nocc"][sel], fingerprint=fp, targets=int(d["targets"]),
                 ctx=a.ctx, freq=a.freq, distracted=bool(a.distracted),
                 seed=int(d["seed"]), limit=int(a.limit))
        print(f"{name} T={a.ctx}: intervention {loss.mean():.4f}, clean {clean.mean():.4f}, "
              f"{(loss > clean).mean():.1%} of targets worse -> {out.name}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
