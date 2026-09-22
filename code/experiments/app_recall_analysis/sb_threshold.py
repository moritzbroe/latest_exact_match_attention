"""The stick-breaking recall models evaluated with small gate values zeroed.

    sb_threshold.py [sb/s0 sb/s1 ...] [--taus 0,0.01,0.03,0.1,0.3]

A plain stick-breaking model (attn "sb": gate sigmoid(alpha <q,k>) at alpha = 1/sqrt(d_h),
c = 0) that solves n=8 fails at large n. If the failure comes from many small gate values
eating the stick before the true match is reached, then zeroing every gate below tau at
evaluation time restores the accuracy. Each run of ../main_recall/out (default: every sb/s*)
is evaluated from its final checkpoint with a reference implementation of stick-breaking
attention in fp32 in which gates below tau are set to zero, at every n of eval_recall.py.
Writes out/sb_threshold/sb_s<seed>.json: {tau: {n: acc}}.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
import lema.model as lm                                                # noqa: E402
from experiments.main_recall.eval_recall import EVAL_NS, OUT as RECALL_OUT, accuracy_at, load  # noqa: E402

OUT_DIR = HERE / "out" / "sb_threshold"
TAU = 0.0


def sb_reference(q, k, v, alpha, c):
    """Stick-breaking attention, strictly causal, latest match first, in fp32 log space as
    the training kernel computes it, with gate values below TAU zeroed. Same (output,
    remainder) signature as lema.model.lema_soft."""
    T = q.shape[-2]
    z = alpha * (torch.einsum("bhid,bhjd->bhij", q.float(), k.float()) - c)
    log_g = torch.nn.functional.logsigmoid(z)                  # log of the gate
    log_1mg = torch.nn.functional.logsigmoid(-z)               # log of one minus the gate
    if TAU > 0:
        small = torch.sigmoid(z) < TAU
        log_g = log_g.masked_fill(small, float("-inf"))
        log_1mg = log_1mg.masked_fill(small, 0.0)
    causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril(-1)      # j < i
    log_1mg = log_1mg.masked_fill(~causal, 0.0)
    cs = torch.cumsum(log_1mg, dim=-1)                                          # sum_{l<=j}
    S = cs[..., -1:] - cs                                                       # sum_{j<l<i}
    w = torch.exp((log_g + S).masked_fill(~causal, float("-inf")))
    o = torch.einsum("bhij,bhjd->bhid", w, v.float())
    return o.to(v.dtype), (1 - w.sum(-1)).detach()


def main():
    global TAU
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="*", help="runs under ../main_recall/out, default sb/s*")
    p.add_argument("--taus", default="0,0.01,0.03,0.1,0.3")
    a = p.parse_args()
    runs = [RECALL_OUT / r for r in a.runs] or sorted((RECALL_OUT / "sb").glob("s*"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lm.lema_soft = sb_reference                        # the "sb" path calls this name
    for run in runs:
        model, res = load(run / "model.pt", "sb"), {}
        for tau in [float(t) for t in a.taus.split(",")]:
            TAU = tau
            res[tau] = {n: accuracy_at(model, n) for n in EVAL_NS}
            print(f"{run.name} tau={tau:<5} "
                  + " ".join(f"{n}:{v:.4f}" for n, v in res[tau].items()), flush=True)
        ev = RECALL_OUT / "eval" / f"sb_{run.name}.json"
        if ev.exists() and 0.0 in res:                # the reference must reproduce the kernel
            ref = {int(n): v for n, v in json.loads(ev.read_text())["curve"].items()}
            gap = max(abs(res[0.0][n] - ref[n]) for n in EVAL_NS)
            print(f"{run.name} tau=0 against the kernel evaluation: max difference {gap:.4f}")
        out = OUT_DIR / f"sb_{run.name}.json"
        out.write_text(json.dumps(res, indent=1))
        print(f"-> {out}")


if __name__ == "__main__":
    main()
