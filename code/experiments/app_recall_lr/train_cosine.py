"""The recall recipe with the hardening learning-rate drop replaced by a cosine.

The paper's LEMA recall recipe (main_recall/train_recall.py, `--c-end 15000`) drops the
learning rate to 5e-5 (`lr_alpha_phase`) for the whole alpha phase. Here that drop is
removed and the run instead follows the baselines' schedule: a whole-run cosine from the
same peak (3e-4) to the baselines' `lr_final` (0.0) with their 1000-step warmup. Everything
else is the recipe: 50000 steps, batch 4096, c linearly from 0 to d_h - 1 over steps 1000
to 15000, alpha from 1/sqrt(64) at step 15000 linearly to exactly 10 at the last step,
backward cap 2, beta 4, probe every 250 steps, never stopped early.

Both accuracies of the hardening come out of the probe the trainer already runs every 250
steps on one held-out batch at the training count n=8 (lema/probes.py): `soft_acc` is the
model as trained (the stick-breaking surrogate at the step's c and alpha), `hard_acc` the
same weights through the exact latest-match op. probes.jsonl also carries step, loss, lr,
c and alpha; log.jsonl carries the loss window, lr, c, alpha and gradient norms.

Checkpoints every 5000 steps (`checkpoint_at`) plus the final model.pt and the trainer's
model_best.pt. After training, every checkpoint is evaluated from disk: hard (exact op) at
every count of eval_recall.EVAL_NS and soft (surrogate at that step's c and alpha, beta as
trained) at n=8, into out/s<seed>/eval.json.

    python train_cosine.py --seed S                      # train + evaluate, out/s<S>/
    python train_cosine.py --seed S --steps 300 --max-n 64   # smoke test, out/s<S>_300/

With `--steps` other than 50000 every phase of the recipe (warmup, c ramp, alpha ramp,
checkpoint cadence) is scaled proportionally, which is what the smoke test uses.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # code/
import torch                                                          # noqa: E402

from lema import TrainConfig, train                                   # noqa: E402
from lema.train import alpha_at, c_at                                 # noqa: E402
from experiments.main_recall.train_recall import D_HEAD, model_cfg, task_at   # noqa: E402
from experiments.main_recall.eval_recall import EVAL_NS, accuracy_at, load    # noqa: E402

HERE = Path(__file__).resolve().parent
FULL = 50000                     # the step count the recipe's phases are quoted at


def make_cfg(a, seed: int, out: Path) -> TrainConfig:
    """The recall recipe at `a.steps` steps, the hardening lr drop replaced by a cosine."""
    f = a.steps / FULL
    warmup = max(1, round(1000 * f))
    c_start = max(1, round(1000 * f))
    c_end = max(c_start + 1, round(a.c_end * f))
    every = max(1, round(5000 * f))
    return TrainConfig(
        steps=a.steps, out=str(out), lr=a.lr,
        lr_final=a.lr_final,                       # whole-run cosine (baselines: 0.0)
        warmup=warmup,                             # baselines' warmup: 1000
        lr_alpha_phase=0.0,                        # the ablation: no drop when alpha rises
        batch=4096,
        c_start=c_start, c_steps=c_end - c_start,  # c over steps c_start .. c_end
        alpha_speed=max(1, round((a.steps - c_end) / (10.0 - D_HEAD ** -0.5))),
        alpha_target=10.0,                         # exactly 10 at the last step
        alpha_bwd_cap=2.0, beta=a.beta,
        seed=seed, probe_every=a.probe_every, early_stop_acc=2.0,   # never stopped early
        checkpoint_at=tuple(range(every, a.steps, every)))


def eval_run(out: Path, cfg: TrainConfig, ns: list[int]) -> dict:
    """Every checkpoint in `out`: hard accuracy at every count in `ns` and soft accuracy
    at n=8, the latter at the c and alpha of the step the checkpoint was written at."""
    a0, c_max = D_HEAD ** -0.5, float(D_HEAD - 1)
    steps = sorted(int(p.name[len("model_step"):-3]) for p in out.glob("model_step*.pt"))
    names = [f"model_step{s}.pt" for s in steps] + ["model.pt", "model_best.pt"]
    res = {"run": str(out), "seed": cfg.seed, "steps": cfg.steps, "eval_ns": list(ns),
           "train": dataclasses.asdict(cfg), "checkpoints": {}}
    for name in names:
        ck_path = out / name
        if not ck_path.exists():
            print(f"  no {ck_path}", flush=True)
            continue
        step = int(torch.load(ck_path, map_location="cpu", weights_only=False)["step"])
        at = min(step, cfg.steps - 1)              # model.pt is stamped with cfg.steps
        c_now, alpha_now = c_at(cfg, at, c_max), alpha_at(cfg, at, a0)
        model = load(ck_path, "lema")              # exact-match LEMA (set_hard(True))
        hard = {n: accuracy_at(model, n) for n in ns}
        model.set_hard(False)                      # the surrogate, as the step trained it
        model.set_hardening(c=c_now, cap=cfg.alpha_bwd_cap, alpha=alpha_now)
        for att in model.lema_layers():
            att.beta = float(cfg.beta)
        soft8 = accuracy_at(model, 8)
        del model
        torch.cuda.empty_cache()
        res["checkpoints"][name] = {
            "step": step, "c": c_now, "alpha": alpha_now,
            "soft_acc_n8": soft8, "hard_acc_n8": hard.get(8),
            "hard_curve": hard}
        print(f"  {name:>20} step {step:>6} c {c_now:6.2f} alpha {alpha_now:6.3f} "
              f"soft@8 {soft8:.4f} hard@8 {hard.get(8, float('nan')):.4f} "
              + " ".join(f"{n}:{v:.4f}" for n, v in hard.items()), flush=True)
    res["final_hard_curve"] = res["checkpoints"].get("model.pt", {}).get("hard_curve")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=FULL)
    p.add_argument("--lr", type=float, default=3e-4, help="peak lr of the cosine")
    p.add_argument("--lr-final", type=float, default=0.0,
                   help="end of the cosine; the baselines' value is 0.0")
    p.add_argument("--c-end", type=int, default=15000,
                   help="step (at 50000 steps) where c reaches d_h - 1 and alpha starts")
    p.add_argument("--beta", type=float, default=4.0, help="STE temperature")
    p.add_argument("--probe-every", type=int, default=250)
    p.add_argument("--max-n", type=int, default=0,
                   help="cap the evaluation counts (smoke tests); 0 = all of EVAL_NS")
    a = p.parse_args()

    out = HERE / "out" / (f"s{a.seed}" + ("" if a.steps == FULL else f"_{a.steps}"))
    cfg = make_cfg(a, a.seed, out)
    print(f"{out}: {dataclasses.asdict(cfg)}", flush=True)
    if not (out / "model.pt").exists():
        train(model_cfg("lema"), task_at(8), cfg)
        torch.cuda.empty_cache()
    ns = [n for n in EVAL_NS if not a.max_n or n <= a.max_n]
    print(f"evaluating {out} at n = {ns}", flush=True)
    res = eval_run(out, cfg, ns)
    (out / "eval.json").write_text(json.dumps(res, indent=1))
    print(f"-> {out / 'eval.json'}", flush=True)


if __name__ == "__main__":
    main()
