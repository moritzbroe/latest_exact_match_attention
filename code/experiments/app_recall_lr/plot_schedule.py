"""Plot the recall hardening schedules from retained, small probe records."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.style import paper


def collect(recipe_root, cosine_root):
    runs = []
    for schedule, root in (("drop", recipe_root), ("cosine", cosine_root)):
        for seed in range(3):
            directory = root / f"s{seed}"
            config = json.loads((directory / "config.json").read_text())
            config["train"].pop("out", None)
            rows = [json.loads(line) for line in
                    (directory / "probes.jsonl").read_text().splitlines() if line.strip()]
            records = [{key: row[key] for key in
                        ("step", "soft_ce", "hard_acc", "lr", "c", "alpha")}
                       for row in rows]
            runs.append(dict(schedule=schedule, seed=seed, config=config, probes=records))
    return dict(task_pairs=8, runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true",
                        help="refresh the retained records from completed runs")
    parser.add_argument("--recipe-root", type=Path,
                        default=HERE.parent / "main_recall/out/lema")
    parser.add_argument("--cosine-root", type=Path, default=HERE / "out")
    args = parser.parse_args()
    target = HERE / "out/schedule.json"
    if args.collect:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(collect(args.recipe_root, args.cosine_root), indent=1) + "\n")
    data = json.loads(target.read_text())
    paper()
    colors = ["#2b6cb0", "#d95f02", "#1b9e77"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35))
    for run in data["runs"]:
        rows = run["probes"]
        steps = [row["step"] / 1000 for row in rows]
        for ax, metric in zip(axes, ("soft_ce", "hard_acc")):
            ax.plot(steps, [row[metric] for row in rows], color=colors[run["seed"]],
                    ls="-" if run["schedule"] == "drop" else "--", lw=1.0, alpha=0.9)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("surrogate loss (nats)")
    axes[1].set_ylabel("exact-match accuracy")
    axes[1].set_ylim(-0.02, 1.03)
    for ax in axes:
        ax.axvline(15, color="0.55", ls=":", lw=0.8, zorder=0)
        ax.set_xlim(0, 50)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_xlabel("training steps (thousands)")
        ax.grid(axis="y", color="0.9", lw=0.5)
        ax.set_axisbelow(True)
    handles = [Line2D([], [], color=color, label=f"seed {seed}")
               for seed, color in enumerate(colors)]
    handles += [Line2D([], [], color="0.2", ls=style, label=label)
                for style, label in (("-", "LR drop"), ("--", "cosine"))]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.01), handlelength=2.0, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.9), pad=0.3, w_pad=1.2)
    output = HERE / "out/recall_lr_schedule.pdf"
    fig.savefig(output)
    print(output)


if __name__ == "__main__":
    main()
