"""GDN recall versus association count at three peak learning rates, seed 0.

    python plot_gdn.py

Reads out/gdn_lr.json (retained per-stage held-out evaluations and provenance).
Writes out/recall_gdn_lr.pdf. No model checkpoints or GPU are needed.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.style import COLOR, paper
from experiments.main_recall.plot_recall import style


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=HERE / 'out' / 'gdn_lr.json')
    parser.add_argument('--out', type=Path, default=HERE / 'out' / 'recall_gdn_lr.pdf')
    parser.add_argument('--collect', action='store_true', help='collect out/lr*/eval/gdn_s0.json')
    args = parser.parse_args()
    if args.collect:
        collected = []
        for lr in (0.0003, 0.001, 0.003):
            path = HERE / 'out' / f'lr{lr}' / 'eval' / 'gdn_s0.json'
            result = json.loads(path.read_text())
            collected.append(dict(lr=lr, seed=0, points=result['points']))
        args.data.parent.mkdir(parents=True, exist_ok=True)
        args.data.write_text(json.dumps(dict(runs=collected), indent=1) + '\n')
    runs = json.loads(args.data.read_text())['runs']
    paper()
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    style(ax)
    styles = {0.0003: ('--', 'o', '#80cdb3'),
              0.001: ('-', '^', COLOR['gdn']),
              0.003: (':', 's', '#006d50')}
    labels = {0.0003: r'$3\cdot10^{-4}$', 0.001: r'$10^{-3}$',
              0.003: r'$3\cdot10^{-3}$'}
    for run in sorted(runs, key=lambda r: r['lr']):
        lr = run['lr']
        if run['seed'] != 0:
            raise ValueError('This figure compares seed 0 at each learning rate')
        points = {int(n): v for n, v in run['points'].items() if int(n) >= 4}
        ns = sorted(points)
        if ns != [2**k for k in range(2, 13)]:
            raise ValueError(f'Incomplete curve at learning rate {lr}')
        ls, marker, color = styles[lr]
        ax.plot(ns, [points[n] for n in ns], label=labels[lr], color=color,
                linestyle=ls, marker=marker, markersize=3.5, linewidth=1.4)
    ax.legend(title='peak learning rate', frameon=True, framealpha=1,
              fancybox=False, edgecolor='0.8', fontsize=7, title_fontsize=7)
    fig.tight_layout(pad=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(args.out)


if __name__ == '__main__':
    main()
