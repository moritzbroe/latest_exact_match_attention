# Learning-rate schedules for synthetic recall

Figures 10 and 11: the LEMA recall recipe against whole-run cosine decay, and GDN at three peak
learning rates.

1. Train both recipes, for S = 0, 1, 2:

       python ../main_recall/train_recall.py lema --seed S
       python train_cosine.py --seed S

2. Collect and plot:

       python plot_schedule.py --collect              # -> out/schedule.json, out/recall_lr_schedule.pdf

Without the runs, `python plot_schedule.py` replots `out/schedule.json`. `--recipe-root` and
`--cosine-root` select other directories containing `s0`, `s1` and `s2`.

## GDN peak learning rates

```sh
for lr in 0.0003 0.001 0.003; do
    python ../main_recall/train_recall.py gdn --seed 0 --lr-baseline "$lr" --out-root "out/lr$lr"
    python ../main_recall/eval_recall.py gdn --out-root "out/lr$lr"
done
python plot_gdn.py --collect                          # -> out/gdn_lr.json, out/recall_gdn_lr.pdf
```

Without the runs, `python plot_gdn.py` replots `out/gdn_lr.json`.
