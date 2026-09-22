# Associative recall

Figures 2, 8 and 12: accuracy against the number of pairs, and the attention pictures.

1. Train, seeds 0 to 2, `arch` one of lema, rope, gdn, sb:

       python train_recall.py <arch> --seed S        # -> out/<arch>/s<S>

2. Evaluate every run at every number of pairs:

       python eval_recall.py                         # -> out/eval/<arch>_s<seed>.json

3. Figures:

       python plot_recall.py                         # -> out/recall.pdf, out/recall_seeds.pdf, out/recall_main.pdf
       python plot_attention.py --grid               # -> out/attention/seeds.pdf

The plotting commands use `out/attention/panels.json` when it is present, so the figures need
no checkpoints or PyTorch.
