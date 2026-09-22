# Mechanism of the recall models

Figure 13 and the competing-association table, from the checkpoints of ../main_recall.

1. Attention targets at 4096 pairs, per seed S, for lema/s<S>/model.pt and rope/s<S>/n4096:

       python mechanism.py lema/s<S>/model.pt --n 4096              # -> out/mechanism/<arch>_s<S>_n4096.json

2. Competing associations, 64 distinct tokens each paired four times with different tokens, for
   lema/s<S>/model.pt, rope/s<S>/n64 and gdn/s<S>/n64:

       python repeated_keys.py lema/s<S>/model.pt --n-queries 64 --key-repeats 4
                                                                   # -> out/repeated_keys/<arch>_s<S>_q64r4.json

3. Stick-breaking attention with small gate values zeroed at evaluation, on the sb/s<S> runs of
   ../main_recall, and the figure:

       python sb_threshold.py                        # -> out/sb_threshold/sb_s<S>.json
       python plot_sb.py                             # -> out/recall_sb.pdf
