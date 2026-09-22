# Stick-breaking attention throughout

Figure 15: the 77M model trained with plain stick-breaking attention, no hardening.

1. Train:

       python train_sb.py                             # -> ../main_lm/out/sb512

2. Validation cross-entropy, the recall analysis and its figure:

       cd ../main_lm && python eval_lm.py sb512
       cd ../main_lm_bigrams && python eval_pertoken.py sb512 && python score.py sb512
       cd ../main_lm_bigrams && python plot_sb_recall.py --out out/sb_recall_77M.pdf
