# Hardening schedule

Figures 6 and 7: the 77M LEMA run with a dense probe, the same run without hardening, and the
same run without the backward cap on alpha.

1. Train the three runs:

       python train_trajectory.py                     # -> out/lema512_h64
       python train_no_hardening.py --head 64         # -> out/lema512_h64_nohard
       python train_no_cap.py                         # -> out/lema512_h64_nocap

2. Validation cross-entropy of the three runs:

       python ../main_lm/eval_lm.py --out out

3. Figures:

       python plot_trajectory.py                      # -> out/trajectory.pdf
       python plot_no_cap.py                          # -> out/no_cap.pdf
