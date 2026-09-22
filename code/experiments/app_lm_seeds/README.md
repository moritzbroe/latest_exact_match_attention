# Variation across seeds

Figure 14 and Table 8: three seeds at model dimensions 256 and 512 for softmax, GDN and LEMA.

1. Per seed, with the main scripts. Runs are named `rope256_s1`, `gdn256_s1`,
   `lema256_h64_s1`, etc.; the extended baselines append `_16k`.

```sh
(cd ../main_lm && python train_lm.py <arch> <dim> --seed <seed>)  # --head 64 for lema
(cd ../main_lm && python eval_lm.py <run> --ctx 2048)
(cd ../main_lm_bigrams && python finetune_ctx.py <run>)  # softmax and GDN only
(cd ../main_lm_bigrams && python eval_pertoken.py <evaluated-run> && python score.py <evaluated-run>)
```

2. Table and figure:

```sh
python losses.py --latex
python plot_seeds.py                                  # -> out/lm_recall_seeds.pdf
```

These read `../main_lm/out/` and the bigram scores in `../main_lm_bigrams/out/`, or the
bucket means there when the scores are absent.
