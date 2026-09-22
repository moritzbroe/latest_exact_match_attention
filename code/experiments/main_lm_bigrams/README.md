# Recall of repeated rare bigrams

Figures 16, 20, 21 and 22: the loss on the second token of a repeated rare bigram against the
distance to its earlier occurrence. Runs are read from ../main_lm/out, extended runs from
out/<run>_16k.

1. Context extension to 16k for rope<dim> and gdn<dim> at every width and for lema1536_h64:

       python finetune_ctx.py <run>                  # -> out/<run>_16k

2. Bigram counts of the training data, once:

       python count_bigrams.py                       # -> out/bigram_counts.npz

3. Per-token losses and target scores, for lema*_h64, lema1024_h{8,16,32} and the extended
   baselines:

       python eval_pertoken.py <run> ...             # -> out/pertoken/<run>_T16384.npz
       python score.py <run> ...                     # -> out/scores/<run>_T16384_f100.npz

4. Intervention baseline, every earlier occurrence of the bigram replaced:

       python intervention_targets.py                # -> out/intervention_T16384_f100.npz
       python eval_intervention.py <run> ...         # -> out/intervention/<run>_T16384_f100.npz

5. The union target set for the 834M models:

       python score.py lema1536_h64 rope1536_16k gdn1536_16k --distracted
       python intervention_targets.py --distracted
       python eval_intervention.py lema1536_h64 rope1536_16k gdn1536_16k --distracted

6. Figures:

       python plot_recall_panel.py --dim 1024 --heads 8,16,32,64 --out out/lm_recall_heads_309M.pdf
       python plot_recall_panel.py --dim 1536 --heads 64 --union --out out/lm_recall_union_834M.pdf
       python plot_sizes_grid.py --ctx 16384 --baselines extended --heads 64 --out out/lm_recall_grid_16k.pdf
       python examples.py                            # -> out/examples/bucket_8192_16384_f100.txt
       python examples_figure.py --distracted --out out/lm_recall_examples.pdf
       python plot_sb_recall.py --out out/sb_recall_77M.pdf   # Figure 15, the model of ../app_sb_ablation

For every run whose scores are absent the plotting scripts read the per-bucket means in
`out/bucket_means.json`; `python measurements.py` refreshes that file.
