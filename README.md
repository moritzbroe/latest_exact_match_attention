# Latest Exact Match Attention

Code and recorded measurements for the paper *Latest Exact Match Attention* (Moritz Brösamle, 2026, arXiv:2609.25802, https://arxiv.org/abs/2609.25802).

    code/lema/          the library: model, attention, training, the inference store
    code/experiments/   the experiments, each with a README and its measurements in out/
    code/tests/         task, schedule, store and GPU tests
    verification/       executable constructions of Theorems 1 and 2, NumPy only

Software environments and run names are listed in `code/README.md`.

## Regenerating the figures and tables
No GPU, no checkpoints, no PyTorch. Run in the folder given, under code/experiments.

    Figure 2, 8   main_recall               python plot_recall.py                 (out/recall_main.pdf, out/recall_seeds.pdf)
    Figure 3      main_lm                   python plot_lm_section.py
    Figure 4      main_inference            python plot_paper.py
    Figure 6      app_hardening             python plot_trajectory.py
    Figure 7      app_hardening             python plot_no_cap.py
    Figure 9      app_recall_discovery      python plot_discovery.py
    Figure 10     app_recall_lr             python plot_schedule.py
    Figure 11     app_recall_lr             python plot_gdn.py
    Figure 12     main_recall               python plot_attention.py --grid
    Figure 13     app_recall_analysis       python plot_sb.py
    Figure 14     app_lm_seeds              python plot_seeds.py
    Figure 15     main_lm_bigrams           python plot_sb_recall.py --out out/sb_recall_77M.pdf
    Figure 16     main_lm_bigrams           python plot_recall_panel.py --dim 1024 --heads 8,16,32,64 --out out/lm_recall_heads_309M.pdf
    Figure 17     app_retrieval             python paper/plot_sniah1_heads.py
    Figure 18     app_lm_analysis           python plot_distances.py out/lema1536_h64.json
    Figure 19     app_lm_context_extension  python plot_extension.py --out out/lm_extension.pdf
    Figure 20     main_lm_bigrams           python plot_sizes_grid.py --ctx 16384 --baselines extended --heads 64 --out out/lm_recall_grid_16k.pdf
    Figure 21     main_lm_bigrams           python examples_figure.py --distracted --out out/lm_recall_examples.pdf
    Figure 22     main_lm_bigrams           python plot_recall_panel.py --dim 1536 --heads 64 --union --out out/lm_recall_union_834M.pdf
    Figure 23     app_retrieval             python paper/plot_ruler.py
    Figure 24     app_retrieval             python paper/plot_ruler_more.py
    Figure 25     app_lm_tokens             python plot_tokens.py
    Figure 26     main_inference            python plot_trained.py
    Figure 27     main_inference            python plot_batched.py
    Table 6, 7, 10  main_lm                 python tables.py
    Table 8       app_lm_seeds              python losses.py --latex
    Table 9       app_lm_lr_ablation        python collect.py --latex
    Table 11, 12  main_codes_count          python table.py --latex
    Table 13      app_lm_context_extension  python ce_table.py --latex --first-tokens 67108864
    Table 14      app_retrieval             python paper/sniah1_table.py
    Table 15      app_lm_tokens             python losses.py --latex

Figures 1 and 5 are typeset in the paper. Tables 1 to 4 belong to the proofs. Tables 5, 16 and 17 have no script; their inputs are app_recall_analysis/out/repeated_keys/ and main_inference/out/.



## Citation

    @article{broesamle2026lema,
      title   = {Latest Exact Match Attention},
      author  = {Moritz Br{\"o}samle},
      journal = {arXiv preprint arXiv:2609.25802},
      year    = {2026}
    }

## License

Apache License 2.0, see `LICENSE`. `code/lema/kernel/` is vendored from the stick-breaking attention kernel of Tan et al. (2025), https://github.com/shawntan/stickbreaking-attention, under the same license; modified files are marked `# lema patch`. `code/experiments/app_retrieval/ruler.py` transcribes the prompt generator of NVIDIA's RULER, Apache License 2.0.
