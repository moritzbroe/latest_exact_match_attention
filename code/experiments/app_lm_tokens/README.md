# Five times the tokens

Table 15 and Figure 25: the model dimension 512 configuration of ../main_lm, all three
architectures, trained on five times the tokens (235500 steps x 32768 tokens = 7.7B instead of
1.5B). The runs are named rope512_5x, gdn512_5x and lema512_h64_5x under ../main_lm/out.

1. Train:

       python train_tokens.py rope 512
       python train_tokens.py gdn 512
       torchrun --standalone --nproc_per_node=4 train_tokens.py lema 512 --head 64

2. Validation cross-entropy, the context extension of the two baselines to 16k and the recall
   analysis:

       cd ../main_lm && python eval_lm.py rope512_5x gdn512_5x lema512_h64_5x [--ctx 16384]
       cd ../main_lm_bigrams && python finetune_ctx.py rope512_5x && python finetune_ctx.py gdn512_5x
       cd ../main_lm_bigrams && python eval_pertoken.py rope512_5x_16k gdn512_5x_16k lema512_h64_5x \
           && python score.py rope512_5x_16k gdn512_5x_16k lema512_h64_5x \
           && python eval_intervention.py rope512_5x_16k gdn512_5x_16k lema512_h64_5x

3. Table and figure:

       cd ../app_lm_tokens
       python losses.py
       python plot_tokens.py                         # -> out/lm_tokens.pdf

Without the per-token arrays, the 16384 column comes from
`../app_lm_context_extension/out/eval/ce_summary.json`.
