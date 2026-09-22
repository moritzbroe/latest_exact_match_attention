# Context extension

Table 13 and Figure 19: validation cross-entropy and recall of the largest models before and
after context extension to 16k.

1. Extend and evaluate the three models, with the scripts of ../main_lm_bigrams (the LEMA
   extension at 16k windows needs `--grad-accum 2 --checkpoint-blocks` on a 24 GB GPU):

       cd ../main_lm_bigrams
       python finetune_ctx.py rope1536; python finetune_ctx.py gdn1536
       python finetune_ctx.py lema1536_h64 --grad-accum 2 --checkpoint-blocks
       python eval_pertoken.py rope1536 gdn1536 lema1536_h64 rope1536_16k gdn1536_16k lema1536_h64_16k --ctx 2048
       python eval_pertoken.py gdn1536 lema1536_h64 rope1536_16k gdn1536_16k lema1536_h64_16k --ctx 16384
       python eval_pertoken.py rope1536 --ctx 16384 --rope-extrapolate
       python score.py rope1536 gdn1536 lema1536_h64 rope1536_16k gdn1536_16k lema1536_h64_16k
       python eval_intervention.py lema1536_h64_16k

2. Table and figure:

       cd ../app_lm_context_extension
       python ce_table.py --latex --first-tokens 67108864
       python plot_extension.py --out out/lm_extension.pdf

Without the per-token arrays, `ce_table.py` reads the means over the first 67108864 tokens
from `out/eval/ce_summary.json`.
