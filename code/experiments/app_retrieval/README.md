# RULER

Figures 17, 23 and 24 and Table 14: needle-in-a-haystack accuracy and the cross-entropy of the
correct answer. ruler.py builds the prompts (a transcription of NVIDIA/RULER). Model arguments
are run names under ../main_lm/out or extended runs ../main_lm_bigrams/out/<run>_16k.

1. Assets, once (the essays, word lists and sentence tokenizer):

       python ruler.py --fetch-assets

2. Accuracy cells, 500 prompts each, into out/gen/<task>-<hay>_<model>_T<ctx>.jsonl. S-NIAH-1
   for the 309M models at every head dimension and the 834M models:

       python gen_eval.py lema1024_h8 lema1024_h16 lema1024_h32 lema1024_h64 gdn1024_16k lema1536_h64 gdn1536_16k \
           --task niah_single_1 --hay noise --ctx 2048 4096 8192 16384 32768 65536 --n 500
       python gen_eval.py rope1024_16k rope1536_16k --task niah_single_1 --hay noise --ctx 2048 4096 8192 16384 --n 500

   Further tasks for the 834M models:

       python gen_eval.py lema1536_h64 --task niah_single_2 --hay essay --ctx 2048 4096 8192 16384 32768 65536 --n 500
       python gen_eval.py gdn1536_16k --task niah_single_2 --hay essay --ctx 2048 4096 8192 16384 32768 --n 500
       python gen_eval.py rope1536_16k --task niah_single_2 --hay essay --ctx 2048 4096 8192 16384 --n 500
       python gen_eval.py lema1536_h64 gdn1536_16k rope1536_16k --task niah_single_3 niah_multikey_1 --hay essay --ctx 2048 4096 8192 16384 --n 500
       python gen_eval.py lema1536_h64 --task niah_multikey_2 --hay needle --ctx 2048 4096 8192 16384 32768 65536 --n 500
       python gen_eval.py gdn1536_16k rope1536_16k --task niah_multikey_2 --hay needle --ctx 2048 4096 8192 16384 --n 500

3. Cross-entropy of the correct answer on the same prompts, into out/answer_ce/:

       python answer_ce.py lema1536_h64 gdn1536_16k --cell niah_single_1-noise niah_single_2-essay --ctx 2048 4096 8192 16384 32768 65536
       python answer_ce.py rope1536_16k --cell niah_single_1-noise niah_single_2-essay --ctx 2048 4096 8192 16384
       python answer_ce.py lema1536_h64 gdn1536_16k rope1536_16k --cell niah_single_3-essay niah_multikey_1-essay niah_multikey_2-needle --ctx 2048 4096 8192 16384
       python answer_ce.py --summary                 # -> out/answer_ce/summary.csv

4. Table and figures:

       python paper/sniah1_table.py
       python paper/plot_ruler.py                    # -> out/sniah1_depth.pdf
       python paper/plot_sniah1_heads.py             # -> out/sniah1_heads_309M.pdf
       python paper/plot_ruler_more.py               # -> out/ruler_more.pdf

5. Fixed point of the filler:

       python fixedpoint/fixedpoint.py lema1024_h64 lema1536_h64   # -> out/fixedpoint/<run>.json
