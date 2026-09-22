# Language models

Figure 3 and Tables 6, 7 and 10: LEMA, softmax transformers and GDN at six sizes on FineWeb-Edu.

1. Data, once (tokenized FineWeb-Edu shards, the last one held out):

       python prepare_data.py                                    # -> ../../data/fineweb_edu100_gpt2

2. Train, one run per architecture and width (dim 256 512 768 1024 1280 1536; LEMA head
   dimension 8 16 32 64 up to 1024, 64 at 1280 and 1536):

       python train_lm.py lema <dim> --head <d_h>
       python train_lm.py rope <dim>
       python train_lm.py gdn <dim>
       torchrun --standalone --nproc_per_node=N train_lm.py ...

3. Validation cross-entropy of every run under out/:

       python eval_lm.py                                         # -> out/<run>/eval.json

4. Tables and the LM figure, which also reads ../main_lm_bigrams and ../app_retrieval results:

       python tables.py
       python plot_lm_section.py                                 # -> out/lm_section_834M.pdf
