# LEMA

- `lema/`: transformer, attention and training code; `cache.py` and `csrc/store.cpp` implement the inference store.
- `experiments/`: main-text (`main_*`) and appendix (`app_*`) experiments. Each README gives its commands; measurements live in `out/` directories.
    main_recall               associative recall
    main_lm                   the language models, their evaluation, the LM figure and tables
    main_lm_bigrams           recall of repeated rare bigrams, context extension training
    main_codes_count          dictionary size per head against context length
    main_inference            generation and prefill benchmarks
    app_recall_analysis       mechanism of the recall models, stick-breaking on recall
    app_recall_discovery      recall baselines trained without a curriculum
    app_recall_lr             learning-rate schedule during hardening, GDN learning rates
    app_hardening             hardening schedule, training without it, backward cap
    app_lm_seeds              additional seeds
    app_lm_lr_ablation        the recipe at half and double learning rate
    app_lm_context_extension  context extension table and figure
    app_lm_tokens             the 77M models on five times the tokens
    app_lm_analysis           hit rates and attention distances of the 834M LEMA model
    app_sb_ablation           stick-breaking attention throughout
    app_retrieval             RULER
- `tests/`: task, schedule, store and GPU tests.
- `../verification/`: executable word-RAM/LEMA constructions, with a separate README.

Scripts locate the library themselves; no package installation is needed.
Experiment names use `rope` for softmax with RoPE, `gdn` for gated DeltaNet, and `sb` for stick-breaking attention. `lema<d>_h<h>` specifies model and head dimensions; `_s1`/`_s2` identify additional seeds, and `_16k` denotes context extension. In inference filenames, `ram` means LEMA's table is in main memory.

## Software

Python 3.12. The plotting and table scripts need NumPy and Matplotlib 3.10.9. The verification suites need NumPy only.

GPU experiments use three environments:

| Experiments | Install |
| --- | --- |
| Training and evaluation | `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128` (brings Triton 3.4), then `pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl flash-linear-attention==0.2.2 numpy matplotlib==3.10.9 ninja tokenizers transformers huggingface_hub pyarrow nltk` |
| LEMA inference and `app_retrieval/gen_eval.py` | the same torch, then `pip install flash-linear-attention==0.5.2 ninja transformers tokenizers numpy nltk` |
| Softmax/GDN inference baselines | `pip install "vllm==0.17.*"` in its own environment (brings PyTorch 2.10) |

The inference store compiles on first use and needs Ninja and a C++17 compiler (GCC >= 9); set `CXX` if necessary.
`lema/kernel/` includes the stick-breaking attention kernel of Tan et al. (2025), https://github.com/shawntan/stickbreaking-attention, with its Apache 2.0 license; modified files are marked `# lema patch`.

## Tests

From this directory, in the training environment:

```sh
python tests/test_task.py
python tests/test_schedules.py
python tests/test_store.py
python tests/test_gpu.py  # CUDA GPU required
```
