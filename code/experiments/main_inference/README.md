# Inference benchmarks

Figures 4, 26 and 27: generation and prefill against context length.

One RTX 3090, 50 GB of host RAM for the LEMA table, swap off, the host side pinned to the
performance cores with `PIN='taskset -c 0-15'`. The softmax and GDN sides run in vLLM:
`VLLM_PY=<python of that environment>`. The GPU must be otherwise idle: vLLM claims 95% of its
memory (`--util`).

1. Every curve of the main figure, random weights and worst-case codes for LEMA:

       ./run_paper.sh                                # -> out/{gen,prefill}_<size>_<method>.jsonl, out/inference.pdf

2. Batched generation of the trained 834M LEMA model against grouped-query softmax and GDN:

       ./run_batched_figure.sh                       # -> out/genB<batch>_08B_*.jsonl (batch 1: out/gen_08B_*.jsonl), out/batched.pdf

3. The trained model's own codes against the worst case:

       python measure_generation.py --method lema-ram --checkpoint ../main_lm/out/lema1536_h64 --store-gb 50 --max-context 687000 --reset --out out/gen_08B_ram_trained.jsonl
       python measure_prefill.py --method lema-ram --checkpoint ../main_lm/out/lema1536_h64 --store-gb 50 --chunk 2048 --max-context 686000 --reset --out out/prefill_08B_ram_trained.jsonl
       python plot_trained.py                        # -> out/trained_vs_worst.pdf

Replot without a GPU:

```sh
python plot_paper.py
python plot_trained.py
python plot_batched.py
```
