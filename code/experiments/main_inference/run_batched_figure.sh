#!/usr/bin/env bash
# The batched-generation figure of the appendix (plot_batched.py): the trained 0.8B LEMA
# model (../main_lm/out/lema1536_h64, its own codes, sampling at temperature 1) against a
# conventional transformer of the same geometry with 8-fold grouped-query attention (3 key/value
# heads for 24 query heads, random weights) and against a GDN of the same width
# (export_gdn.py), at batch sizes 1, 4, 16, 64 and 256.
#
# One curve per (batch, method) in out/genB<b>_08B_trained_ram.jsonl,
# out/genB<b>_08B_gqa8_softmax.jsonl (batch 1 of the softmax side: out/gen_08B_gqa8_softmax.jsonl)
# and out/genB<b>_08B_gdn.jsonl. The LEMA runs stop at 250k tokens up to batch 16 and at 200k
# from batch 64 on, where the table reaches 95% occupancy first; softmax stops at its VRAM
# limit or those token counts, the gated DeltaNet, whose state is constant, at them. The
# figure shows every curve up to plot_batched.py's XMAX of 250k.
#
# Everything runs one after the other on ONE GPU with nothing else on the machine, swap off,
# and the LEMA runs pinned to the performance cores (PIN): unpinned, the scheduler moves the
# exchange threads onto the efficiency cores now and then, which shows as 10-15% plateaus in
# the batch-16 and batch-64 curves. The softmax side is
# vLLM with its default FlashAttention-2 kernel (measure_vllm.py), on a random-weight Llama
# export of the GQA geometry; VLLM_PY names the python of the vLLM environment.
set -uo pipefail
cd "$(dirname "$0")"
RAM_GB=${RAM_GB:-50}
PIN=${PIN:-}                  # e.g. 'taskset -c 0-15'
VLLM_PY=${VLLM_PY:-python3}
CKPT=../main_lm/out/lema1536_h64
[ -f out/hf_08B_gqa8/config.json ] || \
  $VLLM_PY export_llama.py --out out/hf_08B_gqa8 --depth 24 --dim 1536 --num-heads 24 --num-kv-heads 3 --dim-ff 4096
[ -f out/hf_08B_gdn/config.json ] || \
  $VLLM_PY export_gdn.py --out out/hf_08B_gdn --depth 24 --dim 1536 --num-heads 12 --dim-ff 4096

for b in 1 4 16 64 256; do
  if [ $b -ge 64 ]; then mc=200000; ml=0.95; else mc=250000; ml=0.99; fi
  echo "### $(date +%T) trained 0.8B lema-ram batch $b"
  $PIN python3 measure_generation.py --method lema-ram --out out/genB${b}_08B_trained_ram.jsonl \
      --checkpoint $CKPT --store-gb $RAM_GB --points 200 --batch $b --max-context $mc --max-load $ml --reset \
    || echo "!!! FAILED lema batch $b"
done
for b in 1 4 16 64 256; do
  out=out/genB${b}_08B_gqa8_softmax.jsonl; [ $b = 1 ] && out=out/gen_08B_gqa8_softmax.jsonl
  echo "### $(date +%T) gqa8 softmax/vllm 0.8B batch $b"
  $PIN $VLLM_PY measure_vllm.py gen --model out/hf_08B_gqa8 --out $out --batch $b --points 400 \
      --max-context 250000 || echo "!!! FAILED softmax batch $b"
done
# the gated DeltaNet of the same width (12 heads of size 128), same batch sizes; batch 1 is
# the curve of the paper's figure, out/gen_08B_gdn.jsonl from run_paper.sh, and is not redone
for b in 4 16 64 256; do
  echo "### $(date +%T) gdn/vllm 0.8B batch $b"
  $PIN $VLLM_PY measure_vllm.py gen --model out/hf_08B_gdn --method gdn --out out/genB${b}_08B_gdn.jsonl \
      --batch $b --points 400 --max-context 250000 || echo "!!! FAILED gdn batch $b"
done
echo "### $(date +%T) plotting"
python3 plot_batched.py --out out/batched.pdf
