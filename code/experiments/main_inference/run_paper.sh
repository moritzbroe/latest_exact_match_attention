#!/usr/bin/env bash
# The paper's inference figure: three model sizes x three placements of the attention state,
# generation speed vs context and prefill time vs prompt length.
#
# Each (model, method, figure) is one curve file in out/. The softmax side is vLLM
# (measure_vllm.py, see its docstring for the settings and why), run on a random-weight Llama
# export of the geometry (export_llama.py); the LEMA side is our implementation
# (measure_generation.py / measure_prefill.py) with worst-case codes. Both are provisioned
# once and never reallocated, so a curve ends either because the kv cache is full (softmax)
# or because the hash table is nearly full and probe runs lengthen (lema-ram). The third side
# is a GDN of the same widths (export_gdn.py, also through vLLM): its state is
# constant, so its run has no end of its own and is given one: 1.2 times the LEMA curve's
# end, of which plot_paper.py draws the first 1.15 (--gdn-past), so the curve runs past the
# LEMA mark and stops at the frame.
#
# vLLM lives in its own environment: VLLM_PY names its python.
#
# Geometry is hd64 throughout, standard Llama-ish width/depth. The 0.8B matches the LM
# suite's d1536 (depth = dim/64) and is the one that is also run from a TRAINED checkpoint
# (--checkpoint ../main_lm/out/<run>, see plot_trained.py).
set -uo pipefail
cd "$(dirname "$0")"
VLLM_PY=${VLLM_PY:-python3}
PIN=${PIN:-}                  # e.g. 'taskset -c 0-15': the paper's runs pin the host side to the
                              # performance cores, the efficiency cores add 10-15% jitter

# Host RAM for the lema-ram store: 50 GB, with swap turned off on the benchmark machine.
RAM_GB=${RAM_GB:-50}
POINTS=${POINTS:-400}         # records per generation curve
STEP=${STEP:-1024}            # prefill: a record every this many tokens
CHUNK=${CHUNK:-2048}          # prefill: LEMA's chunk, the one vLLM uses for the other two
MAXCTX=${MAXCTX:-0}           # 0 = as far as the budget reaches; set for a quick look

FAILED=()
lema () {  # lema <gen|prefill> <tag> <geometry...>
  local what=$1 tag=$2; shift 2
  local out="out/${what}_${tag}_ram.jsonl"
  echo "### $what $tag lema-ram -> $out"
  if [ "$what" = gen ]; then
    $PIN python3 measure_generation.py --method lema-ram --out "$out" --store-gb "$RAM_GB" \
        --points "$POINTS" --max-context "$MAXCTX" --reset "$@"
  else
    $PIN python3 measure_prefill.py --method lema-ram --out "$out" --store-gb "$RAM_GB" \
        --step "$STEP" --chunk "$CHUNK" --max-context "$MAXCTX" --reset "$@"
  fi || { echo "!!! FAILED: $what $tag lema-ram (continuing)"; FAILED+=("$what/$tag/lema"); }
}
softmax () {  # softmax <gen|prefill> <tag>
  local what=$1 tag=$2 out="out/${what}_${tag}_softmax.jsonl"
  echo "### $what $tag softmax/vllm -> $out"
  if [ "$what" = gen ]; then   # Triton kernel: the one that splits the keys at batch 1
    $PIN $VLLM_PY measure_vllm.py gen --model "out/hf_$tag" --out "$out" --backend TRITON_ATTN \
        --points "$POINTS" --max-context "$MAXCTX"
  else                         # FlashAttention-2: the faster prefill
    $PIN $VLLM_PY measure_vllm.py prefill --model "out/hf_$tag" --out "$out" \
        --step "$STEP" --max-context "$MAXCTX"
  fi || { echo "!!! FAILED: $what $tag softmax (continuing)"; FAILED+=("$what/$tag/softmax"); }
}
gdn () {  # gdn <gen|prefill> <tag> <max-context>; vLLM's defaults, there is no kv-cache to
  local what=$1 tag=$2 mc=$3    # read and thus no attention kernel to choose
  local out="out/${what}_${tag}_gdn.jsonl"
  [ "$MAXCTX" != 0 ] && mc=$MAXCTX          # a quick look stops early here too
  echo "### $what $tag gdn/vllm -> $out"
  if [ "$what" = gen ]; then
    $PIN $VLLM_PY measure_vllm.py gen --model "out/hf_${tag}_gdn" --out "$out" --method gdn \
        --points "$POINTS" --max-context "$mc"
  else
    $PIN $VLLM_PY measure_vllm.py prefill --model "out/hf_${tag}_gdn" --out "$out" --method gdn \
        --step "$STEP" --max-context "$mc"
  fi || { echo "!!! FAILED: $what $tag gdn (continuing)"; FAILED+=("$what/$tag/gdn"); }
}

G08="--depth 24 --dim 1536 --num-heads 24 --dim-ff 4096"
G3B="--depth 28 --dim 3072 --num-heads 48 --dim-ff 8192"
G8B="--depth 32 --dim 4096 --num-heads 64 --dim-ff 14336"
# the same widths with the GDN heads of main_lm: size 128, half as many
N08="--depth 24 --dim 1536 --num-heads 12 --dim-ff 4096"
N3B="--depth 28 --dim 3072 --num-heads 24 --dim-ff 8192"
N8B="--depth 32 --dim 4096 --num-heads 32 --dim-ff 14336"
# how far each gdn run goes: 1.2 times the LEMA curve of that size at load 0.95, which ends
# at 572675/245432/161064 (generation) and 571392/243712/159744 (prefill)
declare -A GDN_END=([gen:08B]=687000 [gen:3B]=295000 [gen:8B]=193000
                    [prefill:08B]=686000 [prefill:3B]=293000 [prefill:8B]=192000)

for spec in "08B:$G08" "3B:$G3B" "8B:$G8B"; do
  tag=${spec%%:*}; geo=${spec#*:}
  [ -f "out/hf_$tag/config.json" ] || $VLLM_PY export_llama.py --out "out/hf_$tag" $geo
done
for spec in "08B:$N08" "3B:$N3B" "8B:$N8B"; do
  tag=${spec%%:*}; geo=${spec#*:}
  [ -f "out/hf_${tag}_gdn/config.json" ] || $VLLM_PY export_gdn.py --out "out/hf_${tag}_gdn" $geo \
    $([ "$tag" = 08B ] && echo --max-position 1048576)   # the 0.8B curves run past 2**19 tokens
done

for what in gen prefill; do
  for spec in "08B:$G08" "3B:$G3B" "8B:$G8B"; do
    tag=${spec%%:*}; geo=${spec#*:}
    softmax $what "$tag"
    gdn $what "$tag" "${GDN_END[$what:$tag]}"
    lema $what "$tag" $geo --head-dim 64
  done
done

if [ ${#FAILED[@]} -gt 0 ]; then echo "### FAILED CURVES: ${FAILED[*]}"; fi
echo "### plotting"
# One figure for the paper: columns = model sizes, rows = generation / prefill.
python3 plot_paper.py --out out/inference.pdf
