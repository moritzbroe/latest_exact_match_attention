"""A random-weight Llama in HuggingFace format for vLLM, of one of our geometries.

    python export_llama.py --out out/hf_08B --depth 24 --dim 1536 --num-heads 24 --dim-ff 4096
    python export_llama.py --out out/hf_08B_gqa8 --depth 24 --dim 1536 --num-heads 24 --num-kv-heads 3 --dim-ff 4096

Same structure as our softmax model (RMSNorm, SwiGLU, RoPE with base 1e4, untied embeddings,
no biases): the parameter count of the export matches ours exactly. Built in bf16 from the
start, an 8B model in fp32 would need 32 GB of host memory. Generation and prefill times do
not depend on the weights, so their values do not matter.
"""
import argparse

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--num-heads", type=int, required=True)
    p.add_argument("--num-kv-heads", type=int, default=0, help="0 = num-heads (multi-head)")
    p.add_argument("--dim-ff", type=int, required=True)
    p.add_argument("--vocab", type=int, default=50304)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    cfg = LlamaConfig(vocab_size=a.vocab, hidden_size=a.dim, intermediate_size=a.dim_ff,
                      num_hidden_layers=a.depth, num_attention_heads=a.num_heads,
                      num_key_value_heads=a.num_kv_heads or a.num_heads,
                      head_dim=a.dim // a.num_heads, hidden_act="silu",
                      max_position_embeddings=262144, rms_norm_eps=1e-5,
                      tie_word_embeddings=False, rope_theta=10000.0, attention_bias=False,
                      mlp_bias=False, torch_dtype=torch.bfloat16)
    torch.manual_seed(a.seed)
    torch.set_default_dtype(torch.bfloat16)
    m = LlamaForCausalLM(cfg)
    m.save_pretrained(a.out, safe_serialization=True)
    print(f"saved {a.out}: {sum(p.numel() for p in m.parameters()) / 1e9:.4f}B params")


if __name__ == "__main__":
    main()
