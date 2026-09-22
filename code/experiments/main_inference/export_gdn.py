"""A random-weight Gated DeltaNet in HuggingFace format for vLLM, of one of our geometries.

    python export_gdn.py --out out/hf_08B_gdn --depth 24 --dim 1536 --num-heads 12 --dim-ff 4096

vLLM serves gated DeltaNet layers through its Qwen3-Next model, so the export is a
`Qwen3NextForCausalLM` with *every* layer a `linear_attention` layer and *every* MLP the
dense `Qwen3NextMLP` (`layer_types`, `mlp_only_layers` and `num_experts = 0`): no
full-attention layer, no mixture of experts, i.e. the same block structure as our GDN models
of `main_lm` (RMSNorm, SwiGLU, no positional encoding, untied embeddings, no biases).

Our GDN models are `fla.layers.GatedDeltaNet(hidden_size=dim, num_heads=H, head_dim=128,
expand_v=1)`, i.e. key and value heads of size 128 and H = dim/128, half as many heads as
the softmax model of the same width has. That is `linear_num_key_heads =
linear_num_value_heads = H`, `linear_key_head_dim = linear_value_head_dim = 128` and the
conv of width 4 here. The gate of the block (the `z` half of `in_proj_qkvz`) is one more
d x d matrix per layer than the softmax model has, so a GDN export has a few percent more
parameters than the Llama export of the same geometry; the script prints the count.

Built in bf16 from the start, like export_llama.py. Generation and prefill times do not
depend on the weights, so their values do not matter.

`max_position_embeddings` defaults to 2**19, not Qwen3-Next's 262144: it is dead config for
these models, which have no positional encoding at all, but vLLM refuses a `max_model_len`
beyond it (the alternative is its env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1). The 0.8B curves
run to 687k tokens, past 2**19, so that export is made with --max-position 1048576.
"""
import argparse

import torch
from transformers import Qwen3NextConfig, Qwen3NextForCausalLM


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--num-heads", type=int, required=True,
                   help="linear key = value heads (dim/head-dim for our models)")
    p.add_argument("--head-dim", type=int, default=128, help="linear key = value head size")
    p.add_argument("--dim-ff", type=int, required=True)
    p.add_argument("--conv", type=int, default=4, help="linear_conv_kernel_dim")
    p.add_argument("--vocab", type=int, default=50304)
    p.add_argument("--max-position", type=int, default=1 << 19,
                   help="only gates vLLM's max_model_len check, see the docstring")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    cfg = Qwen3NextConfig(
        vocab_size=a.vocab, hidden_size=a.dim, intermediate_size=a.dim_ff,
        num_hidden_layers=a.depth,
        # no layer is a full-attention layer, so these describe nothing; kept consistent
        num_attention_heads=a.num_heads, num_key_value_heads=a.num_heads,
        head_dim=a.head_dim,
        layer_types=["linear_attention"] * a.depth,
        linear_num_key_heads=a.num_heads, linear_num_value_heads=a.num_heads,
        linear_key_head_dim=a.head_dim, linear_value_head_dim=a.head_dim,
        linear_conv_kernel_dim=a.conv,
        # every MLP dense: no expert is built either way, both conditions say so
        mlp_only_layers=list(range(a.depth)), num_experts=0, num_experts_per_tok=1,
        decoder_sparse_step=1, moe_intermediate_size=0, shared_expert_intermediate_size=0,
        hidden_act="silu", max_position_embeddings=a.max_position, rms_norm_eps=1e-5,
        tie_word_embeddings=False, rope_theta=10000.0, attention_bias=False,
        torch_dtype=torch.bfloat16)
    torch.manual_seed(a.seed)
    torch.set_default_dtype(torch.bfloat16)
    m = Qwen3NextForCausalLM(cfg)
    m.save_pretrained(a.out, safe_serialization=True)
    print(f"saved {a.out}: {sum(p.numel() for p in m.parameters()) / 1e9:.4f}B params")


if __name__ == "__main__":
    main()
