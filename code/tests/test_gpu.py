"""GPU tests: the capped STE split, the floor clamp, and store-vs-dense hard agreement.
Needs a CUDA GPU: python tests/test_gpu.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from lema import ModelConfig, Recall, Transformer
from lema.attention import lema, lema_soft

assert torch.cuda.is_available(), "GPU tests need a GPU"
torch.manual_seed(0)

# --- STE split at the function level: forward value at alpha_fwd, grads at alpha_bwd ----
q0 = torch.sign(torch.randn(2, 1, 64, 64, device="cuda"))
k0, v0 = q0.clone(), torch.randn(2, 1, 64, 64, device="cuda")
cot = torch.randn_like(v0)


def val_and_grads(fn):
    q, k, v = (t.clone().requires_grad_(True) for t in (q0, k0, v0))
    y = fn(q, k, v)
    y.backward(cot)
    return y.detach(), (q.grad, k.grad, v.grad)


y_split, g_split = val_and_grads(lambda q, k, v: lema(q, k, v, alpha_fwd=8.0,
                                                      alpha_bwd=2.0, c=32.0))
y_fwd, _ = val_and_grads(lambda q, k, v: lema_soft(q, k, v, 8.0, 32.0)[0])
_, g_bwd = val_and_grads(lambda q, k, v: lema_soft(q, k, v, 2.0, 32.0)[0])
assert (y_split - y_fwd).abs().max() < 1e-5, "split forward must equal soft(alpha_fwd)"
for a, b in zip(g_split, g_bwd):
    assert (a - b).abs().max() < 1e-5, "split backward must equal soft(alpha_bwd) grads"
print("test_gpu: STE split OK (value at alpha 8, grads at alpha 2)")

# --- through the model: the cap changes nothing about the forward value, changes the
# --- gradient reaching the query/key projections, and is inert below alpha -----------
task = Recall(16, key_vocab=256)
mcfg = ModelConfig(vocab_size=task.vocab_size, depth=2, dim=128, num_heads=1,
                   d_qk=64, d_v=64, attn="lema", pos="nope")
x, y, mask, _ = next(task.batches("train", 16, seed=0))
x, y, mask = x.cuda(), y.cuda(), mask.cuda()
model = Transformer(mcfg).cuda()
sd = {k: v.clone() for k, v in model.state_dict().items()}


def loss_and_qk_grad(alpha, cap):
    model.zero_grad(set_to_none=True)
    model.set_hardening(c=32.0, cap=cap, alpha=alpha)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = torch.nn.functional.cross_entropy(model(x)[mask].float(), y[mask])
    loss.backward()
    return loss.item(), model.blocks[0].attn.q_proj.weight.grad.clone()


l_capped, g_capped = loss_and_qk_grad(alpha=8.0, cap=2.0)
l_free, g_free = loss_and_qk_grad(alpha=8.0, cap=1e9)
# the split composes y_bwd + (y_fwd - y_bwd).detach() in bf16, so the forward equals
# the uncapped alpha_fwd forward only up to bf16 compose-rounding
assert abs(l_capped - l_free) < 2e-3, "cap must not change the forward value"
assert (g_capped - g_free).abs().max() > 1e-6, "cap must change the q/k gradient"
_, g_a = loss_and_qk_grad(alpha=1.0, cap=2.0)
_, g_b = loss_and_qk_grad(alpha=1.0, cap=1e9)
assert torch.equal(g_a, g_b), "a cap above alpha must be inert"
print("test_gpu: cap semantics OK (forward invariant, q/k gradient differs, inert below alpha)")

# --- store-vs-dense: the hash store must reproduce the dense hard forward --------------
model.load_state_dict(sd)
model.set_hardening(c=float(mcfg.d_qk - 1), cap=2.0, alpha=10.0)
model.set_hard(True)
model.eval()

# fp32, no autocast: the exact-op claim is bit-honest there, while bf16 reduction-order
# differences between the chunked and dense passes can flip a near-tied argmax
# Fed in ANY chunking -- one shot, 16 at a time, token by token -- the store must
# reproduce the dense forward: a chunk's queries see the chunk's earlier keys through the
# store itself, there is no in-chunk matching on the GPU.
with torch.no_grad():
    dense = model(x[:2])
    for step in (x.shape[1], 16, 1):
        store = model.new_cache("ram", max_cache_len=x.shape[1], batch=2)
        parts = [model(x[:2, s:s + step], cache=store, return_logits=True)
                 for s in range(0, x.shape[1], step)]
        chunked = torch.cat(parts, dim=1)
        pd, pc = dense[mask[:2]].argmax(-1), chunked[mask[:2]].argmax(-1)
        agree = (pd == pc).float().mean().item()
        assert agree == 1.0, f"store (chunks of {step}) must match dense ({agree:.3f})"
    assert store.entries() == store.counts().sum().item() > 0
    print("test_gpu: store-vs-dense OK, batch=2, whole sequence and chunks of 16 and 1")

# --- softmax baseline: chunked kv-cached forward must match the dense forward ----------
# bf16: the kv path is the flash kernel, which takes no fp32. Its fused rotary uses bf16
# cos/sin tables where the dense path's rope uses fp32 ones, so agreement is up to bf16
# noise, checked on the logits rather than on an argmax a random model holds by a hair.
# Also with grouped-query attention (4 query heads sharing 2 key/value heads): the dense
# path uses SDPA's enable_gqa, the cached path the flash kernel's native support.
for kv_heads, num_heads in ((2, 2), (2, 4)):
    sm = ModelConfig(vocab_size=task.vocab_size, depth=2, dim=64 * num_heads,
                     num_heads=num_heads, num_kv_heads=kv_heads, d_qk=64, d_v=64,
                     attn="softmax", pos="rope")
    smodel = Transformer(sm).cuda().to(torch.bfloat16).eval()
    with torch.no_grad():
        ref = smodel(x[:1]).float()
        kv = smodel.new_cache(max_cache_len=x.shape[1])
        parts = [smodel(x[:1, s:s + 16], cache=kv) for s in range(0, x.shape[1], 16)]
        got = torch.cat(parts, dim=1).float()
        one = smodel.new_cache(max_cache_len=x.shape[1])
        tok = [smodel(x[:1, s:s + 1], cache=one) for s in range(x.shape[1])]
        tokwise = torch.cat(tok, dim=1).float()
    scale = ref.abs().max().item()
    for name, o in [("chunked", got), ("token-by-token", tokwise)]:
        err = (ref - o).abs().max().item() / scale
        assert err < 0.05, f"kv {name} ({num_heads} q / {kv_heads} kv heads) must match dense to bf16 noise ({err:.3f} rel)"
    print(f"test_gpu: kv-cache chunked + token-by-token match dense (softmax+rope, bf16, {num_heads} q / {kv_heads} kv heads)")

# --- CUDA-graph decoders: a replay must reproduce the eager step it captured -----------
# Uncompiled, so the kernels are identical and the comparison is exact: this checks the
# capture (static buffers, the host round trip between graphs), not compile numerics.
from lema.decode import decoder

with torch.no_grad():
    for name, m, mk in [("softmax", smodel, lambda: smodel.new_cache(max_cache_len=64)),
                        ("lema", model, lambda: model.new_cache("ram", max_cache_len=64))]:
        prompt = x[:1, :8]
        toks = []
        for use_graph in (False, True):
            c = mk()
            m(prompt[:, :-1], cache=c, return_logits=False)
            if use_graph:
                d = decoder(m, c, tok=prompt[:, -1:])
                toks.append([d.step().item() for _ in range(40)])
            else:
                t = prompt[:, -1:]
                out = []
                for _ in range(40):
                    t = m(t, cache=c).argmax(-1)
                    out.append(t.item())
                toks.append(out)
        assert toks[0] == toks[1], f"{name} graph decoder must reproduce eager: {toks}"
print("test_gpu: CUDA-graph decoders reproduce eager decoding (softmax piecewise, lema)")
