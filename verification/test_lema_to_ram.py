"""Check the generic LEMA-to-RAM compiler and its analytic resource bounds."""

import sys

import numpy as np

import lema_to_ram as C
from encoding import TOK, VOCAB
from examples import OUTPUT_TRANSFORMER, TRANSFORMERS, random_transformer
from lema import LemaTransformer
from wordram import WordRAM


class ObservedRAM(WordRAM):
    """Expose the final memory for comparison with direct transformer states."""

    def initial_config(self, x):
        pc, regs, mem = super().initial_config(x)
        self.observed_memory = mem
        return pc, regs, mem


def check_numeric_state(T, M, cells, reference):
    """Compare packed residuals and every latest-value entry, not only logits."""
    dims, fmt = T.dims(), C.fmt_for(T.precision)
    base, mem = (1 << M.w) - cells.size, M.observed_memory
    packed = lambda values: [fmt.pack(float(value)) for value in values]
    actual = [mem.get(base + cells.at["X"] + j, 0) for j in range(dims["d"])]
    assert actual == packed(reference["layers"][-1]["x"][-1]), "final residual differs"
    for li, layer in enumerate(reference["layers"]):
        for hi, head in enumerate(layer["heads"]):
            expected = {tuple(k > 0): packed(v) for k, v in zip(head["k"], head["v"])}
            found = {}
            root = mem.get(base + cells.at["R"] + li * dims["H"] + hi, 0)
            stack = [(root, ())]
            while stack:
                ptr, bits = stack.pop()
                if not ptr:
                    continue
                if len(bits) == dims["d_h"]:
                    found[bits] = [mem.get(ptr + j, 0) for j in range(dims["d_h"])]
                else:
                    stack.extend((mem.get(ptr + bit, 0), (*bits, bool(bit))) for bit in (0, 1))
            assert found == expected, f"dictionary differs at layer {li}, head {hi}"


class Check:
    def __init__(self):
        self.fails = []

    def note(self, name, got, bound=None, exact=False):
        if bound is None:
            print(f"      {name:11s} {got}")
            return
        ok = got == bound if exact else got <= bound
        relation = "=" if exact else "<="
        print(f"      {name:11s} {got}  {relation} derived {bound}   "
              f"{'ok' if ok else 'FAIL'}")
        if not ok:
            self.fails.append(
                f"{name} = {got}, expected {relation} {bound}")

    def fail(self, msg):
        self.fails.append(msg)
        print(f"      FAIL  {msg}")


def generated(T, prompt, limit=8):
    full = T.generate(prompt, limit, stop=TOK["<eos>"])
    if full[-1] != TOK["<eos>"]:
        raise ValueError(f"transformer did not generate <eos> within {limit} tokens")
    return full


def actual_parameters(T):
    """Count the stored scalar slots directly, independently of the formula."""
    total = T.emb.size + T.unemb.size
    for layer in T.layers:
        total += sum(A.size for head in layer["heads"] for A in head)
        total += sum(layer[name].size for name in ("W1", "b", "W2"))
    return total


def check_trace(name, T, prompt, chk):
    dims = T.dims()
    full = generated(T, prompt)
    n, want = len(prompt), full[len(prompt):]
    _, tinfo = T.forward(full[:-1], return_trace=True)
    print(f"  {name}  L={dims['L']} d={dims['d']} H={dims['H']} "
          f"d_h={dims['d_h']} d_ff={dims['d_ff']}  prompt={n}")
    print(f"      transformer -> {len(want)} tokens: "
          f"{' '.join(VOCAB[t] for t in want)}   s_T={tinfo['s_T']}")

    trace = True
    w0 = C.REQUIRED_WORD(T, tinfo["s_T"], n, len(want), trace)
    wB = C.WORD(T, n + len(want), trace)
    chk.note("parameters", actual_parameters(T), C.PARAMETERS(T), exact=True)
    chk.note("w_required", w0)
    chk.note("w_budget", wB)
    if wB < w0:
        chk.fail(f"{name}: theorem word size {wB} below the required {w0}")
    machines = [C.lema_to_ram(T, w=w, trace_output=trace)
                for w in (w0, w0 + 8, wB)]
    machines = [ObservedRAM(M.program, M.r, M.w) for M in machines]
    if any(M.program != machines[0].program for M in machines[1:]):
        chk.fail(f"{name}: emitted program depends on the word size")
    chk.note("registers", machines[0].r, C.REGISTERS(T), exact=True)
    chk.note("|P|", len(machines[0].program),
             C.PROGRAM_LEN(T, trace), exact=True)

    for M in machines:
        ran = M.run_on(list(prompt), max_steps=10 ** 9)
        if ran is False:
            return chk.fail(f"{name}: W={M.w}: exceeded 10^9 steps")
        (m, y), rinfo = ran
        if (m, y) != (len(want), list(want)):
            return chk.fail(f"{name}: W={M.w}: output {(m, y)} != "
                            f"{(len(want), list(want))}")
        print(f"      W={M.w}: output ok")
        chk.note("steps", rinfo["steps"],
                 C.STEPS(T, n, len(want), trace))
        chk.note("space", rinfo["space"],
                 C.SPACE(T, tinfo["s_T"], n, len(want)))
    _, cells, _ = C._compile(T, trace)
    check_numeric_state(T, machines[0], cells, tinfo)
    print("      residual and all dictionaries are bit-identical")


def check_payload(chk):
    """Check the formal <out> payload convention, not just prediction tracing."""
    T, prompt = OUTPUT_TRANSFORMER
    full = generated(T, prompt)
    want = [TOK["1"]]
    if full[len(prompt):] != [TOK["<out>"], *want, TOK["<eos>"]]:
        return chk.fail("output transformer has the wrong reference transcript")
    _, tinfo = T.forward(full[:-1])
    n, t, m = len(prompt), len(full) - len(prompt), len(want)
    w = C.REQUIRED_WORD(T, tinfo["s_T"], n, m, False)
    M = C.lema_to_ram(T, w=w, trace_output=False)
    chk.note("payload |P|", len(M.program),
             C.PROGRAM_LEN(T, False), exact=True)
    ran = M.run_on(prompt, max_steps=10 ** 8)
    if ran is False:
        return chk.fail("payload simulation did not halt")
    (got_m, got_y), rinfo = ran
    if (got_m, got_y) != (m, want):
        return chk.fail(f"payload simulation produced {(got_m, got_y)}")
    chk.note("payload steps", rinfo["steps"], C.STEPS(T, n, t, False))
    chk.note("payload space", rinfo["space"],
             C.SPACE(T, tinfo["s_T"], n, m))


def check_unused_prompt_logits():
    """Generation must not evaluate logits at earlier prompt positions."""
    emb = np.zeros((len(VOCAB), 2))
    emb[TOK["0"]] = (65504, 0)
    emb[TOK["1"]] = (0, 1)
    emb[TOK["<out>"]] = (1, 0)
    unemb = np.zeros_like(emb)
    unemb[TOK["<out>"]] = (0, 1)
    unemb[TOK["<eos>"]] = (2, 0)
    zero_12 = np.zeros((1, 2))
    zero_21 = np.zeros((2, 1))
    layer = {"heads": [(zero_12, zero_12, zero_12, zero_21)],
             "W1": zero_12, "b": np.zeros(1), "W2": zero_21}
    T = LemaTransformer(emb, unemb, [layer], precision="fp16", strict=False)

    try:
        T.forward([TOK["0"], TOK["1"]])
    except OverflowError:
        pass
    else:
        raise AssertionError("all-position diagnostic forward should see the unused overflow")
    pred, _ = T.forward([TOK["0"], TOK["1"]], final_only=True)
    assert pred.tolist() == [TOK["<out>"]]
    assert T.generate([TOK["0"], TOK["1"]], 2, stop=TOK["<eos>"]) == [
        TOK["0"], TOK["1"], TOK["<out>"], TOK["<eos>"]]
    M = C.lema_to_ram(T, w=C.WORD(T, 2), trace_output=False)
    ran = M.run_on([TOK["0"], TOK["1"]], max_steps=10 ** 7)
    assert ran is not False and ran[0] == (0, [])
    print("  unused early-prompt logit overflow is skipped during generation")


def check_size_grid(chk):
    """Check exact symbolic sizes on architectures unlike the run-time cases."""
    specs = [(1, 1, 1, 1, 1), (2, 3, 4, 2, 5),
             (4, 1, 3, 6, 2), (1, 4, 7, 1, 9)]
    for i, (L, H, d, dh, dff) in enumerate(specs):
        T = random_transformer("fp16", seed=20 + i, L=L, H=H, d=d,
                               d_h=dh, d_ff=dff)
        dims = T.dims()
        chk.note(f"grid {i} N", actual_parameters(T), C.PARAMETERS(T), exact=True)
        for trace in (False, True):
            program, cells, _ = C._compile(T, trace)
            chk.note(f"grid {i} P{int(trace)}", len(program),
                     C.PROGRAM_LEN(T, trace), exact=True)
            chk.note(f"grid {i} S", cells.size, C.STATIC_CELLS(T), exact=True)
        print(f"      grid {i}: L={dims['L']} H={dims['H']} d={dims['d']} "
              f"d_h={dims['d_h']} d_ff={dims['d_ff']}")


def check_fractional_state(chk):
    """Exercise rounding and q=k replacement with nonzero attention/MLPs.

    The fixture keeps its EOS logit fixed so all runs terminate, while the
    numerical assertions observe the rest of the residual and every cache.
    Head dimension and MLP width exceed d to exercise the head buffers.
    """
    for precision in ("fp16", "fp32", "fp64"):
        for seed in range(2):
            T0 = random_transformer(precision, seed=80 + seed,
                                    L=2, H=2, d=3, d_h=5, d_ff=9)
            rng = np.random.default_rng(seed)
            scale = lambda A: A.astype(np.float64) * 2.0 ** rng.integers(-5, 1, size=A.shape)
            emb = scale(T0.emb)
            emb[:, 0] = T0.emb[:, 0]  # preserve the fixture's terminating logit
            layers = []
            for layer in T0.layers:
                heads = []
                for Q, K, V, O in layer["heads"]:
                    query = scale(Q)
                    heads.append((query, query.copy(), scale(V), scale(O)))
                layers.append({"heads": heads,
                               **{p: scale(layer[p]) for p in ("W1", "b", "W2")}})
            T = LemaTransformer(emb, T0.unemb, layers, precision=precision, strict=False)
            check_trace(f"fractional {precision}/{seed}", T, [0, 1, 2, 0, 1, 2, 2, 0], chk)


def check_roundtrip(chk):
    """Compile a RAM through both constructions, including a final store."""
    from encoding import decode_output, encode_input
    from ram_to_lema import expected_transcript, ram_to_lema

    original = WordRAM([("store", 0, 0)], 1, 2)
    inp = [3]
    T = ram_to_lema(original)
    prompt = encode_input(inp, original.w)
    transcript = expected_transcript(original, inp)
    out = transcript.index(TOK["<out>"])
    payload = transcript[out + 1:-1]
    w = C.WORD(T, len(prompt) + len(payload), False)
    machine = C.lema_to_ram(T, w=w, trace_output=False)
    ran = machine.run_on(prompt, max_steps=10 ** 8)
    if ran is False:
        return chk.fail("round trip exceeded 10^8 steps")
    (m, actual), info = ran
    if (m, actual) != (len(payload), payload):
        return chk.fail("round trip produced the wrong encoded payload")
    if decode_output(actual, original.w) != original.run_on(inp)[0]:
        return chk.fail("round trip disagrees with the original RAM output")
    _, tinfo = T.forward(transcript[:-1])
    chk.note("round steps", info["steps"], C.STEPS(T, len(prompt), len(transcript) - len(prompt), False))
    chk.note("round space", info["space"], C.SPACE(T, tinfo["s_T"], len(prompt), len(payload)))
    print("      full RAM -> LEMA -> RAM round trip passed")


def main():
    chk = Check()
    for name, T, prompt in TRANSFORMERS:
        try:
            check_trace(name, T, prompt, chk)
        except Exception as exc:
            chk.fail(f"{name}: {type(exc).__name__}: {exc}")
    try:
        check_payload(chk)
    except Exception as exc:
        chk.fail(f"payload: {type(exc).__name__}: {exc}")
    try:
        check_unused_prompt_logits()
    except Exception as exc:
        chk.fail(f"unused prompt logits: {type(exc).__name__}: {exc}")
    try:
        check_size_grid(chk)
    except Exception as exc:
        chk.fail(f"size grid: {type(exc).__name__}: {exc}")
    try:
        check_fractional_state(chk)
    except Exception as exc:
        chk.fail(f"fractional state: {type(exc).__name__}: {exc}")
    try:
        check_roundtrip(chk)
    except Exception as exc:
        chk.fail(f"round trip: {type(exc).__name__}: {exc}")

    print()
    if chk.fails:
        print(f"{len(chk.fails)} failure(s):")
        for failure in chk.fails:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
