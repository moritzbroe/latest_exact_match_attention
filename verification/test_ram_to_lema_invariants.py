"""Check the forward proof's stage contracts and exhaustive small-word ALUs.

The transcript tests inspect actual matrix-evaluated residual states. The ALU
tests apply the neurons of each emitted layer to every operand pair for w=2,...,6
and compare with the independent word-RAM operations, including every stage
of the carry-save multiplication invariant.
"""

import itertools

import numpy as np

import ram_to_lema as C
from encoding import TOK, encode_input, unbits_w
from examples import RAMS
from wordram import WordRAM, _ops


def words(x, f, name, w):
    return ((x[..., f[name]] > 0) * (1 << np.arange(w - 1, -1, -1))).sum(axis=-1)


def apply_neurons(x, layer):
    """Evaluate the neurons of a layer simultaneously, without large dense batches."""
    delta = np.zeros_like(x)
    for cond, updates in layer.neurons:
        match = np.all(x[:, [j for j, _ in cond]] == [v for _, v in cond], axis=1)
        for col, change in updates.items():
            delta[match, col] += change
    out = x + delta
    assert np.all(np.abs(out) == 1), "neuron update left the binary state space"
    return out


def check_alu():
    total = 0
    for w in range(2, 7):
        f = C.layout(w)
        pairs = np.array(list(itertools.product(range(1 << w), repeat=2)))
        a, b = pairs.T
        for op in C.OPS:
            x = np.repeat(C.embeddings(f)[:1], len(pairs), axis=0)
            x[:, f.index("op", C.OPS.index(op))] = 1
            x[:, f.index("ctrl")] = 1
            for k, name in enumerate(("A", "B")):
                x[:, f[name]] = 2 * ((pairs[:, k, None] >> np.arange(w - 1, -1, -1)) & 1) - 1
            if op in ("<", "<=", "==", "!="):
                x[:, f.index("cmp")] = 1
            for stage in range(w):
                layer = C.Layer(f)
                C.add_alu_stage(layer, stage, w, 1)
                x = apply_neurons(x, layer)
                assert np.array_equal(words(x, f, "A", w), a), (w, op, "A changed")
                assert np.array_equal(words(x, f, "B", w), b), (w, op, "B changed")
                if op == "*":
                    modulus = 1 << (stage + 1)
                    carry_sum = words(x, f, "X", w) + words(x, f, "Y", w)
                    assert np.array_equal(carry_sum, a * (b % modulus) // modulus)
                    assert np.all(carry_sum <= a)
                    assert np.array_equal(words(x, f, "Q", w), (a * b) % modulus)
            finish = C.Layer(f)
            C.add_alu_finish(finish, w)
            x = apply_neurons(x, finish)
            fn = (lambda aa, bb: max(0, aa.bit_length() - 1)) if op == "msb" else _ops(w)[op]
            want = np.array([fn(int(aa), int(bb)) for aa, bb in pairs])
            assert np.array_equal(words(x, f, "X", w), want), (w, op)
            total += len(pairs)
        print(f"  w={w}: all 13 operations on every operand pair; stage invariants passed", flush=True)
    print(f"  {total} ALU operand/operation cases passed", flush=True)


def check_trace(name, M, inp):
    w, f = M.w, C.layout(M.w)
    expected = C.expected_transcript(M, inp)
    tokens = np.array(expected[:-1])
    T = C.ram_to_lema(M)
    predictions, info = T.forward(tokens, return_trace=True)
    n = len(encode_input(inp, w))
    assert np.array_equal(predictions[n - 1:], expected[n:]), (name, "next-token prediction")
    layers = [a["x"] for a in info["layers"]]
    windowed, collected, read = layers[w - 1], layers[w], layers[w + 2]
    arithmetic, descriptor = layers[2 * w + 3], layers[2 * w + 4]

    # After layer w every position holds the bits of the w tokens before it,
    # with 0 for every token other than 1 and for positions before the start.
    ones = np.concatenate([np.zeros(w, dtype=np.int64), (tokens == TOK["1"]).astype(np.int64)])
    weights = 1 << np.arange(w - 1, -1, -1)
    window = np.array([(ones[i:i + w] * weights).sum() for i in range(len(tokens))])
    assert np.array_equal(words(windowed, f, "win", w), window), (name, "window")
    for later in layers[w:]:
        assert np.array_equal(later[:, f["win"]], windowed[:, f["win"]]), (name, "win")

    # Parse records from the actual token syntax, independently of role flags.
    out = int(np.flatnonzero(tokens == TOK["<out>"])[0])
    record_positions = np.flatnonzero(tokens == TOK["#"])
    for i in record_positions:
        addr = unbits_w(tokens[i - 2 * w - 1:i - w - 1], w)
        val = unbits_w(tokens[i - w:i], w)
        assert words(windowed[i], f, "win", w) == val
        if i < out:
            assert words(collected[i], f, "addr", w) == addr
            assert (collected[i, f.index("wtype")] > 0) == (tokens[i - 2 * w - 2] == TOK["reg"])
        else:
            assert words(collected[i], f, "X", w) == addr + 1
    noncommits = np.ones(len(tokens), dtype=bool)
    noncommits[record_positions[record_positions < out]] = False
    assert np.all(collected[noncommits, f["addr"]] == -1)
    assert np.all(collected[noncommits, f.index("wtype")] == -1)
    for later in layers[w + 1:]:
        for field in ("addr", "wtype"):
            assert np.array_equal(later[:, f[field]], collected[:, f[field]]), (name, field)
    for later in layers[w + 3:]:
        for field in ("A", "B"):
            assert np.array_equal(later[:, f[field]], read[:, f[field]]), (name, field)

    pc, regs, mem = M.initial_config(inp)
    for pos in np.flatnonzero(tokens == TOK["step"]):
        assert pc is not None
        ins = M.program[pc]
        kind = ins[0]
        aa = bb = 0
        if kind == "op":
            aa, bb = regs[ins[2]], regs[ins[3]]
        elif kind in ("msb", "load"):
            aa = regs[ins[2]]
        elif kind in ("store", "jnz"):
            aa, bb = regs[ins[1]], regs[ins[2]]
        assert words(collected[pos], f, "win", w) == pc
        assert words(collected[pos], f, "Z", w) == 0
        assert words(read[pos], f, "A", w) == aa
        assert words(read[pos], f, "B", w) == bb
        loaded = mem.get(aa, 0) if kind == "load" else 0
        assert words(read[pos], f, "X", w) == loaded
        assert words(read[pos], f, "Y", w) == 0
        assert words(read[pos], f, "Q", w) == 0
        want_result = loaded
        if kind == "op":
            want_result = M.ops[ins[4]](aa, bb)
        elif kind == "msb":
            want_result = max(0, aa.bit_length() - 1)
        assert words(arithmetic[pos], f, "X", w) == want_result
        pc, write = M.step(pc, regs, mem)
        assert (descriptor[pos, f.index("write")] > 0) == (write is not None)
        assert (descriptor[pos, f.index("stop")] > 0) == (pc is None)
        if write is not None:
            dest, addr, val = write
            assert words(descriptor[pos], f, "Y", w) == addr
            assert words(descriptor[pos], f, "X", w) == val
            assert (descriptor[pos, f.index("wreg")] > 0) == (dest == "reg")
        if pc is not None:
            assert words(descriptor[pos], f, "Z", w) == pc
    assert pc is None

    output_sources = [out + 1, *[i for i in record_positions if i > out]]
    m = mem.get(0, 0)
    assert len(output_sources) == m + 1
    for addr, pos in enumerate(output_sources):
        assert words(read[pos], f, "X", w) == addr
        assert words(read[pos], f, "A", w) == mem.get(addr, 0)
        assert words(descriptor[pos], f, "Y", w) == addr
        assert words(descriptor[pos], f, "X", w) == mem.get(addr, 0)
        assert descriptor[pos, f.index("output")] == 1
        assert (descriptor[pos, f.index("last")] > 0) == (addr == m)
    nonsources = tokens != TOK["step"]
    nonsources[output_sources] = False
    for field in ("X", "Y", "Z", "write", "wreg", "stop", "output", "last"):
        assert np.all(descriptor[nonsources, f[field]] == -1), (name, "non-ctrl", field)
    print(f"  {name} {inp}: record, operand, arithmetic, descriptor and output contracts passed", flush=True)


def main():
    check_alu()
    for name, (M, inputs) in RAMS.items():
        for inp in inputs:
            check_trace(name, M, inp)
    edge_cases = [
        ("immediate halt", WordRAM([("halt",)], 1, 2), []),
        ("maximum input/output", WordRAM([("halt",)], 1, 2), [0, 1, 3]),
        ("final register write", WordRAM([("set", 0, 3)], 1, 2), []),
        ("final memory write", WordRAM([("set", 0, 1), ("store", 0, 0)], 1, 2), [3]),
        ("full program jump", WordRAM([("set", 0, 1), ("set", 1, 3),
                                       ("jnz", 0, 1), ("halt",)], 2, 2), []),
    ]
    for name, M, inp in edge_cases:
        check_trace(name, M, inp)
    print("all checks passed")


if __name__ == "__main__":
    main()
