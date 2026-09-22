"""Check the RAM-to-LEMA construction against the theorem's example word-RAMs."""

import sys

import numpy as np

import ram_to_lema as C
from encoding import TOK, VOCAB, decode_output, encode_input
from examples import RAMS
from lema import LemaTransformer


class Check:
    def __init__(self):
        self.fails = []

    def note(self, name, got, bound, exact=False):
        if bound is None:
            print(f"      {name:11s} {got}")
            return
        ok = got == bound if exact else got <= bound
        print(f"      {name:11s} {got}  {'=' if exact else '<='} derived {bound}   "
              f"{'ok' if ok else 'FAIL'}")
        if not ok:
            self.fails.append(f"{name} = {got}, derived {'=' if exact else '<='} {bound}")

    def fail(self, msg):
        self.fails.append(msg)
        print(f"      FAIL  {msg}")


def check_weights(T, w, chk):
    """prop:predictor: integer parameters of magnitude at most max(w+1, 5), from the sets
    the appendix names per matrix."""
    sets = {"emb": {-1, 1}, "unemb": {0, 1}, "W_Q": {-1, 0, 1}, "W_K": {-1, 0, 1},
            "W_V": {0, 1, 2}, "W_O": {0, 1}, "W1": set(range(-2, 3)), "W2": set(range(-2, 3))}
    mats = [("emb", T.emb), ("unemb", T.unemb)]
    for layer in T.layers:
        for head in layer["heads"]:
            mats += list(zip(("W_Q", "W_K", "W_V", "W_O"), head))
        mats += [("W1", layer["W1"]), ("W2", layer["W2"])]
    bound = max(w + 1, 5)
    for kind, A in mats:
        vals = set(np.unique(np.asarray(A, dtype=np.float64)).tolist())
        if not vals <= {float(v) for v in sets[kind]}:
            chk.fail(f"{kind} entries {sorted(vals - {float(v) for v in sets[kind]})} outside {sorted(sets[kind])}")
    for layer in T.layers:
        b = np.asarray(layer["b"], dtype=np.float64)
        if not (np.all(b == np.round(b)) and np.abs(b).max(initial=0) <= bound):
            chk.fail(f"b not integer or above max(w+1, 5) = {bound}")
    chk.note("params", "in the appendix's sets, |bias| <= max(w+1, 5)", None)


def check_one(name, M, x, chk, C):
    print(f"  {name}  |P|={len(M.program)} r={M.r} w={M.w}  x={x}")
    ran = M.run_on(x, max_steps=10 ** 6, return_trace=True)
    if ran is False:
        return chk.fail(f"{name} on {x}: the word-RAM did not halt")
    (m, y), rinfo = ran
    print(f"      RAM         -> (m={m}, y={y})  steps={rinfo['steps']} space={rinfo['space']}")

    # the reference transcript, and a check that it is itself the right thing
    tau = encode_input(x, M.w)
    expected = list(C.expected_transcript(M, x))
    n = len(tau)
    if expected[:n] != tau:
        return chk.fail(f"{name} on {x}: expected_transcript does not start with the prompt")
    if TOK["<out>"] not in expected or expected[-1] != TOK["<eos>"]:
        return chk.fail(f"{name} on {x}: expected_transcript has no <out> ... <eos> block")
    try:
        got = decode_output(expected[expected.index(TOK["<out>"]) + 1:-1], M.w)
    except ValueError as e:
        return chk.fail(f"{name} on {x}: output block does not decode ({e})")
    if got != (m, y):
        return chk.fail(f"{name} on {x}: expected_transcript encodes {got}, the RAM outputs {(m, y)}")

    T = C.ram_to_lema(M)
    check_weights(T, M.w, chk)
    d = T.dims()
    chk.note("|V|", d["vocab"], C.VOCAB_SIZE(M), exact=True)
    for key, claim in [("L", C.DEPTH), ("d", C.WIDTH), ("d_h", C.D_HEAD)]:
        chk.note(key, d[key], claim(M), exact=True)
    chk.note("H", d["H"], C.HEADS(M), exact=True)
    chk.note("d_ff", d["d_ff"], C.D_MLP(M), exact=True)

    # <eos> is the final prediction, never an input and therefore never a key.
    preds, tinfo = T.forward(expected[:-1])
    # positions before the last prompt token predict input data, which nothing can
    # know, so only the generated part is checked
    wrong = np.flatnonzero(preds[n - 1:] != np.array(expected[n:]))
    if len(wrong):
        i = n + int(wrong[0])
        ctx = " ".join(VOCAB[t] for t in expected[max(0, i - 10):i])
        chk.fail(f"{name} on {x}: wrong token at position {i} of {len(expected)}: "
                 f"predicted {VOCAB[preds[i - 1]]!r}, expected {VOCAB[expected[i]]!r}\n"
                 f"              ... {ctx} [{VOCAB[expected[i]]}] ...")

    # Theorem 1 claims a float transformer: the same construction run in IEEE
    # binary16 must predict identically (all intermediates are small integers).
    T16 = LemaTransformer(T.emb, T.unemb,
                          [{"heads": list(l["heads"]),
                            **{p: l[p] for p in ("W1", "b", "W2")}}
                           for l in T.layers],
                          precision="fp16", strict=True, binary=True)
    preds16, _ = T16.forward(expected[:-1])
    if not np.array_equal(preds16, preds):
        chk.fail(f"{name} on {x}: fp16 forward deviates from the integer mode")

    chk.note("t_T", len(expected) - n, C.TOKENS(M, rinfo["steps"], m))
    chk.note("s_T", tinfo["s_T"], C.KEYS(M, rinfo["space"]))
    chk.note("max_abs", tinfo["max_abs"], C.MAX_MAGNITUDE(M))


def main():
    chk = Check()
    for name, (M, xs) in RAMS.items():
        for x in xs:
            try:
                check_one(name, M, x, chk, C)
            except Exception as e:
                chk.fail(f"{name} on {x}: {type(e).__name__}: {e}")
    print()
    if chk.fails:
        print(f"{len(chk.fails)} failure(s):")
        for f in chk.fails:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
