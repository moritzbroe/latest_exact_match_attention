"""Vocabulary and the encoding enc, as defined in the appendix.

    enc(a_1, ..., a_k) = mem bits_w(0) : bits_w(a_1) # ... # mem bits_w(k-1) : bits_w(a_k)

which is k blocks of 2w+2 tokens separated by k-1 occurrences of #, i.e.
k(2w+3)-1 tokens. Tokens are represented by their index in VOCAB.
"""

VOCAB = ["0", "1", "#", ":", "mem", "reg", "pc", "<out>", "<eos>", "step"]
TOK = {s: i for i, s in enumerate(VOCAB)}


def bits_w(a, w):
    """w-bit binary representation of a, most significant bit first."""
    if not 0 <= a < 1 << w:
        raise ValueError(f"{a} is not in [2^{w}]")
    return [TOK["1"] if (a >> (w - 1 - i)) & 1 else TOK["0"] for i in range(w)]


def unbits_w(toks, w):
    if len(toks) != w:
        raise ValueError(f"expected {w} bit tokens, got {len(toks)}")
    a = 0
    for t in toks:
        if t not in (TOK["0"], TOK["1"]):
            raise ValueError(f"{VOCAB[t]!r} is not a bit token")
        a = 2 * a + (t == TOK["1"])
    return a


def enc(words, w):
    k = len(words)
    if k > 1 << w:
        raise ValueError(f"{k} words do not fit in [2^{w}] addresses")
    out = []
    for i, a in enumerate(words):
        if i:
            out.append(TOK["#"])
        out += [TOK["mem"]] + bits_w(i, w) + [TOK[":"]] + bits_w(a, w)
    return out


def encode_input(x, w):
    """enc(n, x_1, ..., x_n) for an input x of length n."""
    return enc([len(x)] + list(x), w)


def decode_output(toks, w):
    """Inverse of enc, returning (m, y). Checks the block structure and that
    the first word m matches the number of words that follow."""
    block = 2 * w + 3
    if (len(toks) + 1) % block != 0:
        raise ValueError("not a whole number of blocks")
    k = (len(toks) + 1) // block
    words = []
    for i in range(k):
        s = i * block
        if i and toks[s - 1] != TOK["#"]:
            raise ValueError(f"missing # before block {i}")
        if toks[s] != TOK["mem"]:
            raise ValueError(f"block {i} does not start with mem")
        if unbits_w(toks[s + 1:s + 1 + w], w) != i:
            raise ValueError(f"block {i} has wrong address")
        if toks[s + 1 + w] != TOK[":"]:
            raise ValueError(f"missing : in block {i}")
        words.append(unbits_w(toks[s + 2 + w:s + 2 + 2 * w], w))
    m = words[0]
    if m != k - 1:
        raise ValueError(f"m = {m} but {k - 1} words follow")
    return m, words[1:]
