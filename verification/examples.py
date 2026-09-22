"""Example word-RAMs and random LEMA transformers to test conversions against."""

import numpy as np

from encoding import TOK, VOCAB
from lema import LemaTransformer
from wordram import WordRAM

# --- word-RAMs, roughly in order of complexity -------------------------------
# Each entry is (program, r, w, [inputs]). All halt on the given inputs.


def asm(lines, tr):
    """Resolve labels: ("label", name) marks a position and ("jnz", i, "name")
    expands to  set tr, addr(name) ; jnz i, tr.  Plain instructions pass through."""
    addr, n = {}, 0
    for line in lines:
        if line[0] == "label":
            addr[line[1]] = n
        else:
            n += 2 if line[0] == "jnz" and isinstance(line[2], str) else 1
    out = []
    for line in lines:
        if line[0] == "label":
            continue
        if line[0] == "jnz" and isinstance(line[2], str):
            out += [("set", tr, addr[line[2]]), ("jnz", line[1], tr)]
        else:
            out.append(line)
    return out


_CONST = ([("set", 0, 7),        # mem[1] <- 7, mem[0] <- 1
           ("set", 1, 1),
           ("store", 1, 0),
           ("set", 2, 0),
           ("store", 2, 1),
           ("halt",)], 3, 4, [[], [1, 2]])

_DOUBLE = (asm([("set", 3, 1),   # mem[i] <- 2 mem[i] for i = 1..n
                ("set", 6, 0),
                ("load", 0, 6),
                ("set", 1, 0),
                ("label", "loop"),
                ("op", 4, 1, 0, "<"),
                ("jnz", 4, "body"),
                ("jnz", 3, "end"),
                ("label", "body"),
                ("op", 1, 1, 3, "+"),
                ("load", 2, 1),
                ("op", 2, 2, 2, "+"),
                ("store", 1, 2),
                ("jnz", 3, "loop"),
                ("label", "end"),
                ("halt",)], 7), 8, 5, [[3], [1, 2, 3, 4]])

_SUM = (asm([("set", 3, 1),      # mem[0] <- 1, mem[1] <- sum of the input
             ("set", 5, 0),
             ("load", 0, 5),
             ("set", 1, 0),
             ("set", 2, 0),
             ("label", "loop"),
             ("op", 4, 1, 0, "<"),
             ("jnz", 4, "body"),
             ("jnz", 3, "end"),
             ("label", "body"),
             ("op", 1, 1, 3, "+"),
             ("load", 4, 1),
             ("op", 2, 2, 4, "+"),
             ("jnz", 3, "loop"),
             ("label", "end"),
             ("set", 5, 1),
             ("store", 5, 2),
             ("set", 5, 0),
             ("store", 5, 3),
             ("halt",)], 6), 7, 5, [[3, 5, 2], [7], [30, 10]])   # 30+10 wraps

_MAX = (asm([("set", 3, 1),      # mem[0] <- 1, mem[1] <- max of the input
             ("set", 6, 0),
             ("load", 0, 6),
             ("set", 1, 0),
             ("set", 2, 0),
             ("label", "loop"),
             ("op", 4, 1, 0, "<"),
             ("jnz", 4, "body"),
             ("jnz", 3, "end"),
             ("label", "body"),
             ("op", 1, 1, 3, "+"),
             ("load", 4, 1),
             ("op", 5, 2, 4, "<"),
             ("jnz", 5, "upd"),
             ("jnz", 3, "loop"),
             ("label", "upd"),
             ("op", 2, 4, 6, "+"),
             ("jnz", 3, "loop"),
             ("label", "end"),
             ("set", 5, 1),
             ("store", 5, 2),
             ("store", 6, 3),
             ("halt",)], 7), 8, 5, [[3, 5, 2], [9, 1, 4, 8]])

_MUL = ([("set", 6, 0),          # mem[0] <- 1, mem[1] <- x_1 * x_2
         ("set", 3, 1),
         ("set", 1, 1),
         ("load", 4, 1),
         ("op", 1, 1, 3, "+"),
         ("load", 5, 1),
         ("op", 2, 4, 5, "*"),
         ("set", 1, 1),
         ("store", 1, 2),
         ("store", 6, 3),
         ("halt",)], 7, 6, [[5, 7], [3, 3], [63, 63]])   # 63*63 wraps mod 64

_ALU = ([("set", 0, 1),          # exercise every operation not covered above
         ("load", 1, 0),
         ("set", 0, 2),
         ("load", 2, 0),
         *[("op", 3 + i, 1, 2, op)
           for i, op in enumerate(["-", "<<", ">>", "&", "|", "^", "<=", "==", "!="])],
         ("halt",)], 12, 4,
        # shift amounts above, exactly at, and below the word size, and by 0
        [[13, 3], [2, 5], [7, 7], [0, 15], [9, 4], [9, 0]])

_MSB = ([("set", 0, 1),          # mem[1] <- msb(mem[1]), mem[0] <- 1
         ("load", 1, 0),
         ("msb", 2, 1),
         ("store", 0, 2),
         ("set", 0, 0),
         ("set", 3, 1),
         ("store", 0, 3),
         ("halt",)], 4, 4, [[0], [1], [2], [3], [8], [15]])

_EDGE = ([("set", 0, 3),         # missing mem[3] is 0; the final write falls through
          ("load", 1, 0),
          ("set", 0, 0),
          ("store", 0, 1)], 2, 2, [[]])

_MZERO = ([("set", 0, 0),        # rewrite the length cell mem[0] mid-run,
           ("set", 1, 3),        # then read it back and fill mem[1..3] <- 3
           ("store", 0, 1),
           ("load", 2, 0),
           ("set", 0, 1), ("store", 0, 2),
           ("set", 0, 2), ("store", 0, 2),
           ("set", 0, 3), ("store", 0, 2),
           ("halt",)], 3, 4, [[7]])

_GRAY = ([("set", 0, 1),         # a larger word size: mem[1] <- x_1 ^ (x_1 >> 1)
          ("load", 1, 0),
          ("set", 2, 1),
          ("op", 3, 1, 2, ">>"),
          ("op", 3, 1, 3, "^"),
          ("store", 0, 3),
          ("set", 0, 0),
          ("store", 0, 2),
          ("halt",)], 4, 8, [[201]])

_UNIT = ([("store", 0, 0)], 1, 1, [[], [1]])   # w = 1; halt by falling through

_DYN = ([("set", 5, 1),          # dispatch on x_1 in {0,1,2} via a computed jump,
         ("set", 6, 14),         # then halt by jumping out of range; line 20 has
         ("set", 1, 1),          # an out-of-range target that is not taken
         ("load", 0, 1),
         ("op", 0, 0, 0, "+"),
         ("set", 2, 8),
         ("op", 2, 2, 0, "+"),
         ("jnz", 5, 2),          # goto 8 + 2 x_1
         ("set", 3, 5),
         ("jnz", 5, 6),
         ("set", 3, 9),
         ("jnz", 5, 6),
         ("set", 3, 13),
         ("jnz", 5, 6),
         ("set", 1, 1),          # tail: mem[1] <- r3, mem[0] <- 1
         ("store", 1, 3),
         ("set", 1, 0),
         ("set", 2, 1),
         ("store", 1, 2),
         ("set", 2, 31),
         ("jnz", 4, 2),          # not taken (r4 = 0)
         ("jnz", 5, 2)], 7, 5, [[0], [1], [2]])

RAMS = {name: (WordRAM(p, r, w), xs) for name, (p, r, w, xs) in
        [("const", _CONST), ("double", _DOUBLE), ("sum", _SUM),
         ("max", _MAX), ("mul", _MUL), ("alu", _ALU), ("msb", _MSB), ("edge", _EDGE),
         ("mzero", _MZERO), ("gray", _GRAY), ("unit", _UNIT), ("dyn", _DYN)]}


# --- random LEMA transformers ------------------------------------------------

def random_transformer(precision="int20", seed=0, L=3, H=2, d=12, d_h=5, d_ff=16):
    """A small random transformer whose first prediction is always <eos>."""
    rng = np.random.default_rng(seed)
    tern = lambda *s: rng.integers(-1, 2, size=s)
    layers = [{"heads": [(tern(d_h, d), tern(d_h, d), tern(d_h, d), tern(d, d_h))
                         for _ in range(H)],
               "W1": tern(d_ff, d), "b": tern(d_ff),
               "W2": tern(d, d_ff)} for _ in range(L)]
    emb = tern(len(VOCAB), d)
    emb[:, 0] = 1
    for layer in layers:
        for _, _, _, WO in layer["heads"]:
            WO[0, :] = 0
        layer["W2"][0, :] = 0
    unemb = np.zeros((len(VOCAB), d), dtype=np.int64)
    unemb[TOK["<eos>"], 0] = 1
    # random parameters may well produce sgn(0) or a tied argmax, both of which are
    # legal for an arbitrary transformer, so strict is off here
    return LemaTransformer(emb, unemb, layers, precision=precision, strict=False)


def output_transformer(precision="fp16"):
    """A tiny transformer generating <out>, 1, <eos> from the prompt 0."""
    d, dh = 3, 1
    emb = np.zeros((len(VOCAB), d), dtype=np.int64)
    emb[TOK["0"], 0] = 1
    emb[TOK["<out>"], 1] = 1
    emb[TOK["1"], 2] = 1
    unemb = np.zeros_like(emb)
    unemb[TOK["<out>"], 0] = 2
    unemb[TOK["1"], 1] = 2
    unemb[TOK["<eos>"], 2] = 2
    Zq = np.ones((dh, d), dtype=np.int64)
    Zv = np.zeros((dh, d), dtype=np.int64)
    Zo = np.zeros((d, dh), dtype=np.int64)
    layer = {"heads": [(Zq, Zq, Zv, Zo)],
             "W1": np.zeros((1, d), dtype=np.int64),
             "b": np.zeros(1, dtype=np.int64),
             "W2": np.zeros((d, 1), dtype=np.int64)}
    return LemaTransformer(emb, unemb, [layer], precision=precision, strict=False)


def random_prompt(n=6, seed=0):
    return list(np.random.default_rng(seed + 99).integers(0, len(VOCAB), n))


TRANSFORMERS = [(p, random_transformer(precision=p, seed=i), random_prompt(seed=i))
                for i, p in enumerate(["int20", "fp16", "fp32", "fp64"])]

OUTPUT_TRANSFORMER = (output_transformer(), [TOK["0"]])
