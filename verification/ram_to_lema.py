"""Compile a word-RAM into the binary-state LEMA transformer of Theorem 1.

The transcript is

    enc(n,x) # pc bits(0) [ step (DEST bits(a) : bits(v) #)? pc bits(pc') ]*
    step (DEST bits(a) : bits(v) #)? <out> enc(m,y) <eos>.

A pre-output # commits the preceding record; step executes an instruction.

Layers 1 to w build a window.  Layer l copies, from the previous position,
the marker flag dist_{l-1} into dist_l and the bit win_{w+2-l} into
win_{w+1-l}, where win_{w+1} stands for the token 1, so afterwards every
position holds in win the bits of the w tokens before it.  Hence a : holds
the address of its record, a # its value and a step token its program
counter.  Layer w+1 moves the address, and the type read off near at the :,
from the : to the # of every committed record, and the address of an output
record into X of its #, where the next cell to emit is X+1.  Execution-phase
lookups always read registers and output-phase lookups always read memory, so
out doubles as the mem flag of every query, and load guards the second
hop, which exists exactly for load and always reads memory.  A jnz reads its
condition register into A and its target register into B, taken jumps write
B into the pc word Z, and a scan of B against |P| (sharing the comparison
flags) raises stop for out-of-range targets.

Every residual coordinate is +-1.  Every head copies fields: its query and
key are signed coordinates of the state, so the binarization is the identity,
its value is a field plus one, and the output matrix adds the value to a
destination field at its default -1, so a hit copies the field and a miss
leaves the default.  Every ReLU neuron fires, with output 1, on one +-1
condition and applies +-2 bit flips.  Neurons of one layer read the same
layer input, and no two neurons that write the same coordinate fire together.
"""

import itertools

import numpy as np

from encoding import TOK, VOCAB, bits_w, enc, encode_input
from lema import LemaTransformer

BLOCK = ["mem", "reg", ":", "pc"]
OPS = ["+", "-", "*", "<<", ">>", "&", "|", "^", "<", "<=", "==", "!=", "msb"]


# Exact architecture sizes and analytic execution bounds.  A neuron tests at
# most max(w+2,6) bits, so every pre-ReLU integer has magnitude at most
# max(2w+3,11).  Non-bias parameters lie in {-2,-1,0,1,2}; biases have
# magnitude at most max(w+1,5).
MAX_MAGNITUDE = lambda M: max(11, 2 * M.w + 3)
PRECISION = lambda M: f"int{MAX_MAGNITUDE(M).bit_length() + 1}"
VOCAB_SIZE = lambda M: len(VOCAB)
DEPTH = lambda M: 2 * M.w + 6
WIDTH = lambda M: 9 * M.w + 62
HEAD_DIM = lambda w: w + 5
D_HEAD = lambda M: HEAD_DIM(M.w)
HEADS = lambda M: 3


def D_MLP(M):
    """Exact padded MLP width, derived by counting the neurons.

    The three potentially largest layers are program preparation, an ALU
    stage, and final state assembly. Lookup and arithmetic have separate
    layer boundaries, and arithmetic finalization uses only 3w+5 neurons.
    """
    w, plen = M.w, len(M.program)
    has_range_check = int(plen < 1 << w)
    nonhalting = sum(ins[0] != "halt" for ins in M.program)
    preparation = plen + w + 2
    alu = 21 * w + 22 + has_range_check
    final = 11 * w + 4 + has_range_check + nonhalting
    return max(preparation, alu, final)


def TOKENS(M, t_M, m):
    """Exact worst-case generated-length bound (every RAM step writes)."""
    return t_M * (3 * M.w + 5) + (m + 1) * (2 * M.w + 3) + 2


def KEYS(M, s_M):
    """Analytically derived upper bound on distinct keys across all heads."""
    return 3 * s_M + 6 * M.w + 25


def pm_bits(a, w):
    """MSB-first +-1 representation."""
    return [1 if (a >> (w - 1 - i)) & 1 else -1 for i in range(w)]


class Fields:
    """Named coordinate ranges of a residual stream, allocated in order."""

    def __init__(self):
        self.at, self.d = {}, 0

    def add(self, name, width=1):
        self.at[name] = slice(self.d, self.d + width)
        self.d += width

    def __getitem__(self, name):
        return self.at[name]

    def index(self, name, i=0):
        return self.at[name].start + i


def layout(w):
    f = Fields()
    f.w = w
    f.add("one")
    f.add("tok", len(VOCAB))
    for j in range(w + 1):
        f.add(f"dist{j}")
    f.add("near", len(BLOCK))
    for name in ("prevout", "out", "commit", "exec", "outsep", "outstart",
                 "ctrl", "wtype"):
        f.add(name)

    # win holds the bits of the w preceding tokens at every position. addr and
    # wtype hold committed records and survive every later layer. A/B are
    # immutable operands; multiplication builds Q using the carry-save
    # workspace X/Y. Arithmetic finalization puts its result in X; assembly
    # supplies Y=address, Z=next pc.
    for name in ("addr", "win", "X", "Y", "Z", "A", "B", "Q"):
        f.add(name, w)

    # carry doubles as the borrow of subtraction, since + and - never run on
    # the same token
    for name in ("q1", "q2", "cmp", "iszero", "carry", "eq", "lt",
                 "write", "wreg", "stop", "output", "last",
                 "load", "store", "jnz"):
        f.add(name)
    f.add("op", len(OPS))
    f.add("next", len(VOCAB))
    return f


def embeddings(f):
    emb = -np.ones((len(VOCAB), f.d), dtype=np.int64)
    emb[:, f["one"]] = 1
    for t, name in enumerate(VOCAB):
        emb[t, f.index("tok", t)] = 1
        if name in BLOCK:
            emb[t, f["dist0"]] = 1
    emb[:, f["eq"]] = 1
    emb[:, f.index("next", TOK["0"])] = 1       # unique harmless default
    return emb


def unembeddings(f):
    U = np.zeros((len(VOCAB), f.d), dtype=np.int64)
    for t in range(len(VOCAB)):
        U[t, f.index("next", t)] = 1
    return U


class Layer:
    """Small declarative builder for heads and firing ReLU neurons."""

    def __init__(self, f):
        self.f, self.heads, self.neurons = f, [], []

    def fire(self, conditions, updates):
        """Add a neuron which is 1 exactly on the supplied +-1 condition."""
        cond = list(conditions)
        if len({i for i, _ in cond}) != len(cond):
            raise ValueError("a condition constrains one coordinate twice")
        self.neurons.append((cond, dict(updates)))

    def finish(self):
        d, n = self.f.d, max(1, len(self.neurons))
        W1, b = np.zeros((n, d), dtype=np.int64), np.zeros(n, dtype=np.int64)
        W2 = np.zeros((d, n), dtype=np.int64)
        for h, (conditions, updates) in enumerate(self.neurons):
            for i, value in conditions:
                W1[h, i] = value
            b[h] = 1 - len(conditions)
            for i, value in updates.items():
                W2[i, h] = value
        return {"heads": self.heads, "W1": W1, "b": b, "W2": W2}


def conditions(f, *flags):
    """Flags are names (meaning +1) or (name, +-1) pairs."""
    out = []
    for flag in flags:
        name, value = flag if isinstance(flag, tuple) else (flag, 1)
        out.append((f.index(name), value))
    return out


def word_conditions(f, name, value, w):
    return [(f.index(name, i), b) for i, b in enumerate(pm_bits(value, w))]


def reset_word(layer, name, base=()):
    f = layer.f
    for i in range(f[name].stop - f[name].start):
        layer.fire([*base, (f.index(name, i), 1)], {f.index(name, i): -2})


def assign_word(layer, dst, src, base=()):
    f = layer.f
    for i in range(f[dst].stop - f[dst].start):
        di, si = f.index(dst, i), f.index(src, i)
        layer.fire([*base, (di, -1), (si, 1)], {di: 2})
        layer.fire([*base, (di, 1), (si, -1)], {di: -2})


def word(f, name):
    s = f[name]
    return list(range(s.start, s.stop))


def copy_head(f, U, V, F, D):
    """The one kind of head: query x[U], key x[V], value x[F]+1, added to D.

    U and V are lists of (coordinate, sign) of equal length, F and D lists of
    coordinates of equal length.  Row r of W_Q and W_K holds the sign at the
    coordinate, so queries and keys are signed coordinates of the binary state
    and sgn is the identity.  Row r of W_V is e_{F_r} + e_one, so the value
    lies in {0, 2}, and W_O adds it to D_r.  Where D is at its default -1, a
    hit sets D_r to the copied coordinate and a miss leaves it.  Shorter
    queries and keys are padded with the constant one, shorter values with
    zero rows, up to the common head dimension.
    """
    if len(U) != len(V) or len(F) != len(D):
        raise ValueError("query/key and value/destination lists must pair up")
    d, d_h = f.d, HEAD_DIM(f.w)
    if max(len(U), len(F)) > d_h:
        raise ValueError("head exceeds the head dimension")
    Q = np.zeros((d_h, d), dtype=np.int64)
    K = np.zeros((d_h, d), dtype=np.int64)
    Vm = np.zeros((d_h, d), dtype=np.int64)
    O = np.zeros((d, d_h), dtype=np.int64)
    Q[:, f.index("one")] = K[:, f.index("one")] = 1
    for r, (c, s) in enumerate(U):
        Q[r, f.index("one")] = 0
        Q[r, c] = s
    for r, (c, s) in enumerate(V):
        K[r, f.index("one")] = 0
        K[r, c] = s
    for r, (src, dst) in enumerate(zip(F, D)):
        Vm[r, src] += 1
        Vm[r, f.index("one")] += 1
        O[dst, r] = 1
    return Q, K, Vm, O


def previous_head(f, srcs, dsts):
    """All keys coincide, hence LEMA returns the immediately previous token."""
    one = (f.index("one"), 1)
    return copy_head(f, [one], [one], srcs, dsts)


def latest_head(f, flag, srcs, dsts):
    """Copy from the latest position whose flag coordinate is +1."""
    return copy_head(f, [(f.index("one"), 1)], [(flag, 1)], srcs, dsts)


def guarded_latest_head(f, guard, flag, srcs, dsts):
    """As latest_head, but only at positions whose guard coordinate is +1."""
    one = (f.index("one"), 1)
    return copy_head(f, [(guard, 1), one], [one, (flag, 1)], srcs, dsts)


def memory_head(f, query_word, active, mem_flag, dst):
    """Read the latest committed (reg/mem,address)->word dictionary entry.

    Query (active, 1, mem_flag, word), key (1, commit, -wtype, addr): a query
    with active = -1 matches no key, and only positions with commit can be
    matched.  Since addr and wtype are written only at committing positions,
    every other position has the same key.  Queries use +1 for memory and -1
    for registers, whereas wtype uses +1 for registers.  Execution-phase
    queries read registers and output-phase queries read memory, so both
    ordinary lookups use out as their mem flag, and the second hop uses
    load for both roles.
    """
    one = (f.index("one"), 1)
    U = [(f.index(active), 1), one, (f.index(mem_flag), 1)]
    U += [(c, 1) for c in word(f, query_word)]
    V = [one, (f.index("commit"), 1), (f.index("wtype"), -1)]
    V += [(c, 1) for c in word(f, "addr")]
    return copy_head(f, U, V, word(f, "win"), word(f, dst))


def broadcast_head(f, name, extra=()):
    """Copy state from the latest control position into all other positions."""
    one = (f.index("one"), 1)
    src = word(f, name) + [f.index(e) for e in extra]
    return copy_head(f, [(f.index("ctrl"), -1), one],
                     [one, (f.index("ctrl"), 1)], src, src)


def structural_layer(f, w):
    layer = Layer(f)
    layer.heads = [
        previous_head(f, [f.index("dist0"), f.index("tok", TOK["1"]),
                          f.index("tok", TOK["<out>"])],
                      [f.index("dist1"), f.index("win", w - 1), f.index("prevout")]),
        latest_head(f, f.index("dist0"),
                    [f.index("tok", TOK[b]) for b in BLOCK], word(f, "near")),
        latest_head(f, f.index("tok", TOK["<out>"]),
                    [f.index("one")], [f.index("out")]),
    ]
    sep = (f.index("tok", TOK["#"]), 1)
    layer.fire([sep, (f.index("out"), -1)], {f.index("commit"): 2})
    layer.fire([(f.index("tok", TOK["step"]), 1)],
               {f.index("exec"): 2, f.index("ctrl"): 2})
    layer.fire([sep, (f.index("out"), 1)],
               {f.index("outsep"): 2, f.index("ctrl"): 2})
    layer.fire([(f.index("tok", TOK["mem"]), 1), (f.index("prevout"), 1)],
               {f.index("outstart"): 2, f.index("ctrl"): 2})
    return layer


def window_layer(f, ell, w):
    """Layer ell shifts the marker flag and the bit one position further."""
    layer = Layer(f)
    layer.heads = [previous_head(f, [f.index(f"dist{ell - 1}"), f.index("win", w + 1 - ell)],
                                 [f.index(f"dist{ell}"), f.index("win", w - ell)])]
    return layer


def records_layer(f, M):
    """Move each record's address (and type) from its : to its #."""
    layer = Layer(f)
    colon = f.index("tok", TOK[":"])
    layer.heads = [
        guarded_latest_head(f, f.index("commit"), colon,
                            word(f, "win") + [f.index("near", BLOCK.index("reg"))],
                            word(f, "addr") + [f.index("wtype")]),
        guarded_latest_head(f, f.index("outsep"), colon, word(f, "win"), word(f, "X")),
    ]
    add_program_prep(layer, M)
    return layer


def add_program_prep(layer, M):
    """Decode the pc into instruction flags and at most two register queries."""
    f, w = layer.f, M.w
    for pc, instr in enumerate(M.program):
        kind = instr[0]
        base = [*conditions(f, "exec"), *word_conditions(f, "win", pc, w)]
        updates = {}
        if kind in ("load", "store", "jnz"):
            updates[f.index(kind)] = 2
        if kind in ("set", "op", "msb", "load", "store"):
            updates[f.index("write")] = 2
        if kind in ("set", "op", "msb", "load"):
            updates[f.index("wreg")] = 2
        if kind == "set":
            pass
        elif kind == "op":
            _, _, j, j2, op = instr
            updates[f.index("q1")] = updates[f.index("q2")] = 2
            updates[f.index("op", OPS.index(op))] = 2
            if op in ("<", "<=", "==", "!="):
                updates[f.index("cmp")] = 2
            for name, value in (("X", j), ("Y", j2)):
                for i, bit in enumerate(pm_bits(value, w)):
                    if bit == 1:
                        updates[f.index(name, i)] = 2
        elif kind == "msb":
            _, _, j = instr
            updates[f.index("q1")] = 2
            updates[f.index("op", OPS.index("msb"))] = 2
            for i, bit in enumerate(pm_bits(j, w)):
                if bit == 1:
                    updates[f.index("X", i)] = 2
        elif kind == "load":
            _, _, j = instr
            updates[f.index("q1")] = 2
            for i, bit in enumerate(pm_bits(j, w)):
                if bit == 1:
                    updates[f.index("X", i)] = 2
        elif kind == "store":
            _, ireg, jreg = instr
            updates[f.index("q1")] = updates[f.index("q2")] = 2
            for name, value in (("X", ireg), ("Y", jreg)):
                for i, bit in enumerate(pm_bits(value, w)):
                    if bit == 1:
                        updates[f.index(name, i)] = 2
        elif kind == "jnz":
            _, ireg, jreg = instr
            updates[f.index("q1")] = updates[f.index("q2")] = 2
            for name, value in (("X", ireg), ("Y", jreg)):
                for i, bit in enumerate(pm_bits(value, w)):
                    if bit == 1:
                        updates[f.index(name, i)] = 2
        elif kind == "halt":
            updates[f.index("stop")] = 2
        layer.fire(base, updates)

    # At an output separator X holds the address of the record just closed.
    # Increment it in place: the neuron of bit s fires if that bit is 0 and
    # all lower bits are 1, and exactly one of them fires since X < 2^w - 1.
    outsep = conditions(f, "outsep")
    layer.fire(outsep, {f.index("q1"): 2, f.index("q2"): 2})
    for k in range(w):
        idx = w - 1 - k
        cond = [*outsep, (f.index("X", idx), -1)]
        updates = {f.index("X", idx): 2}
        for lower in range(k):
            j = w - 1 - lower
            cond.append((f.index("X", j), 1))
            updates[f.index("X", j)] = -2
        layer.fire(cond, updates)
    # Y remains zero, so the second lookup reads the output length mem[0].
    # The first output mem follows <out>, after the final write's commit.
    layer.fire(conditions(f, "outstart"), {f.index("q1"): 2})


def add_lookup_cleanup(layer):
    f = layer.f
    layer.fire([*conditions(f, "jnz"),
                *[(f.index("A", i), -1) for i in range(f["A"].stop - f["A"].start)]],
               {f.index("iszero"): 2})
    reset_word(layer, "X", conditions(f, "exec"))
    reset_word(layer, "Y", conditions(f, "exec"))


def op_conditions(f, op):
    # an op flag is only ever set on an exec token, so it guards alone
    return [(f.index("op", OPS.index(op)), 1)]


def add_alu_stage(layer, stage, w, plen):
    """Advance carry operations from the LSB and comparison scans from the MSB."""
    f, idx = layer.f, w - 1 - stage

    # Addition and subtraction, sharing one binary carry/borrow flag.
    for op, borrow in (("+", False), ("-", True)):
        base = op_conditions(f, op)
        for av, bv, sv in itertools.product((-1, 1), repeat=3):
            a, b, s = av == 1, bv == 1, sv == 1
            if not borrow:
                out, new = (a + b + s) & 1, (a + b + s) >= 2
            else:
                out = a ^ b ^ s
                new = ((not a) and (b or s)) or (b and s)
            updates = {}
            if out:
                updates[f.index("X", idx)] = 2
            if new != s:
                updates[f.index("carry")] = 2 if new else -2
            layer.fire([*base, (f.index("A", idx), av), (f.index("B", idx), bv),
                        (f.index("carry"), sv)], updates)

    # Bitwise operations.
    bit_fns = {"&": lambda a, b: a and b, "|": lambda a, b: a or b,
               "^": lambda a, b: a ^ b}
    for op, fn in bit_fns.items():
        for av, bv in itertools.product((-1, 1), repeat=2):
            if fn(av == 1, bv == 1):
                layer.fire([*op_conditions(f, op), (f.index("A", idx), av),
                            (f.index("B", idx), bv)], {f.index("X", idx): 2})

    # Variable shifts: an out-of-range shift has no matching neuron and yields 0.
    for shift in range(w):
        amount = word_conditions(f, "B", shift, w)
        if shift <= stage:
            src = w - 1 - (stage - shift)
            layer.fire([*op_conditions(f, "<<"), *amount, (f.index("A", src), 1)],
                       {f.index("X", idx): 2})
        if stage + shift < w:
            src = w - 1 - (stage + shift)
            layer.fire([*op_conditions(f, ">>"), *amount, (f.index("A", src), 1)],
                       {f.index("X", idx): 2})

    # Online carry-save multiplication, with both operands immutable.
    # Before stage i, X+Y = floor(A * (B mod 2^i) / 2^i), and Q contains
    # the low i product bits. Adding A*b_i produces sum bits U and unshifted
    # carries Y'; U's low bit goes to fresh Q_i, and (U >> 1,Y') is the next pair.
    mul = op_conditions(f, "*")
    bi = f.index("B", w - 1 - stage)
    low = w - 1
    for sx, cy, ay, by in itertools.product((-1, 1), repeat=4):
        u = (sx == 1) ^ (cy == 1) ^ ((ay == 1) and (by == 1))
        if u:
            layer.fire([*mul, (f.index("X", low), sx), (f.index("Y", low), cy),
                        (f.index("A", low), ay), (bi, by)],
                       {f.index("Q", w - 1 - stage): 2})
    for j in range(w):
        dst = w - 1 - j
        if j == w - 1:
            layer.fire([*mul, (f.index("X", dst), 1)], {f.index("X", dst): -2})
        else:
            src = w - 1 - (j + 1)
            for old, sx, cy, ay, by in itertools.product((-1, 1), repeat=5):
                new = (sx == 1) ^ (cy == 1) ^ ((ay == 1) and (by == 1))
                if new != (old == 1):
                    layer.fire([*mul, (f.index("X", dst), old),
                                (f.index("X", src), sx), (f.index("Y", src), cy),
                                (f.index("A", src), ay), (bi, by)],
                               {f.index("X", dst): 2 if new else -2})
        for sx, cy, ay, by in itertools.product((-1, 1), repeat=4):
            new = sum((sx == 1, cy == 1, (ay == 1) and (by == 1))) >= 2
            if new != (cy == 1):
                layer.fire([*mul, (f.index("X", dst), sx), (f.index("Y", dst), cy),
                            (f.index("A", dst), ay), (bi, by)],
                           {f.index("Y", dst): 2 if new else -2})

    # Comparisons scan from most significant to least significant bit.
    cidx = stage
    cmp_base = [*conditions(f, "cmp"), (f.index("eq"), 1)]
    layer.fire([*cmp_base, (f.index("A", cidx), -1), (f.index("B", cidx), 1)],
               {f.index("eq"): -2, f.index("lt"): 2})
    layer.fire([*cmp_base, (f.index("A", cidx), 1), (f.index("B", cidx), -1)],
               {f.index("eq"): -2})

    # Most significant set bit.  eq is initially true and becomes false at the
    # first 1 encountered by this MSB-to-LSB scan.  X remains zero for A = 0.
    result = w - 1 - stage
    updates = {f.index("eq"): -2}
    for i, bit in enumerate(pm_bits(result, w)):
        if bit == 1:
            updates[f.index("X", i)] = 2
    layer.fire([*op_conditions(f, "msb"), (f.index("eq"), 1),
                (f.index("A", cidx), 1)], updates)

    # The same equality flag determines whether the current output block is last.
    outsep = [*conditions(f, "outsep"), (f.index("eq"), 1)]
    layer.fire([*outsep, (f.index("X", cidx), -1), (f.index("B", cidx), 1)],
               {f.index("eq"): -2})
    layer.fire([*outsep, (f.index("X", cidx), 1), (f.index("B", cidx), -1)],
               {f.index("eq"): -2})
    layer.fire([*conditions(f, "outstart"), (f.index("eq"), 1),
                (f.index("A", cidx), 1)], {f.index("eq"): -2})

    # Jump targets are checked against |P| the same way: afterwards lt holds
    # iff B < |P|.  For |P| = 2^w every target is in range and no rule is needed.
    if plen < 1 << w:
        jnz = [*conditions(f, "jnz"), (f.index("eq"), 1)]
        if pm_bits(plen, w)[cidx] == 1:
            layer.fire([*jnz, (f.index("B", cidx), -1)],
                       {f.index("eq"): -2, f.index("lt"): 2})
        else:
            layer.fire([*jnz, (f.index("B", cidx), 1)], {f.index("eq"): -2})


def add_alu_finish(layer, w):
    """Give every arithmetic operation the same result field X."""
    f = layer.f
    lsb = f.index("X", w - 1)
    layer.fire([*op_conditions(f, "<"), (f.index("lt"), 1)], {lsb: 2})
    layer.fire([*op_conditions(f, "<="), (f.index("lt"), 1)], {lsb: 2})
    layer.fire([*op_conditions(f, "<="), (f.index("eq"), 1)], {lsb: 2})
    layer.fire([*op_conditions(f, "=="), (f.index("eq"), 1)], {lsb: 2})
    layer.fire([*op_conditions(f, "!="), (f.index("eq"), -1)], {lsb: 2})
    assign_word(layer, "X", "Q", op_conditions(f, "*"))
    reset_word(layer, "Y", op_conditions(f, "*"))    # the carry-save leftovers


def add_final_layer(layer, M):
    """Assemble the descriptor (Y address, X value, Z next pc) and flags.

    The arithmetic result is already in X, and Y and Z are at their defaults
    at every step token, so the instruction neurons write into fresh words.
    """
    f, w = layer.f, M.w

    store = conditions(f, "store")
    assign_word(layer, "X", "B", store)
    assign_word(layer, "Y", "A", store)

    outsep = conditions(f, "outsep")
    assign_word(layer, "Y", "X", outsep)
    assign_word(layer, "X", "A", outsep)
    layer.fire(outsep, {f.index("output"): 2})
    layer.fire([*outsep, (f.index("eq"), 1)], {f.index("last"): 2})

    outstart = conditions(f, "outstart")
    assign_word(layer, "X", "A", outstart)
    layer.fire(outstart, {f.index("output"): 2})
    layer.fire([*outstart, (f.index("eq"), 1)], {f.index("last"): 2})

    # Taken jumps: the target was read into B, and lt (computed by the ALU
    # scan) tells whether it is in range.  Z is at its default, so setting
    # the 1-bits of B writes B.
    jump = [*conditions(f, "jnz"), (f.index("iszero"), -1)]
    for i in range(w):
        layer.fire([*jump, (f.index("B", i), 1)], {f.index("Z", i): 2})
    if len(M.program) < 1 << M.w:
        layer.fire([*jump, (f.index("lt"), -1)], {f.index("stop"): 2})

    # Static per-instruction data: destination register, constant, next pc.
    for pc, instr in enumerate(M.program):
        kind = instr[0]
        if kind == "halt":
            continue
        base = [*conditions(f, "exec"), *word_conditions(f, "win", pc, w)]
        if kind == "jnz":                     # the taken case is handled above
            base.append((f.index("iszero"), 1))
        updates = {}
        if kind in ("set", "op", "msb", "load"):
            dest = instr[1]
            for i, bit in enumerate(pm_bits(dest, w)):
                if bit == 1:
                    updates[f.index("Y", i)] = 2
        if kind == "set":
            for i, bit in enumerate(pm_bits(instr[2], w)):
                if bit == 1:
                    updates[f.index("X", i)] = 2
        if pc + 1 < len(M.program):
            for i, bit in enumerate(pm_bits(pc + 1, w)):
                if bit == 1:
                    updates[f.index("Z", i)] = 2
        else:
            updates[f.index("stop")] = 2
        layer.fire(base, updates)


def add_emission(layer, w):
    f = layer.f
    layer.heads = [broadcast_head(f, "X", ("write", "wreg", "stop",
                                                     "output", "last")),
                   broadcast_head(f, "Y"), broadcast_head(f, "Z")]

    def emit(token, base):
        if token == "0":
            return
        layer.fire(base, {f.index("next", TOK["0"]): -2,
                          f.index("next", TOK[token]): 2})

    def emit_bit(word, i, base):
        emit("1", [*base, (f.index(word, i), 1)])

    # Processor tokens either begin a write/pc segment or the next output block.
    emit("mem", [*conditions(f, "ctrl", "output"),
                 (f.index("tok", TOK["#"]), 1)])
    emit("reg", conditions(f, "ctrl", ("output", -1), "write", "wreg"))
    emit("mem", conditions(f, "ctrl", ("output", -1), "write",
                           ("wreg", -1)))
    emit("<out>", conditions(f, "ctrl", ("output", -1), ("write", -1),
                             "stop"))
    emit("pc", conditions(f, "ctrl", ("output", -1), ("write", -1),
                          ("stop", -1)))
    emit("mem", [(f.index("tok", TOK["<out>"]), 1)])
    emit("<out>", [(f.index("tok", TOK["#"]), 1),
                   *conditions(f, ("out", -1), "stop")])
    emit("pc", [(f.index("tok", TOK["#"]), 1),
                *conditions(f, ("out", -1), ("stop", -1))])

    # Block-opening markers spell the first bit; later bit tokens use dist_j.
    for token in ("mem", "reg"):
        emit_bit("Y", 0, [(f.index("tok", TOK[token]), 1)])
    emit_bit("X", 0, [(f.index("tok", TOK[":"]), 1)])
    emit_bit("Z", 0, [(f.index("tok", TOK["pc"]), 1)])

    for token in ("0", "1"):
        tok = (f.index("tok", TOK[token]), 1)
        for marker in ("mem", "reg"):
            near = (f.index("near", BLOCK.index(marker)), 1)
            for j in range(1, w):
                emit_bit("Y", j, [tok, near, (f.index(f"dist{j}"), 1)])
            emit(":", [tok, near, (f.index(f"dist{w}"), 1)])

        near = (f.index("near", BLOCK.index(":")), 1)
        for j in range(1, w):
            emit_bit("X", j, [tok, near, (f.index(f"dist{j}"), 1)])
        end = [tok, near, (f.index(f"dist{w}"), 1)]
        emit("<eos>", [*end, (f.index("output"), 1), (f.index("last"), 1)])
        emit("#", [*end, (f.index("output"), 1), (f.index("last"), -1)])
        emit("#", [*end, (f.index("output"), -1)])

        near = (f.index("near", BLOCK.index("pc")), 1)
        for j in range(1, w):
            emit_bit("Z", j, [tok, near, (f.index(f"dist{j}"), 1)])
        emit("step", [tok, near, (f.index(f"dist{w}"), 1)])


def ram_to_lema(M):
    """Return one binary-residual LEMA transformer simulating M on every input."""
    w, f = M.w, layout(M.w)
    layers = [structural_layer(f, w).finish()]
    for ell in range(2, w + 1):
        layers.append(window_layer(f, ell, w).finish())
    layers.append(records_layer(f, M).finish())

    lookup = Layer(f)
    lookup.heads = [memory_head(f, "X", "q1", "out", "A"),
                    memory_head(f, "Y", "q2", "out", "B")]
    add_lookup_cleanup(lookup)
    layers.append(lookup.finish())

    second_lookup = Layer(f)
    second_lookup.heads = [memory_head(f, "A", "load", "load", "X")]
    layers.append(second_lookup.finish())
    for stage in range(w):
        layer = Layer(f)
        add_alu_stage(layer, stage, w, len(M.program))
        layers.append(layer.finish())

    finish = Layer(f)
    add_alu_finish(finish, w)
    layers.append(finish.finish())
    final = Layer(f)
    add_final_layer(final, M)
    layers.append(final.finish())
    emission = Layer(f)
    add_emission(emission, w)
    layers.append(emission.finish())

    return LemaTransformer(embeddings(f), unembeddings(f), layers,
                           precision=PRECISION(M), strict=True, binary=True)


def expected_transcript(M, x):
    """Reference transcript derived only from the RAM execution trace."""
    ran = M.run_on(x, return_trace=True)
    if ran is False:
        raise ValueError("the word-RAM did not halt on this input")
    (m, y), info = ran
    w = M.w
    toks = encode_input(x, w) + [TOK["#"], TOK["pc"]] + bits_w(0, w)
    for _, write, pc_next in info["trace"]:
        toks.append(TOK["step"])
        if write is not None:
            kind, a, v = write
            toks += [TOK[kind]] + bits_w(a, w) + [TOK[":"]] + bits_w(v, w) + [TOK["#"]]
        if pc_next is None:
            return toks + [TOK["<out>"]] + enc([m] + y, w) + [TOK["<eos>"]]
        toks += [TOK["pc"]] + bits_w(pc_next, w)
    raise AssertionError("a halting trace must contain a final step")
