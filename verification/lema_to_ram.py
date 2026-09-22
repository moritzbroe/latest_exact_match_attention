"""Compile a LEMA transformer into the word-RAM of Theorem 2.

One program P (and r = 9 registers) works for every word size w >= MIN_WORD(T):
all instruction constants are below 2^MIN_WORD, and the static cells and the
kv-cache heap live at the top of memory, at addresses computed at run time as
0 - k mod 2^w.  Correctness then only requires that the space fits below 2^w,
which is the space condition of the theorem.

The program is an interpreter with the weights inlined as constants:

    init        r0 = 1, r8 = 2^w - #cells (the static base), heap pointer
    token loop  choose the current token (prompt cell or fed-back prediction);
                embedding dispatch; per layer and head, the query and key
                projections are computed row by row and binarized into the
                blocks Q and K and the value vector is written into V; the
                walk for Q then copies the retrieved leaf into O, or fills O
                with zeros on a miss, and only afterwards is (K, V) inserted
                (allocating missing nodes and the leaf); W_O projects O into U
                row by row and U is added to S; after all heads S is added to
                X; the MLP runs neuron by neuron with a scalar accumulator, so
                no d_ff-sized state ever exists; unembedding rows feed a
                running strictly-greater argmax; bookkeeping stops on <eos>
    library     FMAC/FMAV (multiply-accumulate into the cell ACC / into an
                address), FPADD, FPGT, called with the return address in r6

Rounding order mirrors lema.py operation for operation, so results are
bit-identical on finite executions. Floats use the standard sign, biased
exponent and fraction fields, including subnormals. Multiplication keeps the
full significand product in the RAM word; addition uses the appendix's exact
alignment for exponent gaps at most p_m+2 and otherwise returns the larger
operand. The exact integer result is then rounded, with msb locating its
leading bit in one instruction.
"""

from fractions import Fraction

import numpy as np

from encoding import TOK
from wordram import WordRAM


# --- float format and reference rounding -------------------------------------


def _pow2(e):
    return Fraction(1 << e) if e >= 0 else Fraction(1, 1 << -e)


def _floor_log2(v):
    """floor(log2(v)) for a positive Fraction."""
    e = v.numerator.bit_length() - v.denominator.bit_length()
    return e - 1 if v < _pow2(e) else e


def _round_even(v):
    """Round a nonnegative Fraction to the nearest integer, ties to even."""
    q, r = divmod(v.numerator, v.denominator)
    twice = 2 * r
    return q + int(twice > v.denominator or
                   (twice == v.denominator and q & 1))

class Fmt:
    def __init__(self, pm, pe):
        self.pm, self.pe = pm, pe
        self.p = 1 + pm + pe
        self.bias = (1 << (pe - 1)) - 1
        self.emin = 1 - self.bias
        self.emax = ((1 << pe) - 2) - self.bias
        self.mlo, self.mhi = 1 << pm, (1 << (pm + 1)) - 1
        self.frac_mask = self.mlo - 1
        self.exp_mask = (1 << pe) - 1
        self.sign_bit = 1 << (self.p - 1)
        # Working exponents are stored as E + off, where a magnitude is M*2^E.
        # This keeps products of the smallest subnormals nonnegative.
        self.off = 4 * self.bias + 4 * pm + 32

    def pack(self, v):
        """Exactly representable v -> packed word (raises otherwise)."""
        v = Fraction(v)
        if v == 0:
            return 0
        s, a = (1, -v) if v < 0 else (0, v)
        if a < _pow2(self.emin):
            q = a / _pow2(self.emin - self.pm)
            if q.denominator != 1 or not 1 <= q < self.mlo:
                raise ValueError(f"{v} is not representable")
            return s * self.sign_bit + int(q)
        e = _floor_log2(a)
        m = a / _pow2(e - self.pm)
        if (m.denominator != 1 or not self.emin <= e <= self.emax
                or not self.mlo <= m <= self.mhi):
            raise ValueError(f"{v} is not representable")
        return s * self.sign_bit + ((e + self.bias) << self.pm) + (int(m) - self.mlo)

    def unpack(self, x):
        if x == 0:
            return Fraction(0)
        s = x >> (self.p - 1)
        ef = (x >> self.pm) & self.exp_mask
        frac = x & self.frac_mask
        if ef == self.exp_mask:
            raise ValueError("infinity and NaN are outside the finite format")
        if ef == 0:
            m, e = frac, self.emin - self.pm
        else:
            m, e = frac + self.mlo, ef - self.bias - self.pm
        return (-1 if s else 1) * m * _pow2(e)

    def rd(self, v):
        """Round an in-range exact value to finite IEEE binary, ties to even."""
        v = Fraction(v)
        if v == 0:
            return 0
        s, a = (1, -v) if v < 0 else (0, v)
        overflow = _pow2(self.emax + 1) - _pow2(self.emax - self.pm - 1)
        if a >= overflow:
            raise OverflowError(f"{v} rounds outside the finite range")
        if a < _pow2(self.emin):
            q = _round_even(a / _pow2(self.emin - self.pm))
            if q == 0:
                return 0
            if q < self.mlo:
                return s * self.sign_bit + q
            return s * self.sign_bit + (1 << self.pm)
        e = _floor_log2(a)
        q = _round_even(a / _pow2(e - self.pm))
        if q == 2 * self.mlo:
            q, e = self.mlo, e + 1
        if e > self.emax:
            raise OverflowError(f"{v} rounds outside the finite range")
        return s * self.sign_bit + ((e + self.bias) << self.pm) + (q - self.mlo)

    def ref_add(self, x, y):
        return self.rd(self.unpack(x) + self.unpack(y))

    def ref_mul(self, x, y):
        return self.rd(self.unpack(x) * self.unpack(y))


def fmt_for(precision):
    if precision == "fp16":
        return Fmt(10, 5)
    if precision == "fp32":
        return Fmt(23, 8)
    if precision == "fp64":
        return Fmt(52, 11)
    if precision.startswith("int"):
        k = int(precision[3:])
        return Fmt(k, max(6, k.bit_length() + 2))
    raise ValueError(f"unknown precision {precision!r}")


# --- word-level algorithms ----------------------------------------------------


def _decode(f, A):
    """Nonzero packed A -> sign, integer significand, power-of-two exponent."""
    s = A >> (f.p - 1)
    ef = (A >> f.pm) & f.exp_mask
    frac = A & f.frac_mask
    if ef == 0:
        return s, frac, f.emin - f.pm
    return s, frac + f.mlo, ef - f.bias - f.pm


def _round_scaled(f, s, M, E):
    """Round the exact value (-1)^s * M * 2^E, with integer M >= 0."""
    if M == 0:
        return 0
    assert M > 0
    lead = M.bit_length() - 1
    top = E + lead
    if top < f.emin:
        target = f.emin - f.pm
        normal = False
    else:
        target = top - f.pm
        normal = True
    shift = target - E
    if shift <= 0:
        q = M << -shift
    elif shift > lead + 1:
        q = 0
    else:
        q = M >> shift
        rem = M & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if rem > half or (rem == half and (q & 1)):
            q += 1
    if not normal:
        if q == 0:
            return 0
        if q < f.mlo:
            return s * f.sign_bit + q
        return s * f.sign_bit + (1 << f.pm)
    if q == 2 * f.mlo:
        q, top = f.mlo, top + 1
    if top > f.emax:
        raise OverflowError("floating-point overflow")
    return s * f.sign_bit + ((top + f.bias) << f.pm) + (q - f.mlo)


def alg_mul(f, A, B):
    if A == 0 or B == 0:
        return 0
    sa, ma, ea = _decode(f, A)
    sb, mb, eb = _decode(f, B)
    return _round_scaled(f, sa ^ sb, ma * mb, ea + eb)


def alg_add(f, A, B):
    if A == 0:
        return B
    if B == 0:
        return A
    sa, ma, ea = _decode(f, A)
    sb, mb, eb = _decode(f, B)
    la, lb = ma.bit_length() - 1, mb.bit_length() - 1
    ma, ea = ma << (f.pm - la), ea - (f.pm - la)
    mb, eb = mb << (f.pm - lb), eb - (f.pm - lb)
    if eb > ea or (eb == ea and mb > ma):
        (sa, ea, ma), (sb, eb, mb) = (sb, eb, mb), (sa, ea, ma)
        A, B = B, A
    d = ea - eb
    if d > f.pm + 2:
        return A
    M = (ma << d) + mb if sa == sb else (ma << d) - mb
    return _round_scaled(f, sa, M, eb)


def alg_gt(f, A, B):
    """1 if unpack(A) > unpack(B) else 0, on packed words."""
    sa, sb = A >> (f.p - 1), B >> (f.p - 1)
    if sa != sb:
        return 1 - sa
    if sa == 0:
        return 1 if A > B else 0
    return 1 if (B & ~(1 << (f.p - 1))) > (A & ~(1 << (f.p - 1))) else 0


# --- assembler ----------------------------------------------------------------

class Asm:
    """Word-RAM macro assembler.  r0 = 1 and r8 = BASE (the top-of-memory
    static base) are set once and never clobbered.  r1..r5 are scratch, r6
    holds return addresses, r7 is the jump-target scratch of every macro jump.
    Clobbers: ldc(rd) -> rd; stc(off,rs,rt) -> rt; opi(rd,ra,c,o,rt) -> rd,rt;
    jnz -> r7; jz -> r5,r7; call -> r6,r7.  Values never live in a scratch
    register across more than a couple of macros; everything is cell-based."""

    R = 9
    ONE, RET, RT, RB = 0, 6, 7, 8

    def __init__(self):
        self.prog, self.labels, self.n = [], {}, 0

    def newlab(self, tag="L"):
        self.n += 1
        return f"{tag}_{self.n}"

    def label(self, name):
        assert name not in self.labels, name
        self.labels[name] = len(self.prog)

    def raw(self, *ins):
        self.prog.append(tuple(ins))

    def imm(self, rd, c):
        self.raw("set", rd, c)                   # c may be a label name

    def op(self, rd, ra, rb, o):
        self.raw("op", rd, ra, rb, o)

    def msb(self, rd, rs):
        self.raw("msb", rd, rs)

    def opi(self, rd, ra, c, o, rt=5):
        self.imm(rt, c)
        self.op(rd, ra, rt, o)

    def mov(self, rd, rs):
        self.op(rd, rs, rs, "&")

    def zero(self, rd):
        self.op(rd, rd, rd, "^")

    def goto(self, lbl):
        self.imm(self.RT, lbl)
        self.raw("jnz", self.ONE, self.RT)

    def jnz(self, rc, lbl):
        self.imm(self.RT, lbl)
        self.raw("jnz", rc, self.RT)

    def jz(self, rc, lbl):
        self.op(5, rc, rc, "^")                  # r5 = 0
        self.op(5, rc, 5, "==")                  # r5 = (rc == 0)
        self.jnz(5, lbl)

    def call(self, lbl):
        ret = self.newlab("ret")
        self.imm(self.RET, ret)
        self.goto(lbl)
        self.label(ret)

    def ret(self):
        self.raw("jnz", self.ONE, self.RET)

    def finish(self):
        out = []
        for ins in self.prog:
            if ins[0] == "set" and isinstance(ins[2], str):
                out.append(("set", ins[1], self.labels[ins[2]]))
            else:
                out.append(ins)
        return out


class Cells:
    """Static cells, addressed as BASE + offset with BASE = 2^w - size."""

    def __init__(self):
        self.at, self.size = {}, 0

    def add(self, name, width=1):
        self.at[name] = self.size
        self.size += width


class Ctx:
    """Assembler plus named static cells plus float format."""

    def __init__(self, a, cells, f):
        self.a, self.cells, self.f = a, cells, f

    def off(self, name, i=0):
        return self.cells.at[name] + i

    def ld(self, rd, name, i=0):                 # rd <- cell
        a = self.a
        a.imm(rd, self.off(name, i))
        a.op(rd, a.RB, rd, "+")
        a.raw("load", rd, rd)

    def st(self, name_i, rs, rt=4):              # cell <- rs
        a = self.a
        name, i = name_i if isinstance(name_i, tuple) else (name_i, 0)
        a.imm(rt, self.off(name, i))
        a.op(rt, a.RB, rt, "+")
        a.raw("store", rt, rs)

    def addr(self, rd, name, i=0):               # rd <- absolute cell address
        a = self.a
        a.imm(rd, self.off(name, i))
        a.op(rd, a.RB, rd, "+")


# --- emitted IEEE floating-point bodies --------------------------------------

def emit_unpack2(c):
    """FA, FB (nonzero) -> sign, significand and offset binary exponent."""
    a, f = c.a, c.f
    for src, s_, e_, m_ in (("FA", "SA", "EA", "MA"),
                            ("FB", "SB", "EB2", "MB")):
        normal, done = a.newlab("normal"), a.newlab("unpacked")
        c.ld(1, src)
        a.opi(2, 1, f.p - 1, ">>")
        c.st(s_, 2)
        a.opi(2, 1, f.pm, ">>")
        a.opi(2, 2, f.exp_mask, "&")
        a.opi(3, 1, f.frac_mask, "&")
        c.st(m_, 3)
        a.jnz(2, normal)
        a.imm(2, f.off + f.emin - f.pm)
        c.st(e_, 2)
        a.goto(done)
        a.label(normal)
        a.opi(2, 2, f.off - f.bias - f.pm, "+")
        c.st(e_, 2)
        c.ld(3, m_)
        a.opi(3, 3, f.mlo, "+")
        c.st(m_, 3)
        a.label(done)


def _emit_round_right(c):
    """QQ <- rounded QQ / 2^WD, using W1=msb(QQ), for WD > 0."""
    a = c.a
    far, up, no_up, done = (a.newlab(x) for x in
                             ("far_shift", "round_up", "no_round", "rounded"))
    c.ld(1, "WD")
    c.ld(2, "W1")
    a.opi(2, 2, 1, "+")
    a.op(3, 2, 1, "<")                          # WD > lead + 1
    a.jnz(3, far)

    c.ld(2, "QQ")
    a.op(2, 2, 1, ">>")
    c.st("QQ", 2)
    a.mov(3, a.ONE)
    a.op(3, 3, 1, "<<")
    a.opi(3, 3, 1, "-")                         # (1 << WD) - 1
    # Recover M from W4, where callers of this helper save it.
    c.ld(2, "W4")
    a.op(2, 2, 3, "&")
    c.st("RR", 2)
    a.opi(1, 1, 1, "-")
    a.mov(3, a.ONE)
    a.op(3, 3, 1, "<<")
    c.st("HH", 3)

    c.ld(1, "RR")
    c.ld(2, "HH")
    a.op(3, 2, 1, "<")                          # remainder > half
    a.jnz(3, up)
    a.op(3, 1, 2, "==")
    a.jz(3, no_up)
    c.ld(1, "QQ")
    a.opi(1, 1, 1, "&")
    a.jz(1, no_up)
    a.label(up)
    c.ld(1, "QQ")
    a.opi(1, 1, 1, "+")
    c.st("QQ", 1)
    a.label(no_up)
    a.goto(done)

    a.label(far)
    a.zero(1)
    c.st("QQ", 1)
    a.label(done)


def emit_round_scaled(c, done):
    """Round the exact value (-1)^SGN * QQ * 2^(EE-off)."""
    a, f = c.a, c.f
    sub, nleft, nright, nready = (a.newlab(x) for x in
                                  ("subnormal", "normal_left", "normal_right",
                                   "normal_ready"))
    sleft, sright, sready = (a.newlab(x) for x in
                             ("sub_left", "sub_right", "sub_ready"))
    carry, nocarry, finite, packn = (a.newlab(x) for x in
                                     ("carry", "no_carry", "finite", "pack_normal"))
    packs, minnormal, zero = (a.newlab(x) for x in
                              ("pack_subnormal", "min_normal", "round_zero"))

    c.ld(1, "QQ")
    a.jz(1, zero)
    a.msb(2, 1)
    c.st("W1", 2)                               # lead
    c.st("W4", 1)                               # original M for remainder
    c.ld(3, "EE")
    a.op(3, 3, 2, "+")
    c.st("W2", 3)                               # top + off
    a.opi(3, 3, f.off + f.emin, "<")
    a.jnz(3, sub)

    # Normal: retain pm+1 significant bits, so shift by lead-pm.
    c.ld(1, "W1")
    a.opi(3, 1, f.pm, "<")
    a.jnz(3, nleft)
    a.opi(3, 1, f.pm, "==")
    a.jz(3, nright)
    a.goto(nready)
    a.label(nleft)
    a.imm(2, f.pm)
    a.op(2, 2, 1, "-")
    c.ld(1, "QQ")
    a.op(1, 1, 2, "<<")
    c.st("QQ", 1)
    a.goto(nready)
    a.label(nright)
    a.opi(1, 1, f.pm, "-")
    c.st("WD", 1)
    _emit_round_right(c)
    a.label(nready)

    c.ld(1, "QQ")
    a.opi(3, 1, 2 * f.mlo, "==")
    a.jnz(3, carry)
    a.goto(nocarry)
    a.label(carry)
    a.imm(1, f.mlo)
    c.st("QQ", 1)
    c.ld(1, "W2")
    a.opi(1, 1, 1, "+")
    c.st("W2", 1)
    a.label(nocarry)
    c.ld(1, "W2")
    a.opi(3, 1, f.off + f.emax, "<=")
    a.jnz(3, finite)
    # Outside the theorem's finite-execution premise. Saturation keeps the
    # emitted program total, but no correctness claim relies on this branch.
    a.imm(1, f.mhi)
    c.st("QQ", 1)
    a.imm(1, f.off + f.emax)
    c.st("W2", 1)
    a.label(finite)
    a.goto(packn)

    # Subnormal: round to the fixed quantum 2^(emin-pm).
    a.label(sub)
    c.ld(1, "EE")
    a.opi(3, 1, f.off + f.emin - f.pm, "<")
    a.jnz(3, sright)
    a.goto(sleft)
    a.label(sleft)
    a.opi(2, 1, f.off + f.emin - f.pm, "-")
    c.ld(1, "QQ")
    a.op(1, 1, 2, "<<")
    c.st("QQ", 1)
    a.goto(sready)
    a.label(sright)
    a.imm(2, f.off + f.emin - f.pm)
    a.op(2, 2, 1, "-")
    c.st("WD", 2)
    _emit_round_right(c)
    a.label(sready)
    c.ld(1, "QQ")
    a.jz(1, zero)
    a.opi(3, 1, f.mlo, "<")
    a.jnz(3, packs)
    a.goto(minnormal)

    a.label(packn)
    c.ld(1, "SGN")
    a.opi(1, 1, f.p - 1, "<<")
    c.ld(2, "W2")
    a.opi(2, 2, f.off - f.bias, "-")
    a.opi(2, 2, f.pm, "<<")
    a.op(1, 1, 2, "+")
    c.ld(2, "QQ")
    a.opi(2, 2, f.mlo, "-")
    a.op(1, 1, 2, "+")
    c.st("FR", 1)
    a.goto(done)

    a.label(packs)
    c.ld(1, "SGN")
    a.opi(1, 1, f.p - 1, "<<")
    c.ld(2, "QQ")
    a.op(1, 1, 2, "+")
    c.st("FR", 1)
    a.goto(done)

    a.label(minnormal)
    c.ld(1, "SGN")
    a.opi(1, 1, f.p - 1, "<<")
    a.opi(1, 1, 1 << f.pm, "+")
    c.st("FR", 1)
    a.goto(done)

    a.label(zero)
    a.zero(1)
    c.st("FR", 1)
    a.goto(done)


def emit_fmul(c):
    """FR <- fl(FA * FB), including subnormal inputs and outputs."""
    a, f = c.a, c.f
    done, zero = a.newlab("fmul_done"), a.newlab("fmul_zero")
    c.ld(1, "FA")
    a.jz(1, zero)
    c.ld(1, "FB")
    a.jz(1, zero)
    emit_unpack2(c)
    c.ld(1, "SA")
    c.ld(2, "SB")
    a.op(1, 1, 2, "^")
    c.st("SGN", 1)
    c.ld(1, "EA")
    c.ld(2, "EB2")
    a.op(1, 1, 2, "+")
    a.opi(1, 1, f.off, "-")
    c.st("EE", 1)
    c.ld(1, "MA")
    c.ld(2, "MB")
    a.op(1, 1, 2, "*")
    c.st("QQ", 1)
    emit_round_scaled(c, done)
    a.label(zero)
    a.zero(1)
    c.st("FR", 1)
    a.label(done)


def _emit_normalize_operand(c, mname, ename):
    """Normalize a nonzero significand to have its leading bit at pm."""
    a, f = c.a, c.f
    c.ld(1, mname)
    a.msb(2, 1)
    a.imm(3, f.pm)
    a.op(3, 3, 2, "-")
    a.op(1, 1, 3, "<<")
    c.st(mname, 1)
    c.ld(1, ename)
    a.op(1, 1, 3, "-")
    c.st(ename, 1)


def emit_fadd(c):
    """FR <- fl(FA + FB), including cancellation and gradual underflow."""
    a, f = c.a, c.f
    done, reta, retb = (a.newlab(x) for x in
                        ("fadd_done", "fadd_a", "fadd_b"))
    swap, ordered, opposite, scaled = (a.newlab(x) for x in
                                       ("swap", "ordered", "opposite",
                                        "fadd_scaled"))
    c.ld(1, "FA")
    a.jz(1, retb)
    c.ld(1, "FB")
    a.jz(1, reta)
    emit_unpack2(c)
    _emit_normalize_operand(c, "MA", "EA")
    _emit_normalize_operand(c, "MB", "EB2")

    c.ld(1, "EB2")
    c.ld(2, "EA")
    a.op(3, 2, 1, "<")                          # EB2 > EA
    a.jnz(3, swap)
    a.op(3, 1, 2, "==")
    a.jz(3, ordered)
    c.ld(1, "MB")
    c.ld(2, "MA")
    a.op(3, 2, 1, "<")                          # MB > MA
    a.jnz(3, swap)
    a.goto(ordered)
    a.label(swap)
    for x, y in (("SA", "SB"), ("EA", "EB2"), ("MA", "MB"), ("FA", "FB")):
        c.ld(1, x)
        c.ld(2, y)
        c.st(x, 2)
        c.st(y, 1)
    a.label(ordered)

    c.ld(1, "SA")
    c.st("SGN", 1)
    c.ld(1, "EA")
    c.ld(2, "EB2")
    a.op(3, 1, 2, "-")                          # exponent difference D
    a.opi(1, 3, f.pm + 2, "<=")
    a.jz(1, reta)                               # distant operand cannot affect rounding

    # Exact magnitude (MA << D) +/- MB, in units of 2^(EB2-off).
    c.ld(1, "MA")
    a.op(1, 1, 3, "<<")
    c.st("QQ", 1)

    c.ld(1, "SA")
    c.ld(2, "SB")
    a.op(3, 1, 2, "==")
    a.jz(3, opposite)
    c.ld(1, "QQ")
    c.ld(2, "MB")
    a.op(1, 1, 2, "+")
    c.st("QQ", 1)
    a.goto(scaled)
    a.label(opposite)
    c.ld(1, "QQ")
    c.ld(2, "MB")
    a.op(1, 1, 2, "-")
    c.st("QQ", 1)
    a.label(scaled)
    c.ld(1, "EB2")
    c.st("EE", 1)
    emit_round_scaled(c, done)

    a.label(reta)
    c.ld(1, "FA")
    c.st("FR", 1)
    a.goto(done)
    a.label(retb)
    c.ld(1, "FB")
    c.st("FR", 1)
    a.label(done)


def emit_fgt(c):
    """r3 <- (FA > FB) for finite packed values."""
    a, f = c.a, c.f
    done, same, negative = (a.newlab(x) for x in
                            ("fgt_done", "fgt_same_sign", "fgt_negative"))
    c.ld(1, "FA")
    a.opi(1, 1, f.p - 1, ">>")
    c.ld(2, "FB")
    a.opi(2, 2, f.p - 1, ">>")
    a.op(3, 1, 2, "==")
    a.jnz(3, same)
    a.op(3, a.ONE, 1, "-")                    # signs differ: 1 - sign(a)
    a.goto(done)
    a.label(same)
    a.jnz(1, negative)
    c.ld(1, "FA")                              # both nonnegative
    c.ld(2, "FB")
    a.op(3, 2, 1, "<")
    a.goto(done)
    a.label(negative)
    c.ld(1, "FA")                              # both negative: reverse magnitudes
    a.opi(1, 1, (1 << (f.p - 1)) - 1, "&")
    c.ld(2, "FB")
    a.opi(2, 2, (1 << (f.p - 1)) - 1, "&")
    a.op(3, 1, 2, "<")
    a.label(done)


# --- the compiler -------------------------------------------------------------

_FCELLS = ["FA", "FB", "FR", "SA", "SB", "EA", "EB2", "MA", "MB", "SGN",
           "EE", "QQ", "RR", "HH", "W1", "W2", "W4",
           "WD"]
_SCELLS = ["HP", "NN", "POS", "CUR", "PRED", "NGEN", "NOUT", "SEEN",
           "ACC", "OV", "TADDR", "PTRQ", "PTRK", "LEAFK", "BESTV", "BESTI"]


def _emit_routines(c):
    """FMAC, FMAV, FPADD, FPGT.  No nesting: each contains its bodies inline."""
    a = c.a
    a.label("FMAC")                  # ACC += rd(r1 * r2); zero shortcut is exact
    a.jz(1, "FMAC_out")
    a.jz(2, "FMAC_out")
    c.st("FA", 1)
    c.st("FB", 2)
    emit_fmul(c)
    c.ld(1, "FR")
    c.st("FA", 1)
    c.ld(1, "ACC")
    c.st("FB", 1)
    emit_fadd(c)
    c.ld(1, "FR")
    c.st("ACC", 1)
    a.label("FMAC_out")
    a.ret()

    a.label("FMAV")                  # mem[r2] += rd(r1 * OV)
    c.st("TADDR", 2)
    a.jz(1, "FMAV_out")
    c.st("FA", 1)
    c.ld(1, "OV")
    c.st("FB", 1)
    emit_fmul(c)
    c.ld(1, "FR")
    c.st("FA", 1)
    c.ld(2, "TADDR")
    a.raw("load", 1, 2)
    c.st("FB", 1)
    emit_fadd(c)
    c.ld(1, "FR")
    c.ld(2, "TADDR")
    a.raw("store", 2, 1)
    a.label("FMAV_out")
    a.ret()

    a.label("FPADD")                 # r3 = rd(r1 + r2)
    c.st("FA", 1)
    c.st("FB", 2)
    emit_fadd(c)
    c.ld(3, "FR")
    a.ret()

    a.label("FPGT")                  # r3 = (FA > FB)
    emit_fgt(c)
    a.ret()


def _emit_dot(c, weights, pk, src="X"):
    """ACC = rounded dot of a packed weight row with the cells of block `src`, row by row."""
    a = c.a
    a.zero(1)
    c.st("ACC", 1)
    for ci, wv in enumerate(weights):
        a.imm(1, pk(wv))
        c.ld(2, src, ci)
        a.call("FMAC")


def _emit_addv(c, name_dst, name_src, d):
    """dst[c] = rd(dst[c] + src[c]) for c = 0..d-1 via FPADD."""
    a = c.a
    for ci in range(d):
        c.ld(1, name_dst, ci)
        c.ld(2, name_src, ci)
        a.call("FPADD")
        c.st((name_dst, ci), 3)


def _compile(T, trace_output):
    f = fmt_for(T.precision)
    dm = T.dims()
    V, L, H, d, dh, dff = (dm["vocab"], dm["L"], dm["H"], dm["d"],
                           dm["d_h"], dm["d_ff"])
    dv = dh                                  # one head dimension (appendix transformer definition)
    cells = Cells()
    for name, width in [("X", d), ("S", d), ("U", d), ("Q", dh), ("K", dh),
                        ("V", dv), ("O", dv), ("R", L * H)]:
        cells.add(name, width)
    for name in _SCELLS + _FCELLS:
        cells.add(name)
    a = Asm()
    c = Ctx(a, cells, f)
    pk = lambda v: f.pack(v.item() if hasattr(v, "item") else v)

    # init: r0 = 1, r8 = BASE = 0 - #cells, heap pointer, n, position
    a.imm(0, 1)
    a.opi(8, 8, cells.size, "-")
    c.st("HP", 8)
    a.imm(1, 0)
    a.raw("load", 1, 1)
    c.st("NN", 1)
    a.imm(1, 1)
    c.st("POS", 1)

    a.label("TOKEN")
    c.ld(1, "POS")                   # current token: prompt cell or prediction
    c.ld(2, "NN")
    a.op(3, 1, 2, "<=")
    a.jz(3, "FROMPRED")
    a.raw("load", 2, 1)
    c.st("CUR", 2)
    a.opi(1, 1, 1, "+")
    c.st("POS", 1)
    a.goto("HAVETOK")
    a.label("FROMPRED")
    c.ld(1, "PRED")
    c.st("CUR", 1)
    a.label("HAVETOK")

    # embedding dispatch
    c.ld(1, "CUR")
    for tok in range(1, V):
        a.opi(3, 1, tok, "==")
        a.jnz(3, f"EMB{tok}")
    for tok in range(V):
        a.label(f"EMB{tok}") if tok else None
        for ci in range(d):
            a.imm(2, pk(T.emb[tok, ci]))
            c.st(("X", ci), 2)
        a.goto("EMBDONE")
    a.label("EMBDONE")

    for l, layer in enumerate(T.layers):
        # attention: S accumulates head outputs
        a.zero(1)
        for ci in range(d):
            c.st(("S", ci), 1)
        for h, (WQ, WK, WV, WO) in enumerate(layer["heads"]):
            for W, block in ((WQ, "Q"), (WK, "K")):   # binarized projections
                for j in range(dh):
                    _emit_dot(c, W[j], pk)
                    c.ld(1, "ACC")
                    a.opi(2, 1, f.p - 1, ">>")
                    a.op(2, a.ONE, 2, "-")       # bit = 1 - sign
                    c.st((block, j), 2)
            for rv in range(dv):                 # the value vector into V
                _emit_dot(c, WV[rv], pk)
                c.ld(1, "ACC")
                c.st(("V", rv), 1)

            c.ld(1, "R", l * H + h)              # look Q up in this head's trie
            c.st("PTRQ", 1)
            for j in range(dh):
                c.ld(2, "Q", j)
                c.ld(3, "PTRQ")
                skip = a.newlab()
                a.jz(3, skip)
                a.op(3, 3, 2, "+")
                a.raw("load", 3, 3)
                c.st("PTRQ", 3)
                a.label(skip)
            miss, copied = a.newlab(), a.newlab()
            c.ld(1, "PTRQ")                      # copy the retrieved leaf into O
            a.jz(1, miss)
            for rv in range(dv):
                c.ld(1, "PTRQ")
                a.opi(1, 1, rv, "+")
                a.raw("load", 1, 1)
                c.st(("O", rv), 1)
            a.goto(copied)
            a.label(miss)
            a.zero(1)                            # a miss delivers zeros
            for rv in range(dv):
                c.st(("O", rv), 1)
            a.label(copied)

            c.addr(1, "R", l * H + h)            # only now insert (K, V)
            c.st("PTRK", 1)
            for j in range(dh):
                c.ld(2, "K", j)
                c.ld(3, "PTRK")
                a.raw("load", 1, 3)
                have = a.newlab()
                a.jnz(1, have)
                c.ld(1, "HP")                    # allocate a node (2 cells)
                a.opi(1, 1, 2, "-")
                c.st("HP", 1)
                a.raw("store", 3, 1)
                a.label(have)
                a.op(1, 1, 2, "+")
                c.st("PTRK", 1)
            c.ld(3, "PTRK")                      # leaf slot
            a.raw("load", 1, 3)
            have = a.newlab()
            a.jnz(1, have)
            c.ld(1, "HP")                        # allocate the leaf (d_h cells)
            a.opi(1, 1, dv, "-")
            c.st("HP", 1)
            a.raw("store", 3, 1)
            a.label(have)
            c.st("LEAFK", 1)
            for rv in range(dv):                 # V into the leaf
                c.ld(1, "LEAFK")
                a.opi(1, 1, rv, "+")
                c.ld(2, "V", rv)
                a.raw("store", 1, 2)

            for ci in range(d):                  # project O into U, row by row
                _emit_dot(c, WO[ci], pk, "O")
                c.ld(1, "ACC")
                c.st(("U", ci), 1)
            _emit_addv(c, "S", "U", d)           # s += head output
        _emit_addv(c, "X", "S", d)               # x += s

        # MLP, neuron by neuron; S becomes the output accumulator
        a.zero(1)
        for ci in range(d):
            c.st(("S", ci), 1)
        for hh in range(dff):
            _emit_dot(c, layer["W1"][hh], pk)
            a.imm(1, pk(layer["b"][hh]))
            c.ld(2, "ACC")
            a.call("FPADD")
            c.st("ACC", 3)
            relu = a.newlab()
            c.ld(1, "ACC")
            a.opi(2, 1, f.p - 1, ">>")
            a.jz(2, relu)
            a.zero(1)
            c.st("ACC", 1)
            a.label(relu)
            c.ld(1, "ACC")
            c.st("OV", 1)
            hskip = a.newlab()
            a.jz(1, hskip)                       # z_h = 0 contributes exactly 0
            for ci in range(d):
                a.imm(1, pk(layer["W2"][ci, hh]))
                c.addr(2, "S", ci)
                a.call("FMAV")
            a.label(hskip)
        _emit_addv(c, "X", "S", d)               # x += mlp

    # Predictions at earlier prompt positions are unused.
    c.ld(1, "POS")
    c.ld(2, "NN")
    a.op(3, 2, 1, "<")
    a.jz(3, "TOKEN")

    # unembedding with running strictly-greater argmax (first max wins)
    for tok in range(V):
        _emit_dot(c, T.unemb[tok], pk)
        if tok == 0:
            c.ld(1, "ACC")
            c.st("BESTV", 1)
            a.zero(1)
            c.st("BESTI", 1)
        else:
            c.ld(1, "ACC")
            c.st("FA", 1)
            c.ld(1, "BESTV")
            c.st("FB", 1)
            a.call("FPGT")
            gskip = a.newlab()
            a.jz(3, gskip)
            c.ld(1, "ACC")
            c.st("BESTV", 1)
            a.imm(1, tok)
            c.st("BESTI", 1)
            a.label(gskip)
    c.ld(1, "BESTI")
    c.st("PRED", 1)

    # bookkeeping; NGEN is counted in trace mode only
    if trace_output:
        c.ld(1, "NGEN")
        a.opi(1, 1, 1, "+")
        c.st("NGEN", 1)
        c.ld(2, "PRED")
        a.raw("store", 1, 2)                     # mem[NGEN] = token, m = NGEN
    else:
        c.ld(2, "PRED")
        nowrite = a.newlab()
        a.opi(3, 2, TOK["<out>"], "==")
        a.jz(3, "NOTOUT")
        a.zero(1)                                # <out>: restart collection
        c.st("NOUT", 1)
        a.mov(1, a.ONE)
        c.st("SEEN", 1)
        a.goto(nowrite)
        a.label("NOTOUT")
        a.opi(3, 2, TOK["<eos>"], "==")
        a.jnz(3, nowrite)                        # <eos> itself is not part of y
        c.ld(1, "SEEN")
        a.jz(1, nowrite)
        c.ld(1, "NOUT")
        a.opi(1, 1, 1, "+")
        c.st("NOUT", 1)
        a.raw("store", 1, 2)                     # mem[NOUT] = token
        a.label(nowrite)
        c.ld(2, "PRED")
    a.opi(3, 2, TOK["<eos>"], "==")
    a.jnz(3, "FIN")
    a.goto("TOKEN")
    a.label("FIN")
    c.ld(1, "NGEN" if trace_output else "NOUT")
    a.zero(2)
    a.raw("store", 2, 1)                         # mem[0] = m
    a.raw("halt")

    _emit_routines(c)
    return a.finish(), cells, f


def lema_to_ram(T, w=None, trace_output=False):
    """Return a WordRAM simulating greedy generation by T at word size w.

    Output: mem[0] = m and mem[1..m] = y, where y is the output in the sense of
    the appendix's generation definition
    (the tokens after the last <out>, excluding <eos>).  With
    trace_output=True, y is instead the full generated token sequence
    including <eos>, which is useful for differential tests. Generation stops
    at the first <eos>, exactly as in the appendix."""
    prog, cells, f = _compile(T, trace_output)
    if w is None:
        w = max(4 * f.p, len(prog).bit_length() + 1,
                cells.size.bit_length() + 2)
    return WordRAM(prog, Asm.R, w)


# --- exact resource accounting ------------------------------------------------
# These constants are derived by expanding the assembler macros (ld/st take
# three instructions, opi two, jz four, and labels none). In particular:
#   round-right = 73, round-scaled = 334, unpack-two = 76;
#   fmul = 14 + 76 + 32 + 334 + 4 = 460;
#   fadd = 14 + 76 + 34 + 25 + 48 + 65 + 334 + 14 = 610;
#   fgt = 13 + 3 + 2 + 9 + 11 = 38 (prefix, different signs,
#          same-sign dispatch, positive case, negative case).
# msb makes all three bodies independent of the precision.
C_MUL, C_ADD, C_GT = 460, 610, 38


def _counts(T):
    """Call-site counts per processed token."""
    d = T.dims()
    V, L, H, dd, dh, dff = (d["vocab"], d["L"], d["H"], d["d"],
                            d["d_h"], d["d_ff"])
    dv = dh
    n_fmac = dd * (L * (H * (2 * dh + 2 * dv) + dff) + V)
    n_fmav = L * dd * dff
    n_fpadd = L * (H * dd + 2 * dd + dff)
    n_fpgt = V - 1
    return V, L, H, dd, dh, dv, dff, n_fmac, n_fmav, n_fpadd, n_fpgt


def _cellcount(T):
    """S_0: the blocks X, S, U, Q, K, V, O, R and the scalar workspace."""
    d = T.dims()
    dv = d["d_h"]
    return (3 * d["d"] + 2 * d["d_h"] + 2 * dv + d["L"] * d["H"]
            + len(_SCELLS) + len(_FCELLS))


def PARAMETERS(T):
    """Number N of scalar parameter slots in the uniformly padded model."""
    d = T.dims()
    V, L, H, dd, dh, dff = (d["vocab"], d["L"], d["H"], d["d"],
                            d["d_h"], d["d_ff"])
    return 2 * V * dd + L * (4 * H * dd * dh + 2 * dd * dff + dff)


def PROGRAM_LEN(T, trace_output=False):
    """Exact length of the emitted program, derived from the compiler blocks."""
    V, L, H, dd, dh, dv, dff, *_ = _counts(T)
    emb = 3 + 4 * (V - 1) + V * (4 * dd + 2)
    head = (dh * (14 * dd + 63) + dv * (7 * dd + 31) + 7 * dd * dv
            + 22 * dd + 39)
    layer = H * head + dff * (13 * dd + 37) + 30 * dd + 2
    unemb = V * (7 * dd + 4) + 10 + 29 * (V - 1) + 6
    routines = ((33 + C_MUL + C_ADD) + (37 + C_MUL + C_ADD)
                + (10 + C_ADD) + (C_GT + 1))
    bookkeeping = 35 if trace_output else 65
    return 15 + 28 + emb + L * layer + unemb + bookkeeping + routines


def PER_TOKEN(T, trace_output=False):
    """Upper bound on executed steps per processed token."""
    V, L, H, dd, dh, dv, dff, n_fmac, n_fmav, n_fpadd, n_fpgt = _counts(T)
    routines = ((33 + C_MUL + C_ADD) + (37 + C_MUL + C_ADD)
                + (10 + C_ADD) + (C_GT + 1))
    body = PROGRAM_LEN(T, trace_output) - 15 - routines
    return (body + n_fmac * (33 + C_MUL + C_ADD)
            + n_fmav * (37 + C_MUL + C_ADD)
            + n_fpadd * (10 + C_ADD) + n_fpgt * (C_GT + 1))


def MIN_WORD(T, trace_output=False):
    """Input-independent numerical/program-size requirement."""
    f = fmt_for(T.precision)
    return max(4 * f.p, (PROGRAM_LEN(T, trace_output) - 1).bit_length(),
               max(1, _cellcount(T) - 1).bit_length())


def REQUIRED_WORD(T, s_T, n, m, trace_output=False):
    """Sufficient word size including the memory needed on this execution."""
    heap = _cellcount(T) + 3 * s_T * T.dims()["d_h"]
    address_span = max(n, m) + 1 + heap
    return max(MIN_WORD(T, trace_output), address_span.bit_length())


def WORD(T, B, trace_output=False):
    """Conservative word size for worst-case cache use when max(|x|, |y|) <= B."""
    d = T.dims()
    heap = _cellcount(T) + 3 * d["L"] * d["H"] * d["d_h"] * (1 << d["d_h"])
    return max(MIN_WORD(T, trace_output), (B + 1 + heap).bit_length())


REGISTERS = lambda T: Asm.R
STATIC_CELLS = lambda T: _cellcount(T)
STEPS = lambda T, n, t, trace_output=False: (
    15 + (n + t - 1) * PER_TOKEN(T, trace_output))
SPACE = lambda T, s_T, n, m: (
    Asm.R + max(n, m) + 1 + _cellcount(T)
    + s_T * 3 * T.dims()["d_h"])
