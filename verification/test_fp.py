"""Check the finite IEEE arithmetic used by the Theorem 2 compiler.

Three small formats are checked exhaustively, including subnormals, rounding
ties and cancellation. Larger formats use targeted boundary cases. Each test
checks both the integer reference algorithm and the emitted word-RAM body.
Overflowing operations are outside the theorem's finite-execution premise.
"""

import random
import sys
from fractions import Fraction

import numpy as np

from lema_to_ram import (Asm, C_ADD, C_GT, C_MUL, Cells, Ctx, Fmt, alg_add,
                         alg_gt, alg_mul, emit_fadd, emit_fgt, emit_fmul,
                         fmt_for)
from wordram import WordRAM

_NUMPY = {"fp16": (np.float16, np.uint16),
          "fp32": (np.float32, np.uint32),
          "fp64": (np.float64, np.uint64)}

_CELLS = ["FA", "FB", "FR", "SA", "SB", "EA", "EB2", "MA", "MB", "SGN",
          "EE", "QQ", "RR", "HH", "W1", "W2", "W4",
          "WD"]


def build_driver(f, kind, w):
    """A RAM computing one float operation: mem[1] op mem[2] -> mem[1]."""
    a, cells = Asm(), Cells()
    for name in _CELLS:
        cells.add(name)
    c = Ctx(a, cells, f)
    a.imm(0, 1)
    a.opi(8, 8, cells.size, "-")
    a.imm(1, 1)
    a.raw("load", 1, 1)
    c.st("FA", 1)
    a.imm(1, 2)
    a.raw("load", 1, 1)
    c.st("FB", 1)
    if kind == "gt":
        emit_fgt(c)
        a.mov(1, 3)
    else:
        (emit_fmul if kind == "mul" else emit_fadd)(c)
        c.ld(1, "FR")
    a.imm(2, 1)
    a.raw("store", 2, 1)
    a.imm(2, 0)
    a.imm(3, 1)
    a.raw("store", 2, 3)
    a.raw("halt")
    M = WordRAM(a.finish(), Asm.R, w)
    body = {"mul": C_MUL, "add": C_ADD, "gt": C_GT}[kind]
    wrapper = 22 if kind != "gt" else 20
    if len(M.program) != wrapper + body:
        raise AssertionError(f"{kind} body has {len(M.program)-wrapper} "
                             f"instructions, analytically expected {body}")
    return M


def all_values(f):
    """All finite packed values, with a single canonical zero."""
    yield 0
    for s in (0, 1):
        for e in range(0, (1 << f.pe) - 1):
            lo = 1 if e == 0 else 0
            for frac in range(lo, f.mlo):
                yield s * f.sign_bit + (e << f.pm) + frac


def boundary_pairs(f, rng, n):
    """Pairs concentrated near zero, the normal boundary and cancellation."""
    vals = list(all_values(f)) if f.p <= 12 else None
    out = []
    for _ in range(n):
        if vals is not None:
            out.append((rng.choice(vals), rng.choice(vals)))
            continue
        # One operand near the subnormal/normal boundary.
        ea = rng.choice([0, 1, 2, 3])
        fa = rng.randrange(f.mlo)
        if ea == 0 and fa == 0:
            fa = 1
        A = rng.randrange(2) * f.sign_bit + (ea << f.pm) + fa
        # The other is either near one, near zero, or close to A.
        mode = rng.randrange(3)
        if mode == 0:
            eb = max(0, min((1 << f.pe) - 2, f.bias + rng.randint(-2, 2)))
            fb = rng.randrange(f.mlo)
            B = rng.randrange(2) * f.sign_bit + (eb << f.pm) + fb
        elif mode == 1:
            eb = rng.choice([0, 1, 2, 3])
            fb = rng.randrange(f.mlo)
            if eb == 0 and fb == 0:
                fb = 1
            B = rng.randrange(2) * f.sign_bit + (eb << f.pm) + fb
        else:
            body = A & (f.sign_bit - 1)
            frac = max(0, min(f.frac_mask,
                              (body & f.frac_mask) + rng.randint(-3, 3)))
            B = ((1 - (A >> (f.p - 1))) * f.sign_bit
                 + (body & ~f.frac_mask) + frac)
        out.append((A, B))
    return out


def addition_pairs(f):
    """Exact-shift and early-return cases around the add alignment cutoff."""
    pow2 = lambda e: Fraction(1 << e) if e >= 0 else Fraction(1, 1 << -e)
    one, odd = f.pack(1), f.pack(1 + pow2(-f.pm))
    pairs = []
    for gap in (f.pm + 1, f.pm + 2, f.pm + 3):
        small = f.pack(pow2(-gap))
        for large in (one, odd):  # even and odd low bits at the half-ulp tie
            for sa in (0, f.sign_bit):
                for sb in (0, f.sign_bit):
                    a, b = large + sa, small + sb
                    pairs.extend(((a, b), (b, a)))

    # A much wider gap, including the minimum subnormal significand.
    minimum = 1
    for sa in (0, f.sign_bit):
        for sb in (0, f.sign_bit):
            a, b = one + sa, minimum + sb
            pairs.extend(((a, b), (b, a)))

    largest = ((f.exp_mask - 1) << f.pm) + f.frac_mask
    pairs.extend(((largest, minimum), (minimum, largest),
                  (largest, minimum + f.sign_bit),
                  (minimum + f.sign_bit, largest)))

    # An aligned addition carrying into the highest finite binade.
    near_top = f.pack((2 - pow2(-f.pm)) * pow2(f.emax - 1))
    top_ulp = f.pack(pow2(f.emax - 1 - f.pm))
    pairs.extend(((near_top, top_ulp), (top_ulp, near_top),
                  (near_top + f.sign_bit, top_ulp + f.sign_bit),
                  (top_ulp + f.sign_bit, near_top + f.sign_bit)))
    top_quarter = f.pack(pow2(f.emax - 1 - f.pm - 2))
    for sa in (0, f.sign_bit):
        for sb in (0, f.sign_bit):
            a, b = near_top + sa, top_quarter + sb
            pairs.extend(((a, b), (b, a)))

    # Cancellation on both sides of the normal/subnormal boundary.
    normal = 1 << f.pm
    max_subnormal = f.frac_mask
    pairs.extend(((normal, max_subnormal + f.sign_bit),
                  (max_subnormal + f.sign_bit, normal),
                  (normal + f.sign_bit, max_subnormal),
                  (max_subnormal, normal + f.sign_bit),
                  (normal, normal + f.sign_bit),
                  (normal + f.sign_bit, normal)))
    return pairs


def check_alg(f, pairs, name, failures):
    subnormal = underflow_zero = finite = 0
    for A, B in pairs:
        for op, alg, ref in (("mul", alg_mul, f.ref_mul),
                             ("add", alg_add, f.ref_add)):
            try:
                want = ref(A, B)
            except OverflowError:
                continue
            got, finite = alg(f, A, B), finite + 1
            if got != want:
                failures.append(
                    f"{name} {op}: A={A:#x} B={B:#x} "
                    f"got={got:#x} want={want:#x}")
                return 0, 0, 0
            exact = (f.unpack(A) * f.unpack(B) if op == "mul"
                     else f.unpack(A) + f.unpack(B))
            if exact != 0 and want == 0:
                underflow_zero += 1
            if want != 0 and ((want >> f.pm) & f.exp_mask) == 0:
                subnormal += 1
        got = alg_gt(f, A, B)
        want = int(f.unpack(A) > f.unpack(B))
        if got != want:
            failures.append(f"{name} gt: A={A:#x} B={B:#x}")
            return 0, 0, 0
    return finite, subnormal, underflow_zero


def check_ram(f, w, pairs, name, failures):
    for kind, alg, ref in (("mul", alg_mul, f.ref_mul),
                           ("add", alg_add, f.ref_add),
                           ("gt", alg_gt, None)):
        M = build_driver(f, kind, w)
        for A, B in pairs:
            if ref is not None:
                try:
                    ref(A, B)
                except OverflowError:
                    continue
            want = alg(f, A, B)
            ran = M.run_on([A, B], max_steps=10000)
            if ran is False:
                failures.append(f"{name} RAM {kind}: did not halt")
                return
            if ran[0][1][0] != want:
                failures.append(
                    f"{name} RAM {kind}: A={A:#x} B={B:#x} "
                    f"got={ran[0][1][0]:#x} want={want:#x}")
                return


def check_numpy(f, precision, pairs, failures):
    """Check that the parameterized format agrees with the IEEE NumPy formats."""
    dtype, uint = _NUMPY[precision]
    with np.errstate(over="ignore", invalid="ignore"):
        for A, B in pairs:
            av = np.array([A], dtype=uint).view(dtype)[0]
            bv = np.array([B], dtype=uint).view(dtype)[0]
            for op, ufunc, ref in (("mul", np.multiply, f.ref_mul),
                                   ("add", np.add, f.ref_add)):
                try:
                    want = ref(A, B)
                except OverflowError:
                    continue
                value = ufunc(av, bv, dtype=dtype)
                got = int(np.array([value], dtype=dtype).view(uint)[0])
                if got & (f.sign_bit - 1) == 0:  # identify the two IEEE zeros
                    got = 0
                if got != want:
                    failures.append(
                        f"{precision} NumPy {op}: A={A:#x} B={B:#x} "
                        f"got={got:#x} want={want:#x}")
                    return


def check_overflow_threshold(f, precision, failures):
    """Check the exact finite/infinite boundary used by the appendix."""
    pow2 = lambda e: Fraction(1 << e) if e >= 0 else Fraction(1, 1 << -e)
    largest = ((f.exp_mask - 1) << f.pm) + f.frac_mask
    half_ulp = f.pack(pow2(f.emax - f.pm - 1))
    quarter_ulp = f.pack(pow2(f.emax - f.pm - 2))
    for sign in (0, f.sign_bit):
        a, b, below = sign + largest, sign + half_ulp, sign + quarter_ulp
        try:
            f.ref_add(a, b)
        except OverflowError:
            pass
        else:
            failures.append(f"{precision}: midpoint at overflow did not overflow")
            return
        if f.ref_add(a, below) != a:
            failures.append(f"{precision}: value below overflow midpoint did not round finite")
            return
        if precision in _NUMPY:
            dtype, uint = _NUMPY[precision]
            av = np.array([a], dtype=uint).view(dtype)[0]
            bv = np.array([b], dtype=uint).view(dtype)[0]
            qv = np.array([below], dtype=uint).view(dtype)[0]
            with np.errstate(over="ignore", invalid="ignore"):
                if not np.isinf(np.add(av, bv, dtype=dtype)):
                    failures.append(f"{precision}: NumPy disagrees at overflow midpoint")
                    return
                got = int(np.array([np.add(av, qv, dtype=dtype)],
                                   dtype=dtype).view(uint)[0])
            if got != a:
                failures.append(f"{precision}: NumPy disagrees below overflow midpoint")
                return


def main():
    failures = []
    rng = random.Random(0)

    for pm, pe in ((1, 2), (2, 3), (3, 4)):
        f = Fmt(pm, pe)
        vals = list(all_values(f))
        pairs = [(A, B) for A in vals for B in vals]
        finite, subnormal, underflow_zero = check_alg(
            f, pairs, f"Fmt({pm},{pe}) exhaustive", failures)
        # The smallest format has no representable quarter-ulp operand at
        # its overflow boundary; the exhaustive pairs cover its arithmetic.
        if pe >= 3:
            check_overflow_threshold(f, f"Fmt({pm},{pe})", failures)
        print(f"  Fmt({pm},{pe}): {len(pairs)} pairs, {finite} finite operations, "
              f"{subnormal} subnormal and {underflow_zero} zero underflows")
        if pm <= 2:
            check_ram(f, 24, pairs, f"Fmt({pm},{pe}) exhaustive", failures)

    for prec in ("fp16", "fp32", "fp64", "int20"):
        f = fmt_for(prec)
        pairs = boundary_pairs(f, rng, 800)
        finite, subnormal, underflow_zero = check_alg(f, pairs, prec, failures)
        check_overflow_threshold(f, prec, failures)
        check_ram(f, max(4 * f.p, 32), pairs[:160], prec, failures)
        targeted = addition_pairs(f)
        check_alg(f, targeted, f"{prec} alignment", failures)
        for w in (max(4 * f.p, 32), 8 * f.p):
            check_ram(f, w, targeted, f"{prec} alignment w={w}", failures)
        if prec in _NUMPY:
            check_numpy(f, prec, pairs, failures)
        if subnormal == 0 or underflow_zero == 0:
            failures.append(
                f"{prec}: boundary sample missed subnormal/zero underflow "
                f"({subnormal}, {underflow_zero})")
        print(f"  {prec}: {finite} finite boundary operations, "
              f"{subnormal} subnormal and {underflow_zero} zero underflows; "
              f"{len(targeted)} alignment pairs at two word sizes")

    print()
    if failures:
        print(f"{len(failures)} failure(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
