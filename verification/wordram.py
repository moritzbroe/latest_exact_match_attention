"""Word-RAM with the semantics defined in the appendix.

The verifier additionally permits w = 1 as a boundary case; the paper starts
at w = 2 only so that its O(log w) precision statement needs no additive 1.

An instruction is a tuple, one of

    ("set",   i, c)          R_i <- c
    ("op",    i, j, j2, s)   R_i <- R_j (s) R_j2
    ("msb",   i, j)          R_i <- msb(R_j), with msb(0) = 0
    ("load",  i, j)          R_i <- mem[R_j]
    ("store", i, j)          mem[R_i] <- R_j
    ("jnz",   i, j)          if R_i != 0 goto R_j
    ("halt",)

with s one of "+ - * << >> & | ^ < <= == !=".  The machine halts when it
executes halt or when the program counter leaves [|P|].
"""


def _ops(w):
    M = 1 << w
    return {
        "+":  lambda a, b: (a + b) % M,
        "-":  lambda a, b: (a - b) % M,
        "*":  lambda a, b: (a * b) % M,
        "<<": lambda a, b: (a << b) % M if b < w else 0,
        ">>": lambda a, b: a >> b if b < w else 0,
        "&":  lambda a, b: a & b,
        "|":  lambda a, b: a | b,
        "^":  lambda a, b: a ^ b,
        "<":  lambda a, b: int(a < b),
        "<=": lambda a, b: int(a <= b),
        "==": lambda a, b: int(a == b),
        "!=": lambda a, b: int(a != b),
    }


_ARITY = {"set": 3, "op": 5, "msb": 3, "load": 3,
          "store": 3, "jnz": 3, "halt": 1}


class WordRAM:
    def __init__(self, program, r, w):
        self.program = list(program)
        self.r = r
        self.w = w
        if not isinstance(w, int) or w < 1:
            raise ValueError(f"word size must be a positive integer, got {w!r}")
        self.ops = _ops(w)
        if not 1 <= r <= 1 << w:
            raise ValueError(f"need 1 <= r <= 2^w, got r={r}, w={w}")
        if not 1 <= len(self.program) <= 1 << w:
            raise ValueError(f"need 1 <= |P| <= 2^w, got |P|={len(self.program)}, w={w}")
        self._check_program()

    def _check_program(self):
        def bad(l, msg):
            raise ValueError(f"instruction {l} = {self.program[l]}: {msg}")

        for l, instr in enumerate(self.program):
            if not instr:
                bad(l, "an instruction cannot be empty")
            kind = instr[0]
            if kind not in _ARITY:
                bad(l, f"unknown instruction {kind!r}")
            if len(instr) != _ARITY[kind]:
                bad(l, f"{kind} takes {_ARITY[kind] - 1} arguments")
            if kind == "set":
                if not 0 <= instr[2] < 1 << self.w:
                    bad(l, f"constant {instr[2]} is not in [2^{self.w}]")
                regs = instr[1:2]
            elif kind == "op":
                if instr[4] not in self.ops:
                    bad(l, f"unknown operation {instr[4]!r}")
                regs = instr[1:4]
            elif kind == "jnz":
                regs = instr[1:3]
            elif kind == "halt":
                regs = ()
            else:
                regs = instr[1:3]
            for i in regs:
                if not 0 <= i < self.r:
                    bad(l, f"register index {i} is not in [{self.r}]")

    def initial_config(self, x):
        """pc, registers, memory on input x (mem = (n, x_1, ..., x_n, 0, 0, ...))."""
        mem = {0: len(x)}
        for i, xi in enumerate(x, start=1):
            mem[i] = xi
        return 0, [0] * self.r, mem

    def step(self, pc, reg, mem):
        """One execution step. Returns (new pc or None, write or None), where a
        write is ("reg", i, v) or ("mem", a, v).  None means the machine has
        halted, i.e. it executed halt or the program counter left [|P|]."""
        instr = self.program[pc]
        kind = instr[0]
        write = None

        if kind == "set":
            _, i, c = instr
            reg[i] = c
            write = ("reg", i, c)
        elif kind == "op":
            _, i, j, j2, s = instr
            reg[i] = self.ops[s](reg[j], reg[j2])
            write = ("reg", i, reg[i])
        elif kind == "msb":
            _, i, j = instr
            reg[i] = reg[j].bit_length() - 1 if reg[j] else 0
            write = ("reg", i, reg[i])
        elif kind == "load":
            _, i, j = instr
            reg[i] = mem.get(reg[j], 0)
            write = ("reg", i, reg[i])
        elif kind == "store":
            _, i, j = instr
            mem[reg[i]] = reg[j]
            write = ("mem", reg[i], reg[j])
        elif kind == "jnz":
            _, i, j = instr
            if reg[i] != 0:
                t = reg[j]
                return (None if t >= len(self.program) else t), None
        elif kind == "halt":
            return None, None

        pc += 1
        return (None if pc >= len(self.program) else pc), write

    def run_on(self, x, max_steps=10 ** 6, return_trace=False):
        """Run on input x. Returns False if max_steps is hit, else ((m, y), info)
        where info holds the number of steps and the space usage, and the list of
        (pc, write, next pc) per step if return_trace=True."""
        n = len(x)
        if not n < 1 << self.w:
            raise ValueError(f"input length {n} is not below 2^{self.w}")
        if not all(0 <= xi < 1 << self.w for xi in x):
            raise ValueError(f"input words are not all in [2^{self.w}]")

        pc, reg, mem = self.initial_config(x)
        accessed, trace, steps = set(), [], 0

        while pc is not None:
            if steps >= max_steps:
                return False
            instr = self.program[pc]
            if instr[0] == "load":
                accessed.add(reg[instr[2]])
            elif instr[0] == "store":
                accessed.add(reg[instr[1]])
            pc_next, write = self.step(pc, reg, mem)
            if return_trace:
                trace.append((pc, write, pc_next))
            pc, steps = pc_next, steps + 1

        m = mem.get(0, 0)
        y = [mem.get(i, 0) for i in range(1, m + 1)]
        lo = max(n, m)
        info = {"steps": steps,
                "space": self.r + lo + 1 + len([a for a in accessed if a > lo])}
        if return_trace:
            info["trace"] = trace
        return (m, y), info
