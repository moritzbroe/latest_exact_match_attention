# Executable word-RAM / LEMA constructions

Run these scripts from the directory containing verification/, with Python and NumPy installed:

```sh
OPENBLAS_NUM_THREADS=1 python verification/test_ram_to_lema.py
OPENBLAS_NUM_THREADS=1 python verification/test_ram_to_lema_invariants.py
OPENBLAS_NUM_THREADS=1 python verification/test_fp.py
OPENBLAS_NUM_THREADS=1 python verification/test_lema_to_ram.py
```

`test_ram_to_lema.py` takes about five minutes, `test_lema_to_ram.py` about two,
`test_ram_to_lema_invariants.py` about twenty seconds and `test_fp.py` a few seconds. Each script exits unsuccessfully
on a failed check.

| Script | Checks |
| --- | --- |
| `test_ram_to_lema.py` | Every generated transcript token, integer/IEEE binary16 agreement on the examples, exact architecture sizes, and token/key/magnitude bounds. |
| `test_ram_to_lema_invariants.py` | All 13 ALU operations on every operand pair for word sizes 2–6 (70,928 cases), every multiplication stage, and actual residual-state contracts for records, operands, lookup, arithmetic, assembly, and output. Includes maximal input/output length and final-write cases. |
| `test_fp.py` | Addition, multiplication, and comparison against exact rational rounding, exhaustively for small formats and on IEEE binary16/32/64 boundary cases; also runs the emitted RAM arithmetic instructions. |
| `test_lema_to_ram.py` | Token outputs, final residuals, every stored key/value entry, program independence from word size, resource bounds, fractional arithmetic with equal query/key projections, and a complete round trip. |

The `int<k>` mode in `lema.py` is an exact-integer verification shortcut, not a transformer
format of the theorem; the binary16 agreement checked on the examples is a statement about
those sizes only.

`lema_to_ram.py` compiles the prescribed floating-point operations and trie updates into the
instruction set of `wordram.py`. Its default output mode returns only the payload between the
generated `<out>` and `<eos>` delimiters; `trace_output=True` returns every prediction,
including delimiters. Word size 1 is tested as a boundary case for the forward compiler; the
theorem assumes word size at least 2.
