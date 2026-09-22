"""The S-NIAH-1 table of the appendix (tab:sniah1): accuracy against context length for
every model, from the generation cells in ../out/gen. No GPU.

    python sniah1_table.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                               # code/
from experiments.app_retrieval.paper import ruler_cells as rc          # noqa: E402

CTXS = [2048, 4096, 8192, 16384, 32768, 65536]
ROWS = [("309M", "softmax, extended", "rope1024_16k"),
        ("", "GDN, extended", "gdn1024_16k"),
        ("", "LEMA", "lema1024_h64"),
        ("834M", "softmax, extended", "rope1536_16k"),
        ("", "GDN, extended", "gdn1536_16k"),
        ("", "LEMA", "lema1536_h64")]


def main():
    got, _ = rc.scan("niah_single_1-noise", [r for _, _, r in ROWS], CTXS)
    print(f"{'':6s} {'model':26s} " + " ".join(f"{c // 1024:>4d}k" for c in CTXS))
    for size, name, run in ROWS:
        cells = got.get(run, {})
        print(f"{size:6s} {name:26s} "
              + " ".join(f"{cells[c][0]:5.2f}" if c in cells else "   --" for c in CTXS))


if __name__ == "__main__":
    main()
