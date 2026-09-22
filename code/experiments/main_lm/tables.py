"""The language-model tables of the appendix, from out/<run>/{config,eval}.json.

    python tables.py            # sizes (tab:lm-models), losses (tab:lm-ce), head sizes (tab:lm-ce-heads)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                              # code/
from experiments.main_lm.plot_lm_section import model_params            # noqa: E402

OUT = HERE / "out"
DIMS = ((256, "29M"), (512, "77M"), (768, "162M"), (1024, "309M"), (1280, "525M"), (1536, "834M"))


def ce(run: str, ctx: str = "2048"):
    f = OUT / run / "eval.json"
    return json.loads(f.read_text())[ctx]["ce"] if f.exists() else None


def fmt(v):
    return "--" if v is None else f"{v:.3f}"


def main():
    print("d     L   softmax/LEMA  GDN    tokens")
    for d, label in DIMS:
        c = json.loads((OUT / f"lema{d}_h64" / "config.json").read_text())
        g = json.loads((OUT / f"gdn{d}" / "config.json").read_text())["model"]
        m, t = c["model"], c["train"]
        print(f"{d:<5} {m['depth']:<3} {model_params(m) / 1e6:>5.0f}M        {model_params(g) / 1e6:>4.0f}M  "
              f"{t['steps'] * t['batch'] * t.get('grad_accum', 1) * 2048 / 1e9:.1f}B")
    print("\nparams  softmax  GDN    LEMA")
    for d, label in DIMS:
        print(f"{label:>6}  {fmt(ce(f'rope{d}')):<12}  {fmt(ce(f'gdn{d}')):<6} {fmt(ce(f'lema{d}_h64'))}")
    print("\nparams  d_h=8   d_h=16  d_h=32  d_h=64")
    for d, label in DIMS[:4]:
        print(f"{label:>6}  " + "  ".join(f"{fmt(ce(f'lema{d}_h{h}')):<6}" for h in (8, 16, 32, 64)))


if __name__ == "__main__":
    main()
