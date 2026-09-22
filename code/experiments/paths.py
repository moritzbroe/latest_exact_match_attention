"""Where things live on disk, in one place. Import as `from experiments.paths import ...`
with `code/` on sys.path (every script puts it there)."""
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
DATA = CODE / "data" / "fineweb_edu100_gpt2"     # main_lm/prepare_data.py writes it
LM_OUT = CODE / "experiments" / "main_lm" / "out"
