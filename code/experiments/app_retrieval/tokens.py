"""GPT-2 tokens: the tokenizer the models were trained with, and the split of an answer
into the tokens it adds to a prompt."""
from __future__ import annotations

from pathlib import Path
import os

import numpy as np

EOT = 50256

# ---------------------------------------------------------------- tokenizer

_TOK = None


def tokenizer(name="gpt2"):
    """The GPT-2 tokenizer the corpus was built with (main_lm/prepare_data.py).

    Cached as out/<name>_tokenizer.json on first use, and loaded from there afterwards,
    so a machine without network still builds byte-identical prompts.
    """
    global _TOK
    if _TOK is None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from tokenizers import Tokenizer
        local = Path(__file__).resolve().parent / "out" / f"{name}_tokenizer.json"
        if local.exists():
            _TOK = Tokenizer.from_file(str(local))
        else:
            _TOK = Tokenizer.from_pretrained(name)
            local.parent.mkdir(parents=True, exist_ok=True)
            _TOK.save(str(local))
    return _TOK


def enc(s: str) -> np.ndarray:
    return np.asarray(tokenizer().encode(s, add_special_tokens=False).ids, dtype=np.int64)


def dec(ids) -> str:
    return tokenizer().decode([int(i) for i in ids])


def split(prefix: str, full: str) -> tuple[np.ndarray, np.ndarray]:
    """(prefix tokens, the tokens `full` adds). Raises if the prefix is not a token
    prefix of `full` -- then the answer could not be scored as written."""
    p, f = enc(prefix), enc(full)
    if not np.array_equal(f[:len(p)], p):
        raise ValueError(f"BPE boundary: {prefix!r} is not a token prefix of {full!r}")
    return p, f[len(p):]


