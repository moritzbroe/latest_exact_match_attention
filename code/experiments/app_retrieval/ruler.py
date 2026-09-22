# Adapted from NVIDIA RULER (https://github.com/NVIDIA/RULER), Copyright (c) 2024, NVIDIA CORPORATION,
# licensed under the Apache License, Version 2.0. The transcribed parts are marked "copied from" below.
"""RULER's needle-in-a-haystack prompts (Hsieh et al. 2024), transcribed from the released
generator (scripts/data/synthetic/niah.py, constants.py, the essay download script and
wonderwords) so that a prompt is built here exactly as RULER builds it: the same draws in
the same order from one random.Random(seed), keys and values drawn with replacement from
sorted(set(adjective-noun)), depths from the 40-value grid, sentence boundaries by nltk's
punkt, the haystack size from the binary search that reserves the 128 generated tokens,
and the per-sample shrink when a sample does not fit. Nothing is checked or resampled.

Tasks: niah_single_1 (haystack `noise`, RULER's repeated sentence), niah_single_2,
niah_single_3 and niah_multikey_1 (haystack `essay`, RULER's Paul Graham essays),
niah_multikey_2 (haystack `needle`, RULER's distractor lines).

    python ruler.py --fetch-assets          # once: essays, wonderwords, punkt
    python ruler.py --selftest --ctx 4096 16384 --n 20
    python ruler.py --show niah_single_2 --hay essay --ctx 4096
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import urllib.request
import uuid as _uuid
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from tokens import EOT, dec, enc, split                                # noqa: E402

OUT = HERE / "out"
ESSAY_JSON = OUT / "PaulGrahamEssays.json"
ADJ_FILE = OUT / "wonderwords_adjectivelist.txt"
NOUN_FILE = OUT / "wonderwords_nounlist.txt"
NLTK_DATA = OUT / "nltk_data"
HF_ROWS = ("https://datasets-server.huggingface.co/rows?dataset=baber%2Fpaul_graham_essays"
           "&config=default&split=train&offset={off}&length=100")
WW_URL = ("https://raw.githubusercontent.com/mrmaxguns/wonderwordsmodule/master/"
          "wonderwords/assets/{name}.txt")

# ---------------------------------------------------------------- RULER's strings

TASK_TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. Make sure to "
    "memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat "
    "are all the special magic {type_needle_v} for {query} mentioned in the provided text?")
ANSWER_PREFIX = (
    " The special magic {type_needle_v} for {query} mentioned in the provided text are")
NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
TOKENS_TO_GENERATE = 128
DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))

RULER_TASKS = {
    # name                 haystacks                key type   value type   nk nv nq
    "niah_single_1":   dict(hays=("noise",),         tk="words", tv="numbers", nk=1, nv=1, nq=1),
    "niah_single_2":   dict(hays=("essay",),         tk="words", tv="numbers", nk=1, nv=1, nq=1),
    "niah_single_3":   dict(hays=("essay",),         tk="words", tv="uuids",   nk=1, nv=1, nq=1),
    "niah_multikey_1": dict(hays=("essay",),         tk="words", tv="numbers", nk=4, nv=1, nq=1),
    "niah_multikey_2": dict(hays=("needle",),        tk="words", tv="numbers", nk=1, nv=1, nq=1),
}

# RULER generates 128 tokens for every niah task.
BUDGET = {t: TOKENS_TO_GENERATE for t in RULER_TASKS}
TASKS = RULER_TASKS
TASK_ORDER = list(RULER_TASKS)


def cells(tasks=None, hays=None):
    """(task, haystack) pairs, restricted to the haystacks a task has."""
    out = []
    for t in (tasks or TASK_ORDER):
        for h in TASKS[t]["hays"]:
            if hays is None or h in hays:
                out.append((t, h))
    return out


def label(task: str, hay: str) -> str:
    """The name a result file and a figure use for one (task, haystack) cell."""
    return f"{task}-{hay}"


# ---------------------------------------------------------------- metric


def string_match_all(pred: str, refs) -> float:
    """RULER's metric for one prediction, copied from
    scripts/eval/synthetic/constants.py:

        sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)

    averaged over prompts and multiplied by 100 by the caller. `postprocess_pred` there
    strips the prediction and maps every control character to a newline; both are no-ops
    for a substring test, and `.strip()` is applied here for faithfulness.
    """
    p = pred.strip().lower()
    return sum(1.0 if r.lower() in p else 0.0 for r in refs) / len(refs)


# ---------------------------------------------------------------- assets


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as r:
        return r.read()


def fetch_assets():
    """The three files the RULER pipeline needs, written once into out/ so a machine
    without network builds the same prompts: the essay corpus in RULER's own
    `{"text": ...}` json, wonderwords' two word lists, and nltk's punkt sentence tokenizer."""
    OUT.mkdir(parents=True, exist_ok=True)
    if not ESSAY_JSON.exists():
        parts = []
        for off in (0, 100, 200):
            d = json.loads(_fetch(HF_ROWS.format(off=off)).decode())
            parts += [row["row"]["text"] for row in d["rows"]]
        ESSAY_JSON.write_text(json.dumps({"text": "".join(parts)}))
        print(f"-> {ESSAY_JSON} ({len(parts)} essays)")
    for name, dst in (("adjectivelist", ADJ_FILE), ("nounlist", NOUN_FILE)):
        if not dst.exists():
            dst.write_bytes(_fetch(WW_URL.format(name=name)))
            print(f"-> {dst} ({len(dst.read_text().split())} words)")
    if not (NLTK_DATA / "tokenizers" / "punkt").exists():
        import nltk
        nltk.download("punkt", download_dir=str(NLTK_DATA))
        nltk.download("punkt_tab", download_dir=str(NLTK_DATA))
        print(f"-> {NLTK_DATA}")


@lru_cache(maxsize=1)
def sent_tokenize():
    """nltk's punkt, from the copy that travels with this experiment. niah.py uses exactly
    this function to find the sentence boundaries a needle is spliced between."""
    import nltk
    if str(NLTK_DATA) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DATA))
    os.environ.setdefault("NLTK_DATA", str(NLTK_DATA))
    from nltk.tokenize import sent_tokenize as st
    st("A test. Another.")                      # fail here, not inside a bank
    return st


@lru_cache(maxsize=1)
def WORDS() -> list:
    """niah.py:

        nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
        adjs  = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
        words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
        words = sorted(list(set(words)))
    """
    if not (ADJ_FILE.exists() and NOUN_FILE.exists()):
        raise SystemExit("wonderwords lists missing -- run `python ruler.py --fetch-assets`")
    adjs = [w for w in ADJ_FILE.read_text().split("\n") if w]
    nouns = [w for w in NOUN_FILE.read_text().split("\n") if w]
    return sorted(set(f"{adj}-{noun}" for adj in adjs for noun in nouns))


@lru_cache(maxsize=1)
def essay_words() -> list:
    """niah.py: `re.sub(r'\\s+', " ", essay).split(" ")`, the same expression."""
    if not ESSAY_JSON.exists():
        raise SystemExit(f"{ESSAY_JSON} missing -- run `python ruler.py --fetch-assets`")
    return re.sub(r"\s+", " ", json.loads(ESSAY_JSON.read_text())["text"]).split(" ")


@lru_cache(maxsize=32)
def _sents(hay: str, nwords: int) -> tuple:
    """niah.py's essay branch, up to `document_sents`: the first `nwords` words of the
    corpus (repeated if it is short), split into sentences by punkt."""
    w = essay_words()
    if nwords <= len(w):
        text = " ".join(w[:nwords])
    else:
        reps = (nwords + len(w) - 1) // len(w)
        text = " ".join((w * reps)[:nwords])
    return tuple(sent_tokenize()(text.strip()))


# ---------------------------------------------------------------- RULER's generator


def _rand_number(rng) -> str:
    return str(rng.randint(10 ** 6, 10 ** 7 - 1))


def _rand_uuid(rng) -> str:
    return str(_uuid.UUID(int=rng.getrandbits(128), version=4))


def _rand(kind: str, rng) -> str:
    if kind == "words":
        return rng.choice(WORDS())
    if kind == "numbers":
        return _rand_number(rng)
    if kind == "uuids":
        return _rand_uuid(rng)
    raise NotImplementedError(kind)


def _gen_input_output(task: str, hay: str, num_haystack: int, rng, seed: int):
    """niah.py's `generate_input_output`, statement for statement.

    Returns the prompt text, the gold values, and one (depth, index) pair per needle so a
    result can be binned by how far back the needle sits: a percentage from RULER's grid
    for the text haystacks, the insertion index over the line count for the other two.
    """
    c = RULER_TASKS[task]
    nk = max(c["nk"], c["nq"])                     # niah.py: num_needle_k = max(k, q)
    nv, nq, tv, tk = c["nv"], c["nq"], c["tv"], c["tk"]

    keys, values, needles = [], [], []
    for _ in range(nk):
        keys.append(_rand(tk, rng))
        value = []
        for _ in range(nv):
            value.append(_rand(tv, rng))
            needles.append(NEEDLE.format(type_needle_v=tv, key=keys[-1], value=value[-1]))
        values.append(value)
    order = list(range(len(needles)))
    random.Random(seed).shuffle(order)             # niah.py shuffles with a FRESH Random
    needles = [needles[i] for i in order]
    nvalue = [n.rsplit("is: ", 1)[1][:-1] for n in needles]

    if hay == "essay":
        sents = _sents(hay, num_haystack)
        depths = rng.sample(DEPTHS, len(needles))
        pos = [0] + sorted(int(len(sents) * (d / 100)) for d in depths) + [len(sents)]
        out = []
        for i in range(1, len(pos)):
            out.append(" ".join(sents[pos[i - 1]:pos[i]]))
            if i - 1 < len(needles):
                out.append(needles[i - 1])
        context = " ".join(out)
        where = [(d, None) for d in sorted(depths)]
    else:
        if hay == "noise":
            sentences = [NOISE] * num_haystack
        else:
            sentences = [NEEDLE.format(type_needle_v=tv, key=_rand(tk, rng),
                                       value=_rand(tv, rng))
                         for _ in range(num_haystack)]
        indexes = sorted(rng.sample(range(num_haystack), len(needles)), reverse=True)
        for index, element in zip(indexes, needles):
            sentences.insert(index, element)
        context = "\n".join(sentences)
        where = [(100.0 * i / max(num_haystack, 1), i) for i in sorted(indexes)]

    indices = rng.sample(range(nk), nq)
    queries = [keys[i] for i in indices]
    answers = [a for i in indices for a in values[i]]
    query = (", ".join(queries[:-1]) + ", and " + queries[-1]) if len(queries) > 1 \
        else queries[0]

    template = TASK_TEMPLATE + ANSWER_PREFIX
    type_needle_v = tv
    if nq * nv == 1:
        template = template.replace("Some", "A").replace("are all", "is")
        template = template.replace("are", "is").replace("answers", "answer")
        type_needle_v = tv[:-1]
    input_text = template.format(type_needle_v=type_needle_v, context=context, query=query)
    # depth of the needle carrying each gold value, in the order `answers` has them
    vdepth = {v: w[0] for v, w in zip(nvalue, where)}
    return input_text, answers, [vdepth.get(a, 0.0) for a in answers]


def _incremental(hay: str, target: int) -> int:
    inc = 500 if hay == "essay" else 25
    if hay != "essay" and target < 4096:
        inc = 5
    return inc


def _sizing(task: str, hay: str, target: int, rng, seed: int) -> int:
    """niah.py's `generate_samples`, the part before the sample loop: estimate the tokens
    per haystack unit from one sample, then binary-search the largest unit count whose
    prompt plus the 128 generated tokens fits the context."""
    inc = _incremental(hay, target)
    txt, _, _ = _gen_input_output(task, hay, inc, rng, seed)
    per = len(enc(txt)) / inc
    lo, hi = inc, max(int((target / per) * 3), inc * 2)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        txt, _, _ = _gen_input_output(task, hay, mid, rng, seed)
        if len(enc(txt)) + TOKENS_TO_GENERATE <= target:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best if best is not None else inc


def _first_ngram(seq: np.ndarray, pat: np.ndarray) -> int:
    m = len(pat)
    if m == 0 or len(seq) < m:
        return -1
    w = np.lib.stride_tricks.sliding_window_view(seq, m)
    hit = np.flatnonzero((w == pat[None, :]).all(1))
    return int(hit[0]) if len(hit) else -1


def _answer_tokens(txt: str, value: str) -> np.ndarray:
    try:
        return split(txt, txt + " " + value)[1]
    except ValueError:
        return enc(" " + value)


def _bank_ruler(task: str, hay: str, target: int, n: int, seed: int) -> dict:
    rng = random.Random(seed)                      # RULER seeds the global module once
    num_haystack = _sizing(task, hay, target, rng, seed)
    inc = _incremental(hay, target)
    rows, shrunk, overlong, ambiguous = [], 0, 0, 0
    for _ in range(n):
        used = num_haystack
        for _ in range(64):                        # niah.py's `while True` shrink loop
            txt, answers, depths = _gen_input_output(task, hay, used, rng, seed)
            ids = enc(txt)
            if len(ids) + TOKENS_TO_GENERATE <= target:
                break
            if used > inc:
                used -= inc
                shrunk += 1
            else:
                overlong += 1
                break
        low = txt.lower()
        amb = any(low.count(a.lower()) != sum(1 for b in answers if b == a)
                  for a in set(answers))
        ambiguous += int(amb)
        vtok = {a: _answer_tokens(txt, a) for a in set(answers)}
        dis = []
        for a in answers:
            p = _first_ngram(ids, vtok[a])
            if p < 0:
                p = len(enc(txt[:low.find(a.lower())]))
            dis.append(len(ids) - p)
        rows.append(dict(ids=ids, ans=vtok[answers[0]], refs=list(answers),
                         ref_depth=[int(round(d)) for d in depths],
                         ref_dist=[int(x) for x in dis],
                         depth=float(min(depths)) / 100.0, dist=int(max(dis)),
                         ambiguous=int(amb)))
    b = _pack(rows, shrunk, overlong)
    b["num_haystack"] = int(num_haystack)
    b["n_ambiguous"] = int(ambiguous)
    return b


# ---------------------------------------------------------------- packing / api


def _pack(rows, resampled: int, failed: int) -> dict:
    T = max(len(r["ids"]) for r in rows)
    A = max(len(r["ans"]) for r in rows)
    ids = np.full((len(rows), T), EOT, dtype=np.int32)
    ans = np.full((len(rows), A), EOT, dtype=np.int32)
    plen = np.zeros(len(rows), np.int32)
    alen = np.zeros(len(rows), np.int32)
    for i, r in enumerate(rows):
        ids[i, :len(r["ids"])] = r["ids"]
        ans[i, :len(r["ans"])] = r["ans"]
        plen[i], alen[i] = len(r["ids"]), len(r["ans"])
    return dict(ids=ids, plen=plen, ans=ans, alen=alen,
                refs=[r["refs"] for r in rows],
                depth=np.asarray([r["depth"] for r in rows], np.float32),
                ref_depth=[r["ref_depth"] for r in rows],
                ref_dist=[r["ref_dist"] for r in rows],
                dist=np.asarray([r["dist"] for r in rows], np.int32),
                ambiguous=np.asarray([r["ambiguous"] for r in rows], np.int32),
                fp=np.int64(int(ids.astype(np.int64).sum())),
                n_resampled=np.int64(resampled), n_failed=np.int64(failed))


def bank(task: str, target: int, n: int, seed: int = 0, hay: str = "essay") -> dict:
    """One bank of `n` prompts for `target` tokens of context.

        ids    [n, Tmax] int32   the PROMPT (no answer appended), right-padded with <|eot|>
        plen   [n] int32         prompt length: RULER sizes it so that plen + 128 <= target
        ans    [n, A] int32      the first reference's answer tokens, for the first-token
                                 diagnostic
        alen   [n] int32
        refs   list[list[str]]   RULER's `outputs`: the gold value(s), in query order
        depth  [n] float32       the target needle's depth, 0..1 (the earliest of them when
                                 several values are queried)
        ref_depth  list[list[int]]  one depth per reference, 0..100
        ref_dist   list[list[int]]  one distance per reference: tokens from the first token
                                 of that value to the end of the prompt
        dist   [n] int32         the largest of those distances
        ambiguous [n] int32      the gold value occurs in the prompt more often than the
                                 construction put it there. RULER does not check and does
                                 not resample, and neither does this; the flag is a
                                 diagnostic.
        fp                       fingerprint of the prompt tokens (their sum)
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; have {TASK_ORDER}")
    if hay not in TASKS[task]["hays"]:
        raise ValueError(f"{task} has haystacks {TASKS[task]['hays']}, not {hay!r}")
    b = _bank_ruler(task, hay, target, n, seed)
    b["meta"] = json.dumps(dict(task=task, hay=hay, target=target, n=n, seed=seed,
                                budget=BUDGET[task]))
    return b


# ---------------------------------------------------------------- self tests


def _selftest(pairs, ctxs, n, seed) -> int:
    """For every (task, haystack, length): every gold string must occur in the prompt's
    text, the prompt must fit `target` minus the 128 generated tokens, and every distance
    must lie inside the prompt. `amb` counts the prompts whose gold value happens to occur
    more often than the construction put it there -- RULER keeps those, so this is reported,
    not failed. `retok` is the mean token difference between the prompt and a re-tokenizing
    of its text; it is 0 for every RULER task, which are built as one string."""
    bad = 0
    print(f"{'cell':28s} {'ctx':>6s} {'plen':>13s} {'alen':>7s} {'refs':>5s} {'gold':>6s} "
          f"{'amb':>4s} {'retok':>6s} {'hay':>7s} {'depth':>11s} {'dist':>14s}")
    for t, hy in pairs:
        for c in ctxs:
            try:
                b = bank(t, c, n, seed, hay=hy)
            except SystemExit as e:
                print(f"{label(t, hy):28s} {c:6d}   SKIPPED: {e}")
                bad += 1
                continue
            k = min(len(b["ids"]), 8)
            retok, counts, miss = [], set(), 0
            for i in range(k):
                ids = b["ids"][i, :b["plen"][i]].astype(np.int64)
                retok.append(abs(len(enc(dec(ids))) - len(ids)))
                txt = dec(ids).lower()
                for r in b["refs"][i]:
                    cnt = txt.count(r.lower())
                    counts.add(cnt)
                    miss += int(cnt < 1)
            fits = bool((b["plen"] + TOKENS_TO_GENERATE <= c).all())
            dist_ok = bool(((b["dist"] >= 1) & (b["dist"] <= b["plen"])).all())
            bad += miss + (0 if dist_ok else 1) + (0 if fits else 1)
            print(f"{label(t, hy):28s} {c:6d} {b['plen'].min():6d}-{b['plen'].max():<6d} "
                  f"{b['alen'].min():3d}-{b['alen'].max():<3d} "
                  f"{len(b['refs'][0]):5d} {'/'.join(str(x) for x in sorted(counts)):>6s} "
                  f"{int(b['ambiguous'].sum()):4d} {np.mean(retok):6.1f} "
                  f"{b.get('num_haystack', 0):7d} "
                  f"{b['depth'].min():.2f}-{b['depth'].max():.2f} "
                  f"{b['dist'].min():6d}-{b['dist'].max():<7d}"
                  + ("  GOLD MISSING" if miss else "")
                  + ("  DIST BAD" if not dist_ok else "")
                  + ("  TOO LONG" if not fits else ""))
    print("\nALL CHECKS PASSED" if bad == 0 else f"\n{bad} FAILURES")
    return 0 if bad == 0 else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", nargs="+", default=None)
    p.add_argument("--hay", nargs="+", default=None)
    p.add_argument("--ctx", nargs="+", type=int, default=[4096, 16384])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--fetch-assets", action="store_true")
    p.add_argument("--show", default=None, help="print one prompt of this task")
    a = p.parse_args()
    if a.fetch_assets:
        fetch_assets()
        return 0
    pairs = cells(a.task, a.hay)
    if a.show:
        hy = (a.hay or TASKS[a.show]["hays"])[0]
        b = bank(a.show, a.ctx[0], 2, a.seed, hay=hy)
        ids = b["ids"][0, :b["plen"][0]]
        txt = dec(ids)
        print(f"--- {label(a.show, hy)} T={a.ctx[0]} plen={b['plen'][0]} "
              f"depth={b['depth'][0]:.2f} dist={b['dist'][0]} refs={b['refs'][0]}")
        print(txt[:1000] + "\n   [...]\n" + txt[-700:])
        return 0
    if a.selftest:
        return _selftest(pairs, a.ctx, a.n, a.seed)
    for t, hy in pairs:
        print(label(t, hy), BUDGET[t])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
