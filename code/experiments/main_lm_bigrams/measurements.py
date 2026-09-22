"""Read bucket means from raw evaluations or compact retained summaries.

Run `python measurements.py` after scoring to refresh out/bucket_means.json,
preserving summaries whose raw evaluations are absent.
Raw files take precedence. Means use float64, as in the paper's plotting scripts.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
EDGES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


@lru_cache(None)
def retained():
    path = OUT / "bucket_means.json"
    return json.loads(path.read_text()) if path.exists() else {}


def means(run, ctx, freq, distracted=False, kind="scores", key="loss", bounds=None):
    tag = "_distracted" if distracted else ""
    name = f"{run}_T{ctx}_f{freq}{tag}"
    bounds = bounds or [(lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:]) if lo < ctx]
    path = OUT / kind / f"{name}.npz"
    if path.exists():
        with np.load(OUT / f"targets_T{ctx}_f{freq}{tag}.npz") as target, np.load(path) as z:
            assert int(z["fingerprint"]) == int(target["fingerprint"]), f"{name}: window mismatch"
            dist = target["dist"]
            if kind == "intervention":
                dist = dist[z["target"].astype(np.int64)]
                canonical = [(lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:]) if lo < ctx]
                bucket = z["bucket"].astype(np.int64)
                assert ((dist >= np.array([lo for lo, _ in canonical])[bucket]) &
                        (dist < np.array([hi for _, hi in canonical])[bucket])).all(), name
            value = z[key].astype(np.float64)
            assert value.shape == dist.shape, f"{name}: target shape mismatch"
            return np.array([float(value[(dist >= lo) & (dist < hi)].mean())
                             for lo, hi in bounds])
    entry = retained().get(kind, {}).get(name)
    if entry is None or key not in entry:
        return None
    with np.load(OUT / f"targets_T{ctx}_f{freq}{tag}.npz") as target:
        assert entry["fingerprint"] == int(target["fingerprint"]), name
    indices = [entry["bounds"].index([lo, hi]) for lo, hi in bounds]
    return np.array(entry[key], dtype=np.float64)[indices]


def main():
    import re
    result = {kind: dict(retained().get(kind, {})) for kind in ("scores", "intervention")}
    updated = 0
    for kind in result:
        for path in sorted((OUT / kind).glob("*.npz")):
            match = re.fullmatch(r"(.+)_T(\d+)_f(\d+)(_distracted)?", path.stem)
            if not match:
                continue
            run, ctx, freq, tag = match.groups()
            ctx, freq = int(ctx), int(freq)
            target_path = OUT / f"targets_T{ctx}_f{freq}{tag or ''}.npz"
            if not target_path.exists():
                continue
            bounds = [(lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:]) if lo < ctx]
            with np.load(target_path) as target, np.load(path) as z:
                row = {"fingerprint": int(target["fingerprint"]), "bounds": bounds}
                for key in ("loss", "clean") if kind == "intervention" else ("loss",):
                    row[key] = means(run, ctx, freq, bool(tag), kind, key, bounds).tolist()
            result[kind][path.stem] = row
            updated += 1
    output = OUT / "bucket_means.json"
    if not updated:
        raise SystemExit(f"no raw evaluations under {OUT}/scores or {OUT}/intervention; "
                         f"{output} left as it is")
    output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(f"{output}: {sum(map(len, result.values()))} summaries ({updated} refreshed)")


if __name__ == "__main__":
    main()
