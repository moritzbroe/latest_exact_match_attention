#!/bin/bash
# The discovery sweep, n by n: the three seeds of one n in parallel, then the verdict, and
# the sweep stops at the first n where no seed discovers the mechanism within the budget.
#   usage: sweep.sh rope [n ...]      default n = 2 3 4 5 6 8 16;  STEPS=100000  SEEDS="0 1 2"
set -u
ARCH=${1:-rope}; shift || true
NS=${*:-2 3 4 5 6 8 16}
STEPS=${STEPS:-100000}
SEEDS=${SEEDS:-0 1 2}
cd "$(dirname "$0")"; mkdir -p out/logs
for n in $NS; do
  for s in $SEEDS; do
    nice -n 10 python3 train_discovery.py $ARCH --n $n --seed $s --steps $STEPS \
      > out/logs/local_${ARCH}_n${n}_s${s}.log 2>&1 &
  done
  wait
  verdict=$(python3 - "$ARCH" "$n" <<'PY'
import json, sys
from pathlib import Path
arch, n = sys.argv[1], int(sys.argv[2])
lines, ok = [], False
for d in sorted(Path("out", arch, f"n{n}").glob("s*")):
    recs = [json.loads(l) for l in (d / "probes.jsonl").read_text().splitlines() if l]
    hit = [r["step"] for r in recs if r["acc"] >= 0.99]
    last = recs[-1]
    if hit:
        ok = True
        lines.append(f"{d.name} discovered at step {hit[0]}")
    else:
        lines.append(f"{d.name} no discovery in {last['step'] + 1} steps "
                     f"(max acc {max(r['acc'] for r in recs):.3f}, final ce {last['ce']:.3f})")
print(f"### {arch} n={n}: " + "; ".join(lines))
print("GO" if ok else "STOP")
PY
)
  echo "$verdict" | head -n 1
  if [ "$(echo "$verdict" | tail -n 1)" = STOP ]; then
    echo "### no seed discovered the mechanism at n=$n within $STEPS steps: stopping"
    break
  fi
done
python3 plot_discovery.py
echo "### DISCOVERY DONE"
