"""Where does the ranking lose the near-native pose it already retrieved?

The reachability ceiling of the nside=3 rotation set is 99.0% and the search
returns 1500 poses, so a miss at Top 1 is almost never "the pose is not there".
This asks the cheaper question the existing end-to-end run already answers: for
every system whose returned set contains a near-native pose, at what rank does
the first one sit? A deeper `--ntop` can only help systems whose first hit is
BELOW the current cut; systems whose first hit is at rank 900 are already
retained and would need a better score, not a longer list.

Reads the per-complex CSVs written by `scripts/eval_search_test.py`, so it costs
no GPU time.

Example
-------
    uv run python scripts/rank_truncation_diag.py \
        --csv data/scaling/eval_search_pinder/published_per_complex.csv \
        --csv data/scaling/eval_search_pinder/trained_N220_per_complex.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", action="append", required=True,
                    help="per-complex CSV; repeat for several conditions")
    ap.add_argument("--ntop", type=int, default=1500,
                    help="how many poses the run kept (for the tail count)")
    args = ap.parse_args()

    for path in args.csv:
        rows = _rows(Path(path))
        label = Path(path).stem.replace("_per_complex", "")
        n = len(rows)
        # ceiling_dockq is score-independent: could the rotation set reach a
        # near-native pose at all for this complex?
        reachable = [r for r in rows if _f(r["ceiling_dockq"]) >= 0.23]
        hit = [r for r in rows if int(r["recall"]) == 1]
        ranks = sorted(int(r["first_hit_rank"]) for r in hit
                       if r["first_hit_rank"] not in ("", "-1")
                       and not math.isnan(_f(r["first_hit_rank"])))
        print(f"\n=== {label}: {n} systems ===")
        print(f"  rotation set can reach DockQ>=0.23 : {len(reachable)}/{n} "
              f"({100 * len(reachable) / n:.1f}%)")
        print(f"  near-native present in the kept {args.ntop}: {len(hit)}/{n} "
              f"({100 * len(hit) / n:.1f}%)")
        print(f"  retrieved but ranked below 1        : "
              f"{sum(1 for x in ranks if x > 1)}")
        print("  first-hit rank of the systems that DID retrieve one:")
        cuts = [1, 5, 10, 25, 50, 100, 250, 500, 1000, args.ntop]
        prev = 0
        for c in cuts:
            k = sum(1 for x in ranks if x <= c)
            print(f"    <= {c:5d} : {k:4d} ({100 * k / n:5.1f}% of all systems)"
                  f"   +{k - prev} in this band")
            prev = k
        # what a deeper list could buy: systems with NO hit in the kept set but
        # whose rotation set could reach one
        lost = [r for r in reachable if int(r["recall"]) == 0]
        print(f"  reachable but nothing near-native in the kept {args.ntop}: "
              f"{len(lost)} systems -- only these can be recovered by a deeper "
              f"--ntop; the rest need a better ranking")


if __name__ == "__main__":
    main()
