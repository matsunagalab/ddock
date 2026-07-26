"""Sanity-check prepared complexes: is the reference structure physically possible?

Why this exists
---------------
Every TEST-set number in this repository up to 2026-07-26 was computed against
reference structures in which the receptor and the ligand occupied the same
space. PINDER ships two kinds of monomer file per system:

* ``pdbs/{system_id}.pdb`` — the complex, chains R and L in a common frame;
* ``test_set_pdbs/{...}-R.pdb`` / ``-L.pdb`` — docking *inputs* for the test
  split, each independently centred on the origin.

Pairing the second kind superimposes the two proteins. Measured over all 250
PINDER-S test systems: closest receptor-ligand heavy-atom distance 0.01-0.03 A,
against 2.72 A for the same system read from its complex file. The consequences
looked like scoring-function bugs for most of a day — the search "never" found a
near-native pose, the baseline AUC was 0.005, the "positives" had 39x the
contacts of the negatives, and training appeared to reach 95.6% success@1 by
learning to count contacts.

A single check would have caught it: two heavy atoms cannot be closer than about
2.4 A (a short hydrogen bond), so a minimum below 2 A means the two chains are
not in a common frame.

The eleven-agent audit of 2026-07-25 (report section 5.7) reviewed the scoring
code from eight angles and never looked at the inputs. Run this instead, first.

Example
-------
    uv run python scripts/check_prep_cache.py \
        --cache-dir data/scaling/prep_cache \
        --ids-file data/scaling/master_ids.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from zdock.dataset import MIN_PLAUSIBLE_CONTACT_ANGSTROM
from zdock.prep_cache import load_prepared
from zdock.search import _rotate_batch


def check(prot, *, chunk: int = 2048) -> dict:
    """Per-complex geometry sanity, all in Angstrom."""
    rec, nat = prot.rec_xyz, prot.native_lig
    best = float("inf")
    n_close = 0
    for i in range(0, rec.shape[0], chunk):
        d = torch.cdist(rec[i:i + chunk], nat)
        best = min(best, float(d.min()))
        n_close += int((d < MIN_PLAUSIBLE_CONTACT_ANGSTROM).sum())
    # every receptor atom sitting at the interface means interpenetration,
    # not a large interface
    frac_iface = 0.0
    for i in range(0, rec.shape[0], chunk):
        frac_iface += float((torch.cdist(rec[i:i + chunk], nat).min(1).values
                             < 4.5).sum())
    frac_iface /= max(1, rec.shape[0])
    # the analytic native pose must reproduce native_lig
    rebuilt = _rotate_batch(prot.lig_ref, prot.q_star.unsqueeze(0))[0] + prot.t_star
    return {
        "id": prot.name, "n_rec": prot.n_rec, "n_lig": prot.n_lig,
        "min_contact_A": best,
        "n_pairs_below_floor": n_close,
        "frac_rec_atoms_within_4.5A": frac_iface,
        "native_reconstruction_err_A": float((rebuilt - nat).abs().max()),
        "lig_ref_centroid_A": float(prot.lig_ref.mean(0).norm()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="data/scaling/prep_cache",
                    dest="cache_dir")
    ap.add_argument("--ids-file", default="data/scaling/master_ids.txt",
                    dest="ids_file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--max-reconstruction-err", type=float, default=1e-2,
                    dest="max_rec_err")
    args = ap.parse_args()

    ids = [ln.strip() for ln in Path(args.ids_file).read_text().splitlines()
           if ln.strip()]
    if args.limit:
        ids = ids[: args.limit]

    rows, missing = [], 0
    for pid in ids:
        prot = load_prepared(args.cache_dir, pid, dtype=torch.float64)
        if prot is None:
            missing += 1
            continue
        rows.append(check(prot))
    if not rows:
        raise SystemExit(f"no usable entries under {args.cache_dir}")

    n = len(rows)
    impossible = [r for r in rows
                  if r["min_contact_A"] < MIN_PLAUSIBLE_CONTACT_ANGSTROM]
    engulfed = [r for r in rows if r["frac_rec_atoms_within_4.5A"] > 0.9]
    bad_rec = [r for r in rows
               if r["native_reconstruction_err_A"] > args.max_rec_err]
    far = [r for r in rows if r["lig_ref_centroid_A"] > 10.0]
    dists = sorted(r["min_contact_A"] for r in rows)

    print(f"{args.cache_dir}: {n} complexes checked ({missing} missing)")
    print(f"  closest receptor-ligand contact: median {dists[n // 2]:.2f} A, "
          f"min {dists[0]:.2f} A, p90 {dists[int(0.9 * n)]:.2f} A")
    print(f"  STERICALLY IMPOSSIBLE (< {MIN_PLAUSIBLE_CONTACT_ANGSTROM} A): "
          f"{len(impossible)}/{n}")
    print(f"  >90% of receptor atoms within 4.5 A of the ligand "
          f"(interpenetration, not a big interface): {len(engulfed)}/{n}")
    print(f"  native pose not reproduced by (q*, t*) to "
          f"{args.max_rec_err} A: {len(bad_rec)}/{n}")
    print(f"  |lig_ref centroid| > 10 A (rotation lever arm): {len(far)}/{n}")

    for lab, group in (("impossible", impossible), ("engulfed", engulfed),
                       ("bad reconstruction", bad_rec)):
        if not group:
            continue
        print(f"\n  worst '{lab}':")
        key = ("min_contact_A" if lab == "impossible"
               else "frac_rec_atoms_within_4.5A" if lab == "engulfed"
               else "native_reconstruction_err_A")
        rev = lab != "impossible"
        for r in sorted(group, key=lambda x: x[key], reverse=rev)[:5]:
            print(f"    {r['id'][:46]:<48}{key} = {r[key]:.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.out}")
    if impossible or bad_rec:
        raise SystemExit(
            f"{len(impossible)} impossible + {len(bad_rec)} unreconstructable "
            f"entries: do not run experiments on this cache")


if __name__ == "__main__":
    main()
