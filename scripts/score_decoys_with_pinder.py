"""Score exported decoys with PINDER's own evaluation harness.

This reports the numbers PINDER reports, not this repository's approximations:
CAPRI hit rates at Oracle / Max(Top 1) / Max(Top 5), split into acceptable,
medium and high, plus median DockQ, iRMSD, LRMSD and Fnat. `BiotiteDockQ`
computes them, so the DockQ here is PINDER's, not the atom-level approximation
used elsewhere in this report (which needed its own 0.23 threshold to stand in
for CAPRI acceptable).

Everything produced by this pipeline is `pinder_s` / `holo`: both partners come
from the bound complex. That is the easiest of PINDER's three monomer settings
and its numbers must not be placed beside another method's `apo` or `predicted`
columns.

Example
-------
    PINDER_BASE_DIR=$PWD/external/pinder \
    uv run python scripts/score_decoys_with_pinder.py \
        --eval-dir data/pinder_eval
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _force_biotite_pdb_backend() -> str:
    """Make PINDER read PDBs with biotite instead of fastpdb.

    fastpdb 1.3.3 pins only `biotite>=0.39` but assigns to
    `biotite.structure.PDBFile.lines`, which biotite 1.7.1 made read-only. Every
    read then raises AttributeError -- measured on PINDER's OWN native files,
    not just ours, so it is an upstream version conflict and not a defect in the
    decoys.

    This matters more than a stack trace would suggest: `MethodMetrics` catches
    per-system exceptions, so a broken reader does not stop the run. It quietly
    scores nothing and still prints a leaderboard. (The same fail-open shape
    silently skipped all 250 systems on our side earlier today.)

    Returns a note for the report so the substitution is never invisible.
    """
    import pinder.core.structure.atoms as atoms

    original = atoms.atom_array_from_pdb_file

    def biotite_default(structure, backend="biotite", extra_fields=None):
        return original(structure, "biotite", extra_fields)

    atoms.atom_array_from_pdb_file = biotite_default
    # BiotiteDockQ imported the symbol directly, so patch it there too
    import pinder.eval.dockq.biotite_dockq as bdq
    if hasattr(bdq, "atom_array_from_pdb_file"):
        bdq.atom_array_from_pdb_file = biotite_default

    import biotite, fastpdb
    return (f"PDB backend forced to biotite (fastpdb "
            f"{getattr(fastpdb, '__version__', '?')} is incompatible with "
            f"biotite {biotite.__version__})")


def roundtrip_check(system_id: str, pdb_dir: str) -> None:
    """Score a native against itself; PINDER must call it DockQ 1.0 / High.

    Without this the whole harness can run, catch every per-system failure, and
    report a leaderboard of zeros that looks like a bad method rather than a
    broken reader.
    """
    from pinder.eval.dockq.biotite_dockq import BiotiteDockQ

    native = Path(pdb_dir) / f"{system_id}.pdb"
    bdq = BiotiteDockQ(native, [native], parallel_io=False)
    m = bdq.calculate()
    dq = float(m.DockQ.iloc[0])
    capri = str(m.CAPRI.iloc[0])
    print(f"round-trip on {system_id}: DockQ={dq:.4f} CAPRI={capri}")
    if dq < 0.99 or capri != "High":
        raise SystemExit(
            f"native scored against itself gives DockQ={dq:.4f} ({capri}), not "
            f"1.0 / High. The evaluator is misreading the structures; any "
            f"leaderboard produced now would be meaningless.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", required=True, dest="eval_dir")
    ap.add_argument("--out", default="data/pinder_eval/leaderboard.csv")
    ap.add_argument("--max-workers", type=int, default=8, dest="max_workers")
    ap.add_argument("--allow-missing", action="store_true", dest="allow_missing",
                    help="tolerate systems with no submitted decoy. NOTE: with "
                         "this false PINDER does NOT reject a short submission "
                         "-- it fills the gap with DockQ=0 / RMSD=100, i.e. a "
                         "penalty. Either way the count of real versus "
                         "penalised systems is printed below and belongs in "
                         "any reported number.")
    ap.add_argument("--roundtrip-id",
                    default="3k1i__D1_O25709--3k1i__A1_O25448",
                    dest="roundtrip_id",
                    help="system used for the native-against-itself check")
    ap.add_argument("--pdb-dir", default="external/pinder/pinder/2024-02/pdbs",
                    dest="pdb_dir")
    # The leaderboard is an aggregate: it cannot say WHICH systems moved, so it
    # cannot answer whether a hit-rate gain is a handful of complexes crossing
    # the CAPRI threshold by a hair. Keep the per-decoy frame.
    ap.add_argument("--metrics-out", default="", dest="metrics_out",
                    help="per-decoy metrics CSV (default: <out> with "
                         "'_per_decoy' appended)")
    args = ap.parse_args()

    print(_force_biotite_pdb_backend())
    roundtrip_check(args.roundtrip_id, args.pdb_dir)

    from pinder.eval.dockq.method import MethodMetrics

    mm = MethodMetrics(Path(args.eval_dir), parallel=True,
                       allow_missing_systems=args.allow_missing,
                       max_workers=args.max_workers)
    metrics = mm.metrics

    # How many systems carry a real decoy and how many are penalty rows.
    # PINDER does not reject a short submission: `add_pinder_set` fills the gap
    # with DockQ = 0 / RMSD = 100 under the name `missing_decoy_1`, so a
    # leaderboard computed over 249 of 250 systems already includes one
    # zero-scored system. Quoting the number without that count overstates
    # nothing, but hides where it came from.
    missing = metrics[metrics.model_name.astype(str)
                      .str.contains("missing_decoy", na=False)]
    real = metrics[~metrics.index.isin(missing.index)]
    print(f"scored {len(metrics)} decoy rows over {metrics.id.nunique()} "
          f"systems, {metrics.method_name.nunique()} method(s)")
    print(f"  real decoys      : {len(real)} rows over "
          f"{real.id.nunique()} systems")
    print(f"  penalty rows     : {len(missing)} over "
          f"{missing.id.nunique()} systems "
          f"(PINDER's DockQ = 0 fill for systems with no submission)")
    if missing.id.nunique():
        print(f"  penalised ids    : "
              f"{sorted(set(missing.id))[:5]}"
              f"{' ...' if missing.id.nunique() > 5 else ''}")
    print()

    mpath = Path(args.metrics_out) if args.metrics_out else \
        Path(args.out).with_name(Path(args.out).stem + "_per_decoy.csv")
    mpath.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(mpath, index=False)
    print(f"per-decoy metrics -> {mpath}")

    # per-decoy CAPRI distribution, before any aggregation
    print("CAPRI class of every scored decoy")
    print(metrics.groupby(["method_name", "CAPRI"]).size()
          .unstack(fill_value=0).to_string())

    lb = mm.get_leaderboard_entry()
    print("\nPINDER leaderboard entry")
    print(lb.to_string())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lb.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    # the hit-rate table is the headline; print it on its own
    cols = [c for c in lb.columns
            if "hit" in c.lower() or "Hit" in c or "DockQ" in c]
    if cols:
        print("\nheadline columns")
        print(lb[[c for c in ("Method", "Dataset", "Monomer") if c in lb.columns]
                 + cols].to_string())


if __name__ == "__main__":
    main()
