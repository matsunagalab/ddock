"""Write top-K poses as two-chain PDBs so PINDER's own evaluator can score them.

Why not just report our own numbers
-----------------------------------
This repository computes DockQ with an atom-level approximation and reports
success@K as a mean over complexes. PINDER's leaderboard uses `BiotiteDockQ`,
reports MEDIANS, and splits CAPRI into four classes rather than a single
acceptable/not threshold. Claiming a result "on PINDER" means being scored by
PINDER, so this exports decoys and `scripts/score_decoys_with_pinder.py` runs
their evaluator on them.

How a decoy is reconstructed
----------------------------
The prep cache keeps coordinates and derived features but no atom names,
residue numbers or chain ids, so a pose cannot be written out from it alone.
Instead the original complex PDB is re-read and its ATOM lines are filtered by
exactly the rule `parse_pdb_plain` uses -- first model, altLoc blank or A, a
(resname, atomname) pair the ZDOCK LUT knows, an element with a van der Waals
radius. That reproduces the atom set and ordering the pipeline scored, so pose
row i corresponds to filtered ligand atom i. Receptor lines are written
unchanged in the receptor-decentred frame; ligand lines get the pose's
coordinates.

Chains are renamed R and L because PINDER's evaluator requires exactly those.

Example
-------
    uv run python scripts/export_decoys_for_pinder.py \
        --ckpt data/scaling/runs_nfixed/N220_seed0/round0_ckpt.pt \
        --out-dir data/pinder_eval/trained_N220 --top-k 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from zdock.atomtypes import _ATOMTYPE_LUT, _VDW_RADIUS, iface_ij
from zdock.dataset import _element_of
from zdock.score import iface_score_matrix, SC_RHO

KEEP_ALT = (" ", "A")


def usable_atom_lines(path: Path, chain: str) -> list[str]:
    """ATOM lines `parse_pdb_plain` would keep, in file order."""
    out = []
    for line in open(path):
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        if line[16] not in KEEP_ALT or line[21] != chain:
            continue
        a = line[12:16].strip()
        r = line[17:20].strip()
        a_norm = "O" if a == "OXT" else a
        if (r, a_norm) not in _ATOMTYPE_LUT or _element_of(a) not in _VDW_RADIUS:
            continue
        out.append(line)
    return out


def rewrite(line: str, xyz, chain: str, serial: int) -> str:
    return (f"ATOM  {serial:5d}" + line[11:21] + chain + line[22:30]
            + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}" + line[54:].rstrip("\n"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="data/shards_pinder/test_pool_reachable.pt")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache", dest="prep")
    ap.add_argument("--pdb-dir", default="external/pinder/pinder/2024-02/pdbs",
                    dest="pdb_dir")
    ap.add_argument("--ckpt", default="", help="empty = published parameters")
    ap.add_argument("--out-dir", required=True, dest="out_dir")
    ap.add_argument("--top-k", type=int, default=5, dest="top_k")
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from zdock.prep_cache import load_prepared
    from zdock.search import _rotate_batch

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        alpha, iface = ck["alpha"].double(), ck["iface"].double()
        clash = ck["clash_weights"].double()
    else:
        alpha = torch.tensor(1.0, dtype=torch.float64)
        iface = iface_ij(dtype=torch.float64, flat=True)
        k = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        clash = alpha * torch.tensor(SC_RHO, dtype=torch.float64).pow(k)
    beta = torch.tensor(args.beta, dtype=torch.float64)

    blob = torch.load(args.pool, map_location="cpu", weights_only=True)
    out_root = Path(args.out_dir)
    n_written = n_skipped = 0
    manifest = []
    for d in blob:
        name = d["name"]
        if args.limit and n_written >= args.limit:
            break
        native = Path(args.pdb_dir) / f"{name}.pdb"
        if not native.is_file():
            n_skipped += 1
            continue
        prot = load_prepared(args.prep, name, dtype=torch.float64)
        if prot is None:
            n_skipped += 1
            continue
        # the two chain ids are the two halves of the PINDER system id
        rec_ch, lig_ch = name.split("--")[0].split("_")[1][0], \
                         name.split("--")[1].split("_")[1][0]
        rec_lines = usable_atom_lines(native, rec_ch)
        lig_lines = usable_atom_lines(native, lig_ch)
        if len(rec_lines) != prot.n_rec or len(lig_lines) != prot.n_lig:
            # the filter did not reproduce the prepared atom set; writing a
            # mismatched decoy would silently score the wrong atoms
            n_skipped += 1
            manifest.append({"id": name, "skipped": "atom count mismatch",
                             "pdb_rec": len(rec_lines), "prep_rec": prot.n_rec,
                             "pdb_lig": len(lig_lines), "prep_lig": prot.n_lig})
            continue

        keep = (d["prov"] == 0).nonzero(as_tuple=True)[0]
        sc = d["sc"][keep].double()
        s = (alpha * sc[:, 0] - (sc[:, 1:4] * clash).sum(-1)
             + (iface_score_matrix(iface) * d["T"][keep].double()).sum(dim=(-2, -1))
             + beta * d["elec"][keep].double())
        order = torch.argsort(s, descending=True)[: args.top_k]

        sysdir = out_root / name
        sysdir.mkdir(parents=True, exist_ok=True)
        rec_xyz = prot.rec_xyz
        for rank, idx in enumerate(order.tolist(), start=1):
            # the pool stores features, not coordinates: rebuild the pose from
            # its identity (rotation index, cell) is not possible here, so the
            # exported decoy uses the stored native placement only when rank==0.
            # Poses come from the pool's own rmsd/dockq bookkeeping instead.
            raise SystemExit(
                "the frozen TEST pool stores features, not pose coordinates; "
                "re-mine with --keep-poses or export from a run that saved them")
        n_written += 1
    print(f"written {n_written}, skipped {n_skipped}")
    if manifest:
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
