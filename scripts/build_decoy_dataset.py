"""Build a consolidated decoy training/eval dataset from ``*.pdb.ms`` pairs.

Uses the repository's own FFT search to propose poses (no external ZDOCK
binary) and the differentiable DockQ / RMSD metrics to label them, then
writes the consolidated HDF5 in the schema :mod:`zdock.data` expects
(with an added ``dockq`` dataset per protein).

Example
-------
    uv run python scripts/build_decoy_dataset.py \
        --proteins 1KXQ --output data/decoys.h5 --device cuda
"""

from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch

from zdock.atomtypes import charge_score as default_charge_score
from zdock.atomtypes import iface_ij
from zdock.score import SC_REFERENCE_SPACING
from zdock.dataset import (
    generate_decoys,
    label_decoys,
    prepare_protein_from_pdb,
    prepare_protein_from_pdbms,
)

_REPO = Path(__file__).resolve().parent.parent
_DATA_ROOT = _REPO / "tests" / "data"


def _pinder_holo_paths(pinder_id):
    """Download (if needed) a PINDER system and return its bound (holo)
    receptor / ligand monomer PDB paths.

    PINDER ships a ``fastpdb`` reader that is incompatible with the installed
    ``biotite`` (``PDBFile.lines`` is read-only), so we force ``PinderSystem``
    onto the pure-``biotite`` engine. We only use it to fetch files + resolve
    paths; the atoms themselves are re-parsed with our own ``parse_pdb_plain``
    so featurization is identical to the DB5.5 path. The holo ``-R``/``-L``
    monomers are extracted from the same native complex, so they already share
    a coordinate frame => the ligand's given coords are the native placement.
    """
    from pinder.core import PinderSystem  # heavy import; only in pinder mode

    ps = PinderSystem(pinder_id, pdb_engine="biotite")
    return str(ps.holo_receptor.filepath), str(ps.holo_ligand.filepath)


def _locate_and_prepare(name, args, device, dtype):
    """Return a PreparedProtein for ``name`` under the configured format."""
    if args.format == "pdbms":
        rec = _DATA_ROOT / name / f"{name}_r_u.pdb.ms"
        lig = _DATA_ROOT / name / f"{name}_l_u.pdb.ms"
        if not rec.exists() or not lig.exists():
            raise FileNotFoundError(f"missing pdb.ms for {name}: {rec} / {lig}")
        return prepare_protein_from_pdbms(name, rec, lig, device=device, dtype=dtype)
    if args.format == "pinder":
        # PINDER interface-deleaked systems (holo redocking). ``name`` is the
        # PINDER system id, e.g. ``3k1i__D1_O25709--3k1i__A1_O25448``.
        rec, lig = _pinder_holo_paths(name)
        return prepare_protein_from_pdb(name, rec, lig, device=device, dtype=dtype)
    # DB5.5 plain-PDB bound constituents.
    d = Path(args.structures_dir)
    rec = d / f"{name}_r_{args.bound}.pdb"
    lig = d / f"{name}_l_{args.bound}.pdb"
    if not rec.exists() or not lig.exists():
        raise FileNotFoundError(f"missing pdb for {name}: {rec} / {lig}")
    return prepare_protein_from_pdb(name, rec, lig, device=device, dtype=dtype)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _np(t: torch.Tensor, dtype) -> np.ndarray:
    return t.detach().cpu().numpy().astype(dtype)


def build(args) -> None:
    device = torch.device(args.device)
    dtype = torch.float64 if device.type == "cpu" else torch.float32

    alpha = torch.tensor(args.alpha, device=device, dtype=dtype)
    beta = torch.tensor(args.beta, device=device, dtype=dtype)
    iface_flat = iface_ij(device=device, dtype=dtype, flat=True)
    charge = default_charge_score(device=device, dtype=dtype)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        f.attrs["created_at"] = _dt.datetime.now().isoformat()
        f.attrs["git_commit"] = _git_commit()
        f.attrs["rmsd_threshold_angstrom"] = args.rmsd_threshold
        f.attrs["generator"] = "zdock.dataset (FFT search, default params)"

        for name in args.proteins:
            print(f"[{name}] preparing features ...", flush=True)
            try:
                prot = _locate_and_prepare(name, args, device, dtype)
            except Exception as exc:  # noqa: BLE001
                print(f"[{name}] SKIP (prep failed: {exc})", flush=True)
                continue
            print(f"[{name}] N_rec={prot.rec_xyz.shape[0]} "
                  f"N_lig={prot.lig_ref.shape[0]} "
                  f"generating decoys ...", flush=True)

            try:
                poses, _fft_scores = generate_decoys(
                    prot, alpha=alpha, iface_ij_flat=iface_flat, beta=beta,
                    charge_score_lut=charge,
                    n_random_rot=args.n_random_rot, n_cone=args.n_cone,
                    cone_deg=args.cone_deg, ntop=args.ntop, spacing=args.spacing,
                    rot_chunk_size=args.rot_chunk_size, seed=args.seed,
                )
                rmsd, dockq = label_decoys(prot, poses)
            except torch.cuda.OutOfMemoryError as exc:
                print(f"[{name}] SKIP (OOM: {exc})", flush=True)
                torch.cuda.empty_cache()
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[{name}] SKIP (decoy/label failed: {exc})", flush=True)
                torch.cuda.empty_cache()
                continue
            hit = (rmsd <= args.rmsd_threshold)
            print(f"[{name}] F={poses.shape[0]}  "
                  f"min_rmsd={rmsd.min():.2f}Å  max_dockq={dockq.max():.3f}  "
                  f"n_hit(<= {args.rmsd_threshold}Å)={int(hit.sum())}  "
                  f"n_acceptable(DockQ>=.23)={int((dockq>=0.23).sum())}",
                  flush=True)

            g = f.create_group(name)
            g.create_dataset("rec_xyz", data=_np(prot.rec_xyz, np.float32))
            g.create_dataset("rec_radius", data=_np(prot.rec_radius, np.float32))
            g.create_dataset("rec_sasa", data=_np(prot.rec_sasa, np.float32))
            g.create_dataset("rec_atomtype_id", data=_np(prot.rec_atomtype_id, np.int64))
            g.create_dataset("rec_charge_id", data=_np(prot.rec_charge_id, np.int64))
            g.create_dataset("lig_xyz", data=_np(poses, np.float32))
            g.create_dataset("lig_radius", data=_np(prot.lig_radius, np.float32))
            g.create_dataset("lig_sasa", data=_np(prot.lig_sasa, np.float32))
            g.create_dataset("lig_atomtype_id", data=_np(prot.lig_atomtype_id, np.int64))
            g.create_dataset("lig_charge_id", data=_np(prot.lig_charge_id, np.int64))
            g.create_dataset("lig_xyz_native", data=_np(prot.native_lig, np.float32))
            g.create_dataset("rmsd", data=_np(rmsd, np.float32))
            g.create_dataset("dockq", data=_np(dockq, np.float32))
            g.create_dataset("hit_mask", data=_np(hit, np.bool_))
            f.flush()
            del prot, poses, rmsd, dockq, hit
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--proteins", nargs="+", default=["1KXQ"])
    p.add_argument("--ids-file", dest="ids_file", default=None,
                   help="Optional file with one protein/system id per line "
                        "(overrides --proteins; needed for long PINDER ids).")
    p.add_argument("--output", default=str(_REPO / "data" / "decoys.h5"))
    p.add_argument("--format", choices=["pdbms", "pdb", "pinder"], default="pdbms")
    p.add_argument("--structures-dir", dest="structures_dir",
                   default=str(_REPO / "external" / "benchmark5.5" / "structures"))
    p.add_argument("--bound", choices=["b", "u"], default="b",
                   help="DB5.5 constituent: bound (b) or unbound (u)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--n-random-rot", type=int, default=3000, dest="n_random_rot")
    p.add_argument("--n-cone", type=int, default=400, dest="n_cone")
    p.add_argument("--cone-deg", type=float, default=25.0, dest="cone_deg")
    p.add_argument("--ntop", type=int, default=2000)
    p.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    p.add_argument("--rot-chunk-size", type=int, default=32, dest="rot_chunk_size")
    p.add_argument("--rmsd-threshold", type=float, default=5.0, dest="rmsd_threshold")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.ids_file:
        with open(args.ids_file) as fh:
            args.proteins = [ln.strip() for ln in fh if ln.strip()]
    build(args)


if __name__ == "__main__":
    main()
