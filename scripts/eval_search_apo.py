"""Dock PINDER's `apo` monomers and export the top poses for official scoring.

Why this is a separate script
-----------------------------
`eval_search_test.py` docks the two chains of a bound complex, so it always has
a native placement to compare against: it reports DockQ, recall, first-hit rank
and the rotation-sampling ceiling. The `apo` setting has none of that. The two
inputs are unbound structures solved separately -- different PDB entries,
different residues present, no shared coordinate frame -- so "the native ligand
position in the receptor frame" does not exist for them, and neither does an
internal DockQ. **Every number here comes from PINDER's harness, not from us.**
This script only searches and writes decoys.

What it does
------------
For each PINDER-S system that has both apo monomers, read the receptor from
`apo_R_pdb` and the ligand from `apo_L_pdb` (chain A of each, the single chain
those files contain), run the identical FFT search, and write the top-K poses as
`{export_dir}/{system_id}/apo/models/model_{rank}.pdb` with chains R and L.
`scripts/score_decoys_with_pinder.py` then scores them against the holo
reference, which is what PINDER's `apo` column means.

The geometry guard is off on purpose. It exists to catch two chains that are not
in a common frame, and apo docking inputs are exactly that by construction --
they are separate structures, so the check has nothing to say.

Example
-------
    PINDER_BASE_DIR=$PWD/external/pinder uv run python scripts/eval_search_apo.py \\
        --ckpt data/scaling/runs_nfixed/N220_seed0/round0_ckpt.pt \\
        --label trained_N220 --export-pdb-dir data/pinder_eval/trained_N220
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_search_test import (ExportError, _usable_atom_lines,  # noqa: E402
                              _write_decoy)
from zdock.atomtypes import charge_score as default_charge_score  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.dataset import parse_pdb_plain, prepare_protein  # noqa: E402
from zdock.prep_cache import has_prepared, load_prepared, save_prepared  # noqa: E402
from zdock.rotation_grid import hopf_quaternions  # noqa: E402
from zdock.score import SC_REFERENCE_SPACING, SC_RHO  # noqa: E402
from zdock.search import _rotate_batch, docking_search  # noqa: E402


def apo_systems(index_path: str, subset: str) -> list[dict]:
    """The subset's systems that have BOTH apo monomers, in index order."""
    import pandas as pd

    ix = pd.read_parquet(index_path)
    sel = ix[ix[subset] & ix.apo_R & ix.apo_L]
    return [{"id": r.id, "rec": r.apo_R_pdb, "lig": r.apo_L_pdb}
            for r in sel.itertuples()]


def prepare(sysrec: dict, pdb_dir: Path, cache_dir: str, device, dtype):
    """Prepared apo receptor/ligand, cached so both conditions share the cost."""
    if cache_dir and has_prepared(cache_dir, sysrec["id"]):
        return load_prepared(cache_dir, sysrec["id"])
    rec_path = pdb_dir / sysrec["rec"]
    lig_path = pdb_dir / sysrec["lig"]
    for p in (rec_path, lig_path):
        if not p.is_file():
            raise FileNotFoundError(f"{sysrec['id']}: missing {p}")
    prot = prepare_protein(
        sysrec["id"],
        parse_pdb_plain(rec_path, "A"), parse_pdb_plain(lig_path, "A"),
        # apo monomers come from different structures; there is no common frame
        # to check, which is exactly what the guard checks for
        check_geometry=False, device=device, dtype=dtype)
    if cache_dir:
        save_prepared(cache_dir, prot.to("cpu", dtype=torch.float32))
    return prot


@torch.no_grad()
def search_and_export(prot, sysrec, pdb_dir, alpha, iface, beta, charge, rho,
                      args) -> int:
    device, dtype = prot.rec_xyz.device, prot.rec_xyz.dtype
    quats = hopf_quaternions(args.hopf_nside, device=device, dtype=dtype)
    quats = quats / quats.norm(dim=-1, keepdim=True)

    rot_chunk = args.rot_chunk
    for attempt in range(args.oom_retries + 1):
        try:
            res = docking_search(
                prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
                prot.rec_atomtype_id, prot.rec_charge_id,
                prot.lig_ref, prot.lig_radius, prot.lig_sasa,
                prot.lig_atomtype_id, prot.lig_charge_id,
                quats, alpha=alpha, iface_ij_flat=iface, beta=beta, sc_rho=rho,
                charge_score_lut=charge, spacing=args.spacing,
                ntop=args.ntop, rot_chunk_size=rot_chunk)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rot_chunk = max(1, rot_chunk // 2)
            if attempt == args.oom_retries:
                raise
    poses = (_rotate_batch(prot.lig_ref, quats[res.quat_indices])
             + res.translations.unsqueeze(1))
    poses = poses[torch.argsort(res.scores, descending=True)]

    # Same reconstruction rule as the holo path: re-read the source files under
    # parse_pdb_plain's filter so that pose row i is filtered atom i, and verify
    # it by coordinates rather than by count alone.
    rec_lines = _usable_atom_lines(pdb_dir / sysrec["rec"], "A")
    lig_lines = _usable_atom_lines(pdb_dir / sysrec["lig"], "A")
    if len(rec_lines) != prot.n_rec or len(lig_lines) != prot.n_lig:
        raise ExportError(
            f"{sysrec['id']}: PDB/prep atom mismatch (rec {len(rec_lines)} vs "
            f"{prot.n_rec}, lig {len(lig_lines)} vs {prot.n_lig})")
    pdb_rec = torch.tensor(
        [[float(x[30:38]), float(x[38:46]), float(x[46:54])] for x in rec_lines],
        dtype=torch.float32)
    cache_rec = prot.rec_xyz.detach().cpu().float()
    shift = pdb_rec.mean(0) - cache_rec.mean(0)
    worst = float((pdb_rec - (cache_rec + shift)).abs().max())
    if worst > 1e-2:
        raise ExportError(
            f"{sysrec['id']}: receptor coordinates disagree with the prepared "
            f"structure by up to {worst:.3g} A, so pose row i is not atom i")

    out = Path(args.export_pdb_dir) / sysrec["id"] / "apo" / "models"
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.pdb"):
        stale.unlink()
    k = min(args.export_top_k, poses.shape[0])
    if k < args.export_top_k:
        raise ExportError(f"{sysrec['id']}: search returned {poses.shape[0]} "
                          f"poses, fewer than the {args.export_top_k} to export")
    rec = prot.rec_xyz.detach().cpu()
    for i in range(k):
        tmp = out / f".model_{i + 1}.pdb.tmp"
        _write_decoy(tmp, rec_lines, lig_lines, rec, poses[i].detach().cpu())
        tmp.rename(out / f"model_{i + 1}.pdb")
    return k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="", help="checkpoint; omit for the published table")
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--rho0", type=float, default=SC_RHO)
    ap.add_argument("--label", default="")
    ap.add_argument("--index", default="external/pinder/pinder/2024-02/index.parquet")
    ap.add_argument("--subset", default="pinder_s")
    ap.add_argument("--pdb-dir", default="external/pinder/pinder/2024-02/pdbs",
                    dest="pdb_dir")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache_apo",
                    dest="prep_cache", help="'' to prepare every time")
    ap.add_argument("--export-pdb-dir", required=True, dest="export_pdb_dir",
                    help="METHOD root: {method}/{id}/apo/models/model_K.pdb")
    ap.add_argument("--export-top-k", type=int, default=5, dest="export_top_k")
    ap.add_argument("--out-dir", default="data/scaling/eval_search_apo", dest="out_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hopf-nside", type=int, default=3, dest="hopf_nside")
    ap.add_argument("--ntop", type=int, default=1500)
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--rot-chunk", type=int, default=8, dest="rot_chunk")
    ap.add_argument("--oom-retries", type=int, default=3, dest="oom_retries")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32
    label = args.label or (Path(args.ckpt).parent.name if args.ckpt else "baseline")

    beta = torch.tensor(3.0, device=device, dtype=dtype)
    charge = default_charge_score(device=device, dtype=dtype)
    iface0 = iface_ij(device=device, dtype=dtype, flat=True)
    if args.ckpt:
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        alpha = blob["alpha"].to(device=device, dtype=dtype)
        iface = blob["iface"].to(device=device, dtype=dtype)
        rho = blob.get("rho", torch.tensor(SC_RHO)).to(device=device, dtype=dtype)
    else:
        alpha = torch.tensor(args.alpha0, device=device, dtype=dtype)
        iface = iface0.clone()
        rho = torch.tensor(args.rho0, device=device, dtype=dtype)
    print(f"[{label}] apo docking; alpha={float(alpha):.4f} rho={float(rho):.4f} "
          f"||dIface||={float((iface - iface0).norm()):.3f}", flush=True)

    systems = apo_systems(args.index, args.subset)
    if args.limit:
        systems = systems[: args.limit]
    si, sn = (int(x) for x in args.shard.split("/"))
    systems = systems[si::sn]
    pdb_dir = Path(args.pdb_dir)
    if args.prep_cache:
        Path(args.prep_cache).mkdir(parents=True, exist_ok=True)

    done, skipped = [], []
    t0 = time.time()
    for i, s in enumerate(systems):
        prot = None
        try:
            prot = prepare(s, pdb_dir, args.prep_cache, device, dtype)
            prot = prot.to(device, dtype=dtype)
            n = search_and_export(prot, s, pdb_dir, alpha, iface, beta, charge,
                                  rho, args)
            done.append({"id": s["id"], "exported": n,
                         "n_rec": prot.n_rec, "n_lig": prot.n_lig})
        except torch.cuda.OutOfMemoryError as exc:
            skipped.append({"id": s["id"], "reason": f"OOM: {str(exc)[:120]}"})
        except ExportError:
            raise                        # never file an export bug as "skipped"
        except Exception as exc:  # noqa: BLE001
            skipped.append({"id": s["id"],
                            "reason": f"{type(exc).__name__}: {exc}"[:160]})
        finally:
            del prot
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(systems)} "
                  f"({(time.time()-t0)/(i+1):.1f}s/complex)", flush=True)

    root = Path(args.export_pdb_dir)
    bad = []
    for r in done:
        models = sorted((root / r["id"] / "apo" / "models").glob("model_*.pdb"))
        ranks = sorted(int(m.stem.split("_")[1]) for m in models)
        if ranks != list(range(1, args.export_top_k + 1)):
            bad.append((r["id"], ranks))
    print(f"\nexport: {len(done)} systems docked, {len(done) - len(bad)} with "
          f"exactly {args.export_top_k} models each")
    if bad:
        raise SystemExit(f"{len(bad)} systems lack models 1..{args.export_top_k} "
                         f"(e.g. {bad[0][0]}: {bad[0][1]})")
    if skipped:
        print(f"export: {len(skipped)} systems were NOT docked and will be "
              f"penalised with DockQ = 0 by PINDER:")
        for s in skipped:
            print(f"    {s['id']}: {s['reason']}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}_apo_summary.json").write_text(json.dumps(
        {"label": label, "subset": args.subset, "monomer": "apo",
         "n_docked": len(done), "n_skipped": len(skipped),
         "alpha": float(alpha), "rho": float(rho),
         "d_iface_norm": float((iface - iface0).norm()),
         "ntop": args.ntop, "hopf_nside": args.hopf_nside,
         "seconds": time.time() - t0,
         "note": "no internal DockQ: apo inputs have no common native frame; "
                 "all metrics come from PINDER's harness",
         "skipped": skipped}, indent=1))
    print(f"  wall {(time.time()-t0)/60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
