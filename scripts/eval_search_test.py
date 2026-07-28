"""End-to-end docking evaluation on the deleaked TEST set: search with the
*trained* parameters, then ask whether the search itself found a near-native
pose.

Why this exists
---------------
Every number in §5.4-§5.6 so far comes from re-ranking a **fixed** candidate
pool built once with default ZDOCK parameters, in which

* the FFT rotation set was seeded with a 25 deg cone around the native
  orientation ``q*`` — information a real docking run does not have, and
* 400 near-native poses were *injected* at the native translation, so a
  positive is guaranteed to be present regardless of what the search found.

That makes it a re-ranking benchmark with the answer planted in the candidate
set. It is internally consistent (the same pool for every N, seed and round, so
comparisons are valid) but it cannot tell us whether the learned parameters
actually dock anything.

This script removes both crutches: rotations are drawn **uniformly at random
with no reference to** ``q*``, nothing is injected, and the candidate set is
exactly what ``docking_search`` returns under the parameters being evaluated.
The rotation set is generated from a fixed seed, so it is identical for every
parameter setting and any difference between conditions is attributable to
scoring rather than to sampling luck.

It also separates the two ways this can fail, which the pooled metric conflates:

* **ceiling** — with a finite uniform rotation set, is a near-native pose even
  reachable? Reported as the DockQ of the closest sampled rotation placed at
  the native translation ``t*``. Parameter-independent.
* **recall** — does the returned top-N contain any pose with DockQ >= 0.23?
* **success@K** — given the returned set, is a near-native pose ranked top-K?
  Reported unconditionally and conditioned on recall.

Example
-------
    PINDER_BASE_DIR=$PWD/external/pinder CUDA_VISIBLE_DEVICES=6 \\
    uv run python scripts/eval_search_test.py \\
        --ckpt data/scaling/runs/N220_seed2/round1_ckpt.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from zdock.atomtypes import charge_score as default_charge_score  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.score import SC_REFERENCE_SPACING, SC_RHO  # noqa: E402
from zdock.dockq import dockq_batch, ligand_rmsd_to_native  # noqa: E402
from zdock.prep_cache import load_prepared  # noqa: E402
from zdock.rotation_grid import hopf_quaternions, random_quaternions  # noqa: E402
from zdock.search import _rotate_batch, docking_search  # noqa: E402

KS = (1, 5, 10, 50, 100)


def _quat_angle_deg(q: torch.Tensor, q_star: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (deg) between each quaternion in ``q`` and ``q_star``,
    accounting for the double cover (q and -q are the same rotation)."""
    d = (q * q_star.unsqueeze(0)).sum(dim=-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(d))


@torch.no_grad()
def _label(prot, poses, budget):
    rmsd = ligand_rmsd_to_native(prot.native_lig, poses)
    per = max(1, prot.n_rec * prot.n_lig)
    chunk = int(max(1, min(64, budget // per)))
    parts = []
    for s in range(0, poses.shape[0], chunk):
        parts.append(dockq_batch(prot.rec_xyz, poses[s:s + chunk],
                                 prot.native_lig).dockq)
    return rmsd, torch.cat(parts, dim=0)



#: PINDER's evaluator requires exactly two chains named R and L.
_PINDER_CHAINS = ("R", "L")


class ExportError(RuntimeError):
    """A decoy could not be written.

    Raised rather than logged because the caller wraps `evaluate_complex` in a
    broad `except Exception` that files failures under "skipped". An export bug
    would therefore look like a hard complex: earlier today a chain-id bug
    skipped all 250 systems while the progress line still read
    "25/250 (21.7s/complex)". A submission missing systems is not a smaller
    submission -- PINDER fills the gaps with DockQ = 0 -- so this stops the run.
    """


def _usable_atom_lines(path, chain: str) -> list[str]:
    """The ATOM lines `parse_pdb_plain` keeps, in file order.

    The prep cache stores coordinates and derived features but no atom names,
    residue numbers or chain ids, so a pose cannot be written to PDB from it
    alone. Re-reading the source complex under the identical filter reproduces
    the atom set and ordering the pipeline scored, so pose row i is filtered
    atom i. Any drift between the two filters would silently write a decoy
    whose atoms do not correspond to the scored ones, which is why the caller
    asserts the counts match.
    """
    from zdock.atomtypes import _ATOMTYPE_LUT, _VDW_RADIUS
    from zdock.dataset import _element_of

    out = []
    for line in open(path):
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        if line[16] not in (" ", "A") or line[21] != chain:
            continue
        a = line[12:16].strip()
        r = line[17:20].strip()
        a_norm = "O" if a == "OXT" else a
        if (r, a_norm) not in _ATOMTYPE_LUT or _element_of(a) not in _VDW_RADIUS:
            continue
        out.append(line)
    return out


def _write_decoy(path, rec_lines, lig_lines, rec_xyz, lig_xyz) -> None:
    """One PDB with the receptor as chain R and the posed ligand as chain L."""
    serial = 0
    with open(path, "w") as fh:
        for lines, xyz, ch in ((rec_lines, rec_xyz, "R"),
                               (lig_lines, lig_xyz, "L")):
            for line, c in zip(lines, xyz.tolist()):
                serial += 1
                # keep everything after the coordinates, including the
                # element symbol in columns 77-78 -- dropping it makes the
                # reader guess, and "CA" then reads as calcium rather than
                # a carbon alpha
                fh.write(f"ATOM  {serial:5d}" + line[11:21] + ch + line[22:30]
                         + f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}"
                         + line[54:].rstrip("\n").ljust(26) + "\n")
            fh.write("TER\n")
        fh.write("END\n")


@torch.no_grad()
def evaluate_complex(prot, alpha, iface, beta, charge, args, rho=SC_RHO):
    device = prot.rec_xyz.device
    dtype = prot.rec_xyz.dtype
    # No native-orientation cone. Default to the SAME Hopf grid the training
    # pools were built on -- comparing a re-search on a different rotation set
    # against a pool built on this one would confound the parameters with the
    # sampling. Hopf nside=3 covers SO(3) to ~18.5 deg at 1944 points against
    # ~28.3 deg for the same number of uniform-random orientations.
    if args.rot_set == "hopf":
        quats = hopf_quaternions(args.hopf_nside, device=device, dtype=dtype)
        quats = quats / quats.norm(dim=-1, keepdim=True)
    else:
        quats = random_quaternions(args.n_rot, seed=args.rot_seed, device=device,
                                   dtype=dtype)

    # Parameter-independent ceiling: best sampled rotation at the ideal
    # translation. Bounds what any scorer could achieve with this rotation set.
    ang = _quat_angle_deg(quats, prot.q_star)
    best_rot = int(ang.argmin())
    ceil_pose = (_rotate_batch(prot.lig_ref, quats[best_rot:best_rot + 1])
                 + prot.t_star.unsqueeze(0).unsqueeze(0))
    ceil_rmsd, ceil_dockq = _label(prot, ceil_pose, args.dockq_budget)

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
    rmsd, dockq = _label(prot, poses, args.dockq_budget)

    # docking_search returns poses ordered by its own score; keep that order.
    scores = res.scores
    order = torch.argsort(scores, descending=True)
    dq_s, rm_s = dockq[order], rmsd[order]
    pos = dq_s >= args.dockq_thr
    hit = pos.nonzero(as_tuple=True)[0]
    n = dq_s.numel()
    row = {
        "name": prot.name,
        "n_out": n,
        "ceiling_dockq": float(ceil_dockq[0]),
        "ceiling_rmsd": float(ceil_rmsd[0]),
        "min_rot_angle_deg": float(ang.min()),
        "recall": int(bool(pos.any())),
        "n_pos_out": int(pos.sum()),
        "first_hit_rank": int(hit[0]) + 1 if hit.numel() else n + 1,
        "best_dockq_out": float(dq_s.max()),
        "min_rmsd_out": float(rm_s.min()),
    }
    for k in KS:
        kk = min(k, n)
        row[f"succ_dockq@{k}"] = int(bool((dq_s[:kk] >= args.dockq_thr).any()))
        row[f"succ_rmsd@{k}"] = int(bool((rm_s[:kk] <= args.rmsd_thr).any()))
        row[f"best_dockq@{k}"] = float(dq_s[:kk].max())

    if getattr(args, "export_pdb_dir", ""):
        row["exported"] = _export_top_k(prot, poses[order], args)
    return row


def _export_top_k(prot, poses_sorted, args) -> int:
    """Write the top-K poses as PINDER-shaped decoys. Returns how many."""
    from pathlib import Path

    src = Path(args.pdb_dir) / f"{prot.name}.pdb"
    if not src.is_file():
        raise ExportError(f"{prot.name}: no source complex at {src}")
    # PINDER already normalises the complex file's chains to R (receptor) and
    # L (ligand), so there is nothing to infer from the system id. Deriving the
    # chain from the id is also wrong: the separator is a DOUBLE underscore
    # ({pdb}__{chain}{copy}_{uniprot}), so `split("_")[1]` is the empty string.
    rec_lines = _usable_atom_lines(src, "R")
    lig_lines = _usable_atom_lines(src, "L")
    if len(rec_lines) != prot.n_rec or len(lig_lines) != prot.n_lig:
        # the filter did not reproduce the prepared atom set, so pose row i is
        # not filtered atom i; writing anyway would score the wrong atoms
        raise ExportError(
            f"{prot.name}: PDB/prep atom mismatch "
            f"(rec {len(rec_lines)} vs {prot.n_rec}, "
            f"lig {len(lig_lines)} vs {prot.n_lig})")

    # Matching counts do not prove matching ORDER. Compare the coordinates:
    # the prep cache is the source complex shifted by the receptor centroid, so
    # PDB row i and prep row i must agree to within rounding once that shift is
    # removed. Measured over the 249 cached systems the worst disagreement is
    # 7.5e-5 A, so 1e-2 is loose enough for PDB's three decimals and tight
    # enough to catch a re-ordered or substituted source file.
    pdb_rec = torch.tensor(
        [[float(x[30:38]), float(x[38:46]), float(x[46:54])] for x in rec_lines],
        dtype=prot.rec_xyz.dtype)
    shift = pdb_rec.mean(0) - prot.rec_xyz.detach().cpu().mean(0)
    worst = float((pdb_rec - (prot.rec_xyz.detach().cpu() + shift)).abs().max())
    if worst > 1e-2:
        raise ExportError(
            f"{prot.name}: receptor coordinates disagree with the prep cache by "
            f"up to {worst:.3g} A after removing the centroid shift, so pose "
            f"row i is not PDB atom i")
    # PINDER's harness expects {method}/{system_id}/{monomer}/models/model_K.pdb
    # and infers the rank from the trailing integer of the file name. The
    # monomer level is "holo" here: both partners come from the bound complex,
    # which is the easiest of PINDER's three settings and must be reported as
    # such rather than compared against apo or predicted numbers.
    out = Path(args.export_pdb_dir) / prot.name / args.monomer / "models"
    out.mkdir(parents=True, exist_ok=True)
    # a re-run must not leave a previous run's models behind: PINDER reads the
    # rank off the file name, so a stale model_5.pdb would be scored as this
    # method's fifth pose
    for stale in out.glob("*.pdb"):
        stale.unlink()
    rec = prot.rec_xyz.detach().cpu()
    k = min(args.export_top_k, poses_sorted.shape[0])
    if k < args.export_top_k:
        raise ExportError(
            f"{prot.name}: search returned {poses_sorted.shape[0]} poses, "
            f"fewer than the {args.export_top_k} to export")
    for i in range(k):
        # write then rename, so an interrupted run leaves no partial PDB that
        # would parse as a valid but truncated structure
        tmp = out / f".model_{i + 1}.pdb.tmp"
        _write_decoy(tmp, rec_lines, lig_lines, rec, poses_sorted[i].detach().cpu())
        tmp.rename(out / f"model_{i + 1}.pdb")
    return k


def _mean(rows, key, mask=None):
    xs = [r[key] for r in rows if (mask is None or mask(r))]
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="", help="checkpoint; omit for default ZDOCK")
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--rho0", type=float, default=SC_RHO)
    ap.add_argument("--label", default="")
    ap.add_argument("--test-ids", default="data/pinder_test_ids.txt", dest="test_ids")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache_test",
                    dest="prep_cache")
    ap.add_argument("--out-dir", default="data/scaling/eval_search", dest="out_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rot-set", choices=("hopf", "uniform"), default="hopf",
                    dest="rot_set")
    ap.add_argument("--hopf-nside", type=int, default=3, dest="hopf_nside")
    ap.add_argument("--n-rot", type=int, default=1900, dest="n_rot",
                    help="uniform rotations (matches the mining budget of "
                         "1500 random + 400 cone, but with no cone)")
    ap.add_argument("--rot-seed", type=int, default=12345, dest="rot_seed",
                    help="fixed so every condition searches the same rotations")
    ap.add_argument("--ntop", type=int, default=1500)
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--rot-chunk", type=int, default=8, dest="rot_chunk")
    ap.add_argument("--dockq-budget", type=int, default=50_000_000, dest="dockq_budget")
    ap.add_argument("--oom-retries", type=int, default=3, dest="oom_retries")
    ap.add_argument("--export-pdb-dir", default="", dest="export_pdb_dir",
                    help="write the top-K poses here as {id}/model_k.pdb with "
                         "chains R and L, for scoring by PINDER's own evaluator")
    ap.add_argument("--export-top-k", type=int, default=5, dest="export_top_k")
    ap.add_argument("--monomer", default="holo",
                    choices=("holo", "apo", "predicted"),
                    help="which PINDER monomer setting these decoys came from. "
                         "This pipeline docks bound structures, so holo.")
    ap.add_argument("--pdb-dir", default="external/pinder/pinder/2024-02/pdbs",
                    dest="pdb_dir",
                    help="source complexes, re-read for atom names and numbering")
    ap.add_argument("--rmsd-threshold", type=float, default=5.0, dest="rmsd_thr")
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="dockq_thr")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1",
                    help="i/n -- evaluate every n-th id starting at i, so several GPUs can split one condition")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    label = args.label or (Path(args.ckpt).parent.name + "_" + Path(args.ckpt).stem
                           if args.ckpt else "baseline")

    beta = torch.tensor(3.0, device=device, dtype=dtype)
    charge = default_charge_score(device=device, dtype=dtype)
    iface0 = iface_ij(device=device, dtype=dtype, flat=True)
    if args.ckpt:
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        alpha = blob["alpha"].to(device=device, dtype=dtype)
        iface = blob["iface"].to(device=device, dtype=dtype)
        # rho is trainable since 2026-07-25; a checkpoint that lacks it predates
        # that and was fitted with rho frozen at the published value.
        rho = blob.get("rho", torch.tensor(SC_RHO)).to(device=device, dtype=dtype)
    else:
        # 1.0, not 0.01: Chen et al. 2003 Eq. (2) has no alpha at all (the PSC
        # scale is set by rho), and measured on the poses a ranker must
        # discriminate among, alpha = 1.02 equalises std(S_PSC) and
        # std(S_IFACE). 0.01 belongs to the older GSC formulation and puts PSC
        # at 1% of IFACE.
        alpha = torch.tensor(args.alpha0, device=device, dtype=dtype)
        iface = iface0.clone()
        rho = torch.tensor(args.rho0, device=device, dtype=dtype)
    print(f"[{label}] alpha={float(alpha):.4f} rho={float(rho):.4f} "
          f"||dIface||={float((iface - iface0).norm()):.3f}", flush=True)

    ids = [ln.strip() for ln in Path(args.test_ids).read_text().splitlines()
           if ln.strip()]
    if args.limit:
        ids = ids[: args.limit]
    si, sn = (int(x) for x in args.shard.split("/"))
    ids = ids[si::sn]

    rows, skipped = [], []
    t0 = time.time()
    for i, pid in enumerate(ids):
        prot_cpu = load_prepared(args.prep_cache, pid)
        if prot_cpu is None:
            skipped.append({"id": pid, "reason": "not in prep cache"})
            continue
        try:
            prot = prot_cpu.to(device, dtype=dtype)
            rows.append(evaluate_complex(prot, alpha, iface, beta, charge, args,
                                         rho=rho))
        except torch.cuda.OutOfMemoryError as exc:
            skipped.append({"id": pid, "n_rec": prot_cpu.n_rec,
                            "n_lig": prot_cpu.n_lig,
                            "reason": f"OOM: {str(exc)[:120]}"})
        except ExportError:
            raise                        # never file an export bug as "skipped"
        except Exception as exc:  # noqa: BLE001
            skipped.append({"id": pid, "reason": f"{type(exc).__name__}: {exc}"[:160]})
        finally:
            del prot_cpu
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(ids)}  ({(time.time()-t0)/(i+1):.1f}s/complex)",
                  flush=True)

    if getattr(args, "export_pdb_dir", ""):
        root = Path(args.export_pdb_dir)
        want = args.export_top_k
        bad = []
        for r in rows:
            models = sorted((root / r["name"] / args.monomer / "models")
                            .glob("model_*.pdb"))
            ranks = sorted(int(m.stem.split("_")[1]) for m in models)
            if ranks != list(range(1, want + 1)):
                bad.append((r["name"], ranks))
        print(f"\nexport: {len(rows)} systems evaluated, "
              f"{len(rows) - len(bad)} with exactly {want} models each")
        if bad:
            raise SystemExit(
                f"{len(bad)} systems do not have models 1..{want} "
                f"(e.g. {bad[0][0]}: {bad[0][1]}). PINDER reads the rank off "
                f"the file name and fills missing systems with DockQ = 0, so a "
                f"partial export would be scored as a worse method rather than "
                f"an incomplete one.")
        if skipped:
            print(f"export: {len(skipped)} systems were NOT evaluated and so "
                  f"have no decoys; PINDER will penalise them with DockQ = 0. "
                  f"Report that count alongside any leaderboard number.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(out_dir / f"{label}_per_complex.csv", "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")

    n = len(rows)
    rec = [r for r in rows if r["recall"]]
    summary = {
        "label": label, "n_evaluated": n, "n_skipped": len(skipped),
        "alpha": float(alpha), "rho": float(rho),
        "d_iface_norm": float((iface - iface0).norm()),
        "n_rot": args.n_rot, "ntop": args.ntop, "rot_seed": args.rot_seed,
        # parameter-independent: what the rotation sampling permits at all
        "ceiling_frac_dockq_ok": sum(r["ceiling_dockq"] >= args.dockq_thr
                                     for r in rows) / max(1, n),
        "ceiling_mean_dockq": _mean(rows, "ceiling_dockq"),
        "mean_min_rot_angle_deg": _mean(rows, "min_rot_angle_deg"),
        # did the search return anything near-native at all
        "recall": len(rec) / max(1, n),
        "mean_n_pos_out": _mean(rows, "n_pos_out"),
        "mean_best_dockq_out": _mean(rows, "best_dockq_out"),
        "seconds": time.time() - t0,
    }
    for k in KS:
        summary[f"succ_dockq@{k}"] = _mean(rows, f"succ_dockq@{k}")
        summary[f"succ_rmsd@{k}"] = _mean(rows, f"succ_rmsd@{k}")
        summary[f"succ_dockq@{k}|recall"] = _mean(rec, f"succ_dockq@{k}") if rec else float("nan")
        summary[f"mean_best_dockq@{k}"] = _mean(rows, f"best_dockq@{k}")
    (out_dir / f"{label}_summary.json").write_text(json.dumps(summary, indent=1))
    if skipped:
        (out_dir / f"{label}_skipped.json").write_text(json.dumps(skipped, indent=1))

    print("=" * 66)
    print(f"[{label}]  end-to-end search on {n} deleaked TEST complexes "
          f"({len(skipped)} skipped)")
    print(f"  rotation-sampling ceiling: {summary['ceiling_frac_dockq_ok']*100:.1f}% "
          f"of complexes could reach DockQ>=0.23 at the ideal translation "
          f"(mean nearest-rotation angle {summary['mean_min_rot_angle_deg']:.1f} deg)")
    print(f"  search recall (any near-native in returned top-{args.ntop}): "
          f"{summary['recall']*100:.1f}%")
    print(f"     K   | succ@K (DockQ) | succ@K | recall | succ@K (RMSD)")
    for k in KS:
        print(f"    {k:>4} |     {summary[f'succ_dockq@{k}']*100:5.1f}%     |"
              f"     {summary[f'succ_dockq@{k}|recall']*100:5.1f}%    |"
              f"     {summary[f'succ_rmsd@{k}']*100:5.1f}%")
    print(f"  mean best DockQ in returned set = {summary['mean_best_dockq_out']:.3f}")
    print(f"  wall {summary['seconds']/60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
