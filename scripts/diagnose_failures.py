"""Why does the search rank a wrong pose above the right one? Term by term.

The rotation ladder (EXPERIMENT_REPORT §5.6.9) ruled out sampling: pushing the
uniform rotation set from 1900 to 54000 orientations lowered the mean
nearest-rotation error from 11.1° to 3.9° and lifted the reachability ceiling,
yet the four failing complexes still returned nothing near-native. So the
correct pose *is* in the candidate space and the score prefers something else.

This script takes the two poses that matter for each complex --- the one the
search actually ranked first, and the best near-native pose available --- and
decomposes the score gap between them:

    S = alpha * (S_fav - S_clash) + S_IFACE + beta * S_ELEC

``S_fav`` (the PSC atom-pair count) and ``S_clash`` (the grid-overlap penalty)
are separated by evaluating the score twice, once with ``rho = 0`` --- which
zeroes the imaginary channel and therefore the clash term --- so no new grid
plumbing is needed.

A positive gap means the term favours the wrong pose. Comparing the profile of
the complexes the search solves against the ones it fails on says which term to
fix, and whether the published ``alpha``, ``D`` and ``rho`` need re-tuning for
this atom-typing.

Example
-------
    PINDER_BASE_DIR=$PWD/external/pinder CUDA_VISIBLE_DEVICES=6 \\
    uv run python scripts/diagnose_failures.py --n-complexes 12
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from zdock.atomtypes import charge_score as default_charge_score  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.dockq import dockq_batch, ligand_rmsd_to_native  # noqa: E402
from zdock.prep_cache import load_prepared  # noqa: E402
from zdock.rotation_grid import (  # noqa: E402
    hopf_quaternions,
    random_quaternions,
    rotation_cone,
)
from zdock.score import (  # noqa: E402
    IFACE_PAIR_OFFSET,
    IFACE_SIGN,
    PSC_D,
    SC_REFERENCE_SPACING,
    SC_RHO,
    docking_score_elec,
)
from zdock.search import _rotate_batch, docking_search  # noqa: E402


@torch.no_grad()
def _dockq(prot, poses, budget):
    per = max(1, prot.n_rec * prot.n_lig)
    chunk = int(max(1, min(64, budget // per)))
    rmsd = ligand_rmsd_to_native(prot.native_lig, poses)
    parts = []
    for s in range(0, poses.shape[0], chunk):
        parts.append(dockq_batch(prot.rec_xyz, poses[s:s + chunk],
                                 prot.native_lig).dockq)
    return rmsd, torch.cat(parts, dim=0)


@torch.no_grad()
def terms(prot, poses, alpha, iface, beta, charge, args):
    """Return the four score components for each pose.

    ``rho = 0`` removes the imaginary (clash) channel entirely, so the PSC
    value it returns is the bare favourable atom-pair count; the clash penalty
    is the difference against the full evaluation.
    """
    def _call(rho):
        return docking_score_elec(
            prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
            prot.rec_atomtype_id, prot.rec_charge_id,
            poses, prot.lig_radius, prot.lig_sasa,
            prot.lig_atomtype_id, prot.lig_charge_id,
            alpha, iface, beta, charge, spacing=args.spacing,
            sc_rho=rho, psc_d=args.psc_d,
            frame_chunk_size=args.frame_chunk, return_components=True)

    psc_full, T, elec = _call(args.sc_rho)
    psc_fav, _, _ = _call(0.0)
    imat = IFACE_PAIR_OFFSET + IFACE_SIGN * iface.view(12, 12).T
    return {
        "fav": psc_fav,
        "clash": psc_fav - psc_full,          # >= 0
        "psc": psc_full,
        "iface": (imat * T).sum(dim=(-2, -1)),
        "elec": beta * elec,
        "total": args.alpha * psc_full + (imat * T).sum(dim=(-2, -1)) + beta * elec,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache", dest="prep_cache")
    ap.add_argument("--prep-manifest", default="data/scaling/prep_manifest.jsonl",
                    dest="prep_manifest")
    ap.add_argument("--grid-voxels", default="data/scaling/grid_voxels.json",
                    dest="grid_voxels")
    ap.add_argument("--max-voxels-at-3a", type=int, default=150_000, dest="max_vox")
    ap.add_argument("--n-complexes", type=int, default=12, dest="n_complexes")
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--sc-rho", type=float, default=SC_RHO, dest="sc_rho")
    ap.add_argument("--psc-d", type=float, default=PSC_D, dest="psc_d")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--n-rot", type=int, default=1900, dest="n_rot")
    ap.add_argument("--rot-seed", type=int, default=12345, dest="rot_seed")
    ap.add_argument("--rot-set", choices=("uniform", "hopf"), default="hopf",
                    dest="rot_set")
    ap.add_argument("--hopf-nside", type=int, default=3, dest="hopf_nside")
    ap.add_argument("--trans-per-rotation", type=int, default=1,
                    dest="trans_per_rotation")
    ap.add_argument("--ntop", type=int, default=2000)
    ap.add_argument("--rot-chunk", type=int, default=4, dest="rot_chunk")
    ap.add_argument("--n-near", type=int, default=200, dest="n_near")
    ap.add_argument("--cone-deg", type=float, default=10.0, dest="cone_deg")
    ap.add_argument("--frame-chunk", type=int, default=25, dest="frame_chunk")
    ap.add_argument("--dockq-budget", type=int, default=50_000_000, dest="dockq_budget")
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="dockq_thr")
    ap.add_argument("--out", default="data/scaling/failure_diagnosis.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    vox = json.loads(Path(args.grid_voxels).read_text())
    ok = [json.loads(l)["id"] for l in Path(args.prep_manifest).read_text().splitlines()
          if l.strip() and json.loads(l)["status"] == "ok"]
    picks = [p for p in ok if vox.get(p, 1 << 60) <= args.max_vox][: args.n_complexes]

    alpha = torch.tensor(args.alpha, device=device, dtype=dtype)
    beta = torch.tensor(args.beta, device=device, dtype=dtype)
    iface = iface_ij(device=device, dtype=dtype, flat=True)
    charge = default_charge_score(device=device, dtype=dtype)

    rows = []
    for pid in picks:
        prot_cpu = load_prepared(args.prep_cache, pid)
        if prot_cpu is None:
            continue
        prot = prot_cpu.to(device, dtype=dtype)
        if args.rot_set == "hopf":
            quats = hopf_quaternions(args.hopf_nside, device=device, dtype=dtype)
            quats = quats / quats.norm(dim=-1, keepdim=True)
        else:
            quats = random_quaternions(args.n_rot, seed=args.rot_seed,
                                       device=device, dtype=dtype)
        res = docking_search(
            prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
            prot.rec_atomtype_id, prot.rec_charge_id,
            prot.lig_ref, prot.lig_radius, prot.lig_sasa,
            prot.lig_atomtype_id, prot.lig_charge_id,
            quats, alpha=alpha, iface_ij_flat=iface, beta=beta,
            charge_score_lut=charge, spacing=args.spacing, ntop=args.ntop,
            rot_chunk_size=args.rot_chunk, sc_rho=args.sc_rho, psc_d=args.psc_d,
            trans_per_rotation=args.trans_per_rotation)
        # res.scores is sorted descending; the rotation that produced entry i is
        # quats[res.quat_indices[i]], NOT quats[i].
        top = int(res.scores.argmax())
        qi = int(res.quat_indices[top])
        won = (_rotate_batch(prot.lig_ref, quats[qi:qi + 1])
               + res.translations[top:top + 1].unsqueeze(1))

        # Three poses, only the last two of which the search can actually emit:
        #   ideal    - exact (q*, t*); off both the rotation set and the lattice
        #   reachable- nearest sampled rotation at the lattice-snapped t*; this
        #              is the best the search could possibly have returned
        #   won      - what it did return
        # ideal vs reachable measures what discretisation costs; reachable vs
        # won measures whether the score prefers the wrong pose.
        ideal = (_rotate_batch(prot.lig_ref, prot.q_star.unsqueeze(0))
                 + prot.t_star.unsqueeze(0).unsqueeze(0))
        d_ang = torch.rad2deg(2.0 * torch.acos(
            (quats * prot.q_star.unsqueeze(0)).sum(-1).abs().clamp(max=1.0)))
        nearest = int(d_ang.argmin())
        t_snap = torch.round(prot.t_star / args.spacing) * args.spacing
        reach = (_rotate_batch(prot.lig_ref, quats[nearest:nearest + 1])
                 + t_snap.unsqueeze(0).unsqueeze(0))

        _, dq_ideal = _dockq(prot, ideal, args.dockq_budget)
        _, dq_reach = _dockq(prot, reach, args.dockq_budget)
        # Recall over the whole returned set: does the search surface the answer
        # at all? Grouping on top-1 alone left the "solved" group at n=1.
        all_poses = (_rotate_batch(prot.lig_ref, quats[res.quat_indices])
                     + res.translations.unsqueeze(1))
        _, dq_all = _dockq(prot, all_poses, args.dockq_budget)
        n_pos_out = int((dq_all >= args.dockq_thr).sum())
        del all_poses
        _, dq_won = _dockq(prot, won, args.dockq_budget)
        t_ideal = terms(prot, ideal, alpha, iface, beta, charge, args)
        t_reach = terms(prot, reach, alpha, iface, beta, charge, args)
        t_won = terms(prot, won, alpha, iface, beta, charge, args)

        row = {"name": pid, "min_rot_angle_deg": float(d_ang.min()),
               "dockq_ideal": float(dq_ideal[0]), "dockq_reach": float(dq_reach[0]),
               "dockq_won": float(dq_won[0]),
               "n_pos_out": n_pos_out, "best_dockq_out": float(dq_all.max()),
               "solved": bool(n_pos_out > 0)}
        for k in ("fav", "clash", "psc", "iface", "elec", "total"):
            row[f"ideal_{k}"] = float(t_ideal[k][0])
            row[f"reach_{k}"] = float(t_reach[k][0])
            row[f"won_{k}"] = float(t_won[k][0])
            row[f"gap_{k}"] = float(t_won[k][0] - t_reach[k][0])     # won - reachable
            row[f"disc_{k}"] = float(t_reach[k][0] - t_ideal[k][0])  # discretisation
        rows.append(row)
        print(f"  [{pid[:26]:<26}] DockQ ideal={row['dockq_ideal']:.2f} "
              f"reach={row['dockq_reach']:.2f} won={row['dockq_won']:.2f} | "
              f"score ideal={row['ideal_total']:8.0f} reach={row['reach_total']:8.0f} "
              f"won={row['won_total']:8.0f}", flush=True)
        del prot, prot_cpu
        if device.type == "cuda":
            torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(rows, indent=1))
    solved = [r for r in rows if r["solved"]]
    failed = [r for r in rows if not r["solved"]]

    def avg(rs, k):
        return sum(r[k] for r in rs) / len(rs) if rs else float("nan")

    print("\n" + "=" * 78)
    print("A) does the search return the best pose it COULD have returned?")
    print("   score(search top-1) - score(nearest sampled rotation @ snapped t*)")
    print("   positive = the score genuinely prefers the wrong pose")
    print("=" * 78)
    hdr = f"{'group':<14}{'n':>3}  " + "".join(f"{k:>11}" for k in
          ("a*fav", "-a*clash", "a*PSC", "IFACE", "b*ELEC", "TOTAL"))
    print(hdr)
    for label, rs in (("recall>0", solved), ("recall=0", failed)):
        if not rs:
            continue
        print(f"{label:<14}{len(rs):>3}  "
              f"{args.alpha*avg(rs,'gap_fav'):>11.1f}"
              f"{-args.alpha*avg(rs,'gap_clash'):>11.1f}"
              f"{args.alpha*avg(rs,'gap_psc'):>11.1f}"
              f"{avg(rs,'gap_iface'):>11.1f}"
              f"{avg(rs,'gap_elec'):>11.1f}"
              f"{avg(rs,'gap_total'):>11.1f}")

    print("\n" + "=" * 78)
    print("B) what does discretisation cost the native pose?")
    print("   score(nearest sampled rotation @ snapped t*) - score(exact q*, t*)")
    print("=" * 78)
    print(hdr)
    for label, rs in (("recall>0", solved), ("recall=0", failed)):
        if not rs:
            continue
        print(f"{label:<14}{len(rs):>3}  "
              f"{args.alpha*avg(rs,'disc_fav'):>11.1f}"
              f"{-args.alpha*avg(rs,'disc_clash'):>11.1f}"
              f"{args.alpha*avg(rs,'disc_psc'):>11.1f}"
              f"{avg(rs,'disc_iface'):>11.1f}"
              f"{avg(rs,'disc_elec'):>11.1f}"
              f"{avg(rs,'disc_total'):>11.1f}")

    print("\n" + "=" * 78)
    print("C) absolute totals and DockQ")
    print("=" * 78)
    print(f"{'group':<14}{'n':>3}{'ideal':>10}{'reach':>10}{'won':>10}"
          f"{'DockQ id':>10}{'DockQ re':>10}{'DockQ won':>10}{'minAng':>8}")
    for label, rs in (("recall>0", solved), ("recall=0", failed)):
        if not rs:
            continue
        print(f"{label:<14}{len(rs):>3}{avg(rs,'ideal_total'):>10.0f}"
              f"{avg(rs,'reach_total'):>10.0f}{avg(rs,'won_total'):>10.0f}"
              f"{avg(rs,'dockq_ideal'):>10.2f}{avg(rs,'dockq_reach'):>10.2f}"
              f"{avg(rs,'dockq_won'):>10.2f}{avg(rs,'min_rot_angle_deg'):>7.1f}°")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
