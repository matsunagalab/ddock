"""Are the top translations of a rotation really "extremely similar"?

`docking_search` keeps one translation per rotation (`trans_per_rotation`
defaults to 1, search.py:626, and no production script overrides it). The
comment at search.py:741 quotes Chen & Weng 2003 for the choice:

    "We used to keep the top 10 translations for each rotation. We
    subsequently discovered that these 10 translations are extremely similar,
    and keeping only the best one helped to remove false positives without
    affecting the ranking of the best hit."

HDOCK keeps ten (PINDER supplement A.7.2) and refines all of them, on a
*coarser* rotation grid than this repository uses, and is ahead. So the two
papers disagree and the one that disagrees with us is winning. This measures
which is right here, before anything is rebuilt around it.

One search per complex with `trans_per_rotation = 10` and a buffer big enough
to keep everything; every arm is then derived by post-filtering the same
candidate set, so the arms differ only in the retention rule and not in any
sampling noise.

Three things get measured.

  * spread -- how far the 2nd..10th translation of a rotation sit from its
    best one. This is Chen & Weng's claim, stated in angstroms.
  * whether the extra translations contain near-native poses that the
    one-per-rotation rule throws away (the ceiling question).
  * whether they survive into a *matched-size* pool and change the rank of the
    first hit (the deployment question). Keeping ten per rotation and then
    cutting to the same `ntop` evicts whole rotations, so more is not
    automatically better.

    CUDA_VISIBLE_DEVICES=3 uv run python scripts/trans_per_rotation_probe.py \
        --ckpt data/scaling/runs_sym/N1000/round0_ckpt.pt \
        --limit 12 --out data/scaling/tpr_probe.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_search_test import _label, _quat_angle_deg  # noqa: E402

from zdock.atomtypes import charge_score as default_charge_score  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.prep_cache import load_prepared  # noqa: E402
from zdock.rotation_grid import hopf_quaternions  # noqa: E402
from zdock.score import SC_REFERENCE_SPACING, SC_RHO  # noqa: E402
from zdock.search import _rotate_batch as _rotate  # noqa: E402
from zdock.search import docking_search  # noqa: E402


@torch.no_grad()
def probe(prot, alpha, iface, beta, charge, args, rho=SC_RHO) -> dict | None:
    device, dtype = prot.rec_xyz.device, prot.rec_xyz.dtype
    quats = hopf_quaternions(args.hopf_nside, device=device, dtype=dtype)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    n_rot = quats.shape[0]

    keep_all = n_rot * args.tpr
    chunk = args.rot_chunk
    for attempt in range(args.oom_retries + 1):
        try:
            res = docking_search(
                prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
                prot.rec_atomtype_id, prot.rec_charge_id,
                prot.lig_ref, prot.lig_radius, prot.lig_sasa,
                prot.lig_atomtype_id, prot.lig_charge_id,
                quats, alpha=alpha, iface_ij_flat=iface, beta=beta, sc_rho=rho,
                charge_score_lut=charge, spacing=args.spacing,
                ntop=keep_all, rot_chunk_size=chunk,
                trans_per_rotation=args.tpr)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            chunk = max(1, chunk // 2)
            if attempt == args.oom_retries:
                return None

    poses = _rotate(prot.lig_ref, quats[res.quat_indices]) + res.translations.unsqueeze(1)
    # DockQ over ~n_rot*tpr poses dominates the runtime, and a pose far from
    # the native cannot reach the acceptable threshold. Screen on ligand RMSD
    # first (cheap) and only score what could plausibly qualify. The cutoff is
    # deliberately far outside the acceptable band so the screen cannot change
    # any DockQ >= thr decision.
    from zdock.dockq import ligand_rmsd_to_native
    rmsd = ligand_rmsd_to_native(prot.native_lig, poses)
    near = rmsd <= args.dockq_rmsd_cut
    dockq = torch.zeros_like(rmsd)
    if bool(near.any()):
        idx = torch.nonzero(near).flatten()
        _, dq = _label(prot, poses[idx], args.dockq_budget)
        dockq[idx] = dq
    score, quat_idx, trans = res.scores, res.quat_indices, res.translations

    # Rank within each rotation by score: 0 is the translation the current
    # `trans_per_rotation=1` rule would have kept.
    order = torch.argsort(score, descending=True)
    within = torch.zeros_like(quat_idx)
    seen: dict[int, int] = {}
    for i in order.tolist():
        q = int(quat_idx[i])
        within[i] = seen.get(q, 0)
        seen[q] = seen.get(q, 0) + 1

    is_best = within == 0
    row = {
        "name": prot.name,
        "n_cand": int(score.numel()),
        "n_rot_filled": len(seen),
        "min_rot_angle_deg": float(_quat_angle_deg(quats, prot.q_star).min()),
    }

    # --- 1. spread: distance from a rotation's best translation to its others
    best_t = {int(quat_idx[i]): trans[i] for i in torch.nonzero(is_best).flatten().tolist()}
    d = []
    for i in torch.nonzero(~is_best).flatten().tolist():
        bt = best_t.get(int(quat_idx[i]))
        if bt is not None:
            d.append(float(torch.linalg.norm(trans[i] - bt)))
    d = np.array(d) if d else np.zeros(1)
    row["spread_median_A"] = float(np.median(d))
    row["spread_p90_A"] = float(np.percentile(d, 90))
    row["spread_frac_under_2A"] = float(np.mean(d < 2.0))
    row["spread_frac_over_5A"] = float(np.mean(d > 5.0))

    thr = args.dockq_thr
    pos = dockq >= thr
    row["n_pos_all"] = int(pos.sum())
    row["n_pos_best"] = int((pos & is_best).sum())
    row["n_pos_extra"] = int((pos & ~is_best).sum())

    # --- 2. ceiling: does dropping the extras lose the near-natives entirely?
    row["best_dockq_all"] = float(dockq.max())
    row["best_dockq_bestonly"] = float(dockq[is_best].max()) if is_best.any() else 0.0

    # --- 3. deployment: matched-size pools, ranked by score
    def pool_stats(mask, ntop, tag):
        idx = torch.nonzero(mask).flatten()
        idx = idx[torch.argsort(score[idx], descending=True)][:ntop]
        dq = dockq[idx]
        hit = (dq >= thr).nonzero(as_tuple=True)[0]
        row[f"recall_{tag}"] = int(bool((dq >= thr).any()))
        row[f"first_hit_{tag}"] = int(hit[0]) + 1 if hit.numel() else 10 ** 6
        row[f"top1_{tag}"] = int(bool(dq[0] >= thr)) if dq.numel() else 0
        row[f"best_dockq_{tag}"] = float(dq.max()) if dq.numel() else 0.0

    pool_stats(is_best, args.ntop, "tpr1")               # current deployment
    pool_stats(torch.ones_like(is_best), args.ntop, "tpr10_matched")
    pool_stats(torch.ones_like(is_best), keep_all, "tpr10_full")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache_test",
                    dest="prep_cache")
    ap.add_argument("--ids-file", default="data/scaling/pinder_s_all_ids.txt",
                    dest="ids_file")
    ap.add_argument("--tpr", type=int, default=10)
    ap.add_argument("--ntop", type=int, default=1500)
    ap.add_argument("--hopf-nside", type=int, default=3, dest="hopf_nside")
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--rot-chunk", type=int, default=8, dest="rot_chunk")
    ap.add_argument("--oom-retries", type=int, default=4, dest="oom_retries")
    ap.add_argument("--dockq-thr", type=float, default=0.23, dest="dockq_thr")
    ap.add_argument("--dockq-budget", type=int, default=200_000_000,
                    dest="dockq_budget")
    ap.add_argument("--dockq-rmsd-cut", type=float, default=30.0,
                    dest="dockq_rmsd_cut",
                    help="skip DockQ for poses further than this from the "
                         "native; far outside the acceptable band")
    ap.add_argument("--rho0", type=float, default=SC_RHO)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device, dtype = torch.device(args.device), torch.float32
    beta = torch.tensor(3.0, device=device, dtype=dtype)
    charge = default_charge_score(device=device, dtype=dtype)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        alpha = ck["alpha"].to(device=device, dtype=dtype)
        iface = ck["iface"].to(device=device, dtype=dtype)
        rho = ck.get("rho", torch.tensor(args.rho0)).to(device=device, dtype=dtype)
    else:
        alpha = torch.tensor(1.0, device=device, dtype=dtype)
        iface = iface_ij(device=device, dtype=dtype, flat=True)
        rho = torch.tensor(args.rho0, device=device, dtype=dtype)

    ids = [i.strip() for i in Path(args.ids_file).read_text().split() if i.strip()]
    si, sn = (int(x) for x in args.shard.split("/"))
    ids = ids[si::sn][: args.limit] if args.limit else ids[si::sn]

    rows, t0 = [], time.time()
    for pid in ids:
        prot = load_prepared(args.prep_cache, pid)
        if prot is None:
            continue
        r = probe(prot.to(device, dtype=dtype), alpha, iface, beta, charge, args, rho)
        if r is None:
            print(f"  {pid}: OOM, skipped", flush=True)
            continue
        rows.append(r)
        print(f"  {r['name'][:34]:36s} spread med {r['spread_median_A']:5.2f} A "
              f"(<2A {r['spread_frac_under_2A']:.2f}) | pos best/extra "
              f"{r['n_pos_best']:4d}/{r['n_pos_extra']:4d} | first hit "
              f"{r['first_hit_tpr1']:>7} -> {r['first_hit_tpr10_matched']:>7}",
              flush=True)

    if not rows:
        raise SystemExit("nothing probed")
    print(f"\n{len(rows)} complexes, {time.time() - t0:.0f} s\n")

    sp = np.array([r["spread_median_A"] for r in rows])
    u2 = np.array([r["spread_frac_under_2A"] for r in rows])
    o5 = np.array([r["spread_frac_over_5A"] for r in rows])
    print("1. spread of the 2nd..Nth translation from the best one, per rotation")
    print(f"   median over complexes : {np.median(sp):.2f} A")
    print(f"   fraction within 2 A   : {np.mean(u2) * 100:.1f}%")
    print(f"   fraction beyond 5 A   : {np.mean(o5) * 100:.1f}%")

    pb = np.array([r["n_pos_best"] for r in rows])
    pe = np.array([r["n_pos_extra"] for r in rows])
    print("\n2. near-native poses (DockQ >= thr) in the full candidate set")
    print(f"   in the kept-one-per-rotation subset : {int(pb.sum())}")
    print(f"   only in the discarded translations  : {int(pe.sum())}")
    da = np.array([r["best_dockq_all"] for r in rows])
    db = np.array([r["best_dockq_bestonly"] for r in rows])
    print(f"   best DockQ  all {np.median(da):.3f}  vs best-only {np.median(db):.3f} "
          f"(median); complexes where dropping extras lowers it: "
          f"{int(np.sum(da > db + 1e-6))}/{len(rows)}")

    print("\n3. pools ranked by score")
    print(f"   {'arm':>16} | {'recall':>7} | {'top-1 hit':>10} | {'median first hit':>17}")
    for tag, lab in [("tpr1", "tpr=1 ntop1500"),
                     ("tpr10_matched", "tpr=10 ntop1500"),
                     ("tpr10_full", "tpr=10 keep all")]:
        rec = np.mean([r[f"recall_{tag}"] for r in rows]) * 100
        t1 = np.mean([r[f"top1_{tag}"] for r in rows]) * 100
        fh = np.median([r[f"first_hit_{tag}"] for r in rows])
        print(f"   {lab:>16} | {rec:6.1f}% | {t1:9.1f}% | {fh:17.0f}")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
