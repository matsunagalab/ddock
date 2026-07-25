"""Which scoring term separates near-native poses from decoys, how large is
each term, and what weighting does the score actually need?

Motivation
----------
After ``S_SC`` was corrected to Chen & Weng 2002 Eq. (1) (surface-surface = +1,
surface-core = -rho, core-core = -rho^2) the FFT search *still* returns nothing
near-native: over 12 complexes at the paper's 1.2 Å spacing the reachability
ceiling is DockQ 0.90 (100% of complexes could reach an acceptable pose) yet
recall is 0.0%. So the remaining failure is in how the three terms are combined,
not in the search space.

Two earlier attempts to measure this were both invalid, in opposite directions:

* random ligand placements inside the receptor bounding box — most decoys
  *interpenetrate* the receptor, which no real search would ever propose, and
  contact-counting terms are dominated by the overlap volume;
* the cached TEST pool — its decoys were *selected* for scoring highly under
  the default parameters, so every term correlated with that score shows an AUC
  near 0 by construction.

This script uses decoys that are neither: each is placed along a random
direction at a controlled surface gap, so it touches the receptor without
interpenetrating, and nothing about the scoring function influenced its
selection.

Reports
-------
1. the magnitude of each weighted term, i.e. what actually drives the argmax;
2. the Mann-Whitney AUC of each term alone (0.5 = no signal);
3. the AUC of the combined score under several ``(alpha, beta)`` weightings,
   including the paper's ``alpha = 0.01, beta = 0.06`` (Eq. (6), p.284) against
   the ``beta = 3.0`` this repository has been using.

Example
-------
    PINDER_BASE_DIR=$PWD/external/pinder CUDA_VISIBLE_DEVICES=6 \\
    uv run python scripts/diagnose_score_terms.py --n-complexes 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from zdock.atomtypes import charge_score as default_charge_score  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.dockq import dockq_batch, ligand_rmsd_to_native  # noqa: E402
from zdock.prep_cache import load_prepared  # noqa: E402
from zdock.rotation_grid import random_quaternions, rotation_cone  # noqa: E402
from zdock.score import (SC_REFERENCE_SPACING, docking_score_elec,  # noqa: E402
                         iface_score_matrix)
from zdock.search import _rotate_batch  # noqa: E402

#: Chen & Weng 2002 Eq. (6): "The default values for scaling factors are
#: alpha = 0.01 and beta = 0.06 in this study."
PAPER_ALPHA, PAPER_BETA = 0.01, 0.06
REPO_BETA = 3.0


def auc(scores: torch.Tensor, pos: torch.Tensor) -> float:
    """Mann-Whitney AUC with **midrank tie correction**.

    Several of the quantities scored here are integer contact counts, i.e.
    maximally tied inputs. Without the midrank block an all-tied vector returns
    0.70 instead of 0.50 and the result depends on the input order (five random
    permutations of one dataset gave 0.859-0.871).
    """
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64,
                               device=scores.device)
    uniq, inv, counts = torch.unique(scores, return_inverse=True,
                                     return_counts=True)
    if int((counts > 1).sum()):
        rank_sum = torch.zeros(uniq.numel(), dtype=torch.float64,
                               device=scores.device)
        rank_sum.index_add_(0, inv, ranks)
        ranks = (rank_sum / counts.to(torch.float64))[inv]
    u = float(ranks[pos].sum()) - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


@torch.no_grad()
def _min_rec_lig_dist(rec_xyz, poses, chunk=4):
    """Per-pose closest receptor-ligand atom distance."""
    out = []
    for s in range(0, poses.shape[0], chunk):
        d = torch.cdist(poses[s:s + chunk], rec_xyz.unsqueeze(0).expand(
            min(chunk, poses.shape[0] - s), -1, -1))
        out.append(d.amin(dim=(-2, -1)))
    return torch.cat(out)


@torch.no_grad()
def build_poses(prot, args):
    """Near-native poses + **surface-touching, non-clashing** decoys.

    Each decoy is placed along a random unit direction at the distance where
    the two bounding extents along that direction just separate, plus a gap, so
    it rests on the receptor surface. Poses are then filtered on the true
    closest-atom distance to guarantee no interpenetration.
    """
    device, dtype = prot.rec_xyz.device, prot.rec_xyz.dtype
    q_near = rotation_cone(prot.q_star, args.n_near, cone_deg=args.cone_deg,
                           seed=args.seed, device=device, dtype=dtype)
    near = _rotate_batch(prot.lig_ref, q_near) + prot.t_star.unsqueeze(0).unsqueeze(0)

    n_try = args.n_decoy * args.oversample
    q_rand = random_quaternions(n_try, seed=args.seed + 1, device=device, dtype=dtype)
    rot = _rotate_batch(prot.lig_ref, q_rand)                     # (n_try, N, 3)

    g = torch.Generator(device="cpu").manual_seed(args.seed + 2)
    u = torch.randn(n_try, 3, generator=g).to(device=device, dtype=dtype)
    u = u / u.norm(dim=-1, keepdim=True)

    # receptor extent along +u, ligand extent along -u  ->  touching offset
    rec_ext = (prot.rec_xyz.unsqueeze(0) * u.unsqueeze(1)).sum(-1).amax(dim=-1)
    lig_ext = (-(rot * u.unsqueeze(1)).sum(-1)).amax(dim=-1)
    gaps = (torch.rand(n_try, generator=g).to(device=device, dtype=dtype)
            * (args.gap_hi - args.gap_lo) + args.gap_lo)
    t = u * (rec_ext + lig_ext + gaps).unsqueeze(-1)
    decoy = rot + t.unsqueeze(1)

    dmin = _min_rec_lig_dist(prot.rec_xyz, decoy, chunk=args.dist_chunk)
    keep = ((dmin >= args.min_contact_dist) & (dmin <= args.max_contact_dist))
    idx = keep.nonzero(as_tuple=True)[0][: args.n_decoy]
    return torch.cat([near, decoy[idx]], dim=0), near.shape[0], int(idx.numel())


@torch.no_grad()
def components(prot, poses, alpha, iface, beta, charge, args):
    """Return ``(S_SC, T, S_ELEC)`` — ``T`` un-contracted so the same poses can
    be re-scored under different atom-pair matrices."""
    sc, T, elec = docking_score_elec(
        prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
        prot.rec_atomtype_id, prot.rec_charge_id,
        poses, prot.lig_radius, prot.lig_sasa,
        prot.lig_atomtype_id, prot.lig_charge_id,
        alpha, iface, beta, charge, spacing=args.spacing,
        frame_chunk_size=args.frame_chunk, return_components=True)
    return sc, T, elec


@torch.no_grad()
def label(prot, poses, budget):
    per = max(1, prot.n_rec * prot.n_lig)
    chunk = int(max(1, min(64, budget // per)))
    rmsd = ligand_rmsd_to_native(prot.native_lig, poses)
    parts = []
    for s in range(0, poses.shape[0], chunk):
        parts.append(dockq_batch(prot.rec_xyz, poses[s:s + chunk],
                                 prot.native_lig).dockq)
    return rmsd, torch.cat(parts, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache", dest="prep_cache")
    ap.add_argument("--prep-manifest", default="data/scaling/prep_manifest.jsonl",
                    dest="prep_manifest")
    ap.add_argument("--grid-voxels", default="data/scaling/grid_voxels.json",
                    dest="grid_voxels")
    ap.add_argument("--max-voxels-at-3a", type=int, default=150_000, dest="max_vox")
    ap.add_argument("--n-complexes", type=int, default=8, dest="n_complexes")
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--n-near", type=int, default=100, dest="n_near")
    ap.add_argument("--n-decoy", type=int, default=300, dest="n_decoy")
    ap.add_argument("--oversample", type=int, default=6)
    ap.add_argument("--gap-lo", type=float, default=-1.0, dest="gap_lo")
    ap.add_argument("--gap-hi", type=float, default=3.0, dest="gap_hi")
    ap.add_argument("--min-contact-dist", type=float, default=3.0,
                    dest="min_contact_dist", help="reject interpenetrating decoys")
    ap.add_argument("--max-contact-dist", type=float, default=6.0,
                    dest="max_contact_dist", help="reject non-touching decoys")
    ap.add_argument("--cone-deg", type=float, default=15.0, dest="cone_deg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-chunk", type=int, default=25, dest="frame_chunk")
    ap.add_argument("--dist-chunk", type=int, default=4, dest="dist_chunk")
    ap.add_argument("--dockq-budget", type=int, default=50_000_000, dest="dockq_budget")
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="dockq_thr")
    ap.add_argument("--match-min", type=int, default=15, dest="match_min",
                    help="min poses per class inside the matched contact band")
    ap.add_argument("--out", default="data/scaling/score_term_diagnosis.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if device.type == "cpu" else torch.float32

    vox = json.loads(Path(args.grid_voxels).read_text())
    ok = [json.loads(l)["id"] for l in Path(args.prep_manifest).read_text().splitlines()
          if l.strip() and json.loads(l)["status"] == "ok"]
    picks = [p for p in ok if vox.get(p, 1 << 60) <= args.max_vox][: args.n_complexes]

    alpha1 = torch.tensor(1.0, device=device, dtype=dtype)   # extract raw S_SC
    beta1 = torch.tensor(1.0, device=device, dtype=dtype)    # extract raw S_ELEC
    iface = iface_ij(device=device, dtype=dtype, flat=True)
    charge = default_charge_score(device=device, dtype=dtype)

    weightings = [("paper alpha=0.01 beta=0.06", PAPER_ALPHA, PAPER_BETA),
                  ("repo  alpha=0.01 beta=3.0", PAPER_ALPHA, REPO_BETA),
                  ("SC only", 1.0, 0.0),
                  ("IFACE only", 0.0, 0.0)]

    rows = []
    for pid in picks:
        prot_cpu = load_prepared(args.prep_cache, pid)
        if prot_cpu is None:
            continue
        prot = prot_cpu.to(device, dtype=dtype)
        poses, n_near, n_dec = build_poses(prot, args)
        if n_dec < 20:
            print(f"  [{pid}] only {n_dec} clash-free decoys — skipped", flush=True)
            del prot, prot_cpu
            continue
        rmsd, dockq = label(prot, poses, args.dockq_budget)
        pos = dockq >= args.dockq_thr
        # keep only genuinely non-native decoys as the negative class
        neg_ok = (~pos)
        if not bool(pos.any()) or not bool(neg_ok.any()):
            del prot, prot_cpu, poses
            continue
        sc, T, el = components(prot, poses, alpha1, iface, beta1, charge, args)
        # The score the FFT search ranks by applies IFACE_SIGN; reconstructing
        # from the raw table gives the NEGATION of the real IFACE term and
        # silently relabels every downstream column.
        imat = iface_score_matrix(iface)
        ones = torch.ones_like(imat)
        IF = (imat * T).sum(dim=(-2, -1))
        CNT = (ones * T).sum(dim=(-2, -1))          # pure contact count
        row = {"name": pid, "n_pos": int(pos.sum()), "n_neg": int(neg_ok.sum()),
               "n_decoy_kept": n_dec,
               "absmean_SC": float(sc.abs().mean()),
               "absmean_IFACE": float(IF.abs().mean()),
               "absmean_ELEC": float(el.abs().mean()),
               "absmean_COUNT": float(CNT.abs().mean()),
               "auc_SC": auc(sc, pos), "auc_IFACE": auc(IF, pos),
               "auc_ELEC": auc(el, pos),
               "auc_COUNT": auc(CNT, pos),          # near-native = more contacts?
               # control: what the AUC would be with the sign convention
               # reversed. `auc_IFACE` is the term actually scored.
               "auc_FLIPPED_IFACE": auc(-IF, pos)}
        # --- contact-count-matched comparison -------------------------------
        # Near-native poses make a large complementary interface while a decoy
        # dropped onto a convex surface only grazes it, so the raw comparison
        # is separable by interface *size* alone (contact-count AUC ~1.0) and
        # cannot say anything about interface *chemistry*. Restricting both
        # classes to the overlapping band of contact counts removes that.
        cnt_pos, cnt_neg = CNT[pos], CNT[~pos]
        lo = max(float(cnt_pos.min()), float(cnt_neg.min()))
        hi = min(float(cnt_pos.max()), float(cnt_neg.max()))
        band = (CNT >= lo) & (CNT <= hi)
        n_bp, n_bn = int((band & pos).sum()), int((band & ~pos).sum())
        row["match_lo"], row["match_hi"] = lo, hi
        row["match_n_pos"], row["match_n_neg"] = n_bp, n_bn
        if n_bp >= args.match_min and n_bn >= args.match_min:
            b = band.nonzero(as_tuple=True)[0]
            pb = pos[b]
            row["match_auc_COUNT"] = auc(CNT[b], pb)
            row["match_auc_IFACE"] = auc(IF[b], pb)
            row["match_auc_FLIPPED_IFACE"] = auc(-IF[b], pb)
            row["match_auc_SC"] = auc(sc[b], pb)
        else:
            for k in ("COUNT", "IFACE", "NEG_IFACE", "SC"):
                row[f"match_auc_{k}"] = float("nan")

        # alpha that puts the two terms on equal footing for this complex
        a_bal = float(IF.abs().mean() / max(1e-12, sc.abs().mean()))
        row["alpha_balanced"] = a_bal
        combos = [("a=0.01, +IFACE", PAPER_ALPHA, +1.0),
                  ("a=0.01, -IFACE", PAPER_ALPHA, -1.0),
                  ("a=balanced, +IFACE", a_bal, +1.0),
                  ("a=balanced, -IFACE", a_bal, -1.0),
                  ("SC alone", 1.0, 0.0)]
        for name, a, s in combos:
            row[f"auc[{name}]"] = auc(a * sc + s * IF, pos)
        for name, a, b in weightings:
            row[f"auc[{name}]"] = auc(a * sc + IF + b * el, pos)
            row[f"absmean_aSC[{name}]"] = float((a * sc).abs().mean())
            row[f"absmean_bEL[{name}]"] = float((b * el).abs().mean())
        rows.append(row)
        print(f"  [{pid[:30]:<30}] AUC  SC={row['auc_SC']:.3f} "
              f"IFACE={row['auc_IFACE']:.3f} flip={row['auc_FLIPPED_IFACE']:.3f} "
              f"count={row['auc_COUNT']:.3f} | bal-IF={row['auc[a=balanced, -IFACE]']:.3f}",
              flush=True)
        del prot, prot_cpu, poses
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("no complex produced a usable positive/negative split")
    Path(args.out).write_text(json.dumps(rows, indent=1))
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n  # noqa: E731

    print("\n" + "=" * 78)
    print(f"{n} complexes, {args.n_near} near-native vs surface-touching "
          f"non-clashing decoys (spacing {args.spacing} Å)")
    print("=" * 78)
    print("1) TERM MAGNITUDE (mean |term|, unweighted)")
    for t in ("SC", "IFACE", "ELEC"):
        print(f"     S_{t:<6} {mean(f'absmean_{t}'):14.4g}")
    print("\n2) TERM DISCRIMINATION (Mann-Whitney AUC; 0.5 = no signal)")
    for t in ("SC", "IFACE", "ELEC"):
        print(f"     S_{t:<6} {mean(f'auc_{t}'):8.4f}")
    print("\n2b) IS THE IFACE TABLE SIGN-FLIPPED, OR IS THIS JUST CONTACT COUNT?")
    print(f"     contact count  Sum n_ij     {mean('auc_COUNT'):8.4f}  "
          f"(near-native should have MORE contacts)")
    print(f"     +IFACE         Sum e_ij n_ij{mean('auc_IFACE'):8.4f}")
    print(f"     flipped IFACE +Sum e_ij n_ij{mean('auc_FLIPPED_IFACE'):8.4f}")
    print("\n2c) SC / IFACE BALANCE")
    print(f"     alpha making |a*S_SC| = |S_IFACE|: {mean('alpha_balanced'):.4f} "
          f"(repo/paper use 0.01)")
    for name in ("a=0.01, +IFACE", "a=0.01, -IFACE", "a=balanced, +IFACE",
                 "a=balanced, -IFACE", "SC alone"):
        print(f"     {name:<24}{mean(f'auc[{name}]'):8.4f}")
    def nanmean(k):
        xs = [r[k] for r in rows if not (isinstance(r[k], float) and math.isnan(r[k]))]
        return (sum(xs) / len(xs)) if xs else float("nan"), len(xs)

    print("\n2d) CONTACT-COUNT-MATCHED (same interface size in both classes)")
    m, k = nanmean("match_auc_COUNT")
    print(f"     usable complexes: {k}/{n}")
    for t in ("COUNT", "SC", "IFACE", "NEG_IFACE"):
        mm, _ = nanmean(f"match_auc_{t}")
        print(f"     {t:<10}{mm:8.4f}")
    print("     (COUNT ~0.5 confirms the size confound is removed; if IFACE stays")
    print("      near 0 the table's sign is inconsistent with Eq.(14), if it")
    print("      returns to ~0.5 the earlier result was pure interface size)")
    print("\n3) COMBINED SCORE under different weightings")
    print(f"     {'weighting':<28}{'AUC':>8}   {'|a*SC|':>12} {'|IFACE|':>12} {'|b*ELEC|':>12}")
    for name, a, b in weightings:
        print(f"     {name:<28}{mean(f'auc[{name}]'):8.4f}   "
              f"{mean(f'absmean_aSC[{name}]'):12.4g} {mean('absmean_IFACE'):12.4g} "
              f"{mean(f'absmean_bEL[{name}]'):12.4g}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
