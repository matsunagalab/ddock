"""Split the 6 A contact count into distance shells, on the cached pools.

The IFACE feature is `T_ij = sum_v L_i(v) H_j(v)`, where `H_j` counts type-j
receptor atoms within `rcut_iface = 6 A` of a cell. Nothing about that is
distance resolved: an atom pair at 1 A and one at 5.9 A contribute equally.
Every method ahead of this one has a distance-dependent pair term, and section
5.14.51 found that recombining the existing 144 numbers non-linearly adds
nothing -- so the next thing to try is a feature that carries information the
present one does not.

Shells are built by differencing cumulative counts:

    T^1 = T(4 A),  T^2 = T(5 A) - T(4 A),  T^3 = T(6 A) - T(5 A)

so `sum_b T^b = T` exactly and a table with equal shell values reproduces the
current model. Only the receptor side changes, which is why the FFT search
would not need extra channels for this (report section 5.14.42 (3)); here we
only need the features, so the point is moot and the cached T is reused as the
6 A cumulant.

Poses are reconstructed from `pose_key = (rotation index, cx, cy, cz)` rather
than re-searched -- `dataset.build_reachable_pool` builds them as
`rotate(lig_ref, quats[qi]) + cell * spacing`. Recomputing T at 6 A and
comparing it against the cached T is therefore a check on the whole
reconstruction, and it runs by default.

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/build_shell_pool.py \
        --pool 'data/scaling/pool_cache/n1000_r0_*of6.pt' \
        --run data/scaling/runs_nfixed/N1000_seed0 \
        --out data/scaling/shell_pool --shard 0/3
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zdock.atomtypes import charge_score as default_charge_score  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.dataset import has_pose_identity  # noqa: E402
from zdock.prep_cache import load_prepared  # noqa: E402
from zdock.rotation_grid import hopf_quaternions  # noqa: E402
from zdock.score import SC_REFERENCE_SPACING, docking_score_elec  # noqa: E402
from zdock.search import _rotate_batch  # noqa: E402


#: vdW radii are assigned by the atom name's first letter (atomtypes._VDW_RADIUS)
#: and every element has a distinct value, so the element is recoverable from
#: `rec_radius` exactly -- no extra field has to be carried through the cache.
ELEMENT_RADIUS = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20}


@torch.no_grad()
def contact_counts(prot, poses, rcut, frame_chunk, spacing, beta, charge,
                   rec_weight=None):
    """`T` at one receptor cutoff, in the same call the deployed score uses."""
    out = []
    for s in range(0, poses.shape[0], frame_chunk):
        _, T, _ = docking_score_elec(
            prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
            prot.rec_atomtype_id, prot.rec_charge_id,
            poses[s:s + frame_chunk], prot.lig_radius, prot.lig_sasa,
            prot.lig_atomtype_id, prot.lig_charge_id,
            torch.zeros((), device=poses.device, dtype=poses.dtype),
            iface_ij(device=poses.device, dtype=poses.dtype, flat=True),
            beta, charge, lig_xyz_for_grid=prot.lig_ref, spacing=spacing,
            frame_chunk_size=frame_chunk, rcut_iface=rcut,
            return_components=True, psc_decompose=True,
            rec_weight=rec_weight)
        out.append(T.cpu())
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache",
                    dest="prep_cache")
    ap.add_argument("--edges", default="4.0,5.0",
                    help="inner shell edges; the outer one is rcut_iface")
    ap.add_argument("--rcut", type=float, default=6.0)
    ap.add_argument("--outer", default="",
                    help="comma-separated cutoffs BEYOND the cached 6 A window, "
                         "e.g. '9.0,12.0'. In this mode the cached T is the "
                         "innermost band and the emitted shells are the outer "
                         "bands only, so a table with zero on them reproduces "
                         "the deployed model exactly. The 6 A recomputation is "
                         "kept as the correctness check on a subset.")
    ap.add_argument("--check-every", type=int, default=20, dest="check_every",
                    help="outer/element mode: recompute T on every Nth complex")
    ap.add_argument("--elements", default="",
                    help="comma-separated elements, e.g. 'C,N,O,S'. Splits the "
                         "receptor side of T by element while keeping the "
                         "12-type alphabet, so the bands partition T exactly "
                         "and an equal table reproduces the deployed model. "
                         "This is the axis the 12-type alphabet throws away: "
                         "'Polar' holds ASN C, CA, CB, CG, N, ND2, O and OD1 "
                         "together, so carbon, nitrogen and oxygen contacts "
                         "are currently indistinguishable.")
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--hopf-nside", type=int, default=3, dest="hopf_nside")
    ap.add_argument("--frame-chunk", type=int, default=16, dest="frame_chunk")
    ap.add_argument("--chunk-elems", type=int, default=800_000_000,
                    dest="chunk_elems")
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--thr", type=float, default=0.23)
    ap.add_argument("--check-tol", type=float, default=1e-3, dest="check_tol")
    ap.add_argument("--no-check", action="store_true", dest="no_check")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--oom-retries", type=int, default=6, dest="oom_retries")
    ap.add_argument("--ids-file", default="", dest="ids_file",
                    help="restrict to these ids -- used to rescue the "
                         "complexes an earlier pass dropped")
    args = ap.parse_args()

    dev, dt = torch.device(args.device), torch.float32
    beta = torch.tensor(args.beta, device=dev, dtype=dt)
    charge = default_charge_score(device=dev, dtype=dt)
    edges = [float(x) for x in args.edges.split(",")] + [args.rcut]
    outer = [float(x) for x in args.outer.split(",")] if args.outer else []
    elements = args.elements.split(",") if args.elements else []

    sp = json.load(open(Path(args.run) / "split.json"))
    want = set(sp["fit_ids"]) | set(sp["val_ids"])

    quats = hopf_quaternions(args.hopf_nside, device=dev, dtype=dt)
    quats = quats / quats.norm(dim=-1, keepdim=True)

    files = sorted(glob.glob(args.pool))
    si, sn = (int(x) for x in args.shard.split("/"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rescue = (set(Path(args.ids_file).read_text().split())
              if args.ids_file else None)
    seen, todo = set(), []
    for f in files:
        for d in torch.load(f, map_location="cpu", weights_only=True)["pools"]:
            if d["name"] in seen or d["name"] not in want:
                continue
            seen.add(d["name"])
            # A complex with no positive, or nothing but positives, cannot
            # enter the hinge fit. Skipping them here is about a third of the
            # work and changes nothing downstream.
            if rescue and d["name"] not in rescue:
                continue
            pk = d["dockq"][d["prov"] == 0] >= args.thr
            if bool(pk.any()) and not bool(pk.all()):
                todo.append(d)
    todo = todo[si::sn]
    if args.limit:
        todo = todo[: args.limit]
    what = ("element bands " + str(elements) if elements else
            "outer bands 6->" + str(outer) if outer else "shells " + str(edges))
    print(f"{len(todo)} complexes on this shard, {what}", flush=True)

    out, t0, checks, dropped = [], time.time(), [], []
    for i, d in enumerate(todo):
        if not has_pose_identity(d["pose_key"]):
            print(f"  {d['name']}: no pose identity, skipped", flush=True)
            continue
        prot = load_prepared(args.prep_cache, d["name"])
        if prot is None:
            print(f"  {d['name']}: not in prep cache, skipped", flush=True)
            continue
        prot = prot.to(dev, dtype=dt)

        m = d["prov"] == 0                      # search-derived poses only
        key = d["pose_key"][m].to(dev)
        poses = (_rotate_batch(prot.lig_ref, quats[key[:, 0]])
                 + (key[:, 1:].to(dt) * args.spacing).unsqueeze(1))

        from zdock.geom import grid_shape
        nx, ny, nz = grid_shape(prot.rec_xyz, prot.lig_ref, spacing=args.spacing)
        fc = int(max(1, min(args.frame_chunk,
                            args.chunk_elems // max(1, nx * ny * nz * 12))))

        T_out = d["T"][m].cpu()                 # the cached 6 A cumulant
        shells = None
        for attempt in range(args.oom_retries + 1):
            try:
                if elements:
                    # Compute the listed elements and take everything else as the
                    # residual band, so B bands cost B-1 passes. The residual is
                    # what makes the set a partition of T by construction; its
                    # non-negativity is then a real check on that construction.
                    shells = []
                    for el in elements:
                        w = (prot.rec_radius - ELEMENT_RADIUS[el]).abs() < 1e-3
                        shells.append(contact_counts(
                            prot, poses, args.rcut, fc, args.spacing, beta, charge,
                            rec_weight=w.to(prot.rec_xyz.dtype)))
                    rest = T_out - sum(shells)
                    if float(rest.min()) < -1e-3:
                        print(f"  {d['name']}: residual element band is negative "
                              f"({float(rest.min()):.3g}) -- the element split does "
                              f"not partition the receptor, stopping")
                        raise SystemExit(1)
                    shells.append(rest)
                    if not (i % args.check_every):
                        chk = contact_counts(prot, poses, args.rcut, fc,
                                             args.spacing, beta, charge)
                        checks.append(float((chk - T_out).abs().max()
                                            / max(1e-9, float(T_out.abs().max()))))
                elif outer:
                    cum = {r: contact_counts(prot, poses, r, fc, args.spacing,
                                             beta, charge) for r in outer}
                    shells, prev = [], T_out
                    for r in outer:
                        shells.append(cum[r] - prev)
                        prev = cum[r]
                    if not (i % args.check_every):
                        chk = contact_counts(prot, poses, args.rcut, fc,
                                             args.spacing, beta, charge)
                        checks.append(float((chk - T_out).abs().max()
                                            / max(1e-9, float(T_out.abs().max()))))
                    for k, sh in enumerate(shells):
                        if float(sh.min()) < -1e-3:
                            print(f"  {d['name']}: outer band {k} has negative "
                                  f"counts ({float(sh.min()):.3g}) -- cumulants are "
                                  f"not monotone, stopping")
                            raise SystemExit(1)
                else:
                    cum = {r: contact_counts(prot, poses, r, fc, args.spacing,
                                             beta, charge) for r in edges[:-1]}
                    if not args.no_check:
                        chk = contact_counts(prot, poses, edges[-1], fc,
                                             args.spacing, beta, charge)
                        checks.append(float((chk - T_out).abs().max()
                                            / max(1e-9, float(T_out.abs().max()))))
                    shells, prev = [], None
                    for r in edges[:-1]:
                        shells.append(cum[r] if prev is None else cum[r] - cum[prev])
                        prev = r
                    shells.append(T_out - cum[edges[-2]])
                if checks and checks[-1] > args.check_tol:
                    print(f"  {d['name']}: recomputed T differs from cache by "
                          f"{checks[-1]:.3g} -- reconstruction is wrong, stopping")
                    raise SystemExit(1)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # Dropping on the first OOM biases the sample towards small
                # complexes, and the missingness is then correlated with the
                # very property under test. Section 5.10.5 lost 33% that way
                # and the excluded group was 2.4x larger. Halve and retry.
                shells = None
                if fc > 1 and attempt < args.oom_retries:
                    fc = max(1, fc // 2)
                    print(f"  {d['name']}: OOM, retrying at frame_chunk {fc}",
                          flush=True)
                else:
                    print(f"  {d['name']}: OOM at frame_chunk {fc}, DROPPED",
                          flush=True)
                    dropped.append(d["name"])
        if shells is None:
            continue
        out.append({"name": d["name"], "sc": d["sc"][m], "elec": d["elec"][m],
                    "dockq": d["dockq"][m], "T": T_out,
                    "Tb": torch.stack(shells, dim=1)})   # (F, B, 12, 12)
        del prot, poses
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(todo)}  {el / (i + 1):.1f}s each, "
                  f"{el / 60:.0f} min so far", flush=True)

    path = out_dir / f"shell_{si}of{sn}.pt"
    torch.save({"edges": elements or (outer if outer else edges),
                "mode": "partition" if (elements or not outer) else "outer",
                "pools": out}, path)
    print(f"\nwrote {path}  ({len(out)} complexes, "
          f"{(time.time() - t0) / 60:.1f} min)")
    if dropped:
        print(f"DROPPED {len(dropped)} complexes to OOM even after "
              f"{args.oom_retries} halvings:")
        for n in dropped:
            print(f"  {n}")
    if checks:
        c = np.array(checks)
        print(f"cache agreement at {edges[-1]} A: max relative difference "
              f"{c.max():.3g} over {len(c)} complexes "
              f"(tolerance {args.check_tol})")


if __name__ == "__main__":
    main()
