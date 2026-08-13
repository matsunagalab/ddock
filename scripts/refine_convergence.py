"""Is the refined pose a local optimum of the score it was refined against?

Report section 5.14.42/5.14.44 raised the possibility that High is limited by
the *optimiser* rather than by the energy. Two facts motivate it. Section
5.14.31 measured that the score's local maximum sits a median 1.0 deg and
0.84 A from the native, i.e. inside the ~5 deg that High requires, yet the
refined Oracle@50 High stalls at 80.8%. And `docking_score_elec` defaults to
`scatter_mode="nearest"` (score.py:514) while the clash channel takes only the
three values {0, rho, rho^2} (score.py:355), so the objective is a staircase on
a 1.2 A grid: rigid moves that shift no atom across a voxel boundary change
nothing at all. `refine_poses.refine` halves its step whenever nothing
improves, so a staircase tread collapses the step to the 0.05 deg tolerance and
the search exits.

This script does not assume any of that. It takes poses that have already been
refined and probes around them. If random probes routinely find a better score
than the refiner reached, the refiner stopped early and the fix is the
optimiser (smoothing, multi-start, larger budget). If probes almost never
improve on it, the refiner is converged and the remaining High deficit is the
energy's fault, which is a much more expensive thing to fix.

    uv run python scripts/refine_convergence.py \
        --in-dir data/pinder_eval/trained_SYM_top50_refined \
        --ckpt data/scaling/runs_sym/N1000/round0_ckpt.pt \
        --limit 40 --frame-chunk 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from refine_poses import Scorer, apply, read_ligand  # noqa: E402

from zdock.atomtypes import charge_score as default_charge_score  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.prep_cache import load_prepared  # noqa: E402
from zdock.score import SC_REFERENCE_SPACING, SC_RHO  # noqa: E402

#: (rotation deg, translation A) probe radii. The smallest is well under one
#: voxel, so it also reports how much of the landscape the grid flattens.
RADII = ((0.25, 0.05), (0.5, 0.1), (1.0, 0.25), (2.0, 0.5), (4.0, 1.0))


def probes(rng: np.random.Generator, n: int, deg: float, shift: float):
    """`n` random 6-vectors with |rotvec| = deg and |translation| = shift."""
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    tvec = rng.normal(size=(n, 3))
    tvec /= np.linalg.norm(tvec, axis=1, keepdims=True)
    return np.concatenate([axis * deg, tvec * shift], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True, dest="in_dir")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--prep-cache", default="data/scaling/prep_cache_test",
                    dest="prep_cache")
    ap.add_argument("--monomer", default="holo")
    ap.add_argument("--model", type=int, default=1,
                    help="which submitted rank to probe around")
    ap.add_argument("--n-probe", type=int, default=64, dest="n_probe",
                    help="random probes per radius")
    ap.add_argument("--spacing", type=float, default=SC_REFERENCE_SPACING)
    ap.add_argument("--frame-chunk", type=int, default=4, dest="frame_chunk")
    ap.add_argument("--rho0", type=float, default=SC_RHO)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device, dtype = torch.device(args.device), torch.float32
    beta = torch.tensor(3.0, device=device, dtype=dtype)
    charge = default_charge_score(device=device, dtype=dtype)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        alpha = ck["alpha"].to(device=device, dtype=dtype)
        iface = ck["iface"].to(device=device, dtype=dtype)
        clash = ck["clash_weights"].to(device=device, dtype=dtype)
    else:
        alpha = torch.tensor(1.0, device=device, dtype=dtype)
        iface = iface_ij(device=device, dtype=dtype, flat=True)
        rho = torch.tensor(args.rho0, device=device, dtype=dtype)
        clash = alpha * rho.pow(torch.tensor([2.0, 3.0, 4.0], device=device,
                                             dtype=dtype))

    root = Path(args.in_dir)
    ids = sorted(p.name for p in root.iterdir() if p.is_dir())
    si, sn = (int(x) for x in args.shard.split("/"))
    ids = ids[si::sn]
    if args.limit:
        ids = ids[: args.limit]

    rng = np.random.default_rng(args.seed)
    rows, t0 = [], time.time()
    for pid in ids:
        prot = load_prepared(args.prep_cache, pid)
        if prot is None:
            continue
        prot = prot.to(device, dtype=dtype)
        scorer = Scorer(prot, alpha, iface, beta, charge, clash, args.spacing,
                        args.frame_chunk)
        fp = root / pid / args.monomer / "models" / f"model_{args.model}.pdb"
        if not fp.exists():
            continue
        base = read_ligand(fp)
        if base.shape[0] != prot.n_lig:
            continue

        s0 = float(scorer(base[None])[0])
        row = {"id": pid, "score": s0}
        for deg, shift in RADII:
            x = probes(rng, args.n_probe, deg, shift)
            poses = np.stack([apply(base, xi) for xi in x])
            s = scorer(poses)
            row[f"best_{deg}"] = float(s.max() - s0)
            row[f"frac_{deg}"] = float((s > s0).mean())
            # how flat is the objective at this radius?
            row[f"same_{deg}"] = float(np.isclose(s, s0, rtol=0, atol=1e-6).mean())
        rows.append(row)
        print(f"  {pid[:34]:36s} s0={s0:12.1f} "
              + " ".join(f"{d}deg:+{row[f'best_{d}']:8.1f}({row[f'frac_{d}']:.2f})"
                         for d, _ in RADII), flush=True)

    if not rows:
        raise SystemExit("nothing probed")
    print(f"\n{len(rows)} systems, {time.time() - t0:.0f} s\n")
    print(f"{'radius':>14} | {'improved':>9} | {'median gain':>12} | "
          f"{'p90 gain':>10} | {'score unchanged':>16}")
    for deg, shift in RADII:
        b = np.array([r[f"best_{deg}"] for r in rows])
        f = np.array([r[f"frac_{deg}"] for r in rows])
        u = np.array([r[f"same_{deg}"] for r in rows])
        print(f"{deg:>6}deg/{shift:<5}A | {np.mean(b > 0) * 100:8.1f}% | "
              f"{np.median(b):12.1f} | {np.percentile(b, 90):10.1f} | "
              f"{np.mean(u) * 100:15.1f}%")
    print("\n'improved' = systems where at least one random probe beat the "
          "refined pose.\n'score unchanged' = probes whose score was bit-identical "
          "to the refined pose\n(the fraction of the neighbourhood the grid "
          "flattens).")

    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
