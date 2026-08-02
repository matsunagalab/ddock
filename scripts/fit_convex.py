"""Fit the IFACE table by convex optimisation instead of SGD.

The score is affine in the 144 numbers (report section 5.14.29), so the training
problem is convex once each complex is assigned a target positive, and can be
solved directly. That removes the learning rate, the step budget, the early
stopping rule and the checkpoint rollback -- two of which were measurably in the
way: the prior kept the table within ||e - e0|| = 1.37 of the published one when
the good region is at 6.6, and `_val_loss` included that prior, so selection
preferred the near table (section 5.14.30).

    min_e  (1/|F|) sum_c max(0, max_n [eps + s_n(e) - s_{p(c)}(e)])
           + (lambda/2) ||e - e0||^2

The anchor p(c) is a latent variable: the deployed metric only asks that SOME
positive rank first, which is a disjunction and not convex. Fixing p(c) convexifies
it, and re-picking each anchor under the current table and re-solving is the usual
CCCP/latent-SVM alternation. Each solve is global; the alternation is monotone in
the surrogate but can settle in different places from different starts, so the
number of rounds is reported rather than assumed.

lambda is chosen by interface-cluster-blocked cross-validation inside the fit
set. The fixed validation set is used once, after everything is chosen, and TEST
never.

Writes a checkpoint in the same format `run_pinder_scaling.py` produces, so
`eval_search_test.py` and everything downstream take it unchanged.

Example
-------
    uv run python scripts/fit_convex.py \
        --pool-glob 'data/scaling/pool_cache/n1000_r0_*of6.pt' \
        --split data/scaling/runs_nfixed/N1000_seed0/split.json \
        --out data/scaling/runs_convex/N1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qp_path import Problem, load, solve  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.score import SC_RHO  # noqa: E402


def reanchor(prob: Problem, e: np.ndarray, thr: float) -> int:
    """Point each complex at whichever positive now scores highest.

    Returns how many anchors moved, so a caller can stop when the assignment is
    stable rather than after a fixed number of rounds.
    """
    ef = e.astype(np.float32)
    moved = 0
    for ci, (name, en) in enumerate(prob.entries):
        s = prob.all_c[ci] + prob.all_A[ci] @ ef
        p_idx = np.flatnonzero(prob.all_pos[ci])
        p = int(p_idx[np.argmax(s[p_idx])])
        if not np.array_equal(prob.Ap[ci], prob.all_A[ci][p]):
            moved += 1
        prob.Ap[ci] = prob.all_A[ci][p]
        prob.cp[ci] = prob.all_c[ci][p]
    return moved


def fit(prob: Problem, e0: np.ndarray, lam: float, eps: float, thr: float,
        rounds: int, idx=None) -> tuple[np.ndarray, int]:
    """Solve, re-anchor, repeat until the anchors stop moving."""
    saved = (prob.Ap.copy(), prob.cp.copy())
    e = solve(prob, e0, lam, idx=idx, eps=eps)
    used = 1
    for _ in range(rounds - 1):
        if reanchor(prob, e, thr) == 0:
            break
        e = solve(prob, e0, lam, idx=idx, eps=eps)
        used += 1
    prob.Ap, prob.cp = saved            # leave the anchors as the caller had them
    return e, used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-glob", required=True, dest="pool_glob")
    ap.add_argument("--split", required=True,
                    help="split.json with fit_ids / val_ids")
    ap.add_argument("--out", required=True, help="directory for the checkpoint")
    ap.add_argument("--ckpt-template", default="", dest="ckpt_template",
                    help="checkpoint to copy alpha/rho/clash from "
                         "(default: the published values)")
    ap.add_argument("--index", default="external/pinder/pinder/2024-02/index.parquet")
    ap.add_argument("--lambdas", default="1e-4,1e-3,1e-2,1e-1,1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=5,
                    help="maximum solve/re-anchor alternations")
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--thr", type=float, default=0.23)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--rho0", type=float, default=SC_RHO)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.ckpt_template:
        ck = torch.load(args.ckpt_template, map_location="cpu", weights_only=True)
        alpha = ck["alpha"].double()
        rho = ck.get("rho", torch.tensor(args.rho0)).double()
        clash = ck["clash_weights"].double() if "clash_weights" in ck else \
            alpha * rho.pow(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
    else:
        ck = None
        alpha = torch.tensor(args.alpha0, dtype=torch.float64)
        rho = torch.tensor(args.rho0, dtype=torch.float64)
        clash = alpha * rho.pow(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
    beta = torch.tensor(args.beta, dtype=torch.float64)
    e0 = iface_ij(dtype=torch.float64, flat=True).numpy()

    sp = json.load(open(args.split))
    fit_e = load(args.pool_glob, set(sp["fit_ids"]), alpha, clash, beta,
                 args.thr, e0)
    P = Problem(fit_e, alpha, clash, beta, args.thr, e0)
    print(f"fit complexes with a positive: {len(fit_e)}")

    import pandas as pd
    ix = pd.read_parquet(args.index, columns=["id", "cluster_id"]).set_index("id")
    cl = [str(ix.cluster_id.get(n, n)) for n, _ in fit_e]
    uniq = sorted(set(cl))
    rng = np.random.default_rng(args.seed)
    fold_of = {c: int(i % args.folds) for i, c in enumerate(rng.permutation(uniq))}
    folds = np.array([fold_of[c] for c in cl])
    print(f"{len(uniq)} interface clusters over {args.folds} blocked folds\n")

    print(f"{'lambda':>9s} {'rounds':>7s} {'CV top-1':>11s} {'CV %':>7s} "
          f"{'fit %':>7s} {'||de||':>7s} {'sec':>6s}")
    best = None
    for lam in [float(x) for x in args.lambdas.split(",")]:
        t0 = time.time()
        ok = tot = 0
        for k in range(args.folds):
            tr = np.flatnonzero(folds != k)
            te = np.flatnonzero(folds == k)
            e, _ = fit(P, e0, lam, args.eps, args.thr, args.rounds, idx=tr)
            a, b = P.top1(e, te)
            ok += a
            tot += b
        e_full, used = fit(P, e0, lam, args.eps, args.thr, args.rounds)
        f_ok, f_tot = P.top1(e_full)
        cv = 100.0 * ok / tot
        print(f"{lam:9.1e} {used:7d} {f'{ok}/{tot}':>11s} {cv:6.2f}% "
              f"{100 * f_ok / f_tot:6.2f}% {np.linalg.norm(e_full - e0):7.3f} "
              f"{time.time() - t0:6.1f}")
        if best is None or cv > best[0]:
            best = (cv, lam, e_full, used)

    cv, lam, e_star, used = best
    print(f"\nchosen by blocked CV: lambda = {lam:.1e} ({used} anchor rounds), "
          f"CV top-1 {cv:.2f}%")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blob = dict(ck) if ck is not None else {}
    blob.update({"alpha": alpha.float(), "rho": rho.float(),
                 "clash_weights": clash.float(),
                 "iface": torch.tensor(e_star, dtype=torch.float32),
                 "iface_mode": "full", "psc_mode": "rho", "round": 0,
                 "rowcol": None, "n_fit": len(sp["fit_ids"]), "seed": args.seed,
                 "config": vars(args), "fitter": "convex",
                 "cv_top1": cv, "lambda": lam, "anchor_rounds": used})
    torch.save(blob, out / "round0_ckpt.pt")
    (out / "split.json").write_text(json.dumps(sp))
    print(f"saved -> {out / 'round0_ckpt.pt'}")

    # one look at the fixed validation set, after everything is chosen
    val_e = load(args.pool_glob, set(sp["val_ids"]), alpha, clash, beta,
                 args.thr, e0)
    V = Problem(val_e, alpha, clash, beta, args.thr, e0)
    print(f"\nfixed validation, {len(val_e)} complexes with a positive")
    for name, e in (("published e0", e0), ("convex fit", e_star)):
        a, b = V.top1(e)
        print(f"  {name:16s} {a}/{b} = {100 * a / b:.2f}%")


if __name__ == "__main__":
    main()
