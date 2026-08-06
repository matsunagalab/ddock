"""Fit the table inside a subspace and write a deployable checkpoint.

`capacity_series.py` found that twelve free directions beat all 144 on the fixed
validation pool -- sym 50.0% against full 46.1%, with add (23) at 48.0% -- and
that adding parameters beyond 144 makes it worse again (report section 5.14.31).
That was a diagnostic, run inside one script and never deployed: no end-to-end
search has ever been run with a subspace table, so it is not known whether the
ordering survives a real search.

This fits one subspace convexly with the same recipe the deployed table uses
(anchored hinge, lambda by interface-cluster-blocked CV) and writes a checkpoint
in `run_pinder_scaling`'s format, so `eval_search_test.py` can dock with it.

    sym   12 free directions -- a global level plus symmetric zero-sum modes
    add   23 -- the level plus independent row and column effects
    full 144 -- the table as it is trained today

Example
-------
    uv run python scripts/fit_subspace.py --mode sym \
        --pool-glob 'data/scaling/pool_cache/n1000_r0_*of6.pt' \
        --run data/scaling/runs_nfixed/N1000_seed0 \
        --out data/scaling/runs_sym/N1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from capacity_series import table_basis  # noqa: E402
from qp_path import Problem, load  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402


def solve_sub(prob: Problem, B: np.ndarray, e0: np.ndarray, lam: float,
              idx=None, eps: float = 1.0, maxiter: int = 400) -> np.ndarray:
    """Same hinge as `qp_path.solve`, but the table moves only along B.

    The table is e0 + B theta, not B theta: the published values are the origin
    the subspace is anchored at, exactly as the full fit regularises e towards
    e0. Fitting B theta alone would throw the published table away and then the
    saved checkpoint (e0 + B theta) would not be the thing that was fitted.
    """
    d = B.shape[1]

    def fg(th):
        h, g = prob.hinge(e0 + B @ th, idx, eps)
        return h + 0.5 * lam * float(th @ th), B.T @ g + lam * th

    res = minimize(fg, np.zeros(d), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxcor": 30})
    return B @ res.x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("sym", "add", "full"))
    ap.add_argument("--pool-glob", required=True, dest="pool_glob")
    ap.add_argument("--run", required=True, help="run dir with split.json / ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", default="external/pinder/pinder/2024-02/index.parquet")
    ap.add_argument("--lambdas", default="1e-4,1e-3,1e-2,1e-1,1,10")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--thr", type=float, default=0.23)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run)
    ck = torch.load(run / "round0_ckpt.pt", map_location="cpu", weights_only=True)
    alpha = ck["alpha"].double()
    clash = ck["clash_weights"].double()
    beta = torch.tensor(args.beta, dtype=torch.float64)
    e0 = iface_ij(dtype=torch.float64, flat=True).numpy()
    sp = json.load(open(run / "split.json"))

    B = table_basis(args.mode)
    fit_e = load(args.pool_glob, set(sp["fit_ids"]), alpha, clash, beta,
                 args.thr, e0)
    val_e = load(args.pool_glob, set(sp["val_ids"]), alpha, clash, beta,
                 args.thr, e0)
    P = Problem(fit_e, alpha, clash, beta, args.thr, e0)
    V = Problem(val_e, alpha, clash, beta, args.thr, e0)
    print(f"{args.mode}: {B.shape[1]} free directions, "
          f"fit {len(fit_e)} / val {len(val_e)} complexes with a positive")

    import pandas as pd
    ix = pd.read_parquet(args.index, columns=["id", "cluster_id"]).set_index("id")
    cl = [str(ix.cluster_id.get(n, n)) for n, _ in fit_e]
    uniq = sorted(set(cl))
    rng = np.random.default_rng(args.seed)
    fold_of = {c: int(i % args.folds) for i, c in enumerate(rng.permutation(uniq))}
    folds = np.array([fold_of[c] for c in cl])

    print(f"\n{'lambda':>9s} {'CV top-1':>11s} {'CV %':>7s} {'fit %':>7s} "
          f"{'||de||':>7s} {'sec':>6s}")
    best = None
    for lam in [float(x) for x in args.lambdas.split(",")]:
        t0 = time.time()
        ok = tot = 0
        for k in range(args.folds):
            de = solve_sub(P, B, e0, lam, idx=np.flatnonzero(folds != k), eps=args.eps)
            a, b = P.top1(e0 + de, np.flatnonzero(folds == k))
            ok += a
            tot += b
        de_full = solve_sub(P, B, e0, lam, eps=args.eps)
        f_ok, f_tot = P.top1(e0 + de_full)
        cv = 100.0 * ok / tot
        print(f"{lam:9.1e} {f'{ok}/{tot}':>11s} {cv:6.2f}% {100 * f_ok / f_tot:6.2f}% "
              f"{np.linalg.norm(de_full):7.3f} {time.time() - t0:6.1f}")
        if best is None or cv > best[0]:
            best = (cv, lam, de_full)

    cv, lam, de = best
    print(f"\nchosen by blocked CV: lambda = {lam:.1e}, CV top-1 {cv:.2f}%")
    for name, e in (("published e0", np.zeros_like(de)),
                    ("current full-144 table", ck["iface"].double().numpy() - e0),
                    (f"{args.mode} fit", de)):
        a, b = V.top1(e0 + e)
        print(f"  fixed validation  {name:24s} {a}/{b} = {100 * a / b:.2f}%")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blob = dict(ck)
    blob.update({"iface": torch.tensor(e0 + de, dtype=torch.float32),
                 "iface_mode": args.mode, "rowcol": None,
                 "fitter": f"convex-{args.mode}", "lambda": lam, "cv_top1": cv,
                 "config": vars(args)})
    torch.save(blob, out / "round0_ckpt.pt")
    (out / "split.json").write_text(json.dumps(sp))
    print(f"saved -> {out / 'round0_ckpt.pt'}")


if __name__ == "__main__":
    main()
