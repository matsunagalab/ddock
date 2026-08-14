"""Solve the ranking hinge to a certified optimum, not to wherever L-BFGS stopped.

Every capacity number in sections 5.14.51-54 came from `scipy.optimize.minimize`
on a max-over-negatives hinge. That objective is non-smooth, L-BFGS-B does not
report success on the larger arms, and the reported fit Top-1 is therefore a
lower bound on what the family attains rather than the family's capacity. The
conclusion drawn from those numbers -- that expressiveness buys fit but not
transfer -- rests on comparing three such lower bounds, which is only sound if
the slack is small and comparable across arms. This measures the slack.

The problem is a QP once written in epigraph form:

    min_{theta, xi >= 0}  (1/C) sum_c xi_c + (lambda/2) |theta|^2
    s.t.                  xi_c >= b_cn + <a_cn, theta>   for every negative n

with b_cn = eps + c_n - c_p(c) and a_cn = A_n - A_p(c), i.e. exactly
`fit_shell.ShellDesign.hinge` with the anchor positive fixed as it already is.

Certification needs both sides, and the trap is that a solver's own duality gap
refers to the *restricted* problem it was handed. Here:

  upper  U = (1/C) sum_c max(0, max_n [b_cn + <a_cn, theta>]) + (lambda/2)|theta|^2
            evaluated over ALL constraints, so it is primal-feasible by
            construction
  lower  D(alpha) = sum_cn alpha_cn b_cn - (1/(2 lambda)) |sum_cn alpha_cn a_cn|^2
            valid for ANY alpha >= 0 with sum_n alpha_cn <= 1/C, which is what
            the KKT conditions of the epigraph form give:
                d/dxi_c :  sum_n alpha_cn + mu_c = 1/C,  mu_c >= 0
                d/dtheta:  theta = -(1/lambda) sum_cn alpha_cn a_cn

The restricted solver's duals, padded with zeros outside the working set and
projected back onto that feasible set, are therefore a valid dual point for the
full problem. The gap U - D is a real certificate; OSQP's internal one is not.

    uv run python scripts/certified_solve.py \
        --pool 'data/scaling/elem_pool*/shell_*of3.pt' \
        --run data/scaling/runs_nfixed/N1000_seed0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fit_shell import ShellDesign, load  # noqa: E402

from zdock.atomtypes import iface_ij  # noqa: E402


class Certified:
    """Epigraph QP over a growing working set, with an honest gap."""

    def __init__(self, design: ShellDesign, lam: float, eps: float = 1.0):
        self.A, self.cn = design.A, design.cn          # negatives: (N,d), (N,)
        self.Ap, self.cp = design.Ap, design.cp        # anchors:   (C,d), (C,)
        self.spans = design.spans
        self.d, self.C = design.d, len(design.spans)
        self.lam, self.eps = lam, eps
        self.owner = np.empty(self.A.shape[0], dtype=np.int64)
        for ci, (lo, hi) in enumerate(self.spans):
            self.owner[lo:hi] = ci

    def violations(self, th):
        """b_cn + <a_cn, theta> for every negative, in one pass."""
        s_neg = self.cn + self.A @ th
        s_pos = self.cp + self.Ap @ th
        return self.eps + s_neg - s_pos[self.owner]

    def upper(self, th, v=None):
        """Primal objective over ALL constraints -- feasible by construction."""
        if v is None:
            v = self.violations(th)
        tot = 0.0
        for ci, (lo, hi) in enumerate(self.spans):
            m = v[lo:hi].max() if hi > lo else -np.inf
            tot += max(0.0, m)
        return tot / self.C + 0.5 * self.lam * float(th @ th)

    def lower(self, idx, alpha):
        """Dual value at a feasible alpha supported on `idx` (zero elsewhere)."""
        if idx.size == 0:
            return 0.0
        b = self.eps + self.cn[idx] - self.cp[self.owner[idx]]
        # sum_cn alpha_cn a_cn  with a_cn = A_n - Ap_{owner(n)}
        g = alpha @ self.A[idx]
        for ci in np.unique(self.owner[idx]):
            g -= alpha[self.owner[idx] == ci].sum() * self.Ap[ci]
        return float(alpha @ b) - float(g @ g) / (2.0 * self.lam)

    def project_dual(self, idx, alpha):
        """Nearest point of {alpha >= 0, sum_n alpha_cn <= 1/C} by clipping and
        per-complex rescaling. Any feasible alpha certifies, so a crude but
        strictly feasible projection is safer than a tight but slightly
        infeasible one."""
        a = np.maximum(alpha, 0.0)
        own = self.owner[idx]
        for ci in np.unique(own):
            m = own == ci
            s = a[m].sum()
            if s > 1.0 / self.C:
                a[m] *= (1.0 / self.C) / s
        return a

    def solve_restricted(self, idx, warm=None, tol=1e-9):
        """QP on the working set. Returns (theta, xi, duals_on_idx)."""
        import osqp
        n_w = idx.size
        rows = self.A[idx] - self.Ap[self.owner[idx]]        # (n_w, d)
        b = self.eps + self.cn[idx] - self.cp[self.owner[idx]]
        own = self.owner[idx]

        # x = [theta (d); xi (C)]
        P = sp.diags(np.concatenate([np.full(self.d, self.lam),
                                     np.zeros(self.C)])).tocsc()
        q = np.concatenate([np.zeros(self.d), np.full(self.C, 1.0 / self.C)])
        # xi_c - <a_cn, theta> >= b_cn
        Acon = sp.hstack([sp.csc_matrix(-rows),
                          sp.csc_matrix((np.ones(n_w), (np.arange(n_w), own)),
                                        shape=(n_w, self.C))])
        # xi_c >= 0
        Acon = sp.vstack([Acon, sp.hstack([sp.csc_matrix((self.C, self.d)),
                                           sp.identity(self.C)])]).tocsc()
        l = np.concatenate([b, np.zeros(self.C)])
        u = np.full(n_w + self.C, np.inf)

        m = osqp.OSQP()
        m.setup(P=P, q=q, A=Acon, l=l, u=u, verbose=False, eps_abs=tol,
                eps_rel=tol, max_iter=200000, polishing=True)
        if warm is not None:
            try:
                m.warm_start(x=warm)
            except Exception:
                pass
        r = m.solve()
        x = np.asarray(r.x, dtype=np.float64)
        y = np.asarray(r.y, dtype=np.float64)[:n_w]
        return x[: self.d], x[self.d:], y, str(r.info.status)


def certify(design, lam, eps=1.0, tol_gap=1e-4, tol_viol=1e-6, max_rounds=60,
            per_round=1, verbose=True):
    P = Certified(design, lam, eps)
    # start from the worst negative of each complex at theta = 0
    v0 = P.violations(np.zeros(P.d))
    idx = np.array([lo + int(np.argmax(v0[lo:hi])) for lo, hi in P.spans
                    if hi > lo], dtype=np.int64)

    th, warm, hist = np.zeros(P.d), None, []
    for rnd in range(max_rounds):
        th, xi, y, status = P.solve_restricted(idx, warm)
        warm = np.concatenate([th, xi])

        # OSQP's sign convention for a one-sided lower bound is settled by
        # checking the stationarity condition rather than assumed.
        best = None
        for s in (1.0, -1.0):
            a = P.project_dual(idx, s * y)
            g = a @ P.A[idx]
            for ci in np.unique(P.owner[idx]):
                g -= a[P.owner[idx] == ci].sum() * P.Ap[ci]
            err = np.linalg.norm(-g / lam - th) / max(1.0, np.linalg.norm(th))
            if best is None or err < best[0]:
                best = (err, a)
        stat_err, alpha = best

        v = P.violations(th)
        U = P.upper(th, v)
        D = P.lower(idx, alpha)
        gap = (U - D) / max(1.0, abs(U))

        # A constraint is violated when it exceeds this complex's slack, not
        # when it is merely positive: xi_c already absorbs every negative that
        # scores below the worst one in the working set. Testing against zero
        # instead re-adds satisfied constraints forever, which is what the
        # first version of this loop did.
        mask = np.ones(v.size, dtype=bool)
        mask[idx] = False
        add, worst = [], 0.0
        for ci, (lo, hi) in enumerate(P.spans):
            sl = np.arange(lo, hi)[mask[lo:hi]]
            if sl.size == 0:
                continue
            k = sl[int(np.argmax(v[sl]))]
            excess = max(0.0, float(v[k]) - max(0.0, float(xi[ci])))
            worst = max(worst, excess)
            if excess > tol_viol:
                add.append(k)
        hist.append({"round": rnd + 1, "n_con": int(idx.size), "U": U, "D": D,
                     "gap": gap, "max_omitted": worst, "stat_err": stat_err,
                     "status": status})
        if verbose:
            print(f"    r{rnd + 1:<3d} |W|={idx.size:6d} U={U:12.6f} D={D:12.6f} "
                  f"gap={gap:.2e} omitted={worst:.3e} stat={stat_err:.1e} {status}",
                  flush=True)
        # gap <= tol certifies optimality for the FULL problem on its own,
        # because U is already evaluated over every constraint.
        if gap <= tol_gap and worst <= tol_viol:
            break
        if not add:
            break
        idx = np.concatenate([idx, np.array(add, dtype=np.int64)])
        warm = np.concatenate([th, xi])
    return th, hist


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--lam", type=float, default=0.001)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--thr", type=float, default=0.23)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--tol-gap", type=float, default=1e-4, dest="tol_gap")
    ap.add_argument("--tol-viol", type=float, default=1e-6, dest="tol_viol")
    ap.add_argument("--max-rounds", type=int, default=60, dest="max_rounds")
    ap.add_argument("--arms", default="sym,full,full+bands")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run = Path(args.run)
    ck = torch.load(run / "round0_ckpt.pt", map_location="cpu", weights_only=True)
    kw = dict(alpha=ck["alpha"].double(), clash=ck["clash_weights"].double(),
              beta=torch.tensor(args.beta, dtype=torch.float64), thr=args.thr,
              e0=iface_ij(dtype=torch.float64, flat=True).numpy())
    sp_ = json.load(open(run / "split.json"))
    t0 = time.time()
    fit_e = load(args.pool, set(sp_["fit_ids"]), args.thr)
    val_e = load(args.pool, set(sp_["val_ids"]), args.thr)
    print(f"loaded fit {len(fit_e)} val {len(val_e)} ({time.time() - t0:.0f}s)\n")

    out = {}
    for arm in args.arms.split(","):
        basis = "full" if arm.startswith("full") else "sym"
        shells = arm.endswith("bands")
        df = ShellDesign(fit_e, use_shells=shells, mode="partition",
                         basis_mode=basis, **kw)
        dv = ShellDesign(val_e, use_shells=shells, mode="partition",
                         basis_mode=basis, col_scale=df.col_scale, **kw)
        print(f"  {arm}  d={df.d}  lambda={args.lam}")
        th, hist = certify(df, args.lam, args.eps, args.tol_gap, args.tol_viol,
                           args.max_rounds)
        fo, fn, _ = df.top1(th)
        vo, vn, _ = dv.top1(th)
        h = hist[-1]
        print(f"    -> fit {fo}/{fn} = {fo / fn * 100:.2f}%   "
              f"val {vo}/{vn} = {vo / vn * 100:.2f}%   "
              f"certified gap {h['gap']:.2e}, max omitted violation "
              f"{h['max_omitted']:.2e}\n", flush=True)
        out[arm] = {"d": df.d, "fit": fo / fn, "val": vo / vn, "hist": hist,
                    "fit_n": [fo, fn], "val_n": [vo, vn]}

    print(f"{'arm':<14} {'d':>4} {'fit':>16} {'val':>16} {'gap':>10} {'omitted':>10}")
    for arm, r in out.items():
        h = r["hist"][-1]
        print(f"{arm:<14} {r['d']:>4} {r['fit_n'][0]:5d}/{r['fit_n'][1]} "
              f"{r['fit'] * 100:6.2f}% {r['val_n'][0]:5d}/{r['val_n'][1]} "
              f"{r['val'] * 100:6.2f}% {h['gap']:10.2e} {h['max_omitted']:10.2e}")
    print("\nU and D bracket the true optimum of the convex surrogate. Top-1 is "
          "read off the certified\nsolution; it is not itself optimised and is "
          "not the family's Top-1 capacity.")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
