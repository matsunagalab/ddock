"""What is the best top-1 rate ANY 144-number IFACE table could reach?

Everything the training loop can change is the 144-entry IFACE vector: the
recipe freezes alpha and rho (`--freeze-psc`; measured alpha = 1.0, rho = 3.5 in
every shipped run). And `iface_score_matrix` is affine in that vector --
`IFACE_SIGN * e.view(12, 12).T` with a zero offset -- so a pose's score is

    s_p(e) = c_p + <A_p, e>,
    c_p    = alpha * sc_0 - w . sc_{1:4} + beta * elec,
    A_p    = IFACE_SIGN * vec(T_p^T)

Then "positive p outranks every negative of its complex" is a system of LINEAR
inequalities in e, and asking whether such an e exists is a linear program. No
training, no GPU, and the answer bounds what any loss function could ever
achieve on this pool.

What this does and does not prove
---------------------------------
For each complex it solves, per positive pose, the LP

    max t   s.t.   <A_p - A_n, e> - (c_n - c_p) >= t  for every negative n
                   |e_i - e0_i| <= r                  for every entry

and calls the complex winnable at radius r if some positive reaches t >= eps.
The complex is counted with ALL its positives, not a top-k sample: a cap would
break the bound, because the excluded positive might have been the separable
one.

**A low number is decisive; a high number is not.** Every complex is solved with
its OWN e here, and the deployed table is shared: complex A can demand an entry
raised while complex B demands it lowered. So this is an upper bound on what a
single table can do, and the gap between it and what training achieves is not
"headroom the optimiser is wasting" -- it may be irreducible conflict. Settling
that needs the mixed-integer version, which is only worth solving if this bound
comes out high.

Three numbers are reported separately, because mixing them hides which failure
is which: how many complexes have any acceptable pose in the pool at all
(retrieval), how many of those are individually winnable (capacity), and what
the trained table actually gets (achieved).

Example
-------
    uv run python scripts/capacity_ceiling.py \
        --pool-glob 'data/scaling/pool_cache/n1000_r0_*of6.pt' \
        --run data/scaling/runs_nfixed/N1000_seed0 --split val \
        --radii 0.5,1.0,2.0,inf
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
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.score import (IFACE_PAIR_OFFSET, IFACE_SIGN,  # noqa: E402
                         iface_score_matrix)


def affine(e_pool: dict, alpha, clash, beta) -> tuple[np.ndarray, np.ndarray]:
    """(A, c) with s_p(e) = c_p + <A_p, e>, in the ordering `e` itself uses."""
    sc, T = e_pool["sc"], e_pool["T"]
    c = (alpha * sc[:, 0] - (sc[:, 1:4] * clash).sum(-1)
         + beta * e_pool["elec"]
         + IFACE_PAIR_OFFSET * T.sum(dim=(-2, -1)))
    A = IFACE_SIGN * T.transpose(-2, -1).reshape(T.shape[0], -1)
    return A.numpy().astype(np.float64), c.numpy().astype(np.float64)


def winnable(A: np.ndarray, c: np.ndarray, pos: np.ndarray, e0: np.ndarray,
             radius: float, eps: float, seed_neg: int = 64,
             max_rounds: int = 40) -> tuple[bool, float]:
    """Is some positive separable from every negative within the box?

    Constraint generation: start from the negatives that already score highest,
    solve, then add whatever the solution violates. Exact -- it stops only when
    no constraint is violated -- but touches a few hundred of the ~1500 rows
    instead of all of them in every LP.
    """
    n_e = A.shape[1]
    neg_idx = np.flatnonzero(~pos)
    pos_idx = np.flatnonzero(pos)
    if neg_idx.size == 0 or pos_idx.size == 0:
        return False, float("nan")
    s0 = c + A @ e0
    order = neg_idx[np.argsort(-s0[neg_idx])]
    best_t = -np.inf
    lo = e0 - radius if np.isfinite(radius) else np.full(n_e, -np.inf)
    hi = e0 + radius if np.isfinite(radius) else np.full(n_e, np.inf)
    bounds = list(zip(lo, hi)) + [(None, None)]          # e, then t
    for p in pos_idx:
        active = list(order[:seed_neg])
        for _ in range(max_rounds):
            D = A[active] - A[p]                          # (m, 144)
            b = -(c[active] - c[p])
            # rows: <A_n - A_p, e> + t <= c_p - c_n   (i.e. margin >= t)
            A_ub = np.hstack([D, np.ones((D.shape[0], 1))])
            res = linprog(np.concatenate([np.zeros(n_e), [-1.0]]),
                          A_ub=A_ub, b_ub=b, bounds=bounds, method="highs")
            if res.status == 3:
                # unbounded: with no box the margin can be scaled up without
                # limit, which means separable, not infeasible
                return True, float("inf")
            if not res.success:
                break                                     # infeasible for this p
            e = res.x[:n_e]
            t = float(res.x[n_e])
            margin = (c[p] + A[p] @ e) - (c[neg_idx] + A[neg_idx] @ e)
            worst = neg_idx[np.argsort(margin)[:seed_neg]]
            bad = [int(i) for i in worst if i not in active
                   and margin[np.searchsorted(neg_idx, i)] < t - 1e-9]
            if not bad:
                best_t = max(best_t, float(margin.min()))
                break
            active.extend(bad)
        if best_t >= eps:
            return True, best_t
    return best_t >= eps, best_t


def shared_table(live, e0: np.ndarray, radius: float, eps: float,
                 seed_neg: int = 32, max_rounds: int = 30):
    """One table for every complex: the globally optimal soft-margin solution.

    The individual bound above is loose because each complex got its own table.
    This solves for a SHARED one, with each complex represented by a single
    target positive -- the one already scoring highest, i.e. the cheapest to
    lift -- by minimising the total hinge

        min_e  sum_c u_c,   u_c >= 0,
               u_c >= eps + <A_n - A_p(c), e> + c_n - c_p(c)   for every negative

    which is a linear program -- so its optimum is global, not a local minimum
    an optimiser happened to reach. A complex with u_c = 0 is genuinely won by
    the returned table, making the count an ACHIEVABLE value: a demonstration
    that a single 144-number table can do at least this well.

    It is not an upper bound. Fixing one target positive per complex, and using
    the hinge rather than the 0/1 count, both leave value on the table.
    """
    n_e = e0.size
    nc = len(live)
    # Which positive to aim at. The count does not care WHICH positive ends up
    # first, so the LP should be given the one that is easiest to lift: the
    # positive already scoring highest under the centre table. Taking the first
    # index instead would hand the LP an arbitrary target and understate what a
    # shared table can do.
    active, targets = [], []
    for name, A, c, pos in live:
        s0 = c + A @ e0
        p_idx = np.flatnonzero(pos)
        p = int(p_idx[np.argmax(s0[p_idx])])
        targets.append(p)
        neg = np.flatnonzero(~pos)
        active.append(list(neg[np.argsort(-s0[neg])][:seed_neg]))

    lo = e0 - radius if np.isfinite(radius) else np.full(n_e, -np.inf)
    hi = e0 + radius if np.isfinite(radius) else np.full(n_e, np.inf)
    bounds = list(zip(lo, hi)) + [(0.0, None)] * nc
    cost = np.concatenate([np.zeros(n_e), np.ones(nc)])
    e = e0.copy()
    for _ in range(max_rounds):
        rows, rhs = [], []
        for ci, (name, A, c, pos) in enumerate(live):
            p = targets[ci]
            for n in active[ci]:
                r = np.zeros(n_e + nc)
                r[:n_e] = A[n] - A[p]
                r[n_e + ci] = -1.0
                rows.append(r)
                rhs.append(c[p] - c[n] - eps)
        res = linprog(cost, A_ub=np.array(rows), b_ub=np.array(rhs),
                      bounds=bounds, method="highs")
        if not res.success:
            return None, 0, res.status
        e = res.x[:n_e]
        added = 0
        for ci, (name, A, c, pos) in enumerate(live):
            p = targets[ci]
            neg = np.flatnonzero(~pos)
            viol = eps + (c[neg] + A[neg] @ e) - (c[p] + A[p] @ e) - res.x[n_e + ci]
            bad = neg[np.argsort(-viol)][:seed_neg]
            new = [int(i) for i in bad if i not in active[ci]
                   and viol[np.searchsorted(neg, i)] > 1e-9]
            active[ci].extend(new)
            added += len(new)
        if added == 0:
            break
    else:
        # Constraint generation that stops on the round limit has not proved
        # anything: the returned table is still a real table, and its top-1 rate
        # is still achievable, but calling it the optimum of the fixed-anchor
        # hinge would be unsupported.
        return e, _count(live, e), 4
    return e, _count(live, e), 0


def _count(live, e: np.ndarray) -> int:
    return sum(1 for _, A, c, pos in live
               if bool(pos[int(np.argmax(c + A @ e))]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-glob", required=True, dest="pool_glob")
    ap.add_argument("--run", required=True, help="run dir with split.json / ckpt")
    ap.add_argument("--split", default="val", choices=("val", "fit"))
    # Solving the shared LP on the same complexes it is scored on is an in-sample
    # fit, and the trained table was fitted elsewhere; comparing the two would
    # flatter the LP. With this flag the shared table is solved on the FIT
    # complexes and then scored on --split, which is the comparison that means
    # something.
    ap.add_argument("--solve-on-fit", action="store_true", dest="solve_on_fit")
    ap.add_argument("--radii", default="0.5,1.0,2.0,inf",
                    help="box radius |e_i - e0_i| <= r; 'inf' for unconstrained")
    # Pre-registered before looking at any result: a margin of 1 raw score unit.
    # Raw pool scores span 5e2-2e3, so this is a strict inequality made numerical,
    # not a meaningful separation requirement.
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--thr", type=float, default=0.23)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--rho0", type=float, default=3.5)
    ap.add_argument("--limit", type=int, default=0)
    # Learning curve for the OPTIMAL table: solve the shared LP on subsets of
    # the fit complexes and score each solution on --split. No loss and no
    # optimiser are involved, so what this measures is whether more data helps
    # the best table there is -- which the N scaling experiments could only
    # answer under one particular loss.
    ap.add_argument("--curve", default="",
                    help="comma-separated fit-set sizes, e.g. 50,100,200,400")
    ap.add_argument("--curve-repeats", type=int, default=3, dest="curve_repeats")
    ap.add_argument("--curve-seed", type=int, default=0, dest="curve_seed")
    args = ap.parse_args()

    run = Path(args.run)
    ck = torch.load(run / "round0_ckpt.pt", map_location="cpu", weights_only=True)
    alpha = ck["alpha"].double()
    rho = ck.get("rho", torch.tensor(args.rho0)).double()
    clash = ck["clash_weights"].double() if "clash_weights" in ck else \
        alpha * rho.pow(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
    beta = torch.tensor(args.beta, dtype=torch.float64)
    e_trained = ck["iface"].double()
    e0 = iface_ij(dtype=torch.float64, flat=True)
    print(f"published table as the box centre; trained table sits "
          f"||de||_2 = {float((e_trained - e0).norm()):.3f}, "
          f"||de||_inf = {float((e_trained - e0).abs().max()):.3f} away")

    ids = json.load(open(run / "split.json"))[f"{args.split}_ids"]
    ids = ids[: args.limit] if args.limit else ids
    want = set(ids)
    pool = {}
    for f in sorted(glob.glob(args.pool_glob)):
        for d in torch.load(f, map_location="cpu", weights_only=True)["pools"]:
            if d["name"] not in want or d["name"] in pool:
                continue
            m = d["prov"] == 0
            if int(m.sum()) < 2:
                continue
            pool[d["name"]] = {"sc": d["sc"][m].double(), "T": d["T"][m].double(),
                               "elec": d["elec"][m].double(),
                               "dockq": d["dockq"][m].double()}
    print(f"{len(pool)} of {len(ids)} {args.split} complexes have a pool\n")

    # one-off check that the affine form reproduces the scorer it claims to
    name0 = next(iter(pool))
    A, c = affine(pool[name0], alpha, clash, beta)
    ref = (alpha * pool[name0]["sc"][:, 0]
           - (pool[name0]["sc"][:, 1:4] * clash).sum(-1)
           + (iface_score_matrix(e_trained) * pool[name0]["T"]).sum(dim=(-2, -1))
           + beta * pool[name0]["elec"]).numpy()
    err = np.abs(ref - (c + A @ e_trained.numpy())).max()
    print(f"affine reconstruction check on {name0[:40]}: max |diff| = {err:.3g}")
    if err > 1e-6:
        raise SystemExit("the affine decomposition does not reproduce the score")

    e0n, etn = e0.numpy(), e_trained.numpy()
    reach, achieved = 0, 0
    live = []
    for name, e in pool.items():
        A, c = affine(e, alpha, clash, beta)
        pos = (e["dockq"] >= args.thr).numpy()
        if not pos.any() or pos.all():
            continue
        reach += 1
        s = c + A @ etn
        achieved += bool(pos[int(np.argmax(s))])
        live.append((name, A, c, pos))
    n = len(pool)
    print(f"\nretrieval: {reach}/{n} complexes have an acceptable pose in the pool "
          f"({100 * reach / n:.1f}%)")
    print(f"achieved : the trained table ranks one first in {achieved}/{n} "
          f"({100 * achieved / n:.1f}%), i.e. {100 * achieved / reach:.1f}% of "
          f"the reachable ones\n")

    print(f"{'radius':>8s} {'winnable/reachable':>20s} {'% of all':>9s} "
          f"{'% of reachable':>15s} {'seconds':>8s}")
    for tok in args.radii.split(","):
        r = float("inf") if tok.strip() == "inf" else float(tok)
        t0 = time.time()
        ok = 0
        for name, A, c, pos in live:
            w, _ = winnable(A, c, pos, e0n, r, args.eps)
            ok += bool(w)
        print(f"{tok:>8s} {f'{ok}/{reach}':>20s} {100 * ok / n:8.1f}% "
              f"{100 * ok / reach:14.1f}% {time.time() - t0:8.1f}")

    solve_live = live
    if args.solve_on_fit:
        fit_ids = set(json.load(open(run / "split.json"))["fit_ids"])
        solve_pool = {}
        for f in sorted(glob.glob(args.pool_glob)):
            for d in torch.load(f, map_location="cpu", weights_only=True)["pools"]:
                if d["name"] not in fit_ids or d["name"] in solve_pool:
                    continue
                m = d["prov"] == 0
                if int(m.sum()) < 2:
                    continue
                solve_pool[d["name"]] = {
                    "sc": d["sc"][m].double(), "T": d["T"][m].double(),
                    "elec": d["elec"][m].double(), "dockq": d["dockq"][m].double()}
        solve_live = []
        for name, e in solve_pool.items():
            A_, c_ = affine(e, alpha, clash, beta)
            pos_ = (e["dockq"] >= args.thr).numpy()
            if pos_.any() and not pos_.all():
                solve_live.append((name, A_, c_, pos_))
        print(f"\nshared table solved on {len(solve_live)} FIT complexes with a "
              f"positive, then scored on the {reach} {args.split} ones")

    print(f"\n{'radius':>8s} {'SHARED table wins':>19s} {'% of all':>9s} "
          f"{'% of reachable':>15s} {'seconds':>8s}")
    for tok in args.radii.split(","):
        r = float("inf") if tok.strip() == "inf" else float(tok)
        t0 = time.time()
        e_star, won, st = shared_table(solve_live, e0n, r, args.eps)
        if e_star is None:
            print(f"{tok:>8s} {'LP failed (status ' + str(st) + ')':>19s}")
            continue
        note = "" if st == 0 else "  NOT CONVERGED (achievable, not optimal)"
        if args.solve_on_fit:
            won = sum(1 for _, A_, c_, pos_ in live
                      if bool(pos_[int(np.argmax(c_ + A_ @ e_star))]))
        print(f"{tok:>8s} {f'{won}/{reach}':>19s} {100 * won / n:8.1f}% "
              f"{100 * won / reach:14.1f}% {time.time() - t0:8.1f}{note}")

    if args.curve:
        if not args.solve_on_fit:
            raise SystemExit("--curve needs --solve-on-fit")
        r = float("inf") if args.radii.split(",")[-1].strip() == "inf" \
            else float(args.radii.split(",")[-1])
        print(f"\nlearning curve for the optimal shared table "
              f"(radius {args.radii.split(',')[-1]}, scored on the {reach} "
              f"{args.split} complexes)")
        print(f"{'fit size':>9s} {'wins/reachable':>16s} {'% of reachable':>15s} "
              f"{'(repeats)':>12s}")
        rng = np.random.default_rng(args.curve_seed)
        sizes = [int(x) for x in args.curve.split(",")] + [len(solve_live)]
        for k in sizes:
            k = min(k, len(solve_live))
            vals = []
            for _ in range(1 if k == len(solve_live) else args.curve_repeats):
                idx = rng.choice(len(solve_live), size=k, replace=False)
                sub = [solve_live[i] for i in idx]
                e_star, _, st = shared_table(sub, e0n, r, args.eps)
                if e_star is None:
                    continue
                vals.append(sum(1 for _, A_, c_, pos_ in live
                                if bool(pos_[int(np.argmax(c_ + A_ @ e_star))])))
            if not vals:
                print(f"{k:9d} {'LP failed':>16s}")
                continue
            m = sum(vals) / len(vals)
            sd = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
            print(f"{k:9d} {f'{m:.1f}/{reach}':>16s} {100 * m / reach:14.1f}% "
                  f"{f'+-{sd:.1f}, n={len(vals)}':>12s}")

    print("\nA low ceiling is decisive: no 144-number table can beat it. A high "
          "one is not -- each complex was solved with its own table, and the "
          "deployed one is shared.")


if __name__ == "__main__":
    main()
