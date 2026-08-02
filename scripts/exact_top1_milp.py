"""The true best top-1 rate a single shared table can reach, with a certificate.

Everything measured so far about the table's limits came from a surrogate. The
convex fit minimises a hinge against one anchor positive per complex, so its
score -- 54.0% on the fit complexes, 46.7% transferred -- is a value that was
ACHIEVED, not the maximum. The per-complex oracle (89.5-100%) is an upper bound
for a different, much larger model: one table per complex. Between the two sits
the number nobody has: the best top-1 rate of a single shared table.

It is a mixed-integer program, because "some positive ranks first" is a
disjunction:

    max_(e, y, z)  sum_c y_c
    s.t.  sum_{p in P_c} z_cp = y_c                                for each c
          <A_p - A_n, e> >= c_n - c_p + eps - M(1 - z_cp)          for each c, p, n
          ||e - e0||_inf <= R,     y, z binary

y_c says complex c is solved; z_cp picks which positive is meant to win. The
big-M is computed per constraint from the box, so it is valid rather than
guessed. Negatives enter lazily: start from the ones that already score highest
and add whatever the incumbent violates, which is exact because it stops only
when nothing is violated.

The solver returns both an incumbent (a table that really achieves that count,
so a lower bound) and a bound from branch and bound (no table can beat it). The
gap between them is reported rather than hidden, and the gap to the hinge
solution is the surrogate gap -- how much the convex fit's choice of loss costs,
as opposed to the model's capacity.

Small on purpose. The point is a certificate on 25-200 complexes, not a table to
deploy: a shared-capacity claim needs the count to stay low as the subset grows,
and a surrogate claim needs only that the MILP beats the hinge on the same set.

Example
-------
    uv run python scripts/exact_top1_milp.py \
        --pool-glob 'data/scaling/pool_cache/n1000_r0_*of6.pt' \
        --run data/scaling/runs_nfixed/N1000_seed0 --n 25 --radius 1.0
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
from scipy.optimize import LinearConstraint, milp, Bounds
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qp_path import Problem, load, solve  # noqa: E402
from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.score import IFACE_PAIR_OFFSET, IFACE_SIGN  # noqa: E402


def features(e, alpha, clash, beta, e0, thr):
    """(A, c, positives) for one complex: s_p(e) = c_p + <A_p, e>."""
    T = e["T"]
    A = (IFACE_SIGN * T.transpose(-2, -1).reshape(T.shape[0], -1)).numpy()
    sc = e["sc"].numpy()
    c = (float(alpha) * sc[:, 0] - sc[:, 1:4] @ clash.numpy()
         + float(beta) * e["elec"].numpy()
         + IFACE_PAIR_OFFSET * T.sum(dim=(-2, -1)).numpy())
    return A, c, (e["dockq"] >= thr).numpy()


def build_and_solve(cx, e0, radius, eps, seed_neg, rounds, time_limit, verbose):
    """Cutting-plane MILP. Returns (incumbent count, upper bound, e, rounds)."""
    n_e = e0.size
    n_c = len(cx)
    n_z = sum(len(c["pos_idx"]) for c in cx)
    lo = e0 - radius
    hi = e0 + radius
    # variable order: e (n_e), z (n_z), y (n_c)
    zoff = n_e
    yoff = n_e + n_z
    n_var = n_e + n_z + n_c
    integrality = np.concatenate([np.zeros(n_e), np.ones(n_z + n_c)])
    bounds = Bounds(np.concatenate([lo, np.zeros(n_z + n_c)]),
                    np.concatenate([hi, np.ones(n_z + n_c)]))
    obj = np.zeros(n_var)
    obj[yoff:] = -1.0                                   # maximise sum y

    # sum_p z_cp - y_c = 0
    rows, cols, vals = [], [], []
    zbase = 0
    for ci, c in enumerate(cx):
        for j in range(len(c["pos_idx"])):
            rows.append(ci); cols.append(zoff + zbase + j); vals.append(1.0)
        rows.append(ci); cols.append(yoff + ci); vals.append(-1.0)
        c["zbase"] = zbase
        zbase += len(c["pos_idx"])
    link = LinearConstraint(csr_matrix((vals, (rows, cols)), shape=(n_c, n_var)),
                            0.0, 0.0)

    # Seed the active set from BOTH the published table and the hinge solution.
    # Seeding only from e0 leaves the relaxation blind to the negatives that
    # matter near the solution, so the first rounds propose tables that violate
    # constraints nobody has written down yet.
    active = [{p: sorted(set(c["order"][:seed_neg])
                         | set(c.get("order_warm", c["order"])[:seed_neg]))
               for p in range(len(c["pos_idx"]))} for c in cx]
    best = None
    for rnd in range(1, rounds + 1):
        rows, cols, vals, ub = [], [], [], []
        r = 0
        for ci, c in enumerate(cx):
            A, cc = c["A"], c["c"]
            for j, p in enumerate(c["pos_idx"]):
                for n in active[ci][j]:
                    d = A[n] - A[p]                     # <d, e> + M z <= rhs
                    m = float(d @ e0 + radius * np.abs(d).sum()
                              - (cc[p] - cc[n] - eps)) + 1.0
                    m = max(m, 1.0)
                    for k in np.nonzero(d)[0]:
                        rows.append(r); cols.append(int(k)); vals.append(float(d[k]))
                    rows.append(r); cols.append(zoff + c["zbase"] + j); vals.append(m)
                    ub.append(float(cc[p] - cc[n] - eps + m))
                    r += 1
        con = LinearConstraint(csr_matrix((vals, (rows, cols)), shape=(r, n_var)),
                               -np.inf, np.asarray(ub))
        res = milp(c=obj, constraints=[link, con], integrality=integrality,
                   bounds=bounds,
                   options={"time_limit": time_limit, "disp": False})
        if res.x is None:
            return None, None, None, rnd
        e = res.x[:n_e]
        inc = int(round(-res.fun))
        bnd = int(np.floor(-res.mip_dual_bound + 1e-6)) \
            if res.mip_dual_bound is not None else n_c
        # who is actually solved by this e, and what did it violate?
        added, solved = 0, 0
        for ci, c in enumerate(cx):
            s = c["c"] + c["A"] @ e
            solved += bool(c["pos"][int(np.argmax(s))])
            for j, p in enumerate(c["pos_idx"]):
                if res.x[zoff + c["zbase"] + j] < 0.5:
                    continue
                viol = s[c["neg_idx"]] - s[p] + eps
                bad = c["neg_idx"][np.argsort(-viol)][:seed_neg]
                new = [int(i) for i in bad if i not in active[ci][j]
                       and viol[np.searchsorted(c["neg_idx"], i)] > 1e-9]
                active[ci][j].extend(new)
                added += len(new)
        if verbose:
            print(f"  round {rnd}: rows {r}, MILP says {inc}, bound {bnd}, "
                  f"actually solved {solved}, added {added} cuts", flush=True)
        best = (solved, bnd, e, rnd)
        if added == 0:
            return best
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-glob", required=True, dest="pool_glob")
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="fit", choices=("fit", "val"))
    ap.add_argument("--n", type=int, default=25, help="complexes in the subset")
    ap.add_argument("--subsets", type=int, default=3)
    ap.add_argument("--radius", type=float, default=1.0,
                    help="||e - e0||_inf box; also sets the big-M")
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--thr", type=float, default=0.23)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--max-pos", type=int, default=0, dest="max_pos",
                    help="cap positives per complex (0 = all; a cap breaks the "
                         "upper bound and is only for a quick incumbent)")
    ap.add_argument("--seed-neg", type=int, default=24, dest="seed_neg")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--time-limit", type=float, default=300.0, dest="time_limit")
    ap.add_argument("--seed", type=int, default=0)
    # The incumbent is not just a bound, it is a table -- fitted by maximising
    # the deployed metric directly. Saving it lets the obvious question be
    # answered: does a table that is exactly optimal on n complexes generalise?
    ap.add_argument("--save-tables", default="", dest="save_tables",
                    help="directory for each subset's incumbent table (.npy)")
    ap.add_argument("--val-pool-glob", default="", dest="val_pool_glob",
                    help="score every incumbent on the validation pool too")
    args = ap.parse_args()

    run = Path(args.run)
    ck = torch.load(run / "round0_ckpt.pt", map_location="cpu", weights_only=True)
    alpha = ck["alpha"].double()
    clash = ck["clash_weights"].double()
    beta = torch.tensor(args.beta, dtype=torch.float64)
    e0 = iface_ij(dtype=torch.float64, flat=True).numpy()
    ids = set(json.load(open(run / "split.json"))[f"{args.split}_ids"])
    entries = load(args.pool_glob, ids, alpha, clash, beta, args.thr, e0)
    print(f"{len(entries)} {args.split} complexes with a positive\n")

    rng = np.random.default_rng(args.seed)
    print(f"{'subset':>7s} {'n':>4s} {'hinge':>7s} {'MILP':>7s} {'bound':>7s} "
          f"{'gap':>5s} {'rounds':>7s} {'sec':>7s}")
    for s in range(args.subsets):
        pick = rng.choice(len(entries), size=min(args.n, len(entries)),
                          replace=False)
        sub = [entries[i] for i in pick]
        cx = []
        for name, e in sub:
            A, c, pos = features(e, alpha, clash, beta, e0, args.thr)
            p_idx = np.flatnonzero(pos)
            if args.max_pos:
                s0 = c + A @ e0
                p_idx = p_idx[np.argsort(-s0[p_idx])][: args.max_pos]
            neg = np.flatnonzero(~pos)
            s0 = c + A @ e0
            cx.append({"A": A, "c": c, "pos": pos, "pos_idx": p_idx,
                       "neg_idx": neg, "order": neg[np.argsort(-s0[neg])]})
        t0 = time.time()
        P = Problem(sub, alpha, clash, beta, args.thr, e0)
        e_h = solve(P, e0, 1e-2, eps=args.eps)
        e_h = np.clip(e_h, e0 - args.radius, e0 + args.radius)
        h_ok, h_n = P.top1(e_h)
        for c in cx:                      # warm-start cuts from the hinge table
            sh = c["c"] + c["A"] @ e_h
            c["order_warm"] = c["neg_idx"][np.argsort(-sh[c["neg_idx"]])]
        out = build_and_solve(cx, e0, args.radius, args.eps, args.seed_neg,
                              args.rounds, args.time_limit, verbose=True)
        if out[0] is None:
            print(f"{s:7d} {len(sub):4d}  MILP failed")
            continue
        inc, bnd, e_m, rnd = out
        extra = ""
        if args.val_pool_glob:
            if "V" not in dir():
                val_ids = set(json.load(open(run / "split.json"))["val_ids"])
                V = Problem(load(args.val_pool_glob, val_ids, alpha, clash, beta,
                                 args.thr, e0), alpha, clash, beta, args.thr, e0)
            vh = V.top1(e_h)
            vm = V.top1(e_m)
            v0 = V.top1(e0)
            extra = (f"   val: e0 {100 * v0[0] / v0[1]:.1f}%  "
                     f"hinge {100 * vh[0] / vh[1]:.1f}%  "
                     f"MILP {100 * vm[0] / vm[1]:.1f}%")
        if args.save_tables:
            Path(args.save_tables).mkdir(parents=True, exist_ok=True)
            np.save(Path(args.save_tables) / f"milp_n{len(sub)}_s{s}.npy", e_m)
        print(f"{s:7d} {len(sub):4d} {h_ok:7d} {inc:7d} {min(bnd, len(sub)):7d} "
              f"{min(bnd, len(sub)) - inc:5d} {rnd:7d} {time.time() - t0:7.1f}"
              f"{extra}")

    print("\nMILP > hinge means the surrogate is leaving value on the table.")
    print("MILP == bound means the certificate is tight; a gap means the search "
          "hit its limit and only the incumbent is proven achievable.")


if __name__ == "__main__":
    main()
