"""How far does performance go as the model is given more parameters, and does
data start to matter when it does?

Two questions that only make sense together. A 144-number table saturates at
200-400 training complexes (report section 5.14.27), so more data is useless --
but that is a statement about 144 numbers, not about the method. If a larger
model both scores higher and keeps improving with data, the saturation was the
model's, and collecting more complexes becomes worth doing.

Every model here is affine in its parameters, so each is fitted by the same
convex program and each solve is global. That keeps the comparison free of
learning rates, initialisation and early stopping: a difference between two rows
is a difference between two model classes, not between two optimisation runs.

The families, in order of size:

  psc      5    only alpha, the three clash weights and beta; table frozen
  sym     12    symmetric zero-sum directions of the table
  add     23    additive (row + column) directions
  full   144    the table as it is trained today
  full+   149   the table AND alpha / clash / beta together
  bucket 288    two tables, chosen by how large the complex is
  bucket 576    four tables

The bucketed models are the interesting ones. Fitting one table to every complex
loses a lot: complexes that are individually separable 89.5-100% of the time end
up at about 56% under a single shared table (section 5.14.29). Conditioning the
table on something observable *before* ranking -- here the complex's typical
contact count, a size proxy available at deployment -- tests whether that loss is
recoverable by letting the table vary.

What this cannot test: distance shells or finer atom types. Those need contact
counts the pool does not store, so they need the features recomputed from
structures first.

Example
-------
    uv run python scripts/capacity_series.py \
        --pool-glob 'data/scaling/pool_cache/n1000_r0_*of6.pt' \
        --run data/scaling/runs_nfixed/N1000_seed0 \
        --sizes 50,100,200,400
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
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.score import IFACE_SIGN  # noqa: E402


def zero_sum_basis(n: int) -> np.ndarray:
    """Orthonormal (n, n-1) basis of the zero-sum subspace (Helmert)."""
    B = np.zeros((n, n - 1))
    for k in range(1, n):
        B[:k, k - 1] = 1.0 / np.sqrt(k * (k + 1))
        B[k, k - 1] = -k / np.sqrt(k * (k + 1))
    return B


def table_basis(mode: str) -> np.ndarray:
    """(144, d) matrix mapping free coefficients onto a table offset."""
    if mode == "full":
        return np.eye(144)
    V = zero_sum_basis(12)
    cols = [np.ones((12, 12)).reshape(-1) / 12.0]        # the global level
    if mode == "sym":
        for j in range(11):
            a = V[:, j]
            cols.append(((a[:, None] + a[None, :]) / np.sqrt(24.0)).reshape(-1))
    elif mode == "add":
        for j in range(11):
            cols.append((np.tile(V[:, j][:, None], (1, 12)) / np.sqrt(12.0)).reshape(-1))
        for j in range(11):
            cols.append((np.tile(V[:, j][None, :], (12, 1)) / np.sqrt(12.0)).reshape(-1))
    else:
        raise ValueError(mode)
    return np.stack(cols, axis=1)


class Design:
    """Per-complex features for one model family.

    Every family is affine in its parameters, so a complex reduces to a matrix
    whose rows are each pose's feature vector, plus the constant part of the
    score. Fitting is then the same convex program regardless of family.
    """

    def __init__(self, entries, alpha, clash, beta, thr, e0, family,
                 n_bucket=1, edges=None):
        self.family, self.n_bucket, self.entries = family, n_bucket, entries
        self.base = None if family == "psc" else table_basis(
            "full" if family.startswith(("full", "bucket")) else family)
        # Size proxy: the typical number of contacts a pose of this complex
        # makes. Known before any ranking, so a deployed system could switch on
        # it. Bucket edges come from the FIT set and are passed in, never
        # recomputed on the set being scored.
        self.sizes = np.array([float(e["T"].sum(dim=(-2, -1)).median())
                               for _, e in entries])
        self.edges = np.array([]) if n_bucket == 1 else (
            np.quantile(self.sizes, np.linspace(0, 1, n_bucket + 1)[1:-1])
            if edges is None else edges)
        self.bucket = np.searchsorted(self.edges, self.sizes)
        self.dim = self._dim()

        neg_rows, pos_rows, spans = [], [], []
        self.all_rows, self.all_pos, self.all_const = [], [], []
        off = 0
        for ci, (name, e) in enumerate(entries):
            F = self._features(e, ci, alpha, clash, beta)
            const = self._const(e, alpha, clash, beta, e0)
            pos = (e["dockq"] >= thr).numpy()
            p_idx = np.flatnonzero(pos)
            p = int(p_idx[np.argmax(const[p_idx])])   # anchor at theta = 0
            neg = np.flatnonzero(~pos)
            neg_rows.append(F[neg].astype(np.float32))
            pos_rows.append(F[p].astype(np.float32))
            spans.append((off, off + neg.size, const[neg], const[p]))
            off += neg.size
            self.all_rows.append(F.astype(np.float32))
            self.all_pos.append(pos)
            self.all_const.append(const.astype(np.float32))
        self.Fn = np.concatenate(neg_rows)
        self.Fp = np.stack(pos_rows)
        self.spans = [(a, b) for a, b, _, _ in spans]
        self.cn = np.concatenate([c for _, _, c, _ in spans]).astype(np.float32)
        self.cp = np.array([c for _, _, _, c in spans], dtype=np.float32)

    def _dim(self) -> int:
        if self.family == "psc":
            return 5
        d = self.base.shape[1]
        if self.family == "full+":
            return d + 5
        if self.family.startswith("bucket"):
            return d * self.n_bucket
        return d

    @staticmethod
    def _const(e, alpha, clash, beta, e0) -> np.ndarray:
        """The score at parameters = 0, i.e. the published model."""
        T = e["T"]
        A = (IFACE_SIGN * T.transpose(-2, -1).reshape(T.shape[0], -1)).numpy()
        sc = e["sc"].numpy()
        return (float(alpha) * sc[:, 0] - sc[:, 1:4] @ clash.numpy()
                + float(beta) * e["elec"].numpy() + A @ e0)

    def _features(self, e, ci, alpha, clash, beta) -> np.ndarray:
        T = e["T"]
        A = (IFACE_SIGN * T.transpose(-2, -1).reshape(T.shape[0], -1)).numpy()
        sc = e["sc"].numpy()
        psc = np.column_stack([sc[:, 0], -sc[:, 1], -sc[:, 2], -sc[:, 3],
                               e["elec"].numpy()])
        if self.family == "psc":
            return psc
        AB = A @ self.base
        if self.family == "full+":
            return np.hstack([AB, psc])
        if self.family.startswith("bucket"):
            out = np.zeros((A.shape[0], self.dim))
            b = int(self.bucket[ci])
            out[:, b * AB.shape[1]:(b + 1) * AB.shape[1]] = AB
            return out
        return AB

    def hinge(self, th, idx=None, eps=1.0):
        thf = th.astype(np.float32)
        sn = self.cn + self.Fn @ thf
        sp = self.cp + self.Fp @ thf
        sel = range(len(self.spans)) if idx is None else idx
        tot, grad, m = 0.0, np.zeros(th.size), 0
        for ci in sel:
            lo, hi = self.spans[ci]
            k = lo + int(np.argmax(sn[lo:hi]))
            v = float(sn[k] - sp[ci]) + eps
            m += 1
            if v > 0:
                tot += v
                grad += self.Fn[k] - self.Fp[ci]
        return tot / max(1, m), grad / max(1, m)

    def top1(self, th, idx=None):
        thf = th.astype(np.float32)
        sel = range(len(self.entries)) if idx is None else idx
        ok = n = 0
        for ci in sel:
            s = self.all_const[ci] + self.all_rows[ci] @ thf
            ok += bool(self.all_pos[ci][int(np.argmax(s))])
            n += 1
        return ok, n


def solve(d: Design, lam: float, idx=None, eps=1.0, maxiter=300) -> np.ndarray:
    def fg(th):
        h, g = d.hinge(th, idx, eps)
        return h + 0.5 * lam * float(th @ th), g + lam * th
    return minimize(fg, np.zeros(d.dim), jac=True, method="L-BFGS-B",
                    options={"maxiter": maxiter, "maxcor": 30}).x


def load(pattern, ids, thr):
    out, seen = [], set()
    for f in sorted(glob.glob(pattern)):
        for d in torch.load(f, map_location="cpu", weights_only=True)["pools"]:
            if d["name"] in seen or d["name"] not in ids:
                continue
            seen.add(d["name"])
            m = d["prov"] == 0
            if int(m.sum()) < 2:
                continue
            e = {"sc": d["sc"][m].double(), "T": d["T"][m].double(),
                 "elec": d["elec"][m].double(), "dockq": d["dockq"][m].double()}
            pos = (e["dockq"] >= thr).numpy()
            if pos.any() and not pos.all():
                out.append((d["name"], e))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-glob", required=True, dest="pool_glob")
    ap.add_argument("--run", required=True)
    ap.add_argument("--index", default="external/pinder/pinder/2024-02/index.parquet")
    ap.add_argument("--sizes", default="50,100,200,400")
    ap.add_argument("--lambdas", default="1e-2,1e-1,1,10")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
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

    fit_e = load(args.pool_glob, set(sp["fit_ids"]), args.thr)
    val_e = load(args.pool_glob, set(sp["val_ids"]), args.thr)
    print(f"fit {len(fit_e)} / validation {len(val_e)} complexes with a positive")

    import pandas as pd
    ix = pd.read_parquet(args.index, columns=["id", "cluster_id"]).set_index("id")
    cl = [str(ix.cluster_id.get(n, n)) for n, _ in fit_e]
    uniq = sorted(set(cl))
    rng = np.random.default_rng(args.seed)
    fold_of = {c: int(i % args.folds) for i, c in enumerate(rng.permutation(uniq))}
    folds = np.array([fold_of[c] for c in cl])

    fams = [("psc", 1), ("sym", 1), ("add", 1), ("full", 1), ("full+", 1),
            ("bucket", 2), ("bucket", 4)]
    sizes = [int(x) for x in args.sizes.split(",")] + [len(fit_e)]
    lams = [float(x) for x in args.lambdas.split(",")]

    print(f"\n{'family':>8s} {'params':>7s} {'lambda':>7s} " +
          " ".join(f"{f'n={k}':>9s}" for k in sizes))
    for fam, nb in fams:
        F = Design(fit_e, alpha, clash, beta, args.thr, e0, fam, nb)
        # the validation design must use the FIT set's bucket edges
        V = Design(val_e, alpha, clash, beta, args.thr, e0, fam, nb,
                   edges=F.edges)
        best_lam, best_cv = lams[0], -1.0
        for lam in lams:
            ok = tot = 0
            for k in range(args.folds):
                th = solve(F, lam, idx=np.flatnonzero(folds != k), eps=args.eps)
                a, b = F.top1(th, np.flatnonzero(folds == k))
                ok += a
                tot += b
            if 100.0 * ok / tot > best_cv:
                best_cv, best_lam = 100.0 * ok / tot, lam
        row = []
        for k in sizes:
            k = min(k, len(fit_e))
            vals = []
            for r in range(1 if k == len(fit_e) else args.repeats):
                idx = np.random.default_rng(args.seed + r).choice(
                    len(fit_e), size=k, replace=False)
                th = solve(F, best_lam, idx=idx, eps=args.eps)
                a, b = V.top1(th)
                vals.append(100.0 * a / b)
            row.append(f"{np.mean(vals):8.1f}%")
        name = fam if nb == 1 else f"{fam}{nb}"
        print(f"{name:>8s} {F.dim:7d} {best_lam:7.2g} " + " ".join(row), flush=True)

    a, b = V.top1(np.zeros(V.dim))
    print(f"\npublished table on the same validation complexes: "
          f"{a}/{b} = {100 * a / b:.1f}%")


if __name__ == "__main__":
    main()
