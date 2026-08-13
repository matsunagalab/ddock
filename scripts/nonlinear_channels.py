"""Break pair-additivity without leaving the FFT, and without leaving convexity.

Section 5.14.50 closed off every alternative: candidate retrieval saturates
(Oracle@25 = 100%), the refiner is already at a local optimum, refining 20x
more poses changes nothing, more translations per rotation changes nothing.
What is left is the ranking function, and section 5.14.34 showed that no
re-fitting of the shared 144-number table recovers it. The function form has to
change.

The FFT does not actually require pair-additivity. It requires that the score
be a pointwise function of cross-correlations:

    S(t) = g(G_1(t), ..., G_r(t)),  G_k = IFFT[F(R_k) . conj(F(L_k))]

`g` is evaluated on the translation grid after the inverse transform, so the
nonlinearity itself is free. The channels are not: the deployed score sums the
twelve ligand transforms in the frequency domain and inverts once, whereas
keeping G_1..G_r apart costs one inverse transform each. At r = 4 that is 4
forward + 4 inverse against the present 12 forward + 1 inverse, so still
cheaper -- but "no extra FFTs" would be wrong, and an earlier version of this
docstring said it.
Choosing a *fixed* basis of products keeps the score affine in its parameters,
so the convex fit, the cached-feature rescoring and the LP capacity diagnostics
all survive intact -- which is the whole reason to prefer this over bolting a
second discriminator on top.

Everything here reads the cached pools. `T` is exactly the 144 raw correlation
values at each pose's translation, so the products are already in hand and no
GPU or new features are needed.

Families, all nested so that the added coefficients being zero reproduces the
one before it:

    sym      12  the current best subspace, e = e0 + B_sym theta
    symquad  12 + products of a few aggregated channels
    symnorm  12, with every feature divided by (n / median_c n)^p, n = sum T
             -- the extensive-vs-intensive correction, p fixed per fit

    uv run python scripts/nonlinear_channels.py \
        --pool-glob 'data/scaling/pool_cache/n1000_r0_*of6.pt' \
        --run data/scaling/runs_nfixed/N1000_seed0
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from capacity_series import table_basis  # noqa: E402

from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.score import IFACE_PAIR_OFFSET, IFACE_SIGN  # noqa: E402


def channel_vectors(e0_mat: np.ndarray, n_modes: int) -> tuple[np.ndarray, list[str]]:
    """Linear functionals of the 144 raw features that define the channels.

    One is the total contact count, which is what an extensive score cannot
    normalise away by itself. The rest are the leading eigenmodes of the
    published table -- PIPER truncates its own 18x18 matrix to two such modes
    before handing it to the FFT, and e0 here carries 98.4% of its Frobenius
    energy in three, so a handful of modes is the natural small basis.
    """
    w, V = np.linalg.eigh(e0_mat)
    order = np.argsort(-np.abs(w))
    cols = [np.ones((12, 12)).reshape(-1)]
    names = ["contact"]
    for r in range(n_modes):
        v = V[:, order[r]]
        cols.append(np.outer(v, v).reshape(-1))
        names.append(f"mode{r + 1}")
    return np.stack(cols, axis=1), names


class Design:
    """One model family, as `score = c + Phi @ theta` for every pose.

    Centring a feature within a complex shifts every pose of that complex by
    the same amount, so it changes neither the ranking nor the hinge. It is
    applied to the added columns only, to keep them conditioned without
    disturbing the exact nesting with the deployed recipe.
    """

    def __init__(self, entries, alpha, clash, beta, thr, e0, family,
                 n_modes=3, p=0.0, gamma_col=False, col_scale=None):
        self.family, self.names = family, []
        B = table_basis("sym")
        G, self.chan_names = channel_vectors(
            (IFACE_SIGN * e0.reshape(12, 12).T).T, n_modes)

        self.Phi, self.c, self.pos = [], [], []
        for name, e in entries:
            sc, T = e["sc"], e["T"]
            a = (IFACE_SIGN * T.transpose(-2, -1)
                 .reshape(T.shape[0], -1)).numpy().astype(np.float64)
            cb = (alpha * sc[:, 0] - (sc[:, 1:4] * clash).sum(-1)
                  + beta * e["elec"]
                  + IFACE_PAIR_OFFSET * T.sum(dim=(-2, -1))).numpy()

            phi = a @ B                       # (F, 12)
            c = cb + a @ e0

            if family == "symquad":
                ch = a @ G                    # (F, n_chan)
                ch = ch / (np.abs(ch).mean(axis=0, keepdims=True) + 1e-12)
                prods, pn = [], []
                for i in range(ch.shape[1]):
                    for j in range(i, ch.shape[1]):
                        prods.append(ch[:, i] * ch[:, j])
                        pn.append(f"{self.chan_names[i]}*{self.chan_names[j]}")
                extra = np.stack(prods, axis=1)
                extra -= extra.mean(axis=0, keepdims=True)
                phi = np.concatenate([phi, extra], axis=1)
                self.prod_names = pn
            elif family == "symnorm":
                n = T.sum(dim=(-2, -1)).numpy()
                w = (n / max(1e-9, float(np.median(n)))) ** p
                w = np.clip(w, 1e-3, 1e3)
                phi, c = phi / w[:, None], c / w
            elif family == "symlogn":
                # Strictly nested one-degree-of-freedom contact correction:
                # S = S_sym - gamma * log(n / median_c n), gamma = 0 is sym.
                # Additive and bounded, so unlike dividing the whole score it
                # cannot amplify a near-zero-contact pose by a large factor,
                # and it leaves the clash penalty and the electrostatics on
                # their own scale.
                n = T.sum(dim=(-2, -1)).numpy()
                ln = -np.log(np.clip(n / max(1e-9, float(np.median(n))),
                                     1e-6, None))
                phi = np.concatenate([phi, ln[:, None]], axis=1)
            elif family != "sym":
                raise ValueError(family)

            self.Phi.append(phi.astype(np.float64))
            self.c.append(c.astype(np.float64))
            self.pos.append((e["dockq"] >= thr).numpy())
            self.names.append(name)

        self.d = self.Phi[0].shape[1]
        # Scale the added columns to unit spread so one lambda means the same
        # thing for the twelve subspace directions and for the products.
        #
        # The scale has to be FITTED ONCE AND PASSED IN. Recomputing it per
        # design makes the same theta mean a different function on validation
        # than on fit, which silently invalidates the comparison the whole
        # screen exists to make. (It did, until codex caught it.)
        if self.d > 12:
            if col_scale is None:
                col_scale = np.concatenate(
                    [np.abs(P[:, 12:]) for P in self.Phi]).mean(0) + 1e-12
            for P in self.Phi:
                P[:, 12:] /= col_scale
        self.col_scale = col_scale

        # Stack every negative into one matrix so a gradient step is a single
        # matmul rather than one per complex. The anchor positive is the one
        # already scoring highest at theta = 0, i.e. the cheapest to lift; the
        # count this is a surrogate for does not care which positive wins.
        Aneg, cneg, Apos, cpos, spans, off = [], [], [], [], [], 0
        for P, c, pos in zip(self.Phi, self.c, self.pos):
            s0 = c.copy()
            pi = np.flatnonzero(pos)
            ap = int(pi[np.argmax(s0[pi])])
            neg = np.flatnonzero(~pos)
            Aneg.append(P[neg])
            cneg.append(c[neg])
            Apos.append(P[ap])
            cpos.append(c[ap])
            spans.append((off, off + neg.size))
            off += neg.size
        self.A = np.concatenate(Aneg)
        self.cn = np.concatenate(cneg)
        self.Ap = np.stack(Apos)
        self.cp = np.array(cpos)
        self.spans = spans
        # every pose, for the 0/1 count -- the hinge only needs the worst
        # negative, but top-1 needs the argmax over the whole pool
        self.allA = np.concatenate(self.Phi)
        self.allc = np.concatenate(self.c)
        self.allpos = np.concatenate(self.pos)
        self.allspans, o = [], 0
        for P in self.Phi:
            self.allspans.append((o, o + P.shape[0]))
            o += P.shape[0]
        self.Phi, self.c, self.pos = None, None, None

    def hinge(self, th, idx=None, eps: float = 1.0):
        s_neg = self.cn + self.A @ th
        s_pos = self.cp + self.Ap @ th
        sel = range(len(self.spans)) if idx is None else idx
        tot, grad, m = 0.0, np.zeros(self.d), 0
        for ci in sel:
            lo, hi = self.spans[ci]
            k = lo + int(np.argmax(s_neg[lo:hi]))
            v = float(s_neg[k] - s_pos[ci]) + eps
            m += 1
            if v > 0:
                tot += v
                grad += self.A[k] - self.Ap[ci]
        return tot / max(1, m), grad / max(1, m)

    def top1(self, th, idx=None):
        s = self.allc + self.allA @ th
        sel = range(len(self.allspans)) if idx is None else idx
        ok = n = 0
        for ci in sel:
            lo, hi = self.allspans[ci]
            ok += bool(self.allpos[lo + int(np.argmax(s[lo:hi]))])
            n += 1
        return ok, n


def solve(design: Design, lam: float, idx=None, eps: float = 1.0,
          maxiter: int = 400, lam_extra: float | None = None) -> np.ndarray:
    """Group-wise ridge. The twelve subspace directions and the added product
    columns are on different scales and answer to different priors, so a single
    penalty would confound "the products do not help" with "the products were
    regularised like a table"."""
    pen = np.full(design.d, lam)
    if lam_extra is not None and design.d > 12:
        pen[12:] = lam_extra

    def fg(th):
        h, g = design.hinge(th, idx, eps)
        return h + 0.5 * float(th @ (pen * th)), g + pen * th
    res = minimize(fg, np.zeros(design.d), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxcor": 30})
    return res.x


def load(pattern: str, ids: set[str], thr: float):
    entries, seen = [], set()
    for f in sorted(glob.glob(pattern)):
        for d in torch.load(f, map_location="cpu", weights_only=True)["pools"]:
            if d["name"] in seen or d["name"] not in ids:
                continue
            seen.add(d["name"])
            m = d["prov"] == 0          # search-derived poses only
            if int(m.sum()) < 2:
                continue
            e = {"sc": d["sc"][m].double(), "T": d["T"][m].double(),
                 "elec": d["elec"][m].double(), "dockq": d["dockq"][m].double()}
            pos = (e["dockq"] >= thr).numpy()
            if pos.any() and not pos.all():
                entries.append((d["name"], e))
    return entries


def blocked_folds(names, index_path, folds, seed):
    import pandas as pd
    ix = pd.read_parquet(index_path, columns=["id", "cluster_id"]).set_index("id")
    cl = [str(ix.cluster_id.get(n, n)) for n in names]
    uniq = sorted(set(cl))
    rng = np.random.default_rng(seed)
    assign = {c: i for c, i in zip(uniq, rng.permutation(len(uniq)) % folds)}
    return np.array([assign[c] for c in cl])


def evaluate(design_fit, design_val, lambdas, fold, folds, eps, label,
             lam_extras=(None,)):
    best, best_cv = None, -1.0
    for lam in lambdas:
        for le in lam_extras:
            ok = n = 0
            for k in range(folds):
                tr = np.flatnonzero(fold != k)
                te = np.flatnonzero(fold == k)
                th = solve(design_fit, lam, idx=tr, eps=eps, lam_extra=le)
                a, b = design_fit.top1(th, idx=te)
                ok += a
                n += b
            cv = ok / max(1, n)
            if cv > best_cv:
                best_cv, best = cv, (lam, le)
    lam, le = best
    label = label + (f" [lq={le:g}]" if le is not None else "")
    th = solve(design_fit, lam, eps=eps, lam_extra=le)
    fo, fn = design_fit.top1(th)
    vo, vn = design_val.top1(th)
    print(f"  {label:<34} lambda={lam:<8g} CV {best_cv * 100:6.2f}%   "
          f"fit {fo}/{fn} = {fo / fn * 100:6.2f}%   "
          f"val {vo}/{vn} = {vo / vn * 100:6.2f}%")
    return {"label": label, "lam": lam, "cv": best_cv,
            "fit": fo / fn, "val": vo / vn, "theta": th, "vo": vo, "vn": vn}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-glob", required=True, dest="pool_glob")
    ap.add_argument("--run", required=True)
    ap.add_argument("--index", default="external/pinder/pinder/2024-02/index.parquet")
    ap.add_argument("--lambdas", default="1e-3,1e-2,1e-1,1,10")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--thr", type=float, default=0.23)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--rho0", type=float, default=3.5)
    ap.add_argument("--n-modes", type=int, default=3, dest="n_modes")
    ap.add_argument("--modes-grid", default="3,5", dest="modes_grid")
    ap.add_argument("--quad-lambdas", default="1e-3,1e-2,1e-1,1,10,100",
                    dest="quad_lambdas")
    ap.add_argument("--ps", default="0.1,0.25,0.5,0.75,1.0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run)
    ck = torch.load(run / "round0_ckpt.pt", map_location="cpu", weights_only=True)
    alpha = ck["alpha"].double()
    rho = ck.get("rho", torch.tensor(args.rho0)).double()
    clash = (ck["clash_weights"].double() if "clash_weights" in ck else
             alpha * rho.pow(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)))
    beta = torch.tensor(args.beta, dtype=torch.float64)
    e0 = iface_ij(dtype=torch.float64, flat=True).numpy()

    sp = json.load(open(run / "split.json"))
    t0 = time.time()
    fit_e = load(args.pool_glob, set(sp["fit_ids"]), args.thr)
    val_e = load(args.pool_glob, set(sp["val_ids"]), args.thr)
    print(f"loaded fit {len(fit_e)}  val {len(val_e)}  ({time.time() - t0:.0f}s)")

    fold = blocked_folds([n for n, _ in fit_e], args.index, args.folds, args.seed)
    lambdas = [float(x) for x in args.lambdas.split(",")]
    kw = dict(alpha=alpha, clash=clash, beta=beta, thr=args.thr, e0=e0)

    print("\nfamilies (nested: added coefficients at zero reproduce the row above)")
    res = []
    df = Design(fit_e, family="sym", **kw)
    dv = Design(val_e, family="sym", **kw)
    base = evaluate(df, dv, lambdas, fold, args.folds, args.eps, "sym (12)")
    res.append(base)

    lq = [float(x) for x in args.quad_lambdas.split(",")]
    for nm in [int(x) for x in args.modes_grid.split(",")]:
        df = Design(fit_e, family="symquad", n_modes=nm, **kw)
        dv = Design(val_e, family="symquad", n_modes=nm,
                    col_scale=df.col_scale, **kw)
        res.append(evaluate(df, dv, lambdas, fold, args.folds, args.eps,
                            f"sym + quad m{nm} ({df.d})", lam_extras=lq))
        print(f"    channels {df.chan_names}  products {len(df.prod_names)}")

    df = Design(fit_e, family="symlogn", **kw)
    dv = Design(val_e, family="symlogn", col_scale=df.col_scale, **kw)
    res.append(evaluate(df, dv, lambdas, fold, args.folds, args.eps,
                        "sym + gamma*log n (13)", lam_extras=lq))

    for p in [float(x) for x in args.ps.split(",")]:
        df = Design(fit_e, family="symnorm", p=p, **kw)
        dv = Design(val_e, family="symnorm", p=p, col_scale=None, **kw)
        res.append(evaluate(df, dv, lambdas, fold, args.folds, args.eps,
                            f"sym / n^{p}"))

    print("\nvalidation top-1, best first")
    for r in sorted(res, key=lambda r: -r["val"]):
        d = (r["val"] - base["val"]) * 100
        print(f"  {r['label']:<34} {r['vo']:3d}/{r['vn']:3d} = "
              f"{r['val'] * 100:6.2f}%   vs sym {d:+6.2f} pp")


if __name__ == "__main__":
    main()
