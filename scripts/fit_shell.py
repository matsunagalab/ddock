"""Does distance resolution buy anything the 6 A contact count does not?

Section 5.14.51 ruled out recombining the existing 144 numbers non-linearly:
products of aggregated channels and a contact-count correction both moved
validation by exactly zero complexes. Every method ahead of this one carries a
distance-dependent pair term, so the remaining single-score option is a feature
that is not a function of the present one.

The model is section 5.14.38's, and it is strictly nested:

    S_IFACE = <W(e_base), T>
              + sum_b <W(B_table sum_r H_band[b,r] eta_r), T_b>

The columns of B_table and H_band are orthonormal; H_band is also zero-sum.
Thus eta = 0 reproduces the deployed model exactly, while |eta|^2 is exactly
the sum of squared Frobenius norms of the per-band table contrasts.  This is
the same function family as eliminating the last band, but not the same ridge
geometry.  For three bands it leaves 24 free directions on top of the 12 of
the base, which is the "effective 24" the plan quotes.

Group-wise ridge throughout: the base directions and the shell contrasts are
different objects and a shared penalty would confound "distance does not help"
with "the contrasts were regularised like a table". Both lambdas are chosen by
interface-cluster blocked CV inside fit; the fixed validation set is scored
once, afterwards.

    uv run python scripts/fit_shell.py \
        --shell-glob 'data/scaling/shell_pool/shell_*of3.pt' \
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

from capacity_series import table_basis, zero_sum_basis  # noqa: E402

from zdock.atomtypes import iface_ij  # noqa: E402
from zdock.score import IFACE_PAIR_OFFSET, IFACE_SIGN  # noqa: E402


def band_contrast_basis(n_bands: int) -> np.ndarray:
    """Orthonormal basis of band contrasts whose columns sum to zero."""
    if n_bands < 2:
        raise ValueError("partition contrasts require at least two bands")
    return zero_sum_basis(n_bands)


class ShellDesign:
    """Linear score design in an orthonormal table x band parameterisation."""

    def __init__(self, entries, alpha, clash, beta, thr, e0, use_shells=True,
                 col_scale=None, mode="partition", basis_mode="sym"):
        B = table_basis(basis_mode)
        self.base_dim = B.shape[1]
        self.table_basis = B
        self.band_basis = None
        self.Phi, self.c, self.pos, self.names = [], [], [], []
        for name, e in entries:
            sc, T, Tb = e["sc"], e["T"], e["Tb"]
            F = T.shape[0]
            a = (IFACE_SIGN * T.transpose(-2, -1).reshape(F, -1)).numpy()
            c = (alpha * sc[:, 0] - (sc[:, 1:4] * clash).sum(-1)
                 + beta * e["elec"]
                 + IFACE_PAIR_OFFSET * T.sum(dim=(-2, -1))).numpy() + a @ e0
            phi = a @ B
            if use_shells:
                nb = Tb.shape[1]
                ab = [(IFACE_SIGN * Tb[:, k].transpose(-2, -1)
                       .reshape(F, -1)).numpy() for k in range(nb)]
                if mode == "partition":
                    # H has orthonormal, zero-sum columns.  If eta_r is the
                    # table coefficient for column r, delta_G_b = sum_r
                    # H[b,r] B eta_r and therefore
                    # sum_b ||delta_G_b||_F^2 = sum_r ||eta_r||_2^2.
                    H = band_contrast_basis(nb)
                    if self.band_basis is None:
                        self.band_basis = H
                    elif self.band_basis.shape != H.shape:
                        raise ValueError("all entries must use the same bands")
                    add = []
                    for r in range(H.shape[1]):
                        mixed = sum(H[k, r] * ab[k] for k in range(nb))
                        add.append(mixed @ B)
                else:
                    # Bands outside the 6 A window are not a re-partition of T
                    # but extra information.  The identity band basis keeps
                    # their independent table perturbations orthonormal.
                    H = np.eye(nb)
                    if self.band_basis is None:
                        self.band_basis = H
                    elif self.band_basis.shape != H.shape:
                        raise ValueError("all entries must use the same bands")
                    add = [ab[k] @ B for k in range(nb)]
                phi = np.concatenate([phi] + add, axis=1)
            self.Phi.append(phi.astype(np.float64))
            self.c.append(c.astype(np.float64))
            self.pos.append((e["dockq"] >= thr).numpy())
            self.names.append(name)

        self.d = self.Phi[0].shape[1]
        # `col_scale` remains in the public constructor because existing
        # callers pass the fit design's value into validation.  Data-dependent
        # feature scaling is deliberately no longer applied: it turns raw
        # ridge into sum_j scale_j^2 * physical_coefficient_j^2.  Unit scales
        # make that legacy call pattern harmless while preserving the common
        # Frobenius geometry.
        n_extra = self.d - self.base_dim
        if col_scale is not None and np.asarray(col_scale).shape != (n_extra,):
            raise ValueError(
                f"col_scale has shape {np.asarray(col_scale).shape}, "
                f"expected {(n_extra,)}"
            )
        self.col_scale = np.ones(n_extra, dtype=np.float64)

        A, cn, Ap, cp, spans, off = [], [], [], [], [], 0
        for P, c, pos in zip(self.Phi, self.c, self.pos):
            pi = np.flatnonzero(pos)
            ap = int(pi[np.argmax(c[pi])])
            neg = np.flatnonzero(~pos)
            A.append(P[neg]); cn.append(c[neg])
            Ap.append(P[ap]); cp.append(c[ap])
            spans.append((off, off + neg.size)); off += neg.size
        self.A, self.cn = np.concatenate(A), np.concatenate(cn)
        self.Ap, self.cp = np.stack(Ap), np.array(cp)
        self.spans = spans
        self.allA = np.concatenate(self.Phi)
        self.allc = np.concatenate(self.c)
        self.allpos = np.concatenate(self.pos)
        self.allspans, o = [], 0
        for P in self.Phi:
            self.allspans.append((o, o + P.shape[0])); o += P.shape[0]
        self.Phi = self.c = self.pos = None

    def hinge(self, th, idx=None, eps=1.0):
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
        hit, ok, n = [], 0, 0
        for ci in sel:
            lo, hi = self.allspans[ci]
            w = bool(self.allpos[lo + int(np.argmax(s[lo:hi]))])
            hit.append(w); ok += w; n += 1
        return ok, n, np.array(hit)


def solve(design, lam, lam_extra=None, idx=None, eps=1.0, maxiter=400):
    pen = np.full(design.d, lam)
    if lam_extra is not None and design.d > design.base_dim:
        pen[design.base_dim:] = lam_extra

    def fg(th):
        h, g = design.hinge(th, idx, eps)
        return h + 0.5 * float(th @ (pen * th)), g + pen * th
    r = minimize(fg, np.zeros(design.d), jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter, "maxcor": 30})
    return r.x, bool(r.success)


def load(pattern, ids, thr):
    entries, seen = [], set()
    for f in sorted(glob.glob(pattern)):
        for d in torch.load(f, map_location="cpu", weights_only=True)["pools"]:
            if d["name"] in seen or d["name"] not in ids:
                continue
            seen.add(d["name"])
            e = {k: d[k].double() for k in ("sc", "T", "Tb", "elec", "dockq")}
            pos = (e["dockq"] >= thr).numpy()
            if pos.any() and not pos.all():
                entries.append((d["name"], e))
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shell-glob", required=True, dest="shell_glob")
    ap.add_argument("--run", required=True)
    ap.add_argument("--index", default="external/pinder/pinder/2024-02/index.parquet")
    ap.add_argument("--lambdas", default="1e-3,1e-2,1e-1,1,10")
    ap.add_argument("--shell-lambdas", default="1e-2,1e-1,1,10,100",
                    dest="shell_lambdas")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--thr", type=float, default=0.23)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--rho0", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="partition",
                    choices=("partition", "outer"))
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
    fit_e = load(args.shell_glob, set(sp["fit_ids"]), args.thr)
    val_e = load(args.shell_glob, set(sp["val_ids"]), args.thr)
    nb = fit_e[0][1]["Tb"].shape[1]
    print(f"loaded fit {len(fit_e)}  val {len(val_e)}  shells {nb}  "
          f"({time.time() - t0:.0f}s)")

    import pandas as pd
    ix = pd.read_parquet(args.index, columns=["id", "cluster_id"]).set_index("id")
    cl = [str(ix.cluster_id.get(n, n)) for n, _ in fit_e]
    uniq = sorted(set(cl))
    rng = np.random.default_rng(args.seed)
    assign = {c: i for c, i in zip(uniq, rng.permutation(len(uniq)) % args.folds)}
    fold = np.array([assign[c] for c in cl])

    kw = dict(alpha=alpha, clash=clash, beta=beta, thr=args.thr, e0=e0)
    lam_g = [float(x) for x in args.lambdas.split(",")]
    lam_s = [float(x) for x in args.shell_lambdas.split(",")]

    out = {}
    mode = args.mode
    ndim = 12 * (nb if mode == "outer" else nb - 1)
    for label, shells, extras in (
            ("sym base (12)", False, [None]),
            (f"sym + {nb} {mode} band(s) (12+{ndim})", True, lam_s)):
        df = ShellDesign(fit_e, use_shells=shells, mode=mode, **kw)
        dv = ShellDesign(val_e, use_shells=shells, mode=mode,
                         col_scale=df.col_scale, **kw)
        best, best_cv = None, -1.0
        for lam in lam_g:
            for le in extras:
                ok = n = 0
                for k in range(args.folds):
                    th, _ = solve(df, lam, le, idx=np.flatnonzero(fold != k),
                                  eps=args.eps)
                    a, b, _ = df.top1(th, idx=np.flatnonzero(fold == k))
                    ok += a; n += b
                if ok / max(1, n) > best_cv:
                    best_cv, best = ok / max(1, n), (lam, le)
        lam, le = best
        th, conv = solve(df, lam, le, eps=args.eps)
        fo, fn, _ = df.top1(th)
        vo, vn, hit = dv.top1(th)
        out[label] = hit
        print(f"  {label:<24} lam={lam:<6g} lam_shell={str(le):<6} "
              f"CV {best_cv * 100:6.2f}%  fit {fo / fn * 100:6.2f}%  "
              f"val {vo}/{vn} = {vo / vn * 100:6.2f}%"
              + ("" if conv else "   [L-BFGS did not report success]"))

    a, b = list(out.values())
    w, l = int((~a & b).sum()), int((a & ~b).sum())
    from scipy.stats import binomtest
    p = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0
    print(f"\npaired on the same {len(a)} validation complexes: "
          f"shells win {w}, lose {l}, exact McNemar p = {p:.3f}")
    print("fixed pool re-ranking only; section 5.14.39 measured that a fixed-pool "
          "difference shrinks a lot end-to-end, so this is a screen, not a result.")


if __name__ == "__main__":
    main()
