"""Compare two scoring-parameter settings on the frozen TEST pool, with power.

Why not success@K
-----------------
``success@K`` over 241 complexes moves in steps of 1/241 = 0.41 pp and every
number measured so far lives between 0.0% and 4.1%. Re-testing the differences
the report quoted: Fisher exact gives p = 0.62 / 0.34 / 0.17 at K = 1 / 10 /
100, and exact McNemar gives p = 1.00 / 0.25 / 0.25. **None is significant.**
At n = 241, alpha = 0.05, 80% power, the smallest detectable rate from a 1.7%
baseline is 6.5%; exact McNemar needs >= 6 one-directional discordant complexes
(>= 2.5 pp). Every effect in the report is below those floors.

What this uses instead
----------------------
Per-complex continuous readouts, compared with a **paired** test over the same
complexes:

* ``auc``  -- Mann-Whitney AUC between the complex's positives and negatives,
  with midrank tie correction. The primary readout: it uses the whole ranking
  rather than a threshold crossing, and it has no floor.
* ``first_hit_pct`` -- how deep the ranking must be read before the first
  positive, as a fraction of the pool.
* ``best_dockq@K`` -- quality of the best pose in the top K.

Paired Wilcoxon signed-rank is reported alongside the paired t-test because the
per-complex differences are not remotely normal. success@K is kept as a legacy
column, annotated with its exact McNemar p so it cannot be read as a result.

Restricting to search-found positives
-------------------------------------
``--prov search`` computes everything using only poses the FFT search actually
returned, dropping the enumerated near-native candidates. That is the
end-to-end-relevant subset: a complex whose only positives were enumerated is
one the search failed on, and including them measures re-ranking of a set the
search would never offer.

Example
-------
    uv run python scripts/compare_conditions.py \
        --pool data/shards_pinder/test_feats_reachable.pt \
        --baseline-alpha 1.0 --baseline-rho 3.5 \
        --ckpt data/scaling/runs/N220_seed0/round0_ckpt.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from zdock.atomtypes import iface_ij
from zdock.evaluate import evaluate_ranking
from zdock.score import iface_score_matrix, psc_score_from_terms, SC_RHO

KS = (1, 5, 10, 50, 100)


def score_pool(d, alpha, iface, beta, clash):
    """``clash`` = (w_ss, w_sc, w_cc); see run_pinder_scaling.Params.

    Taking the three weights explicitly lets one code path score both the
    paper-constrained model (w_k = alpha*rho^k) and the relaxed one.
    """
    sc = d["sc"]
    if sc.ndim == 2:
        s_psc = alpha * sc[:, 0] - (sc[:, 1:4] * clash).sum(-1)
    else:
        s_psc = alpha * sc
    return (s_psc + (iface_score_matrix(iface) * d["T"]).sum(dim=(-2, -1))
            + beta * d["elec"])


def clash_from_ckpt(ck, alpha, rho):
    """Weights a checkpoint implies, whichever mode produced it."""
    if "clash_weights" in ck:
        return ck["clash_weights"].double()
    k = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
    return alpha * rho.pow(k)


def mann_whitney_auc(scores: torch.Tensor, pos: torch.Tensor) -> float:
    """P(random positive outranks random negative), midranks for ties."""
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64,
                                device=scores.device)
    uniq, inv, counts = torch.unique(scores, return_inverse=True, return_counts=True)
    if int((counts > 1).sum()):
        rs = torch.zeros(uniq.numel(), dtype=torch.float64, device=scores.device)
        rs.index_add_(0, inv, ranks)
        ranks = (rs / counts.to(torch.float64))[inv]
    u = float(ranks[pos].sum()) - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def control_aucs(d, thr, prov_filter):
    """Trivial rankers that need no parameters at all.

    A pool whose positives and negatives differ mainly in how much surface they
    bury can be separated by counting contacts, and then a "trained" model that
    reaches AUC 0.96 has demonstrated nothing about interface chemistry -- it
    has rediscovered burial, and usually less well than the counter. Measured on
    the 250-complex reachable pool: positives have a median of 34,856 contacts
    against 910 for the negatives (39x), and for 172 of 249 complexes NO
    negative falls inside the positives' contact-count range at all. So these
    controls are printed next to every comparison, not on request.
    """
    keep = torch.ones(d["sc"].shape[0], dtype=torch.bool)
    if prov_filter == "search":
        keep = d.get("prov", torch.zeros_like(keep, dtype=torch.int16)) == 0
    pos = d["dockq"][keep] >= thr
    T, sc = d["T"][keep].double(), d["sc"][keep].double()
    out = {"auc_contact_count": mann_whitney_auc(T.sum(dim=(-2, -1)), pos)}
    if sc.ndim == 2:
        out["auc_psc_pair_count"] = mann_whitney_auc(sc[:, 0], pos)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos and n_neg:
        cp, cn = T.sum(dim=(-2, -1))[pos], T.sum(dim=(-2, -1))[~pos]
        lo, hi = float(cp.min()), float(cp.max())
        out["frac_neg_in_pos_contact_range"] = float(
            ((cn >= lo) & (cn <= hi)).double().mean())
        out["contact_ratio_pos_over_neg"] = float(cp.median() / cn.median().clamp_min(1))
    return out


def per_complex(d, alpha, iface, beta, thr, prov_filter, clash):
    keep = torch.ones(d["sc"].shape[0], dtype=torch.bool)
    if prov_filter == "search":
        keep = d.get("prov", torch.zeros_like(keep, dtype=torch.int16)) == 0
    if int(keep.sum()) < 2:
        return None
    s = score_pool({k: (v[keep] if torch.is_tensor(v) and v.shape[:1] == keep.shape
                        else v) for k, v in d.items()}, alpha, iface, beta, clash)
    dockq, rmsd = d["dockq"][keep], d["rmsd"][keep]
    pos = dockq >= thr
    rep = evaluate_ranking(s, rmsd, dockq, ks=KS, rmsd_threshold=5.0,
                           dockq_threshold=thr)
    order = torch.argsort(s, descending=True)
    hit = pos[order].nonzero(as_tuple=True)[0]
    n = s.numel()
    row = {"name": d["name"], "n_poses": n, "n_pos": int(pos.sum()),
           "auc": mann_whitney_auc(s, pos),
           "first_hit_pct": ((int(hit[0]) + 1) if hit.numel() else n + 1) / n,
           "best_dockq_overall": float(dockq.max())}
    for k in KS:
        row[f"succ_dockq@{k}"] = int(rep.success_dockq[k])
        row[f"best_dockq@{k}"] = rep.best_dockq_at[k]
    return row


# ---- statistics (no scipy dependency) ------------------------------------
def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def wilcoxon_signed_rank(diff: list[float]) -> tuple[float, float]:
    """Two-sided p via the normal approximation with tie and zero handling.

    Returns ``(W, p)``. Exact for the sample sizes here is unnecessary: n is in
    the hundreds, where the normal approximation is accurate to <1e-3.
    """
    d = [x for x in diff if x != 0.0]
    n = len(d)
    if n < 6:
        return float("nan"), float("nan")
    mag = sorted((abs(x), i) for i, x in enumerate(d))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and mag[j + 1][0] == mag[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[mag[t][1]] = r
        i = j + 1
    w_pos = sum(ranks[i] for i in range(n) if d[i] > 0)
    w_neg = sum(ranks[i] for i in range(n) if d[i] < 0)
    w = min(w_pos, w_neg)
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd == 0:
        return w, float("nan")
    z = (abs(w - mu) - 0.5) / sd
    return w, 2.0 * _norm_sf(z)


def paired_t(diff: list[float]) -> float:
    """Two-sided paired t-test p-value.

    Uses scipy's t distribution when available; otherwise the normal
    approximation, which for n in the hundreds is within a factor of ~2 of the
    exact value in the far tail (measured: 3.7e-6 vs 6.7e-6 at n = 200). Only
    the Wilcoxon result is quoted as primary, so this is a cross-check either
    way — the per-complex differences are nowhere near normal.
    """
    n = len(diff)
    if n < 3:
        return float("nan")
    m = sum(diff) / n
    var = sum((x - m) ** 2 for x in diff) / (n - 1)
    if var <= 0:
        return float("nan")
    t = m / math.sqrt(var / n)
    try:
        from scipy import stats
        return float(2.0 * stats.t.sf(abs(t), df=n - 1))
    except ImportError:
        return 2.0 * _norm_sf(abs(t))


def mcnemar_exact(a: list[int], b: list[int]) -> tuple[int, int, float]:
    """Exact two-sided McNemar on paired 0/1 outcomes."""
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return b01, b10, min(1.0, p)


def bootstrap_ci(diff: list[float], n_boot: int = 10000, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    t = torch.tensor(diff, dtype=torch.float64)
    n = t.numel()
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    means = t[idx].mean(dim=1).sort().values
    return float(means[int(0.025 * n_boot)]), float(means[int(0.975 * n_boot)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="data/shards_pinder/test_feats_reachable.pt")
    ap.add_argument("--ckpt", required=True,
                    help="round*_ckpt.pt of the condition being tested")
    ap.add_argument("--baseline-alpha", type=float, default=1.0,
                    dest="base_alpha")
    ap.add_argument("--baseline-rho", type=float, default=SC_RHO, dest="base_rho")
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--dockq-threshold", type=float, default=0.23, dest="thr")
    ap.add_argument("--prov", choices=("all", "search"), default="all",
                    help="search: use only poses the FFT search returned")
    ap.add_argument("--out-dir", default="data/scaling/compare", dest="out_dir")
    args = ap.parse_args()

    blob = torch.load(args.pool, map_location="cpu", weights_only=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    beta = torch.tensor(args.beta, dtype=torch.float64)
    ba = torch.tensor(args.base_alpha, dtype=torch.float64)
    br = torch.tensor(args.base_rho, dtype=torch.float64)
    base = dict(alpha=ba, iface=iface_ij(dtype=torch.float64, flat=True),
                clash=clash_from_ckpt({}, ba, br))
    ta = ck["alpha"].double()
    tr = ck.get("rho", torch.tensor(args.base_rho)).double()
    trained = dict(alpha=ta, iface=ck["iface"].double(),
                   clash=clash_from_ckpt(ck, ta, tr))
    print(f"pool     : {args.pool}  ({len(blob)} complexes, prov={args.prov})")
    fmt = lambda c: "(" + ", ".join(f"{float(x):.2f}" for x in c) + ")"
    print(f"baseline : alpha={float(base['alpha']):.4f} "
          f"clash(ss,sc,cc)={fmt(base['clash'])}")
    print(f"trained  : alpha={float(trained['alpha']):.4f} "
          f"clash(ss,sc,cc)={fmt(trained['clash'])} "
          f"mode={ck.get('psc_mode','rho')} "
          f"||d_iface||={float((trained['iface']-base['iface']).norm()):.4f}")

    rows_b, rows_t, rows_c = [], [], []
    for d in blob:
        dd = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v)
              for k, v in d.items()}
        rb = per_complex(dd, base["alpha"], base["iface"], beta,
                         args.thr, args.prov, base["clash"])
        rt = per_complex(dd, trained["alpha"], trained["iface"], beta,
                         args.thr, args.prov, trained["clash"])
        if rb is None or rt is None:
            continue
        if math.isnan(rb["auc"]) or math.isnan(rt["auc"]):
            continue           # no positive or no negative -> AUC undefined
        rows_b.append(rb)
        rows_t.append(rt)
        rows_c.append(control_aucs(dd, args.thr, args.prov))
    n = len(rows_b)
    n_dropped = len(blob) - n
    print(f"\n{n} complexes usable for a paired test "
          f"({n_dropped} dropped: no positive or no negative under this filter)")

    out = {"pool": args.pool, "ckpt": args.ckpt, "prov": args.prov,
           "n_paired": n, "n_dropped": n_dropped, "metrics": {}}
    if n == 0:
        # Not an error, and an important reading in its own right: under
        # `--prov search` an empty set means the FFT search returned no
        # near-native pose for ANY complex, so there is nothing to re-rank.
        # That is the baseline condition measured on this pool (0 of 250).
        # Report it and stop rather than raising on `rows_b[0]`.
        print("\nNo complex has BOTH a positive and a negative under this "
              "filter, so no paired comparison is defined.")
        if args.prov == "search":
            print("With --prov search that means the search itself surfaced no "
                  "near-native pose anywhere; re-run once training has moved "
                  "the parameters enough for it to.")
        od = Path(args.out_dir)
        od.mkdir(parents=True, exist_ok=True)
        tag = (Path(args.ckpt).parent.name + "_" + Path(args.ckpt).stem
               + f"_{args.prov}")
        (od / f"{tag}.json").write_text(json.dumps(out, indent=1))
        print(f"wrote {od}/{tag}.json")
        return
    print(f"\n{'metric':<22}{'baseline':>11}{'trained':>11}{'delta':>11}"
          f"{'Wilcoxon p':>13}{'t p':>10}")
    for key, better in (("auc", "higher"), ("first_hit_pct", "lower"),
                        ("best_dockq@1", "higher"), ("best_dockq@10", "higher"),
                        ("best_dockq@100", "higher")):
        vb = [r[key] for r in rows_b]
        vt = [r[key] for r in rows_t]
        diff = [b - a for a, b in zip(vb, vt)]
        mb, mt = sum(vb) / n, sum(vt) / n
        _, pw = wilcoxon_signed_rank(diff)
        pt = paired_t(diff)
        lo, hi = bootstrap_ci(diff)
        print(f"{key + ' (' + better + ')':<22}{mb:11.4f}{mt:11.4f}{mt-mb:+11.4f}"
              f"{pw:13.3g}{pt:10.3g}")
        out["metrics"][key] = {"baseline": mb, "trained": mt, "delta": mt - mb,
                               "wilcoxon_p": pw, "paired_t_p": pt,
                               "boot_ci95": [lo, hi], "better": better}

    # --- parameter-free controls -----------------------------------------
    print(f"\n{'control (no parameters at all)':<42}{'mean AUC':>10}{'median':>9}")
    for key, lab in (("auc_contact_count", "  pure contact count  sum_ij n_ij"),
                     ("auc_psc_pair_count", "  PSC favourable pair count c_pair")):
        v = [r[key] for r in rows_c if key in r and not math.isnan(r[key])]
        if not v:
            continue
        v_sorted = sorted(v)
        m = sum(v) / len(v)
        print(f"{lab:<42}{m:>10.4f}{v_sorted[len(v)//2]:>9.4f}")
        out["metrics"][key] = {"mean": m, "median": v_sorted[len(v) // 2]}
    fr = [r["frac_neg_in_pos_contact_range"] for r in rows_c
          if "frac_neg_in_pos_contact_range" in r]
    if fr:
        n_disjoint = sum(1 for x in fr if x == 0.0)
        ratio = [r["contact_ratio_pos_over_neg"] for r in rows_c
                 if "contact_ratio_pos_over_neg" in r]
        print(f"  positives bury {sorted(ratio)[len(ratio)//2]:.1f}x more than "
              f"negatives (median); {n_disjoint}/{len(fr)} complexes have NO "
              f"negative in the positives' contact range")
        out["metrics"]["burial_separation"] = {
            "median_contact_ratio": sorted(ratio)[len(ratio) // 2],
            "n_complexes_fully_disjoint": n_disjoint, "n": len(fr)}
        if n_disjoint > 0.25 * len(fr):
            print("  >> WARNING: the two classes are largely separable by "
                  "burial alone, so this pool cannot show whether the learned "
                  "pair potential contributes anything beyond contact count.")

    print(f"\n{'success@K (legacy, floor-bound)':<32}{'base':>8}{'trained':>9}"
          f"{'0->1':>7}{'1->0':>7}{'McNemar p':>12}")
    for k in KS:
        a = [r[f"succ_dockq@{k}"] for r in rows_b]
        b = [r[f"succ_dockq@{k}"] for r in rows_t]
        b01, b10, p = mcnemar_exact(a, b)
        print(f"{'  K = ' + str(k):<32}{sum(a)/n*100:7.1f}%{sum(b)/n*100:8.1f}%"
              f"{b01:7d}{b10:7d}{p:12.3g}")
        out["metrics"][f"succ_dockq@{k}"] = {
            "baseline": sum(a) / n, "trained": sum(b) / n,
            "discordant_0to1": b01, "discordant_1to0": b10, "mcnemar_p": p}

    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    tag = Path(args.ckpt).parent.name + "_" + Path(args.ckpt).stem + f"_{args.prov}"
    (od / f"{tag}.json").write_text(json.dumps(out, indent=1))
    cols = list(rows_b[0].keys())
    with open(od / f"{tag}_per_complex.csv", "w") as fh:
        fh.write("condition," + ",".join(cols) + "\n")
        for cond, rows in (("baseline", rows_b), ("trained", rows_t)):
            for r in rows:
                fh.write(cond + "," + ",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nwrote {od}/{tag}.json and the per-complex table")


if __name__ == "__main__":
    main()
