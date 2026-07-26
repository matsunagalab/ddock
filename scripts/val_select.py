"""Model selection on the VALIDATION split, with a metric independent of the
training objective.

margin 0.5 and margin 0.0 optimise different objectives, so their validation
*losses* are not comparable. Codex's rule: re-score the same validation
complexes with a common, objective-independent readout (success@1, AUC,
best DockQ@1) and pick the loss there, not on TEST.

The round-0 pool cache is seed-independent, so one read serves every run; only
`split.json["val_ids"]` differs per seed.
"""
import glob, json, math, sys
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
from compare_conditions import (clash_from_ckpt, mann_whitney_auc, score_pool)
from zdock.atomtypes import iface_ij

THR, BETA = 0.23, torch.tensor(3.0, dtype=torch.float64)

pools = {}
for f in sorted(glob.glob("data/scaling/pool_cache/*_r0_*.pt")):
    for d in torch.load(f, map_location="cpu", weights_only=True)["pools"]:
        pools[d["name"]] = d
print(f"pool cache: {len(pools)} complexes", flush=True)


def metrics(ids, alpha, iface, clash):
    """`hit` covers EVERY complex with a search pose set.

    A complex whose search returned no near-native pose is a failure, not an
    undefined value: dropping it (as the AUC must) inflates success@1 and, on
    this validation split, leaves only 33 of 55 complexes and no resolution at
    all. AUC and best DockQ@1 stay on the restricted set where they are defined.
    """
    n = s1 = n_all = 0
    aucs, dq1 = [], []
    hit = {}
    for pid in ids:
        d = pools.get(pid)
        if d is None:
            continue
        dd = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v)
              for k, v in d.items()}
        keep = dd["prov"] == 0                       # search-derived poses only
        if int(keep.sum()) == 0:
            continue
        pos = dd["dockq"][keep] >= THR
        s = score_pool({k: (v[keep] if torch.is_tensor(v) and v.shape[:1] == keep.shape
                            else v) for k, v in dd.items()}, alpha, iface, BETA, clash)
        top = int(s.argmax())
        dq = float(dd["dockq"][keep][top])
        h = int(dq >= THR)
        n_all += 1
        s1 += h
        hit[pid] = h
        if int(pos.sum()) == 0 or int((~pos).sum()) == 0:
            continue
        n += 1
        aucs.append(mann_whitney_auc(s, pos))
        dq1.append(dq)
    return dict(n=n, n_all=n_all, success1=s1 / max(1, n_all),
                auc=sum(aucs) / max(1, len(aucs)),
                dq1=sum(dq1) / max(1, len(dq1)), hit=hit)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)


base_if = iface_ij(dtype=torch.float64, flat=True)
ba = torch.tensor(1.0, dtype=torch.float64)
base_clash = clash_from_ckpt({}, ba, torch.tensor(3.5, dtype=torch.float64))

rows = []
for m in ("full", "add", "sym"):
    for g in ("m5", "m0"):
        for s in (0, 1, 2):
            rd = Path(f"data/scaling/runs_{m}_{g}/N220_seed{s}")
            sp = json.loads((rd / "split.json").read_text())
            ids = sp["val_ids"]
            ck = torch.load(rd / "round0_ckpt.pt", map_location="cpu", weights_only=True)
            ta = ck["alpha"].double()
            tr = ck.get("rho", torch.tensor(3.5)).double()
            mb = metrics(ids, ba, base_if, base_clash)
            mt = metrics(ids, ta, ck["iface"].double(), clash_from_ckpt(ck, ta, tr))
            b2t = sum(1 for k in mt["hit"] if mt["hit"][k] and not mb["hit"][k])
            t2b = sum(1 for k in mt["hit"] if mb["hit"][k] and not mt["hit"][k])
            rows.append(dict(mode=m, margin=g, seed=s, n=mt["n"],
                             n_all=mt["n_all"],
                             s1_base=mb["success1"], s1=mt["success1"],
                             auc_base=mb["auc"], auc=mt["auc"],
                             dq1_base=mb["dq1"], dq1=mt["dq1"],
                             win=b2t, lose=t2b, p=mcnemar_exact(b2t, t2b)))
            r = rows[-1]
            print(f"{m:5s} {g} seed{s}: n_all={r['n_all']:3d} n_auc={r['n']:3d}"
                  f"  success@1 {r['s1_base']*100:5.1f}"
                  f" -> {r['s1']*100:5.1f}%  ({r['win']}勝{r['lose']}敗 p={r['p']:.3g})"
                  f"  AUC {r['auc_base']:.4f} -> {r['auc']:.4f}"
                  f"  bestDQ@1 {r['dq1_base']:.4f} -> {r['dq1']:.4f}", flush=True)

mean = lambda v: sum(v) / len(v)
print("\nvalidation, seed-averaged (n=3 optimizer seeds, NOT independent samples):")
print(f"{'mode':5s} {'mgn':4s} {'success@1 %':>13s} {'AUC':>9s} {'bestDQ@1':>10s}")
for m in ("full", "add", "sym"):
    for g in ("m5", "m0"):
        c = [r for r in rows if r["mode"] == m and r["margin"] == g]
        print(f"{m:5s} {g:4s} {mean([r['s1'] for r in c])*100:13.2f} "
              f"{mean([r['auc'] for r in c]):9.4f} {mean([r['dq1'] for r in c]):10.4f}")
json.dump(rows, open("data/scaling/val_selection.json", "w"), indent=1)
print("\nwrote data/scaling/val_selection.json")
