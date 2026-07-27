"""Do the trained parameters rank the search poses the same way on both pools?

The tensor audit found the pose set and order identical and sc/T/rmsd/DockQ
bit-identical, with ELEC differing by <=3.6e-7 -- float32 rounding from mining
the two caches on different GPUs, not a change of content. What matters is
whether that rounding moves any decision, so this scores both pools with the
checkpoint the report's headline numbers came from and compares the outcome.
"""
import glob, sys
import torch

sys.path.insert(0, "scripts")
from compare_conditions import clash_from_ckpt, mann_whitney_auc, score_pool
from zdock.atomtypes import iface_ij

THR = 0.23
BETA = torch.tensor(3.0, dtype=torch.float64)


def load(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        for d in torch.load(f, map_location="cpu", weights_only=True)["pools"]:
            out[d["name"]] = d
    return out


old = load('data/scaling/pool_cache/n220_r0_*_bg1.*of3.pt')
new = load('data/scaling/pool_cache/n220_r0_*_bg1_pk2.*of5.pt')
shared = sorted(set(old) & set(new))
print(f"{len(shared)} complexes in both caches")

ck = torch.load("data/scaling/runs_full_m5/N220_seed0/round0_ckpt.pt",
                map_location="cpu", weights_only=True)
ta = ck["alpha"].double()
cond = {
    "baseline": (torch.tensor(1.0, dtype=torch.float64),
                 iface_ij(dtype=torch.float64, flat=True),
                 clash_from_ckpt({}, torch.tensor(1.0, dtype=torch.float64),
                                 torch.tensor(3.5, dtype=torch.float64))),
    "trained": (ta, ck["iface"].double(),
                clash_from_ckpt(ck, ta, torch.tensor(3.5, dtype=torch.float64))),
}

for label, (alpha, iface, clash) in cond.items():
    n_top1_same = n_rank_same = n_auc_same = n = 0
    worst_auc = worst_score = 0.0
    for name in shared:
        rows = {}
        for tag, blob in (("old", old), ("new", new)):
            d = blob[name]
            keep = (d["prov"] == 0).nonzero(as_tuple=True)[0]
            dd = {k: (v[keep].double() if torch.is_tensor(v) and v.is_floating_point()
                      else v[keep] if torch.is_tensor(v) else v)
                  for k, v in d.items() if k != "name"}
            rows[tag] = (score_pool(dd, alpha, iface, BETA, clash), dd["dockq"])
        s_o, dq_o = rows["old"]
        s_n, dq_n = rows["new"]
        if s_o.shape != s_n.shape:
            continue
        n += 1
        worst_score = max(worst_score, float((s_o - s_n).abs().max()))
        n_top1_same += int(s_o.argmax()) == int(s_n.argmax())
        n_rank_same += bool(torch.equal(torch.argsort(s_o, descending=True),
                                        torch.argsort(s_n, descending=True)))
        pos = dq_o >= THR
        if int(pos.sum()) and int((~pos).sum()):
            a_o = mann_whitney_auc(s_o, pos)
            a_n = mann_whitney_auc(s_n, dq_n >= THR)
            worst_auc = max(worst_auc, abs(a_o - a_n))
            n_auc_same += abs(a_o - a_n) < 1e-12
    print(f"\n[{label}] over {n} complexes, search-derived poses only")
    print(f"  identical top-1 pose        : {n_top1_same}/{n}")
    print(f"  identical FULL ranking      : {n_rank_same}/{n}")
    print(f"  identical AUC (<1e-12)      : {n_auc_same}/{n}")
    print(f"  max |score difference|      : {worst_score:.3e}")
    print(f"  max |AUC difference|        : {worst_auc:.3e}")
