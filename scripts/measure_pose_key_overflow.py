"""How much did the overflowing pose key over-drop enumerated candidates?

The dedup in `generate_pool_reachable` removes an enumerated pose that the
search already returned. Under the broken packing it removed any enumerated
pose sharing a translation cell with a search pose of the same rotation
*parity*, because all 1944 rotation indices collapsed to two keys.

Search poses were never dropped -- the filter only runs on the enumerated side
-- so this can only have under-counted enumerated positives.
"""
import sys
import torch

sys.path.insert(0, "src")
from zdock.atomtypes import charge_score as default_charge_score, iface_ij
from zdock.prep_cache import load_prepared
from zdock.rotation_grid import hopf_quaternions, so3_geodesic_deg
from zdock.search import docking_search
from zdock.dataset import _rotate_batch

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dt = torch.float64
ids = [ln.strip() for ln in open("data/scaling/master_ids.txt") if ln.strip()]
excl = set(open("data/scaling/excluded_bad_geometry.txt").read().split())

n_done, tot_old, tot_new, tot_gen = 0, 0, 0, 0
for pid in ids:
    if pid in excl:
        continue
    prot_cpu = load_prepared("data/scaling/prep_cache", pid, dtype=dt)
    if prot_cpu is None:
        continue
    if prot_cpu.n_rec * prot_cpu.n_lig > 4_000_000:
        continue                      # keep this diagnostic off the OOM edge
    prot = prot_cpu.to(dev, dtype=dt)
    q = hopf_quaternions(3, device=dev, dtype=dt)
    q = q / q.norm(dim=-1, keepdim=True)
    try:
        res = docking_search(
            prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
            prot.rec_atomtype_id, prot.rec_charge_id,
            prot.lig_ref, prot.lig_radius, prot.lig_sasa,
            prot.lig_atomtype_id, prot.lig_charge_id, q,
            alpha=torch.tensor(1.0, device=dev, dtype=dt),
            iface_ij_flat=iface_ij(device=dev, dtype=dt, flat=True),
            beta=torch.tensor(3.0, device=dev, dtype=dt),
            charge_score_lut=default_charge_score(device=dev, dtype=dt),
            spacing=1.2, ntop=1500, rot_chunk_size=4)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache(); del prot, prot_cpu; continue
    neg_qi = res.quat_indices
    neg_cell = torch.round(res.translations / 1.2).to(torch.long)

    d = so3_geodesic_deg(q, prot.q_star.unsqueeze(0).to(device=dev, dtype=dt))[:, 0]
    near = torch.argsort(d)[:8]
    off = torch.arange(-1, 2, device=dev, dtype=torch.long)
    oz, oy, ox = torch.meshgrid(off, off, off, indexing="ij")
    off3 = torch.stack([ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)], -1)
    tc = torch.round(prot.t_star / 1.2).to(torch.long)
    pos_cell = (tc.unsqueeze(0) + off3).repeat(8, 1)
    pos_qi = near.repeat_interleave(off3.shape[0])

    def old_key(qi, cell):
        big = 1 << 21
        c = cell + (big // 2)
        return ((qi.to(torch.int64) * big + c[:, 0]) * big + c[:, 1]) * big + c[:, 2]

    seen_old = set(old_key(neg_qi, neg_cell).tolist())
    kept_old = sum(1 for x in old_key(pos_qi, pos_cell).tolist() if x not in seen_old)
    seen_new = set(map(tuple, torch.cat(
        [neg_qi.reshape(-1, 1), neg_cell], 1).tolist()))
    kept_new = sum(1 for x in torch.cat([pos_qi.reshape(-1, 1), pos_cell], 1).tolist()
                   if tuple(x) not in seen_new)
    tot_gen += pos_qi.numel(); tot_old += kept_old; tot_new += kept_new
    n_done += 1
    print(f"{pid[:40]:<42} generated {pos_qi.numel():4d}  "
          f"kept(old) {kept_old:4d}  kept(new) {kept_new:4d}  "
          f"over-dropped {kept_new - kept_old:4d}", flush=True)
    del prot, prot_cpu
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    if n_done >= 12:
        break

print(f"\n{n_done} complexes: generated {tot_gen}, kept(old) {tot_old}, "
      f"kept(new) {tot_new}")
print(f"enumerated candidates wrongly dropped: {tot_new - tot_old} "
      f"({100.0 * (tot_new - tot_old) / max(1, tot_gen):.2f}% of generated)")
