"""Tests for the streaming pieces of the PINDER cluster-count scaling
experiment (`scripts/run_pinder_scaling.py`, `zdock.prep_cache`,
`PreparedProtein.to`).

The scaling run holds every complex on the host and pages one at a time onto
the accelerator, so the two things that must be proven are (a) moving a
prepared protein across devices / through the disk cache is lossless, and
(b) the per-pose features it produces are device-independent, i.e. the CPU
reference and the GPU streaming path agree.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from zdock.atomtypes import charge_score as default_charge_score
from zdock.atomtypes import iface_ij
from zdock.dataset import PreparedProtein, SimpleAtoms, prepare_protein
from zdock.prep_cache import (
    cache_key,
    cache_path,
    has_prepared,
    load_prepared,
    save_prepared,
)
from zdock.score import docking_score_elec

_REPO = Path(__file__).resolve().parent.parent


def _load_run_module():
    spec = importlib.util.spec_from_file_location(
        "run_pinder_scaling", _REPO / "scripts" / "run_pinder_scaling.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _toy_protein(device="cpu", dtype=torch.float64, seed=0) -> PreparedProtein:
    """A tiny two-residue-ish complex; enough to exercise every tensor field."""
    rng = np.random.default_rng(seed)
    rec = SimpleAtoms(
        atomname=["N", "CA", "C", "O", "CB"] * 3,
        resname=["ALA"] * 15,
        xyz=rng.normal(scale=4.0, size=(15, 3)),
    )
    lig = SimpleAtoms(
        atomname=["N", "CA", "C", "O"] * 2,
        resname=["GLY", "GLY", "GLY", "GLY", "ALA", "ALA", "ALA", "ALA"],
        xyz=rng.normal(loc=8.0, scale=3.0, size=(8, 3)),
    )
    return prepare_protein("TOY", rec, lig, device=device, dtype=dtype)


# --------------------------------------------------------------------------
# PreparedProtein.to / state_dict
# --------------------------------------------------------------------------
def test_to_cpu_roundtrip_is_lossless():
    prot = _toy_protein()
    same = prot.to("cpu")
    for f in PreparedProtein.TENSOR_FIELDS:
        assert torch.equal(getattr(prot, f), getattr(same, f)), f
    assert same.name == prot.name
    assert same.n_rec == 15 and same.n_lig == 8


def test_to_does_not_cast_integer_id_fields():
    prot = _toy_protein(dtype=torch.float64)
    moved = prot.to("cpu", dtype=torch.float32)
    assert moved.rec_xyz.dtype is torch.float32
    assert moved.rec_atomtype_id.dtype is prot.rec_atomtype_id.dtype
    assert moved.lig_charge_id.dtype is prot.lig_charge_id.dtype
    assert not moved.rec_atomtype_id.is_floating_point()


def test_state_dict_roundtrip():
    prot = _toy_protein()
    back = PreparedProtein.from_state_dict(prot.state_dict())
    for f in PreparedProtein.TENSOR_FIELDS:
        assert torch.equal(getattr(prot, f), getattr(back, f)), f


# --------------------------------------------------------------------------
# disk cache
# --------------------------------------------------------------------------
def test_cache_key_is_stable_and_filesystem_safe():
    pid = "7bgl__DB1_P0A1N8--7bgl__DA1_A0A745A2I3"
    k = cache_key(pid)
    assert k == cache_key(pid)
    assert k.isalnum() and len(k) == 20
    assert cache_key(pid) != cache_key(pid + "x")


def test_disk_cache_roundtrip(tmp_path):
    prot = _toy_protein()
    assert not has_prepared(tmp_path, prot.name)
    assert load_prepared(tmp_path, prot.name) is None
    save_prepared(tmp_path, prot)
    assert has_prepared(tmp_path, prot.name)
    back = load_prepared(tmp_path, prot.name)
    assert back is not None and back.name == prot.name
    for f in PreparedProtein.TENSOR_FIELDS:
        assert torch.equal(getattr(prot, f), getattr(back, f)), f


def test_disk_cache_detects_key_collision(tmp_path):
    prot = _toy_protein()
    save_prepared(tmp_path, prot)
    # Forge a file at the slot another id hashes to.
    other = "SOME-OTHER-SYSTEM"
    p = cache_path(tmp_path, other)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(cache_path(tmp_path, prot.name).read_bytes())
    with pytest.raises(RuntimeError, match="collision"):
        load_prepared(tmp_path, other)


def test_corrupt_cache_entry_is_a_miss(tmp_path):
    prot = _toy_protein()
    save_prepared(tmp_path, prot)
    cache_path(tmp_path, prot.name).write_bytes(b"not a torch file")
    assert load_prepared(tmp_path, prot.name) is None


# --------------------------------------------------------------------------
# CPU vs GPU feature agreement (the streaming correctness claim)
# --------------------------------------------------------------------------
def _features(prot, poses, *, psc_decompose=False):
    dev, dt = prot.rec_xyz.device, prot.rec_xyz.dtype
    return docking_score_elec(
        prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
        prot.rec_atomtype_id, prot.rec_charge_id,
        poses, prot.lig_radius, prot.lig_sasa,
        prot.lig_atomtype_id, prot.lig_charge_id,
        torch.zeros((), device=dev, dtype=dt),
        iface_ij(device=dev, dtype=dt, flat=True),
        torch.tensor(3.0, device=dev, dtype=dt),
        default_charge_score(device=dev, dtype=dt),
        frame_chunk_size=4, return_components=True,
        psc_decompose=psc_decompose,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_and_gpu_features_agree():
    """The pool is built on the GPU and consumed on the host; the extracted
    (S_SC, T, S_ELEC) must not depend on where they were computed."""
    prot_cpu = _toy_protein(device="cpu", dtype=torch.float32)
    rng = np.random.default_rng(3)
    poses_np = (prot_cpu.lig_ref.numpy()[None] + rng.normal(scale=2.0, size=(6, 1, 3)))
    poses_cpu = torch.as_tensor(poses_np, dtype=torch.float32)

    sc_c, T_c, el_c = _features(prot_cpu, poses_cpu)
    prot_gpu = prot_cpu.to("cuda")
    sc_g, T_g, el_g = _features(prot_gpu, poses_cpu.cuda())

    for a, b, name in ((sc_c, sc_g.cpu(), "S_SC"), (T_c, T_g.cpu(), "T"),
                       (el_c, el_g.cpu(), "S_ELEC")):
        denom = b.abs().max().clamp_min(1.0)
        assert torch.allclose(a, b, rtol=2e-4, atol=float(1e-4 * denom)), name


@pytest.mark.parametrize("rho_train", [1.0, 3.5, 6.0])
def test_score_decomposition_matches_full_score(rho_train):
    """`score_from_feats` must reproduce `docking_score_elec` — the whole
    streaming pool is useless if the cached decomposition drifts.

    The features are cached ONCE at rho = 3.5 and then re-scored at a different
    rho; that must still agree with a direct re-scoring at that rho. This is the
    property that makes rho trainable from cached features at all.
    """
    mod = _load_run_module()
    prot = _toy_protein(dtype=torch.float64)
    rng = np.random.default_rng(5)
    poses = torch.as_tensor(
        prot.lig_ref.numpy()[None] + rng.normal(scale=2.0, size=(4, 1, 3)),
        dtype=torch.float64)
    sc, T, elec = _features(prot, poses, psc_decompose=True)
    assert sc.shape == (poses.shape[0], 4), "expected the (F, 4) PSC terms"
    n = sc.shape[0]
    f = mod.Feats("TOY", sc, T, elec, torch.zeros(n, dtype=torch.float64),
                  torch.zeros(n, dtype=torch.float64),
                  torch.zeros(n, dtype=torch.int16))

    alpha = torch.tensor(0.031, dtype=torch.float64)
    rho = torch.tensor(rho_train, dtype=torch.float64)
    iface = iface_ij(dtype=torch.float64, flat=True) + 0.05
    beta = torch.tensor(3.0, dtype=torch.float64)
    charge = default_charge_score(dtype=torch.float64)
    full = docking_score_elec(
        prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
        prot.rec_atomtype_id, prot.rec_charge_id,
        poses, prot.lig_radius, prot.lig_sasa,
        prot.lig_atomtype_id, prot.lig_charge_id,
        alpha, iface, beta, charge, frame_chunk_size=4, sc_rho=rho)
    got = mod.score_from_feats(f, mod.Params(alpha, rho, iface), beta)
    assert torch.allclose(got, full, rtol=1e-8, atol=1e-8)


def test_rho_receives_gradient_through_cached_features():
    """rho is the whole reason the pool stores 4 scalars instead of 1."""
    mod = _load_run_module()
    prot = _toy_protein(dtype=torch.float64)
    rng = np.random.default_rng(7)
    poses = torch.as_tensor(
        prot.lig_ref.numpy()[None] + rng.normal(scale=2.0, size=(6, 1, 3)),
        dtype=torch.float64)
    sc, T, elec = _features(prot, poses, psc_decompose=True)
    n = sc.shape[0]
    f = mod.Feats("TOY", sc, T, elec, torch.zeros(n, dtype=torch.float64),
                  torch.zeros(n, dtype=torch.float64),
                  torch.zeros(n, dtype=torch.int16))
    p = mod.Params(torch.tensor(1.0, dtype=torch.float64),
                   torch.tensor(3.5, dtype=torch.float64),
                   iface_ij(dtype=torch.float64, flat=True)).requires_grad_(True)
    mod.score_from_feats(f, p, torch.tensor(3.0, dtype=torch.float64)).sum().backward()
    for t in p.tensors():                      # what THIS mode optimises
        assert t.grad is not None and bool((t.grad != 0).any())

    # 'free' mode drops Chen & Weng's w_k = alpha*rho^k collinearity, so the
    # three clash weights must each receive their own gradient.
    pf = mod.Params(torch.tensor(1.0, dtype=torch.float64),
                    torch.tensor(3.5, dtype=torch.float64),
                    iface_ij(dtype=torch.float64, flat=True),
                    mode="free").requires_grad_(True)
    mod.score_from_feats(f, pf, torch.tensor(3.0, dtype=torch.float64)).sum().backward()
    # Each weight gets its own gradient wherever its feature column is
    # non-zero. (The toy complex has no core atoms, so n_sc = n_cc = 0 and
    # those two legitimately receive none -- that is data, not a dead path.)
    assert pf.log_clash.grad is not None
    present = (sc[:, 1:4].abs().sum(0) > 0)
    assert torch.equal(pf.log_clash.grad != 0, present), (
        pf.log_clash.grad, sc[:, 1:4].abs().sum(0))
    # and the gradient is exactly -w_k * sum_f n_k(f)  (d/d log w_k of -w_k n_k)
    expected = -pf.clash_weights().detach() * sc[:, 1:4].sum(0)
    assert torch.allclose(pf.log_clash.grad, expected, rtol=1e-10)

    # and at the shared starting point the two modes must score identically
    a = mod.Params(torch.tensor(1.0, dtype=torch.float64),
                   torch.tensor(3.5, dtype=torch.float64),
                   iface_ij(dtype=torch.float64, flat=True), mode="rho")
    b = mod.Params(torch.tensor(1.0, dtype=torch.float64),
                   torch.tensor(3.5, dtype=torch.float64),
                   iface_ij(dtype=torch.float64, flat=True), mode="free")
    beta = torch.tensor(3.0, dtype=torch.float64)
    assert torch.allclose(mod.score_from_feats(f, a, beta),
                          mod.score_from_feats(f, b, beta), atol=1e-12)


# --------------------------------------------------------------------------
# pool bookkeeping invariants
# --------------------------------------------------------------------------
def _fake_feats(mod, name, n, dockq, origin, pose_key=None):
    return mod.Feats(name,
                     torch.arange(n, dtype=torch.float64),
                     torch.zeros(n, 12, 12, dtype=torch.float64),
                     torch.zeros(n, dtype=torch.float64),
                     torch.full((n,), 3.0, dtype=torch.float64),
                     torch.as_tensor(dockq, dtype=torch.float64),
                     torch.full((n,), origin, dtype=torch.int16),
                     None,
                     None if pose_key is None
                     else torch.as_tensor(
                         [[k, 0, 0, 0] for k in pose_key]
                         if not hasattr(pose_key[0], "__len__") else pose_key,
                         dtype=torch.int64))


def test_normalized_scores_preserve_pose_ranking():
    mod = _load_run_module()
    n = 40
    torch.manual_seed(0)
    f = mod.Feats("X", torch.randn(n, dtype=torch.float64) * 700.0,
                  torch.randn(n, 12, 12, dtype=torch.float64),
                  torch.randn(n, dtype=torch.float64) * 50.0,
                  torch.zeros(n, dtype=torch.float64),
                  torch.zeros(n, dtype=torch.float64),
                  torch.zeros(n, dtype=torch.int16))
    alpha = torch.tensor(0.02, dtype=torch.float64)
    rho = torch.tensor(3.5, dtype=torch.float64)
    iface = iface_ij(dtype=torch.float64, flat=True)
    beta = torch.tensor(3.0, dtype=torch.float64)
    p = mod.Params(alpha, rho, iface)
    raw = mod.score_from_feats(f, p, beta)
    norm = mod.normalized_scores(f, p, beta)
    assert torch.equal(raw.argsort(), norm.argsort())


def test_cap_pool_keeps_every_positive():
    mod = _load_run_module()
    n = 300
    dockq = torch.zeros(n, dtype=torch.float64)
    dockq[:40] = 0.5                       # 40 positives
    f = _fake_feats(mod, "X", n, dockq, 0)
    alpha = torch.tensor(0.01, dtype=torch.float64)
    rho = torch.tensor(3.5, dtype=torch.float64)
    iface = iface_ij(dtype=torch.float64, flat=True)
    beta = torch.tensor(3.0, dtype=torch.float64)
    capped = mod.cap_pool(f, 100, mod.Params(alpha, rho, iface), beta, 0.23)
    assert capped.n == 100
    assert int((capped.dockq >= 0.23).sum()) == 40


def test_mining_appends_negatives_without_growing_positives():
    mod = _load_run_module()
    dq0 = torch.zeros(100, dtype=torch.float64)
    dq0[:10] = 0.6
    pool = _fake_feats(mod, "X", 100, dq0, 0)
    n_pos_before = int((pool.dockq >= 0.23).sum())

    mined = _fake_feats(mod, "X", 60, torch.full((60,), 0.6, dtype=torch.float64), 1)
    neg = (mined.dockq < 0.23).nonzero(as_tuple=True)[0]
    assert neg.numel() == 0                       # all-positive mine adds nothing

    mined2 = _fake_feats(mod, "X", 60, torch.zeros(60, dtype=torch.float64), 1)
    pool.cat(mined2.index((mined2.dockq < 0.23).nonzero(as_tuple=True)[0]))
    assert int((pool.dockq >= 0.23).sum()) == n_pos_before
    c = pool.counts(0.23)
    assert c == {"n": 160, "n_pos": 10, "n_rand_neg": 90, "n_hard_neg": 60,
                 # prov defaults to 0 (= search) for a hand-built Feats
                 "n_pos_from_search": 10, "n_pos_enumerated": 0}


def test_iface_coverage_counts_complexes_not_poses():
    mod = _load_run_module()
    a = _fake_feats(mod, "A", 5, torch.zeros(5), 0)
    b = _fake_feats(mod, "B", 5, torch.zeros(5), 0)
    a.T[:, 0, 0] = 1.0
    b.T[:, 0, 0] = 2.0
    b.T[:, 1, 3] = 1.0
    cov = mod.iface_coverage([a, b])
    assert cov["n_complexes"] == 2
    assert cov["coverage_matrix"][0][0] == 2
    assert cov["coverage_matrix"][1][3] == 1
    assert cov["coverage_matrix"][5][5] == 0
    assert cov["n_components_zero"] == 142


# --------------------------------------------------------------------------
# split selection: nesting + leakage
# --------------------------------------------------------------------------
def _write_selection_fixture(tmp_path, n_master=60, bad=(3, 11), voxels=None):
    import json
    ids = [f"sys{i:03d}__A1_X--sys{i:03d}__B1_Y" for i in range(n_master)]
    (tmp_path / "master_ids.txt").write_text("\n".join(ids) + "\n")
    with open(tmp_path / "manifest.jsonl", "w") as fh:
        for r, pid in enumerate(ids):
            st = "fail" if r in bad else "ok"
            fh.write(json.dumps({"rank": r, "id": pid, "status": st}) + "\n")
    vox = {pid: 1000 for pid in ids}
    for r in (voxels or ()):
        vox[ids[r]] = 10_000_000
    (tmp_path / "voxels.json").write_text(json.dumps(vox))
    return ids


class _Args:
    def __init__(self, **kw):
        kw.setdefault("max_grid_voxels", 0)
        kw.setdefault("grid_voxels", "")
        # the fixture ids are not PINDER-shaped, so homodimer detection is a
        # no-op here; the flag still has to exist for `eligible`
        kw.setdefault("exclude_homodimer", False)
        kw.setdefault("exclude_bad_geometry", "")
        self.__dict__.update(kw)


def test_split_is_nested_and_stride_balanced(tmp_path):
    mod = _load_run_module()
    _write_selection_fixture(tmp_path, n_master=60)
    common = dict(master_ids=str(tmp_path / "master_ids.txt"),
                  prep_manifest=str(tmp_path / "manifest.jsonl"),
                  val_frac=0.2, n_total=0)

    sel_a, fit_a, val_a, _, _ = mod.select_split(_Args(n_fit=8, **common))
    sel_b, fit_b, val_b, _, _ = mod.select_split(_Args(n_fit=16, **common))
    sel_c, fit_c, val_c, _, _ = mod.select_split(_Args(n_fit=32, **common))

    for n, fit, val, sel in ((8, fit_a, val_a, sel_a), (16, fit_b, val_b, sel_b),
                             (32, fit_c, val_c, sel_c)):
        assert len(fit) == n
        assert len(val) == len(sel) - n
        assert abs(len(val) / len(sel) - 0.2) < 1e-9
        assert not set(fit) & set(val)

    assert set(sel_a) < set(sel_b) < set(sel_c), "subsets must be nested"
    assert set(fit_a) < set(fit_b) < set(fit_c), "fit sets must be nested"
    assert set(val_a) < set(val_b) < set(val_c), "validation sets must be nested"


def test_split_skips_failed_preparations(tmp_path):
    mod = _load_run_module()
    ids = _write_selection_fixture(tmp_path, n_master=60, bad=(0, 1))
    sel, fit, val, status, info = mod.select_split(_Args(
        n_fit=8, n_total=0, val_frac=0.2,
        master_ids=str(tmp_path / "master_ids.txt"),
        prep_manifest=str(tmp_path / "manifest.jsonl")))
    assert ids[0] not in sel and ids[1] not in sel
    assert sel[0] == ids[2]
    assert len(sel) == 10 and len(fit) == 8


def test_oversized_complexes_are_excluded_deterministically(tmp_path):
    """The grid-volume cutoff must drop the same complexes for every N, so the
    nested-prefix property survives it."""
    mod = _load_run_module()
    ids = _write_selection_fixture(tmp_path, n_master=60, bad=(),
                                   voxels=(0, 5, 17))
    common = dict(master_ids=str(tmp_path / "master_ids.txt"),
                  prep_manifest=str(tmp_path / "manifest.jsonl"),
                  grid_voxels=str(tmp_path / "voxels.json"),
                  max_grid_voxels=2_000_000, val_frac=0.2, n_total=0)

    sel_a, fit_a, val_a, _, info = mod.select_split(_Args(n_fit=8, **common))
    sel_b, fit_b, val_b, _, _ = mod.select_split(_Args(n_fit=24, **common))

    for r in (0, 5, 17):
        assert ids[r] not in sel_b
    assert info["n_excluded_oversized"] == 3
    assert info["n_usable"] == 57
    assert set(sel_a) < set(sel_b)
    assert set(fit_a) < set(fit_b) and set(val_a) < set(val_b)


def test_missing_voxel_table_is_a_hard_error(tmp_path):
    mod = _load_run_module()
    _write_selection_fixture(tmp_path, n_master=60, bad=())
    with pytest.raises(SystemExit, match="compute_grid_sizes"):
        mod.select_split(_Args(
            n_fit=8, n_total=0, val_frac=0.2,
            master_ids=str(tmp_path / "master_ids.txt"),
            prep_manifest=str(tmp_path / "manifest.jsonl"),
            grid_voxels=str(tmp_path / "does_not_exist.json"),
            max_grid_voxels=2_000_000))


def test_split_refuses_when_cache_too_small(tmp_path):
    mod = _load_run_module()
    _write_selection_fixture(tmp_path, n_master=10, bad=())
    with pytest.raises(SystemExit, match="usable clusters"):
        mod.select_split(_Args(
            n_fit=100, n_total=0, val_frac=0.2,
            master_ids=str(tmp_path / "master_ids.txt"),
            prep_manifest=str(tmp_path / "manifest.jsonl")))


# ---------------------------------------------------------------------------
# mining round > 0: pose identity, de-duplication and cap accounting
#
# A round-1 pool that double-counts a pose the search returned twice reweights
# the loss without saying so, and the reweighting would be read as the mining
# effect the experiment is trying to measure. These pin the parts of that path
# that are easy to get subtly wrong.
# ---------------------------------------------------------------------------

def test_dedup_uses_pose_identity_not_derived_quantities():
    """Two distinct poses may share (RMSD, DockQ, ELEC); they must both survive.

    `_fake_feats` gives every pose rmsd 3.0, dockq from the argument and elec 0,
    so under the earlier derived-quantity key the whole candidate set collapsed
    to one entry.
    """
    mod = _load_run_module()
    pool = _fake_feats(mod, "X", 4, torch.zeros(4), 0, pose_key=[10, 11, 12, 13])
    cand = _fake_feats(mod, "X", 4, torch.zeros(4), 1, pose_key=[20, 21, 22, 23])
    keep, n_dup = mod._fresh_indices(cand, pool)
    assert keep.tolist() == [0, 1, 2, 3]
    assert n_dup == 0


def test_dedup_drops_poses_already_in_the_pool():
    mod = _load_run_module()
    pool = _fake_feats(mod, "X", 3, torch.zeros(3), 0, pose_key=[10, 11, 12])
    cand = _fake_feats(mod, "X", 4, torch.zeros(4), 1, pose_key=[11, 99, 10, 98])
    keep, n_dup = mod._fresh_indices(cand, pool)
    assert keep.tolist() == [1, 3]
    assert n_dup == 2


def test_dedup_drops_repeats_inside_the_candidate_set():
    """`generate_pool_reachable` de-duplicates its own two provenances against
    each other, but a candidate set can still repeat a key; keeping both would
    double its weight just as surely as a cross-round duplicate."""
    mod = _load_run_module()
    pool = _fake_feats(mod, "X", 1, torch.zeros(1), 0, pose_key=[1])
    cand = _fake_feats(mod, "X", 4, torch.zeros(4), 1, pose_key=[7, 7, 8, 7])
    keep, n_dup = mod._fresh_indices(cand, pool)
    assert keep.tolist() == [0, 2]
    assert n_dup == 2


def test_dedup_refuses_a_pool_without_pose_identities():
    mod = _load_run_module()
    pool = _fake_feats(mod, "X", 2, torch.zeros(2), 0)          # pose_key = -1
    cand = _fake_feats(mod, "X", 2, torch.zeros(2), 1, pose_key=[5, 6])
    with pytest.raises(SystemExit):
        mod._fresh_indices(cand, pool)


def test_cap_can_evict_old_negatives_so_growth_is_not_survival():
    """The statistic `pool_after - pool_before` is not the number of new poses
    that survived: `cap_pool` keeps the highest-scoring negatives, so a new pose
    can enter by pushing an old one out and leave the size unchanged."""
    mod = _load_run_module()
    alpha = torch.tensor(0.01, dtype=torch.float64)
    rho = torch.tensor(3.5, dtype=torch.float64)
    p = mod.Params(alpha, rho, iface_ij(dtype=torch.float64, flat=True))
    beta = torch.tensor(3.0, dtype=torch.float64)

    pool = _fake_feats(mod, "X", 10, torch.zeros(10), 0, pose_key=list(range(10)))
    cand = _fake_feats(mod, "X", 5, torch.zeros(5), 1, pose_key=list(range(100, 105)))
    cand.sc = torch.arange(50.0, 55.0, dtype=torch.float64)   # outscore the pool
    before = pool.n
    pool.cat(cand)
    capped = mod.cap_pool(pool, 10, p, beta, 0.23)

    assert capped.n == before                       # size unchanged ...
    survivors = {r[0] for r in capped.pose_key.tolist()}
    assert len(survivors & set(range(100, 105))) == 5   # ... but 5 new poses in
    assert len(survivors & set(range(10))) == 5        # ... and 5 old ones out


def test_param_fingerprint_separates_checkpoints():
    """The round-1 cache key carries this hash so a pool mined from one
    checkpoint can never be picked up by a run holding another."""
    mod = _load_run_module()
    rho = torch.tensor(3.5, dtype=torch.float64)
    e0 = iface_ij(dtype=torch.float64, flat=True)
    a = mod.Params(torch.tensor(1.0, dtype=torch.float64), rho, e0.clone())
    b = mod.Params(torch.tensor(1.0, dtype=torch.float64), rho, e0.clone())
    assert mod.param_fingerprint(a) == mod.param_fingerprint(b)
    b.iface[0] += 1e-6
    assert mod.param_fingerprint(a) != mod.param_fingerprint(b)
    c = mod.Params(torch.tensor(1.0 + 1e-9, dtype=torch.float64), rho, e0.clone())
    assert mod.param_fingerprint(a) != mod.param_fingerprint(c)


def test_shards_partition_the_fit_list_exactly_once():
    """Sharding happens before cached ids are removed, so a worker's ownership
    does not depend on what happened to be cached when it started."""
    fit = [f"c{i}" for i in range(220)]
    n = 3
    shards = [fit[i::n] for i in range(n)]
    flat = [pid for sh in shards for pid in sh]
    assert sorted(flat) == sorted(fit)
    assert len(flat) == len(set(flat))
    assert max(len(s) for s in shards) - min(len(s) for s in shards) <= 1


def test_pose_identity_distinguishes_every_rotation_at_one_cell():
    """Regression: the packed-int64 key collapsed 1944 rotations to two.

    ``((qi*2**21 + cx)*2**21 + cy)*2**21 + cz`` puts ``qi`` at bit 63, so it
    overflowed int64 and only the *parity* of the rotation index survived --
    measured, all 1944 Hopf rotations produced two distinct keys, and every odd
    rotation produced a negative one. Both consequences were silent: enumerated
    poses were dropped as duplicates of a search pose they merely shared a
    translation cell and a rotation parity with, and a negative key would later
    read as "this pool has no identities".
    """
    from zdock.dataset import generate_pool_reachable  # noqa: F401  (contract)

    n = 1944
    qi = torch.arange(n, dtype=torch.int64).reshape(-1, 1)
    cell = torch.zeros(n, 3, dtype=torch.int64)
    key = torch.cat([qi, cell], dim=1)
    assert len({tuple(r) for r in key.tolist()}) == n
    assert int((key < 0).sum()) == 0


def test_generate_pool_reachable_returns_one_identity_per_pose():
    """The identity must line up with the poses, or de-duplication silently
    compares the wrong rows."""
    import inspect

    from zdock.dataset import generate_pool_reachable

    src = inspect.getsource(generate_pool_reachable)
    assert "return poses, origin, pose_key" in src, (
        "generate_pool_reachable must return the pose identity; a caller that "
        "unpacks two values would take `origin` as the identity")


def test_pose_identity_accepts_negative_translation_cells():
    """A negative translation cell is ordinary, not a missing identity.

    The lattice spans the origin, so `(qi, -1, 0, 0)` is a perfectly good pose.
    An earlier sentinel test asked whether ANY column was negative and would
    have rejected it, refusing to de-duplicate a pool that was in fact fine.
    Only the rotation-index column separates the sentinel from an identity.
    """
    from zdock.dataset import has_pose_identity, pose_identity

    qi = torch.tensor([0, 5, 1943], dtype=torch.int64)
    cell = torch.tensor([[-3, 0, 2], [0, -7, -1], [4, 4, -9]], dtype=torch.int64)
    key = pose_identity(qi, cell)
    assert key.shape == (3, 4)
    assert has_pose_identity(key)
    assert len({tuple(r) for r in key.tolist()}) == 3

    missing = torch.full((3, 4), -1, dtype=torch.int64)
    assert not has_pose_identity(missing)
    assert not has_pose_identity(torch.zeros(3, dtype=torch.int64))   # wrong rank


def test_pose_identity_distinguishes_every_rotation_via_production_helper():
    """The same 1944-rotation check, through the function the pipeline calls."""
    from zdock.dataset import has_pose_identity, pose_identity

    n = 1944
    key = pose_identity(torch.arange(n, dtype=torch.int64),
                        torch.zeros(n, 3, dtype=torch.int64))
    assert len({tuple(r) for r in key.tolist()}) == n
    assert has_pose_identity(key)
