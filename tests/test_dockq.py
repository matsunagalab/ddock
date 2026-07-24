"""Tests for `zdock.dockq` — atom-level DockQ v2 approximation.

Cross-checks:
- Native-identical pose → Fnat=1, iRMSD=0, LRMSD=0, DockQ=1.
- Shifted pose (small displacement) → intermediate DockQ tier.
- Far pose → DockQ < 0.23 (Incorrect tier).
- Formula sanity: DockQ matches the (Fnat + 1/(1+(iR/1.5)²) + 1/(1+(LR/8.5)²))/3
  identity at an arbitrary non-trivial pose.
- Autograd flows through DockQ w.r.t. pose coords (for future
  differentiable-ranking use — DockQ is used as a target here, but
  non-NaN gradient is a useful property check).
"""

from __future__ import annotations

import math

import pytest
import torch

from zdock.dockq import (
    CAPRI_ACCEPTABLE,
    CAPRI_MEDIUM,
    capri_tier,
    dockq_batch,
    interface_atom_masks,
    native_contacts,
)


def _build_synthetic_complex():
    """Tiny hand-placed complex: receptor at origin spread, ligand
    placed near receptor so that many atoms are within the 5 Å contact
    cutoff in the reference."""
    dtype = torch.float64
    rec_xyz = torch.tensor([
        [0.0,  0.0,  0.0],
        [1.0,  0.0,  0.0],
        [0.0,  1.0,  0.0],
        [0.0,  0.0,  1.0],
        [1.0,  1.0,  0.0],
        [5.0,  0.0,  0.0],      # a far rec atom (no native contact)
    ], dtype=dtype)
    native_lig_xyz = torch.tensor([
        [2.5,  0.0,  0.0],      # within 5 Å of rec atoms 0..4
        [2.5,  1.0,  0.0],
        [3.0,  0.0,  1.0],
    ], dtype=dtype)
    return rec_xyz, native_lig_xyz


def test_native_identical_pose_scores_unity():
    rec_xyz, native = _build_synthetic_complex()
    poses = native.unsqueeze(0)              # F=1, identical to native
    c = dockq_batch(rec_xyz, poses, native)
    assert torch.allclose(c.fnat, torch.tensor([1.0], dtype=poses.dtype))
    assert torch.allclose(c.i_rmsd, torch.tensor([0.0], dtype=poses.dtype), atol=1e-12)
    assert torch.allclose(c.l_rmsd, torch.tensor([0.0], dtype=poses.dtype), atol=1e-12)
    assert torch.allclose(c.dockq, torch.tensor([1.0], dtype=poses.dtype), atol=1e-12)


def test_far_pose_is_incorrect():
    rec_xyz, native = _build_synthetic_complex()
    far_pose = native + torch.tensor([50.0, 0.0, 0.0], dtype=native.dtype)
    poses = far_pose.unsqueeze(0)
    c = dockq_batch(rec_xyz, poses, native)
    # No contacts preserved, huge RMSDs ⇒ DockQ should be essentially 0
    assert c.fnat[0].item() == 0.0
    assert c.dockq[0].item() < CAPRI_ACCEPTABLE


def test_shifted_pose_is_intermediate():
    """A pose offset by ~1 Å should give high Fnat but nonzero RMSDs,
    landing in the Medium or High tier (neither 1.0 nor < 0.23)."""
    rec_xyz, native = _build_synthetic_complex()
    pose = native + torch.tensor([0.5, 0.5, 0.0], dtype=native.dtype)
    poses = pose.unsqueeze(0)
    c = dockq_batch(rec_xyz, poses, native)
    assert 0.0 < c.fnat[0].item() <= 1.0
    assert c.dockq[0].item() < 1.0
    assert c.dockq[0].item() >= CAPRI_ACCEPTABLE
    assert capri_tier(c.dockq)[0].item() in (1, 2, 3)


def test_dockq_formula_consistency():
    """At an arbitrary pose, reproduce DockQ from (Fnat, iRMSD, LRMSD)
    via the closed-form definition — catches any miswiring of the
    three terms."""
    rec_xyz, native = _build_synthetic_complex()
    pose = native + torch.tensor([1.2, -0.4, 0.7], dtype=native.dtype)
    poses = pose.unsqueeze(0)
    c = dockq_batch(rec_xyz, poses, native)
    expected = (
        c.fnat
        + 1.0 / (1.0 + (c.i_rmsd / 1.5).pow(2))
        + 1.0 / (1.0 + (c.l_rmsd / 8.5).pow(2))
    ) / 3.0
    assert torch.allclose(c.dockq, expected, atol=1e-12)


def test_batch_equals_elementwise():
    """Batched output for F poses must match per-pose scalar outputs."""
    rec_xyz, native = _build_synthetic_complex()
    rng = torch.Generator().manual_seed(0)
    offsets = torch.randn(4, 3, generator=rng, dtype=native.dtype) * 2.0
    poses = native.unsqueeze(0) + offsets.unsqueeze(1)   # (4, N_lig, 3)
    c_batch = dockq_batch(rec_xyz, poses, native)
    for i in range(4):
        c_one = dockq_batch(rec_xyz, poses[i:i+1], native)
        assert torch.allclose(c_batch.fnat[i:i+1], c_one.fnat)
        assert torch.allclose(c_batch.i_rmsd[i:i+1], c_one.i_rmsd, atol=1e-12)
        assert torch.allclose(c_batch.l_rmsd[i:i+1], c_one.l_rmsd, atol=1e-12)
        assert torch.allclose(c_batch.dockq[i:i+1], c_one.dockq, atol=1e-12)


def test_native_contacts_consistency():
    """`native_contacts` and `interface_atom_masks` should both flag
    atoms the native pose brings into contact."""
    rec_xyz, native = _build_synthetic_complex()
    mask = native_contacts(rec_xyz, native, cutoff=5.0)
    assert mask.shape == (rec_xyz.shape[0], native.shape[0])
    assert mask.any()
    rec_iface, lig_iface = interface_atom_masks(rec_xyz, native, cutoff=8.0)
    # Every atom that has a native contact is in the interface.
    assert (mask.any(dim=1) <= rec_iface).all()
    assert (mask.any(dim=0) <= lig_iface).all()


def test_dockq_autograd_through_pose():
    """DockQ is differentiable w.r.t. the pose coords — useful for
    future differentiable-ranking experiments and a sanity check that
    the metric doesn't hit a non-diff code path (e.g. argmax).
    """
    rec_xyz, native = _build_synthetic_complex()
    pose = (native + torch.tensor([0.3, 0.2, -0.1], dtype=native.dtype))
    poses = pose.unsqueeze(0).clone().requires_grad_(True)
    c = dockq_batch(rec_xyz, poses, native)
    c.dockq.sum().backward()
    assert poses.grad is not None
    assert not torch.isnan(poses.grad).any()


def test_capri_tier_boundaries():
    vals = torch.tensor([0.0, 0.22, 0.23, 0.48, 0.49, 0.79, 0.80, 0.99])
    tiers = capri_tier(vals)
    assert tiers.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
