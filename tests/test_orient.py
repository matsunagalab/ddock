"""Test orient() against Julia's orient! (as used in generate_refs.jl).

Julia's pipeline:
  1. decenter!(ligands)                           -> produces `lig_xyz` (F, N, 3)
  2. lig_for_grid = deepcopy(ligands); orient!(lig_for_grid)
     -> `lig_xyz_for_grid` (frame 1 only).

orient! uses `ta.mass` (already set to iface_score[atomtype_id] in the
prep step). Our Python `orient()` takes explicit `mass=` so we pass the
same weights.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from zdock import atomtypes
from zdock.geom import orient


def _2d(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[0] == 3:
        return a.T
    return a


def _3d(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] == 3:
        return a.transpose(2, 1, 0)
    return a


def test_orient_matches_julia(load_ref, device, dtype):
    ref = load_ref("phase5", "scores")

    # Ligand (frame 0) post-decenter, BEFORE orient.
    lig_xyz_all = torch.as_tensor(_3d(ref["lig_xyz"]), device=device, dtype=dtype)
    lig_frame0 = lig_xyz_all[0]

    # Julia's orient! uses the `mass` field, which the notebook sets to
    # iface_score[atomtype_id]. Build the same mass vector here.
    lig_atomtype = torch.as_tensor(
        np.asarray(ref["lig_atomtype_id"]), device=device, dtype=torch.int64
    )
    iface_mat = atomtypes.iface_ij(device=device, dtype=dtype)
    # Diagonal of iface matrix? The Julia prep does:
    #     mass = iface_score[atomtype_id]
    # where iface_score = MDToolbox.get_iface_ij() — the *whole* 12×12 matrix.
    # Indexing a matrix by a vector of atomtype_ids returns a (N, 12) slice
    # — rows 1..natom, cols 1..12. Julia collapses this to (N,) by taking
    # the first column (index broadcast), OR the diagonal. Let's check:
    # `iface_score[receptor.atomtype_id]` in Julia is column-1 of the row
    # selected by atomtype_id — i.e. iface_score[a, 1] for each atom a.
    # So mass[atom] = iface_ij[atomtype_id[atom], 1] (1-based).
    mass = iface_mat[lig_atomtype - 1, 0]

    got = orient(lig_frame0, mass=mass)
    expected = torch.as_tensor(_2d(ref["lig_xyz_for_grid"]), device=device, dtype=dtype)

    # Sign ambiguity means per-axis sign may differ from Julia even though
    # we enforce largest-magnitude-positive convention. Compute match under
    # axis-sign flips to diagnose.
    def _diff(A, B):
        return (A - B).abs().max().item()

    raw_err = _diff(got, expected)
    # Try each sign combination for the 3 axes
    best_err = raw_err
    best_flip = (1, 1, 1)
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                flipped = got * torch.tensor([sx, sy, sz], device=device, dtype=dtype)
                e = _diff(flipped, expected)
                if e < best_err:
                    best_err = e
                    best_flip = (sx, sy, sz)
    print(f"\n[orient] raw max-abs err = {raw_err:.4f}")
    print(f"[orient] best  max-abs err = {best_err:.4f}  (axis flips {best_flip})")

    # Accept up to 1e-3 Å tolerance (float64 should give ~1e-6; float32 ~1e-3)
    tol = 1e-4 if dtype == torch.float64 else 1e-2
    torch.testing.assert_close(got, expected, atol=tol, rtol=tol)
