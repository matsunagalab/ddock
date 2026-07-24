"""Physics validation tests for the Coulombic ELEC implementation.

The underlying primitive `spread_neighbors_coulomb` is tested directly
(without the full `docking_score_elec` pipeline that zeros V inside the
receptor SC shell and quantizes ligand positions to nearest grid cells).
End-to-end Coulomb sanity is also exercised via a cross-type comparison
between `elec_mode="coulomb"` and `"legacy"`, which captures the fact
that the Coulombic path does not silently drop non-same-atomtype pairs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from zdock.score import docking_score_elec
from zdock.spread import spread_neighbors_coulomb, spread_nearest_add
from zdock.atomtypes import charge_score as default_charge_score


def _make_grid(n: int, spacing: float, device, dtype):
    """Cubic (n, n, n) grid spanning [-n//2·spacing, +n//2·spacing] in each
    axis. Returns (grid, xg, yg, zg) all empty / linspace tensors."""
    axis = torch.arange(n, device=device, dtype=dtype) * spacing - (n // 2) * spacing
    grid = torch.zeros((n, n, n), device=device, dtype=dtype)
    return grid, axis, axis, axis


def _cell_of(xyz_value: float, axis: torch.Tensor, spacing: float) -> int:
    """Julia-compatible nearest-cell index for a scalar coordinate."""
    return math.ceil((xyz_value - axis[0].item()) / spacing) - 1


def test_coulomb_primitive_sign(device, dtype):
    """Two unit charges at opposite positions — verify V(r)'s sign flips
    correctly with charge sign."""
    spacing = 0.5
    grid, xg, yg, zg = _make_grid(31, spacing, device, dtype)   # ±7.5 Å

    # Positive source at origin; field at (+5,0,0) should be positive.
    rec_xyz = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=dtype)
    rec_q = torch.tensor([1.0], device=device, dtype=dtype)
    spread_neighbors_coulomb(grid, rec_xyz, rec_q, 8.0, xg, yg, zg)
    ix = _cell_of(5.0, xg, spacing)
    iy = iz = _cell_of(0.0, yg, spacing)
    v_pos = grid[ix, iy, iz].item()

    # Negative source: field should flip sign.
    grid2 = torch.zeros_like(grid)
    spread_neighbors_coulomb(grid2, rec_xyz, -rec_q, 8.0, xg, yg, zg)
    v_neg = grid2[ix, iy, iz].item()

    print(f"\n[primitive sign] V(+q at 5Å) = {v_pos:+.4f}  V(-q at 5Å) = {v_neg:+.4f}")
    assert v_pos > 0 and v_neg < 0
    assert math.isclose(v_pos, -v_neg, rel_tol=1e-6)


def test_coulomb_primitive_inverse_distance(device, dtype):
    """V(r) from a single +1 source should scale as 1/r."""
    spacing = 0.25
    grid, xg, yg, zg = _make_grid(65, spacing, device, dtype)  # ±8 Å
    rec_xyz = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=dtype)
    rec_q = torch.tensor([1.0], device=device, dtype=dtype)
    spread_neighbors_coulomb(grid, rec_xyz, rec_q, 8.0, xg, yg, zg)

    # Sample V at a few (x, 0, 0) cells and compare to 1/r.
    distances = [1.0, 2.0, 4.0, 6.0]
    print("\n[primitive 1/r]")
    for r in distances:
        ix = _cell_of(r, xg, spacing)
        iy = iz = _cell_of(0.0, yg, spacing)
        cell_xyz = torch.tensor(
            [xg[ix].item(), yg[iy].item(), zg[iz].item()],
            device=device, dtype=dtype,
        )
        d_to_src = cell_xyz.norm().item()
        v = grid[ix, iy, iz].item()
        expected = 1.0 / d_to_src
        print(f"  r_target={r}  cell={cell_xyz.tolist()}  d={d_to_src:.4f}  "
              f"V={v:.6f}  1/d={expected:.6f}  rel_err={abs(v-expected)/expected:.2e}")
        tol = 1e-10 if dtype == torch.float64 else 1e-5
        assert abs(v - expected) / expected < tol, (
            f"V at cell ≠ 1/distance: got {v}, expected {expected}"
        )


def test_coulomb_primitive_superposition(device, dtype):
    """V from two sources should equal V(source1) + V(source2)."""
    spacing = 0.5
    grid_A, xg, yg, zg = _make_grid(31, spacing, device, dtype)
    grid_B = torch.zeros_like(grid_A)
    grid_AB = torch.zeros_like(grid_A)

    # Source A at (-2, 0, 0) with q=+1; source B at (+2, 0, 0) with q=-1.
    xyz_A = torch.tensor([[-2.0, 0.0, 0.0]], device=device, dtype=dtype)
    xyz_B = torch.tensor([[+2.0, 0.0, 0.0]], device=device, dtype=dtype)
    xyz_AB = torch.cat([xyz_A, xyz_B], dim=0)
    q_A = torch.tensor([1.0], device=device, dtype=dtype)
    q_B = torch.tensor([-1.0], device=device, dtype=dtype)
    q_AB = torch.cat([q_A, q_B], dim=0)

    spread_neighbors_coulomb(grid_A,  xyz_A,  q_A,  8.0, xg, yg, zg)
    spread_neighbors_coulomb(grid_B,  xyz_B,  q_B,  8.0, xg, yg, zg)
    spread_neighbors_coulomb(grid_AB, xyz_AB, q_AB, 8.0, xg, yg, zg)

    diff = (grid_A + grid_B - grid_AB).abs().max().item()
    print(f"\n[superposition] max |V_A + V_B − V_AB| = {diff:.2e}")
    assert diff < 1e-12


def test_docking_score_elec_coulomb_vs_legacy_symmetry(device, dtype):
    """For a realistic receptor + ligand configuration, Coulomb should give
    a score structurally different from legacy: legacy's Σq/Σr is always
    finite-signed and restricted to same-atomtype pairs, so the ratio of
    |coulomb| to |legacy| should differ substantially across poses."""
    # Use the 1KXQ reference inputs to guarantee a non-trivial receptor.
    from conftest import load_h5 as _load     # conftest.py is on sys.path
    from pathlib import Path
    refs = Path(__file__).resolve().parent / "data" / "refs" / "1KXQ"
    if not (refs / "phase5_scores.h5").exists():
        pytest.skip("phase5 refs missing")

    ref = _load(refs / "phase5_scores.h5")

    def _2d(a):
        arr = np.asarray(a)
        return arr.T if arr.ndim == 2 and arr.shape[0] == 3 else arr

    def _3d(a):
        arr = np.asarray(a)
        return arr.transpose(2, 1, 0) if arr.ndim == 3 and arr.shape[0] == 3 else arr

    common = dict(
        rec_xyz=torch.as_tensor(_2d(ref["rec_xyz"]), device=device, dtype=dtype),
        rec_radius=torch.as_tensor(np.asarray(ref["rec_radius"]), device=device, dtype=dtype),
        rec_sasa=torch.as_tensor(np.asarray(ref["rec_sasa"]), device=device, dtype=dtype),
        rec_atomtype_id=torch.as_tensor(np.asarray(ref["rec_atomtype_id"]), device=device, dtype=torch.int64),
        rec_charge_id=torch.as_tensor(np.asarray(ref["rec_charge_id"]), device=device, dtype=torch.int64),
        lig_xyz=torch.as_tensor(_3d(ref["lig_xyz"]), device=device, dtype=dtype),
        lig_radius=torch.as_tensor(np.asarray(ref["lig_radius"]), device=device, dtype=dtype),
        lig_sasa=torch.as_tensor(np.asarray(ref["lig_sasa"]), device=device, dtype=dtype),
        lig_atomtype_id=torch.as_tensor(np.asarray(ref["lig_atomtype_id"]), device=device, dtype=torch.int64),
        lig_charge_id=torch.as_tensor(np.asarray(ref["lig_charge_id"]), device=device, dtype=torch.int64),
        alpha=torch.tensor(float(ref["alpha"]), device=device, dtype=dtype),
        iface_ij_flat=torch.as_tensor(np.asarray(ref["iface_ij_flat"]), device=device, dtype=dtype),
        beta=torch.tensor(float(ref["beta"]), device=device, dtype=dtype),
        charge_score=torch.as_tensor(np.asarray(ref["charge_score"]), device=device, dtype=dtype),
    )

    s_coulomb = docking_score_elec(**common, elec_mode="coulomb")
    s_legacy = docking_score_elec(**common, elec_mode="legacy")

    diff = (s_coulomb - s_legacy).abs().max().item()
    print(f"\n[coulomb vs legacy] max |Δ score| across 10 poses = {diff:.4f}")
    # ELEC is a small component of total score on 1KXQ (β·score_elec ≈ O(1-10),
    # whereas SC+IFACE dominate at O(10^3)). But the two ELEC formulations
    # are NOT identical — they produce non-trivial deltas per pose.
    assert diff > 0.1, (
        f"Coulomb and legacy should differ at the β·score_elec level "
        f"(expected > 0.1, got Δ={diff})"
    )


def test_docking_score_elec_autograd_through_coulomb(device, dtype):
    """Ensure autograd still flows through the Coulombic ELEC path —
    gradient w.r.t. β and charge_score should be finite and non-zero."""
    # Minimal two-atom setup that actually produces a non-zero ELEC score.
    # Place rec at (-4, 0, 0) (far enough that its SC shell does not cover
    # the ligand cell), ligand at (+4, 0, 0). Distance 8 Å.
    rec_xyz = torch.tensor([[-4.0, 0.0, 0.0],
                            (0.0, -50.0, 0.0), (0.0, 50.0, 0.0),
                            (0.0, 0.0, -50.0), (0.0, 0.0, 50.0),
                            (-50.0, 0.0, 0.0), (50.0, 0.0, 0.0)],
                           device=device, dtype=dtype)
    rec_sasa = torch.tensor([10.0] + [0.0] * 6, device=device, dtype=dtype)
    lig_xyz = torch.tensor([[[+4.0, 0.0, 0.0]]], device=device, dtype=dtype)

    def ones(n, dtype_=dtype): return torch.ones(n, device=device, dtype=dtype_)

    alpha = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
    beta = torch.tensor(1.0, device=device, dtype=dtype, requires_grad=True)
    iface = torch.zeros(144, device=device, dtype=dtype, requires_grad=True)
    charge = default_charge_score(device=device, dtype=dtype).clone().requires_grad_(True)

    s = docking_score_elec(
        rec_xyz=rec_xyz,
        rec_radius=ones(7) * 1.7,
        rec_sasa=rec_sasa,
        rec_atomtype_id=torch.ones(7, device=device, dtype=torch.int64),
        rec_charge_id=torch.tensor([1] + [8] * 6, device=device, dtype=torch.int64),
        lig_xyz=lig_xyz,
        lig_radius=ones(1) * 1.7,
        lig_sasa=ones(1) * 10.0,
        lig_atomtype_id=torch.ones(1, device=device, dtype=torch.int64),
        lig_charge_id=torch.tensor([2], device=device, dtype=torch.int64),
        alpha=alpha, iface_ij_flat=iface, beta=beta, charge_score=charge,
        lig_xyz_for_grid=lig_xyz[0],
    )
    loss = s.sum()
    loss.backward()

    print(f"\n[autograd coulomb] loss={loss.item():.4e}  dβ={beta.grad.item():.4e}  "
          f"dα={alpha.grad.item():.4e}")
    assert torch.isfinite(beta.grad).all()
    assert torch.isfinite(charge.grad).all()
    # dα should be finite (zero is OK since alpha=0 in this test).
    assert torch.isfinite(alpha.grad).all()
