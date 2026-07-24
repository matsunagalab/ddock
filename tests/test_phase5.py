"""Phase 5 end-to-end test: docking_score_elec forward on 1KXQ top-10 poses.

The reference scores (`score_elec_total` in phase5_scores.h5) were generated
by Julia's `docking_score_elec` after our B1 / B3 / B5 / B8 fixes. We load
the prepared inputs (post-decenter receptor, per-frame ligand, and a
separate pre-oriented ligand for grid-bounds only — we don't reimplement
MDToolbox's `orient!`) and compare."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from zdock.geom import generate_grid
from zdock.score import docking_score_elec


def _2d(arr) -> np.ndarray:
    """Julia stores (N, 3) as column-major → h5py sees (3, N). Transpose."""
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[0] == 3:
        return a.T
    return a


def _3d(arr) -> np.ndarray:
    """Julia stores (F, N, 3) → h5py sees (3, N, F). Unpermute."""
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] == 3:
        # Julia axes [F, N, 3] → h5 axes [3, N, F]; we want [F, N, 3].
        return a.transpose(2, 1, 0)
    return a


def test_docking_score_elec_matches_julia(load_ref, device, dtype):
    ref = load_ref("phase5", "scores")

    alpha = torch.as_tensor(float(ref["alpha"]), device=device, dtype=dtype)
    beta = torch.as_tensor(float(ref["beta"]), device=device, dtype=dtype)
    iface_flat = torch.as_tensor(
        np.asarray(ref["iface_ij_flat"]), device=device, dtype=dtype
    )
    charge_score = torch.as_tensor(
        np.asarray(ref["charge_score"]), device=device, dtype=dtype
    )

    rec_xyz = torch.as_tensor(_2d(ref["rec_xyz"]), device=device, dtype=dtype)
    rec_radius = torch.as_tensor(np.asarray(ref["rec_radius"]), device=device, dtype=dtype)
    rec_sasa = torch.as_tensor(np.asarray(ref["rec_sasa"]), device=device, dtype=dtype)
    rec_atomtype = torch.as_tensor(
        np.asarray(ref["rec_atomtype_id"]), device=device, dtype=torch.int64
    )
    rec_charge = torch.as_tensor(
        np.asarray(ref["rec_charge_id"]), device=device, dtype=torch.int64
    )

    lig_xyz = torch.as_tensor(_3d(ref["lig_xyz"]), device=device, dtype=dtype)
    lig_xyz_for_grid = torch.as_tensor(_2d(ref["lig_xyz_for_grid"]), device=device, dtype=dtype)
    lig_radius = torch.as_tensor(np.asarray(ref["lig_radius"]), device=device, dtype=dtype)
    lig_sasa = torch.as_tensor(np.asarray(ref["lig_sasa"]), device=device, dtype=dtype)
    lig_atomtype = torch.as_tensor(
        np.asarray(ref["lig_atomtype_id"]), device=device, dtype=torch.int64
    )
    lig_charge = torch.as_tensor(
        np.asarray(ref["lig_charge_id"]), device=device, dtype=torch.int64
    )

    # Compare against Julia's Coulombic reference (physically correct,
    # matches PyTorch's default `elec_mode="coulomb"`). The legacy Σq/Σr
    # reference is preserved in the HDF5 under "score_elec_total" for
    # thesis reproduction but isn't used here.
    expected = torch.as_tensor(
        np.asarray(ref["score_coulomb_total"]), device=device, dtype=dtype
    )

    # Timing for reporting (not asserted).
    import time
    t0 = time.perf_counter()
    got = docking_score_elec(
        rec_xyz, rec_radius, rec_sasa, rec_atomtype, rec_charge,
        lig_xyz, lig_radius, lig_sasa, lig_atomtype, lig_charge,
        alpha, iface_flat, beta, charge_score,
        # lig_xyz_for_grid now comes from Python's own orient() — the
        # previous pinning to `lig_xyz_for_grid=...` was only needed while
        # orient was unported.
    )
    if device.type != "cpu":
        torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"\n[perf] docking_score_elec on {device} {dtype}: {dt*1000:.1f} ms "
          f"for {lig_xyz.shape[0]} poses "
          f"({dt*1000/lig_xyz.shape[0]:.1f} ms/pose)")

    # End-to-end scores agree with Julia within ~5% on 1KXQ top-10. Residual
    # comes from boundary-cell handling in the grid spreads (Julia's exact
    # `ceil` grid mapping of atoms on cell boundaries) and from not fully
    # porting `orient!` — we use Julia-preoriented ligand for grid bounds
    # but the scoring uses the decentered (non-oriented) positions, which
    # is what Julia also does. Tightening is Phase-6 gradient work.
    atol = 100.0
    rtol = 0.05 if dtype == torch.float64 else 0.1
    print(f"[perf] max abs err {(got - expected).abs().max().item():.2f}, "
          f"max rel err {((got - expected) / expected).abs().max().item():.4f}")
    torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)
