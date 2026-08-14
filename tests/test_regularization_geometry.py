"""Regression tests for parameterisation-invariant table regularisation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from capacity_series import table_basis  # noqa: E402
from fit_shell import ShellDesign, band_contrast_basis  # noqa: E402


def test_common_frobenius_geometry_for_table_and_band_bases():
    """The same deployed table perturbation must have the same ridge cost."""
    rng = np.random.default_rng(20260814)

    # A sym perturbation represented in the 12-dimensional basis and in the
    # full 144-dimensional table must receive exactly the same penalty.
    q_sym = table_basis("sym")
    theta_sym = rng.normal(size=q_sym.shape[1])
    delta_full = q_sym @ theta_sym
    np.testing.assert_allclose(q_sym.T @ q_sym, np.eye(q_sym.shape[1]), atol=1e-14)
    np.testing.assert_allclose(theta_sym @ theta_sym,
                               delta_full @ delta_full, atol=1e-12)

    # The Helmert band coordinates likewise equal the sum of physical
    # Frobenius energies of all zero-sum per-band table contrasts.
    h_band = band_contrast_basis(3)
    eta = rng.normal(size=(h_band.shape[1], q_sym.shape[1]))
    per_band_coeff = h_band @ eta
    per_band_tables = per_band_coeff @ q_sym.T
    np.testing.assert_allclose(per_band_coeff.sum(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(np.sum(eta * eta),
                               np.sum(per_band_tables * per_band_tables),
                               atol=1e-12)


def test_shell_design_does_not_rescale_full_table_coordinates():
    """ShellDesign must preserve the same score for the same physical table."""
    rng = np.random.default_rng(7)
    n_pose = 3
    entry = {
        "sc": torch.zeros((n_pose, 4), dtype=torch.float64),
        "T": torch.from_numpy(rng.normal(size=(n_pose, 12, 12))),
        "Tb": torch.from_numpy(rng.normal(size=(n_pose, 3, 12, 12))),
        "elec": torch.zeros(n_pose, dtype=torch.float64),
        "dockq": torch.tensor([0.3, 0.0, 0.0], dtype=torch.float64),
    }
    common = dict(
        entries=[("synthetic", entry)],
        alpha=torch.tensor(0.0, dtype=torch.float64),
        clash=torch.zeros(3, dtype=torch.float64),
        beta=torch.tensor(0.0, dtype=torch.float64),
        thr=0.23,
        e0=np.zeros(144),
        use_shells=False,
    )
    sym = ShellDesign(basis_mode="sym", **common)
    full = ShellDesign(basis_mode="full", **common)

    theta_sym = rng.normal(size=12)
    delta_full = table_basis("sym") @ theta_sym
    np.testing.assert_allclose(sym.allA @ theta_sym,
                               full.allA @ delta_full, atol=1e-12)
    assert sym.col_scale.shape == (0,)
    assert full.col_scale.shape == (0,)
