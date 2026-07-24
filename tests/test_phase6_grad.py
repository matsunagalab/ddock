"""Phase 6: gradient integrity tests.

Three-way comparison:
  * PyTorch autograd gradient of `docking_score_elec`
  * PyTorch finite-difference gradient (uses the same forward)
  * Julia finite-difference gradient (exported via
    `docking/tests/julia_ref/gradcheck_fd_export.jl`)

Julia's hand-written `rrule(::typeof(docking_score_elec))` is NOT used
here — we established in A-4 that it has bugs (B6/B7). Julia FD is the
ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from zdock.score import docking_score_elec


def _2d(a):
    arr = np.asarray(a)
    return arr.T if arr.ndim == 2 and arr.shape[0] == 3 else arr


def _3d(a):
    arr = np.asarray(a)
    return arr.transpose(2, 1, 0) if arr.ndim == 3 and arr.shape[0] == 3 else arr


@pytest.fixture(scope="module")
def prepared(load_ref):
    """Load the inputs that gradcheck_fd_export.jl used (first 5 poses)."""
    ref5 = load_ref("phase5", "scores")
    refg = load_ref("phase6", "fd_grads")
    n_pose = int(refg["n_pose"])
    return {
        "alpha_nom": float(refg["alpha"]),
        "beta_nom": float(refg["beta"]),
        "iface_flat_nom": np.asarray(refg["iface_ij_flat"]),
        "charge_score_nom": np.asarray(refg["charge_score"]),
        "rec_xyz": _2d(ref5["rec_xyz"]),
        "rec_radius": np.asarray(ref5["rec_radius"]),
        "rec_sasa": np.asarray(ref5["rec_sasa"]),
        "rec_atomtype": np.asarray(ref5["rec_atomtype_id"]),
        "rec_charge": np.asarray(ref5["rec_charge_id"]),
        "lig_xyz": _3d(ref5["lig_xyz"])[:n_pose],
        "lig_radius": np.asarray(ref5["lig_radius"]),
        "lig_sasa": np.asarray(ref5["lig_sasa"]),
        "lig_atomtype": np.asarray(ref5["lig_atomtype_id"]),
        "lig_charge": np.asarray(ref5["lig_charge_id"]),
        "n_pose": n_pose,
        "alpha_fd": float(refg["alpha_fd"]),
        "beta_fd": float(refg["beta_fd"]),
        "iface_fd": np.asarray(refg["iface_fd"]),
        "charge_fd": np.asarray(refg["charge_fd"]),
        "halpha": float(refg["halpha"]),
        "hbeta": float(refg["hbeta"]),
        "hiface": float(refg["hiface"]),
        "hcharge": float(refg["hcharge"]),
    }


def _build_loss(prepared, device, dtype):
    """Return a closure loss(alpha, iface, beta, charge) -> scalar."""
    rec_xyz = torch.as_tensor(prepared["rec_xyz"], device=device, dtype=dtype)
    rec_radius = torch.as_tensor(prepared["rec_radius"], device=device, dtype=dtype)
    rec_sasa = torch.as_tensor(prepared["rec_sasa"], device=device, dtype=dtype)
    rec_atomtype = torch.as_tensor(prepared["rec_atomtype"], device=device, dtype=torch.int64)
    rec_charge = torch.as_tensor(prepared["rec_charge"], device=device, dtype=torch.int64)
    lig_xyz = torch.as_tensor(prepared["lig_xyz"], device=device, dtype=dtype)
    lig_radius = torch.as_tensor(prepared["lig_radius"], device=device, dtype=dtype)
    lig_sasa = torch.as_tensor(prepared["lig_sasa"], device=device, dtype=dtype)
    lig_atomtype = torch.as_tensor(prepared["lig_atomtype"], device=device, dtype=torch.int64)
    lig_charge = torch.as_tensor(prepared["lig_charge"], device=device, dtype=torch.int64)

    def loss(alpha, iface, beta, charge):
        scores = docking_score_elec(
            rec_xyz, rec_radius, rec_sasa, rec_atomtype, rec_charge,
            lig_xyz, lig_radius, lig_sasa, lig_atomtype, lig_charge,
            alpha, iface, beta, charge,
        )
        return scores.sum()

    return loss


def test_autograd_matches_julia_fd(prepared, device, dtype):
    """PyTorch autograd against Julia central-FD reference for α, β,
    three representative iface elements, and three charge elements.

    Testing all 144+11 elements would be slow — we sample to keep the test
    under a few seconds. Full-matrix agreement is verified indirectly
    when the training loop converges in B-7.
    """
    loss_fn = _build_loss(prepared, device, dtype)

    alpha = torch.tensor(prepared["alpha_nom"], device=device, dtype=dtype, requires_grad=True)
    beta = torch.tensor(prepared["beta_nom"], device=device, dtype=dtype, requires_grad=True)
    iface = torch.as_tensor(prepared["iface_flat_nom"], device=device, dtype=dtype).clone().requires_grad_(True)
    charge = torch.as_tensor(prepared["charge_score_nom"], device=device, dtype=dtype).clone().requires_grad_(True)

    l = loss_fn(alpha, iface, beta, charge)
    (da, dif, db, dch) = torch.autograd.grad(l, (alpha, iface, beta, charge))

    # Ground truth from Julia FD
    alpha_fd = prepared["alpha_fd"]
    beta_fd = prepared["beta_fd"]
    iface_fd = prepared["iface_fd"]
    charge_fd = prepared["charge_fd"]

    rtol = 5e-3 if dtype == torch.float64 else 2e-2
    atol = 1.0    # scores are O(10^5) for α; tiny absolute errors are fine

    # α and β — small relative tolerance
    print(f"\n[grad] α  autograd={da.item():.6e}  julia_fd={alpha_fd:.6e}")
    print(f"[grad] β  autograd={db.item():.6e}  julia_fd={beta_fd:.6e}")
    assert abs(da.item() - alpha_fd) <= atol + rtol * abs(alpha_fd), (
        f"dα autograd {da.item()} disagrees with Julia FD {alpha_fd}"
    )
    assert abs(db.item() - beta_fd) <= atol + rtol * abs(beta_fd), (
        f"dβ autograd {db.item()} disagrees with Julia FD {beta_fd}"
    )

    # Representative iface and charge elements (same as gradcheck.jl)
    for k in (0, 12, 71):
        ad = dif[k].item()
        fd = iface_fd[k]
        print(f"[grad] iface[{k}]  autograd={ad:.6e}  julia_fd={fd:.6e}")
        assert abs(ad - fd) <= atol + rtol * abs(fd), (
            f"diface[{k}] mismatch: autograd {ad} vs Julia FD {fd}"
        )
    for l_ in (0, 5, 10):
        ad = dch[l_].item()
        fd = charge_fd[l_]
        print(f"[grad] charge[{l_}]  autograd={ad:.6e}  julia_fd={fd:.6e}")
        assert abs(ad - fd) <= atol + rtol * abs(fd), (
            f"dcharge[{l_}] mismatch: autograd {ad} vs Julia FD {fd}"
        )
