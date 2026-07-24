"""Phase 7: end-to-end Adam training smoke test.

We run a short training (30 epochs on 1KXQ top-10 decoys) and verify the
loss descends. Full 200-epoch three-protein training requires generating
the 1F51 / 2VDB reference inputs and isn't exercised here — the machinery
(autograd + Adam + B2-fixed loss) is identical though.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from zdock.train import ProteinInputs, total_loss, train


def _2d(a):
    arr = np.asarray(a)
    return arr.T if arr.ndim == 2 and arr.shape[0] == 3 else arr


def _3d(a):
    arr = np.asarray(a)
    return arr.transpose(2, 1, 0) if arr.ndim == 3 and arr.shape[0] == 3 else arr


def build_1kxq(load_ref, device, dtype) -> ProteinInputs:
    ref = load_ref("phase5", "scores")

    def T(key, int_=False):
        arr = np.asarray(ref[key])
        if arr.ndim == 3:
            arr = _3d(arr)
        elif arr.ndim == 2:
            arr = _2d(arr)
        dtype_ = torch.int64 if int_ else dtype
        return torch.as_tensor(arr, device=device, dtype=dtype_)

    F = int(ref["n_pose"])
    # Hit/Miss by first-3 are Hit (synthetic — real RMSD split would use
    # `*.zd3.0.2.fg.fixed.out.rmsds`). For the smoke test, what matters is
    # that both classes are non-empty so the loss has gradient signal.
    hit_mask = torch.zeros(F, dtype=torch.bool, device=device)
    hit_mask[:3] = True

    return ProteinInputs(
        rec_xyz=T("rec_xyz"),
        rec_radius=T("rec_radius"),
        rec_sasa=T("rec_sasa"),
        rec_atomtype_id=T("rec_atomtype_id", int_=True),
        rec_charge_id=T("rec_charge_id", int_=True),
        lig_xyz=T("lig_xyz"),
        lig_radius=T("lig_radius"),
        lig_sasa=T("lig_sasa"),
        lig_atomtype_id=T("lig_atomtype_id", int_=True),
        lig_charge_id=T("lig_charge_id", int_=True),
        hit_mask=hit_mask,
    )


def test_train_smoke_loss_decreases(load_ref, device, dtype):
    """30 epochs on CPU (float64). Quick smoke test for CI parity."""
    p = build_1kxq(load_ref, device, dtype)
    out = train([p], n_epoch=30, lr=0.01, device=device, dtype=dtype,
                progress_every=10)
    hist = out["history"]["loss"]
    print(f"\n[train] initial={hist[0]:.4e}  final={hist[-1]:.4e}  "
          f"reduction={(hist[0]-hist[-1])/hist[0]*100:.1f}%")
    assert hist[-1] < hist[0], (
        f"loss did not decrease: init={hist[0]} final={hist[-1]}"
    )
    assert hist[-1] < hist[0] * 0.95
    assert not torch.allclose(out["alpha"], torch.tensor(0.01, device=device, dtype=dtype))


def test_frame_chunking_matches_unchunked(load_ref, device, dtype):
    """docking_score_elec must return (bit-exact on CPU float64, close on
    GPU float32) the same scores and parameter gradients whether we run
    with or without frame_chunk_size. Guards the memory-saving path from
    silent numerical drift."""
    from zdock.atomtypes import iface_ij, charge_score as default_charge_score
    from zdock.score import docking_score_elec

    p = build_1kxq(load_ref, device, dtype)

    alpha = torch.tensor(0.01, device=device, dtype=dtype, requires_grad=True)
    beta = torch.tensor(3.0, device=device, dtype=dtype, requires_grad=True)
    iface = iface_ij(device=device, dtype=dtype, flat=True).clone().detach().requires_grad_(True)
    charge = default_charge_score(device=device, dtype=dtype).clone().detach().requires_grad_(True)

    def run(chunk):
        return docking_score_elec(
            p.rec_xyz, p.rec_radius, p.rec_sasa, p.rec_atomtype_id, p.rec_charge_id,
            p.lig_xyz, p.lig_radius, p.lig_sasa, p.lig_atomtype_id, p.lig_charge_id,
            alpha, iface, beta, charge,
            frame_chunk_size=chunk,
        )

    s_full = run(None)
    s_chunk = run(3)

    # CPU float64 is bit-exact; float32 scatter_add is non-associative so
    # chunking changes the reduction order and we need small tolerance.
    if dtype == torch.float64:
        atol, rtol = 1e-10, 0.0
    else:
        atol, rtol = 1e-4, 1e-5
    assert torch.allclose(s_full, s_chunk, atol=atol, rtol=rtol), (
        f"score mismatch: max |diff|={(s_full - s_chunk).abs().max().item():.3e}"
    )

    g_full = torch.autograd.grad(s_full.sum(), (alpha, iface, beta, charge), retain_graph=True)
    g_chunk = torch.autograd.grad(s_chunk.sum(), (alpha, iface, beta, charge))
    for name, a, b in zip(("alpha", "iface", "beta", "charge"), g_full, g_chunk):
        assert torch.allclose(a, b, atol=atol, rtol=rtol), (
            f"grad[{name}] mismatch: max |diff|={(a - b).abs().max().item():.3e}"
        )


@pytest.mark.slow
def test_train_200_epoch_1kxq(load_ref, device, dtype):
    """Full 200-epoch training on 1KXQ alone (matching thesis schedule).

    Run with `pytest -m slow` to opt in. Proves loss continues to descend
    across the full 200 epochs and α lands near a physically-plausible
    value (α ~ 0.01). β is held fixed at 3.0 by `train()`."""
    p = build_1kxq(load_ref, device, dtype)
    out = train([p], n_epoch=200, lr=0.01, device=device, dtype=dtype,
                progress_every=25)
    hist = out["history"]["loss"]
    print(f"\n[train-200] initial={hist[0]:.4e}  final={hist[-1]:.4e}  "
          f"reduction={(hist[0]-hist[-1])/hist[0]*100:.1f}%")
    assert hist[-1] < hist[0] * 0.5   # expect ≥50% drop after 200 epochs
    print(f"[train-200] α = {out['alpha'].item():.4e}")
    assert abs(out["alpha"]) < 1.0, "α drifted out of plausible range"


# ---------------------------------------------------------- input-validation


def test_train_rejects_nonpositive_progress_every():
    """Regression: `progress_every=0` previously triggered a
    ZeroDivisionError inside the epoch loop; it must now fail fast with a
    clear ValueError, without needing a real ProteinInputs."""
    with pytest.raises(ValueError, match="progress_every"):
        train([], n_epoch=1, progress_every=0)
    with pytest.raises(ValueError, match="progress_every"):
        train([], n_epoch=1, progress_every=-5)


def test_train_rejects_empty_proteins():
    """Regression: `proteins=[]` previously reached `torch.stack([])` and
    raised an opaque RuntimeError. It must now fail early with a clear
    ValueError."""
    with pytest.raises(ValueError, match="empty"):
        train([], n_epoch=1, progress_every=1)


def test_total_loss_rejects_empty_proteins():
    """`total_loss` must also reject empty input directly — callers that
    bypass `train` (e.g. tests) should still see a useful ValueError
    instead of `torch.stack([])` failing mysteriously."""
    alpha = torch.tensor(0.01, dtype=torch.float64)
    beta = torch.tensor(3.0, dtype=torch.float64)
    iface = torch.zeros(144, dtype=torch.float64)
    charge = torch.zeros(11, dtype=torch.float64)
    with pytest.raises(ValueError, match="empty"):
        total_loss([], alpha, iface, beta, charge, [])


# ---------------------------------------------------- consolidated h5 path


_SMOKE_H5 = Path("/tmp/smoke.h5")


@pytest.mark.skipif(
    not _SMOKE_H5.exists(),
    reason=f"smoke dataset missing at {_SMOKE_H5}; run "
           "`uv run python scripts/build_training_dataset.py --proteins 1KXQ "
           "--max-poses 100 --output /tmp/smoke.h5`",
)
def test_train_with_consolidated_h5(device, dtype):
    """End-to-end: load the smoke h5 via `zdock.data` and train 5 epochs."""
    from zdock.data import load_training_dataset

    proteins = load_training_dataset(
        _SMOKE_H5, device=device, dtype=dtype, protein_names=["1KXQ"],
    )
    assert len(proteins) == 1
    p = proteins[0]
    # Sanity: enough poses and at least one hit so the loss has gradient signal.
    assert p.lig_xyz.shape[0] >= 10
    assert p.hit_mask.any() and (~p.hit_mask).any()

    out = train(
        [p], n_epoch=5, lr=0.01, device=device, dtype=dtype, progress_every=5,
    )
    hist = out["history"]["loss"]
    assert len(hist) == 5
    # With only 5 epochs the loss can still wobble; just require no NaN.
    assert all(torch.isfinite(torch.tensor(x)) for x in hist), (
        f"loss history contains non-finite values: {hist}"
    )
