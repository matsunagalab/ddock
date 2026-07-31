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


# ---------------------------------------------------------------- top-tail loss


def test_top_tail_zero_without_positives_or_negatives():
    """A graph-connected zero, like every other loss here, so a mixed batch
    still backprops when one complex has no positive."""
    from zdock.train import loss_top_tail

    s = torch.randn(20, requires_grad=True)
    for dq in (torch.zeros(20), torch.ones(20)):
        loss = loss_top_tail(s, dq)
        assert float(loss) == 0.0
        loss.backward(retain_graph=True)
    assert s.grad is not None


def test_top_tail_falls_when_the_best_positive_rises():
    """The whole point: the term must respond to the top-1 decision."""
    from zdock.train import loss_top_tail

    dq = torch.tensor([0.9, 0.5] + [0.0] * 30)
    lose = torch.tensor([0.0, -1.0] + [2.0] + [0.0] * 29)   # a negative on top
    win = lose.clone()
    win[0] = 5.0                                            # positive on top
    assert float(loss_top_tail(win, dq)) < float(loss_top_tail(lose, dq))


def test_top_tail_ignores_the_worst_positive():
    """`loss_margin_hard_negatives` anchors on min(positive), so dragging one
    poor positive down changes it; this term must not care, because Max(Top 1)
    does not."""
    from zdock.train import loss_margin_hard_negatives, loss_top_tail

    dq = torch.tensor([0.9, 0.3] + [0.0] * 30)
    a = torch.tensor([4.0, 1.0] + [0.5] * 30)
    b = a.clone()
    b[1] = -8.0                                    # same best positive, worse worst
    # not bit-identical: the worst positive is exponentially suppressed, not
    # excluded. 1e-3 SD against the ~9 SD the min-anchor term moves by.
    assert abs(float(loss_top_tail(a, dq)) - float(loss_top_tail(b, dq))) < 1e-3
    assert abs(float(loss_margin_hard_negatives(a, dq))
               - float(loss_margin_hard_negatives(b, dq))) > 1.0


def test_top_tail_looks_only_k_deep_into_the_negative_tail():
    """Negatives below the top-k must not dilute the term -- that dilution is
    exactly what makes the min-anchor hinge a broad regulariser."""
    from zdock.train import loss_top_tail

    dq = torch.cat([torch.tensor([0.9]), torch.zeros(200)])
    s = torch.cat([torch.tensor([3.0]), torch.full((5,), 2.0),
                   torch.full((195,), -5.0)])
    few = torch.cat([torch.tensor([3.0]), torch.full((5,), 2.0)])
    dq_few = torch.cat([torch.tensor([0.9]), torch.zeros(5)])
    assert float(loss_top_tail(s, dq, k=5)) == \
        pytest.approx(float(loss_top_tail(few, dq_few, k=5)))


def test_top_tail_soft_max_sits_between_mean_and_max():
    """tau -> 0 approaches the true maxima; a large tau pulls both sides
    towards their means, which lowers the positive side and so RAISES the
    penalty."""
    from zdock.train import loss_top_tail

    dq = torch.tensor([0.9, 0.8] + [0.0] * 10)
    s = torch.tensor([2.0, 1.0] + [0.0] * 10)
    sharp = float(loss_top_tail(s, dq, tau_pos=0.01, tau_neg=0.01, k=4))
    soft = float(loss_top_tail(s, dq, tau_pos=2.0, tau_neg=2.0, k=4))
    # softplus(margin + n - p) decreases in p, and mean-exp <= max, so a
    # sharper p sits higher and gives the smaller penalty
    assert sharp < soft


def test_top_tail_does_not_reward_a_finely_sampled_basin():
    """A sum-exp positive side would grow like tau*log(count), paying a complex
    for having many near-native poses rather than for ranking one first."""
    from zdock.train import loss_top_tail

    s_few = torch.tensor([2.0, 1.9] + [0.0] * 50)
    dq_few = torch.tensor([0.9, 0.9] + [0.0] * 50)
    s_many = torch.tensor([2.0] + [1.9] * 60 + [0.0] * 50)
    dq_many = torch.tensor([0.9] * 61 + [0.0] * 50)
    assert abs(float(loss_top_tail(s_few, dq_few))
               - float(loss_top_tail(s_many, dq_many))) < 0.05


# ------------------------------------------------------- pairwise shaping loss


def test_shape_zero_when_nothing_is_near_native():
    """gamma = u(max DockQ), so a pool of pure garbage contributes nothing."""
    from zdock.train import loss_shape_pairwise

    s = torch.randn(50, requires_grad=True)
    loss = loss_shape_pairwise(s, torch.zeros(50))
    assert float(loss) == 0.0
    loss.backward()
    assert s.grad is not None


def test_shape_weight_scales_with_how_close_the_best_pose_is():
    """The whole point of using the absolute scale: a complex whose best pose is
    DockQ 0.02 must matter far less than one at 0.20."""
    from zdock.train import loss_shape_pairwise

    s = torch.tensor([0.0, 1.0] + [0.5] * 20)      # the better pose ranks below
    near = torch.tensor([0.20, 0.0] + [0.0] * 20)
    far = torch.tensor([0.02, 0.0] + [0.0] * 20)
    a = float(loss_shape_pairwise(s, near))
    b = float(loss_shape_pairwise(s, far))
    assert a > b > 0
    # QUADRATIC, not linear: gamma gates the complex and (u_i - u_j) weights the
    # pair, and both carry u(q_max) when the rival pose has DockQ 0. So the
    # median zero-positive complex (best DockQ 0.073) is down-weighted to
    # (0.073/0.23)^2 = 0.10, not 0.32 -- shaping is driven by the near misses.
    assert a / b == pytest.approx((0.20 / 0.02) ** 2, rel=1e-3)


def test_shape_rewards_putting_the_better_pose_first():
    from zdock.train import loss_shape_pairwise

    dq = torch.tensor([0.20, 0.01] + [0.0] * 20)
    bad = torch.tensor([0.0, 2.0] + [0.0] * 20)     # worse pose on top
    good = torch.tensor([2.0, 0.0] + [0.0] * 20)
    assert float(loss_shape_pairwise(good, dq)) < float(loss_shape_pairwise(bad, dq))


def test_shape_ignores_pairs_that_are_not_meaningfully_different():
    """delta_q keeps DockQ noise from being taught as an ordering."""
    from zdock.train import loss_shape_pairwise

    s = torch.tensor([0.0, 1.0] + [0.0] * 20)
    dq = torch.tensor([0.101, 0.100] + [0.099] * 20)   # every gap below delta_q
    assert float(loss_shape_pairwise(s, dq, delta_q=0.02)) == 0.0
    # and the same pool with one genuinely better pose is not zero
    dq2 = dq.clone()
    dq2[0] = 0.200
    assert float(loss_shape_pairwise(s, dq2, delta_q=0.02)) > 0.0


def test_shape_saturates_at_the_acceptable_threshold():
    """u is clamped at 1, so an acceptable pose and a high-quality one carry the
    same relevance -- above the threshold this term is not the right tool."""
    from zdock.train import loss_shape_pairwise

    s = torch.tensor([0.0, 1.0] + [0.0] * 20)
    at = torch.tensor([0.23, 0.0] + [0.0] * 20)
    above = torch.tensor([0.90, 0.0] + [0.0] * 20)
    assert float(loss_shape_pairwise(s, at)) == \
        pytest.approx(float(loss_shape_pairwise(s, above)))
