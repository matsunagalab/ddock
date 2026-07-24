"""Adam training loop for the 156 learnable scoring parameters (α,
iface_flat, charge_score), with B2 loss-function bug fixed. β is held
fixed at 3.0 because `score_elec` is linear (coulomb mode) or quadratic
(legacy mode) in `charge_score`, so any β can be absorbed into an
overall scaling of `charge_score` — training β separately just adds a
scale-redundant degree of freedom that worsens Adam dynamics.

The Julia notebook (cell 62) wrote the loss as:

    l = sum(...) / 9
      + sum(...) / 91
      + ...

which — due to Julia's expression-continuation rules — only assigned the
first term to `l`; the `+` lines started fresh expressions and were
discarded. The port below puts the six MSE terms inside a single `sum()`
call so all of them contribute to the gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from .atomtypes import charge_score as default_charge_score
from .atomtypes import iface_ij
from .score import docking_score_elec


@dataclass
class ProteinInputs:
    """Bundle of prepared inputs for one receptor–ligand pair, as
    docking_score_elec expects them (see `score.py` docstring for the
    preconditions on decenter / mass init / charge id)."""
    rec_xyz: torch.Tensor
    rec_radius: torch.Tensor
    rec_sasa: torch.Tensor
    rec_atomtype_id: torch.Tensor
    rec_charge_id: torch.Tensor
    lig_xyz: torch.Tensor              # (F, N_lig, 3)
    lig_radius: torch.Tensor
    lig_sasa: torch.Tensor
    lig_atomtype_id: torch.Tensor
    lig_charge_id: torch.Tensor
    hit_mask: torch.Tensor             # (F,) bool: True for Positive poses
    rmsd: torch.Tensor | None = None   # (F,) optional raw RMSD (Å) to re-threshold later
    dockq: torch.Tensor | None = None  # (F,) optional DockQ [0, 1] per pose
                                       # — required by loss="dockq_rank"

    def call(
        self,
        alpha: torch.Tensor,
        iface: torch.Tensor,
        beta: torch.Tensor,
        charge: torch.Tensor,
        *,
        frame_chunk_size: int | None = None,
    ) -> torch.Tensor:
        return docking_score_elec(
            self.rec_xyz, self.rec_radius, self.rec_sasa,
            self.rec_atomtype_id, self.rec_charge_id,
            self.lig_xyz, self.lig_radius, self.lig_sasa,
            self.lig_atomtype_id, self.lig_charge_id,
            alpha, iface, beta, charge,
            frame_chunk_size=frame_chunk_size,
        )


def make_ideal_targets(initial_scores: torch.Tensor, hit_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Julia's "ideal score distribution" scheme: all Hit (Positive) poses
    get the *maximum* of the pre-training score distribution as their
    target (we want them ranked highest), all Miss (Negative) poses get
    the *minimum* target. Returns (hit_target, miss_target) per pose —
    scalars broadcast to match."""
    hit_target = initial_scores.max()
    miss_target = initial_scores.min()
    return hit_target, miss_target


def loss_b2_fixed(
    scores: torch.Tensor,
    hit_mask: torch.Tensor,
    hit_target: torch.Tensor,
    miss_target: torch.Tensor,
) -> torch.Tensor:
    """Two-term MSE split by Hit/Miss with per-class normalization. This
    is the per-protein loss; the full Julia loss sums three such
    per-protein contributions (six terms total).

    B2 fix: make sure *both* sides actually contribute to the returned
    tensor. In the original Julia notebook the `+` at the start of each
    line broke expression continuation, so only the first term was kept.
    """
    hit_sq = ((scores[hit_mask] - hit_target) ** 2)
    miss_sq = ((scores[~hit_mask] - miss_target) ** 2)
    # Use mean per class so protein size doesn't bias the gradient;
    # sum is also valid — the Julia notebook used `/ n_hit` and `/ n_miss`
    # constants, equivalent to `mean()` here.
    hit_term = hit_sq.mean() if hit_sq.numel() > 0 else torch.zeros((), device=scores.device, dtype=scores.dtype)
    miss_term = miss_sq.mean() if miss_sq.numel() > 0 else torch.zeros((), device=scores.device, dtype=scores.dtype)
    return hit_term + miss_term


def loss_listnet(
    scores: torch.Tensor,
    rmsd: torch.Tensor,
    *,
    temperature: float = 5.0,
) -> torch.Tensor:
    """ListNet listwise cross-entropy ranking loss: push the score
    ranking to match the ideal ranking induced by ascending RMSD.

        target_i = softmax(-rmsd / T)_i     # low RMSD -> high target prob
        pred_i   = softmax(scores)_i
        loss     = -sum_i target_i * log pred_i

    Temperature T (in Å) controls target peakedness: T -> 0 concentrates
    mass on the single lowest-RMSD pose (sparse gradient); T -> infty
    approaches uniform. Default 5.0 Å gives a reasonable spread across
    the BM4 top-K RMSD range (~2-30 Å).

    Unlike `loss_b2_fixed`, this uses the raw RMSD values directly and
    does not depend on a Hit/Miss threshold — continuous ordering
    information within each class is preserved.
    """
    if rmsd.numel() == 0:
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    target = torch.softmax(-rmsd / temperature, dim=0)
    log_pred = torch.log_softmax(scores, dim=0)
    return -(target * log_pred).sum()


def loss_listnet_dockq(
    scores: torch.Tensor,
    dockq: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> torch.Tensor:
    """ListNet listwise cross-entropy with DockQ as the relevance signal.

        target_i = softmax(dockq / T)_i   # HIGH dockq -> high target prob
        pred_i   = softmax(scores)_i
        loss     = -sum_i target_i * log pred_i

    Sign is flipped relative to `loss_listnet` (RMSD is lower-is-better;
    DockQ is higher-is-better). Default ``temperature=0.2`` puts most of
    the target mass on the top-DockQ decoys while still carrying
    gradient through the Medium/High tier — DockQ lives in [0, 1] so
    the temperature scale is dimensionless and much smaller than the
    RMSD-temperature default.

    Returns a scalar loss. Input poses with all-zero DockQ (no positive
    examples) give a uniform target ⇒ the loss equals
    ``log F - mean(log softmax(scores))`` which still has gradient.
    """
    if dockq.numel() == 0:
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    target = torch.softmax(dockq / temperature, dim=0)
    log_pred = torch.log_softmax(scores, dim=0)
    return -(target * log_pred).sum()


def loss_margin_hard_negatives(
    scores: torch.Tensor,
    dockq: torch.Tensor,
    *,
    positive_threshold: float = 0.23,
    margin: float = 1.0,
) -> torch.Tensor:
    """Hinge-style penalty on high-scoring non-positives.

    Define positives as poses with DockQ >= ``positive_threshold``
    (CAPRI "acceptable" by default). For every negative pose whose
    score exceeds ``min(positive scores) - margin``, add a hinge
    penalty:

        L = mean_neg( max(0, margin + score_neg - min_pos_score) )

    Intuition: the scorer must rank the WORST positive above every
    negative by at least ``margin``. Targets the "FFT finds a
    high-scoring non-native" failure mode directly.

    Edge cases:
    - No positives in the batch → return zero (no signal to give).
    - No negatives → return zero.
    """
    pos_mask = dockq >= positive_threshold
    neg_mask = ~pos_mask
    if not pos_mask.any() or not neg_mask.any():
        # Graph-connected zero: keeps autograd happy in a training
        # loop that mixes proteins with and without positive examples.
        # A protein with no positives contributes no margin signal but
        # also doesn't block backward().
        return (scores * 0.0).sum()
    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]
    min_pos = pos_scores.min()
    hinge = torch.clamp(margin + neg_scores - min_pos, min=0.0)
    return hinge.mean()


def total_loss(
    proteins: Iterable[ProteinInputs],
    alpha: torch.Tensor,
    iface: torch.Tensor,
    beta: torch.Tensor,
    charge: torch.Tensor,
    targets: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Sum per-protein losses. `targets[i]` = (hit_target, miss_target)
    for proteins[i]."""
    proteins = list(proteins)
    if len(proteins) == 0:
        raise ValueError("total_loss requires at least one protein; got an empty list")
    if len(proteins) != len(targets):
        raise ValueError(
            f"proteins and targets must have equal length; got "
            f"{len(proteins)} proteins, {len(targets)} targets"
        )
    parts = []
    for p, (ht, mt) in zip(proteins, targets):
        scores = p.call(alpha, iface, beta, charge)
        parts.append(loss_b2_fixed(scores, p.hit_mask, ht, mt))
    return torch.stack(parts).sum()


_VALID_LOSSES = ("split_mse", "rank", "dockq_rank", "dockq_margin")


def train(
    proteins: list[ProteinInputs],
    *,
    n_epoch: int = 200,
    lr: float = 0.01,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    progress_every: int = 10,
    log: Callable[[str], None] = print,
    frame_chunk_size: int | None = None,
    loss: str = "split_mse",
    listnet_temperature: float = 5.0,
    dockq_temperature: float = 0.2,
    margin_positive_threshold: float = 0.23,
    margin: float = 1.0,
) -> dict:
    """Run the full Adam optimization and return the trained parameters
    plus a loss history.

    `loss` selects the per-protein objective (each is a *pure* term —
    they are exposed separately so experiments can compare them
    head-to-head; for a combined-term run, call `train()` twice or
    compose externally):

      - "split_mse" (default): the Julia `loss_b2_fixed` — Hit poses
        pulled to max(initial_scores), Miss poses to min. Targets
        are frozen at pre-training values.
      - "rank": ListNet on RMSD — `loss_listnet(scores, rmsd,
        temperature=listnet_temperature)`. Requires every protein
        to carry `rmsd`.
      - "dockq_rank": ListNet on DockQ — `loss_listnet_dockq(scores,
        dockq, temperature=dockq_temperature)`. Requires every
        protein to carry `dockq`. Higher DockQ → higher target prob,
        so the sign is flipped relative to "rank".
      - "dockq_margin": hard-negative hinge on DockQ —
        `loss_margin_hard_negatives(scores, dockq,
        positive_threshold=margin_positive_threshold,
        margin=margin)`. Directly penalises negatives (DockQ below
        threshold) that outscore the weakest positive; the
        pin-point antidote to the observed "FFT top-1 is
        non-native" failure mode. Requires every protein to carry
        `dockq`.

    The per-epoch loss is the sum of per-protein losses. We compute + run
    backward *per protein* so only one protein's autograd graph lives at
    a time (the Adam step still sees the sum of all per-protein gradients,
    which is mathematically identical to `sum(...).backward()`). This
    keeps peak VRAM from scaling with the number of proteins in the
    training list.

    `frame_chunk_size` is forwarded into `docking_score_elec` so the
    per-frame (F) memory also stays bounded — see that function's
    docstring.

    Raises ValueError if `progress_every <= 0`, `proteins` is empty, or
    `loss` is unknown.
    """
    if progress_every <= 0:
        raise ValueError(
            f"progress_every must be a positive integer, got {progress_every}"
        )
    if len(proteins) == 0:
        raise ValueError("train requires at least one protein; got an empty list")
    if loss not in _VALID_LOSSES:
        raise ValueError(
            f"unknown loss {loss!r}; must be one of {_VALID_LOSSES}"
        )
    if loss == "rank":
        for i, p in enumerate(proteins):
            if p.rmsd is None:
                raise ValueError(
                    f"loss='rank' requires rmsd on every protein, but "
                    f"proteins[{i}] has rmsd=None"
                )
    if loss in ("dockq_rank", "dockq_margin"):
        for i, p in enumerate(proteins):
            if p.dockq is None:
                raise ValueError(
                    f"loss={loss!r} requires dockq on every protein, but "
                    f"proteins[{i}] has dockq=None"
                )
    # Initial parameters — same as Julia `train_param-apart.ipynb` cell
    # 27-28 defaults.
    alpha = torch.tensor(0.01, device=device, dtype=dtype, requires_grad=True)
    beta = torch.tensor(3.0, device=device, dtype=dtype)
    iface_init = iface_ij(device=device, dtype=dtype, flat=True).clone()
    iface = iface_init.detach().requires_grad_(True)
    charge_init = default_charge_score(device=device, dtype=dtype).clone()
    charge = charge_init.detach().requires_grad_(True)

    # split_mse freezes per-protein targets from pre-training scores;
    # rank doesn't use targets at all.
    if loss == "split_mse":
        with torch.no_grad():
            targets = []
            for p in proteins:
                s0 = p.call(
                    alpha, iface, beta, charge,
                    frame_chunk_size=frame_chunk_size,
                )
                targets.append(make_ideal_targets(s0, p.hit_mask))
    else:
        targets = [None] * len(proteins)

    opt = torch.optim.Adam([alpha, iface, charge], lr=lr)
    history = {"loss": []}

    for epoch in range(n_epoch):
        opt.zero_grad()
        epoch_loss = 0.0
        for p, tgt in zip(proteins, targets):
            scores = p.call(
                alpha, iface, beta, charge,
                frame_chunk_size=frame_chunk_size,
            )
            if loss == "rank":
                lp = loss_listnet(
                    scores, p.rmsd, temperature=listnet_temperature,
                )
            elif loss == "dockq_rank":
                lp = loss_listnet_dockq(
                    scores, p.dockq, temperature=dockq_temperature,
                )
            elif loss == "dockq_margin":
                lp = loss_margin_hard_negatives(
                    scores, p.dockq,
                    positive_threshold=margin_positive_threshold,
                    margin=margin,
                )
            else:  # split_mse
                ht, mt = tgt
                lp = loss_b2_fixed(scores, p.hit_mask, ht, mt)
            lp.backward()
            epoch_loss += float(lp.detach())
        opt.step()
        history["loss"].append(epoch_loss)
        if epoch % progress_every == 0 or epoch == n_epoch - 1:
            log(f"epoch {epoch:4d}  loss={epoch_loss:.6e}")

    return {
        "alpha": alpha.detach(),
        "iface": iface.detach(),
        "charge": charge.detach(),
        "history": history,
    }
