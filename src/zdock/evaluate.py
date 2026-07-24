"""Ranking-quality evaluation for a docking scorer.

Given candidate poses with precomputed RMSD / DockQ labels and a set of
156 scoring parameters, rank the poses by ``docking_score_elec`` and
report CAPRI-style success rates: does a near-native pose appear within
the top-K by score?

Because the repository's DockQ is an atom-level approximation (no
residue/backbone metadata), we report success under **both** a DockQ
threshold and a ligand-RMSD threshold, so a reader can cross-check the
two definitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .train import ProteinInputs


@dataclass
class RankingReport:
    n_poses: int
    n_hit_rmsd: int
    n_hit_dockq: int
    # success@K under RMSD and DockQ definitions
    success_rmsd: dict[int, bool]
    success_dockq: dict[int, bool]
    # best-quality-in-top-K
    best_dockq_at: dict[int, float]
    min_rmsd_at: dict[int, float]
    # rank (1-based) of the single best pose by each label
    rank_of_best_dockq: int
    rank_of_best_rmsd: int


@torch.no_grad()
def score_poses(
    prot: ProteinInputs,
    alpha: torch.Tensor,
    iface_flat: torch.Tensor,
    beta: torch.Tensor,
    charge: torch.Tensor,
    *,
    frame_chunk_size: int | None = 256,
) -> torch.Tensor:
    return prot.call(alpha, iface_flat, beta, charge,
                     frame_chunk_size=frame_chunk_size)


def evaluate_ranking(
    scores: torch.Tensor,
    rmsd: torch.Tensor,
    dockq: torch.Tensor,
    *,
    ks: tuple[int, ...] = (1, 5, 10, 50, 100),
    rmsd_threshold: float = 5.0,
    dockq_threshold: float = 0.23,
) -> RankingReport:
    order = torch.argsort(scores, descending=True)
    rmsd_s = rmsd[order]
    dockq_s = dockq[order]
    n = scores.numel()

    hit_rmsd = rmsd <= rmsd_threshold
    hit_dockq = dockq >= dockq_threshold

    success_rmsd, success_dockq = {}, {}
    best_dockq_at, min_rmsd_at = {}, {}
    for k in ks:
        kk = min(k, n)
        success_rmsd[k] = bool((rmsd_s[:kk] <= rmsd_threshold).any())
        success_dockq[k] = bool((dockq_s[:kk] >= dockq_threshold).any())
        best_dockq_at[k] = float(dockq_s[:kk].max())
        min_rmsd_at[k] = float(rmsd_s[:kk].min())

    rank_best_dockq = int((order == int(dockq.argmax())).nonzero()[0, 0]) + 1
    rank_best_rmsd = int((order == int(rmsd.argmin())).nonzero()[0, 0]) + 1

    return RankingReport(
        n_poses=n,
        n_hit_rmsd=int(hit_rmsd.sum()),
        n_hit_dockq=int(hit_dockq.sum()),
        success_rmsd=success_rmsd,
        success_dockq=success_dockq,
        best_dockq_at=best_dockq_at,
        min_rmsd_at=min_rmsd_at,
        rank_of_best_dockq=rank_best_dockq,
        rank_of_best_rmsd=rank_best_rmsd,
    )


def format_report(name: str, rep: RankingReport) -> str:
    ks = sorted(rep.success_rmsd)
    lines = [
        f"[{name}] F={rep.n_poses}  hits(rmsd)={rep.n_hit_rmsd}  "
        f"hits(dockq)={rep.n_hit_dockq}  "
        f"rank_best_dockq={rep.rank_of_best_dockq}  "
        f"rank_best_rmsd={rep.rank_of_best_rmsd}",
    ]
    hdr = "   K  | succ_rmsd succ_dockq | minRMSD@K  bestDockQ@K"
    lines.append(hdr)
    for k in ks:
        lines.append(
            f"  {k:>4} |    {int(rep.success_rmsd[k])}         {int(rep.success_dockq[k])}     "
            f"|   {rep.min_rmsd_at[k]:6.2f}     {rep.best_dockq_at[k]:5.3f}"
        )
    return "\n".join(lines)
