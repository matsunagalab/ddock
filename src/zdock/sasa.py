"""Solvent-accessible surface area (SASA) via the golden-section spiral
approximation, ported from `docking.jl::compute_sasa` (lines 921–977).

Per atom i, we distribute `npoint` (default 960) probe points on the sphere
of radius r_i + probe centered at atom i, then count how many are *not*
inside r_j + probe of any other atom j. SASA[i] = 4π(r_i+probe)² × (accessible
/ npoint).

The GPU-friendly strategy:

  * Generate the unit-sphere sample once on-device via `golden_section_spiral`.
  * Batch atoms in chunks of `atom_chunk` (default 128) to bound the peak
    memory of the (chunk × M × N) distance tensor.
  * Use a pairwise cutoff mask (``d² < (2·r_max + 2·probe)²``) to cheaply
    discard far-away atoms from the inner `any(...)` reduction without
    shuffling data into ragged neighbor lists.

All ops are pure `torch` broadcasts / reductions — no Python loops over
atoms or sphere points in the hot path.
"""

from __future__ import annotations

import math

import torch

from .geom import golden_section_spiral


def compute_sasa(
    xyz: torch.Tensor,
    radius: torch.Tensor,
    *,
    probe_radius: float = 1.4,
    npoint: int = 960,
    atom_chunk: int | None = None,
) -> torch.Tensor:
    """Return SASA per atom as a (N,) tensor.

    Parameters
    ----------
    xyz      : (N, 3) atom coordinates on the target device.
    radius   : (N,)   vdW radii, same device/dtype as xyz.
    probe_radius : solvent probe radius in Å (default 1.4 = water).
    npoint   : number of sphere points per atom (default 960, matches Julia).
    atom_chunk : how many atoms to process per batch. Lower values reduce
                 peak memory on small accelerators (e.g. MPS). The default
                 keeps the distance tensor under ~512 MB for typical protein
                 sizes in float32.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3), got {tuple(xyz.shape)}")
    if radius.shape != (xyz.shape[0],):
        raise ValueError(
            f"radius shape {tuple(radius.shape)} inconsistent with xyz N={xyz.shape[0]}"
        )
    if xyz.device != radius.device:
        raise ValueError("xyz and radius must share the same device")

    device = xyz.device
    dtype = xyz.dtype
    N = xyz.shape[0]

    # Default atom_chunk depends on device: CPU float64 segfaults above ~64
    # (libomp thread race) and is memory-happy, so stay conservative. GPU /
    # MPS prefers large chunks to amortize kernel launch overhead — the
    # (B, M, N) float32 tensor is ~3.75N MB per chunk of size B=256.
    if atom_chunk is None:
        # Neighbor-packed SASA drops the inner dim from N → K_max (<< N) so
        # the chunk tensor shrinks by ~100×. We can afford large chunks
        # on GPU / MPS for minimum launch overhead.
        atom_chunk = 16 if device.type == "cpu" else 512

    # Unit sphere sample (M, 3).
    sphere = golden_section_spiral(npoint, device=device, dtype=dtype)

    r_plus = radius + probe_radius                         # (N,)
    r_plus_sq = r_plus * r_plus                            # (N,)

    # Atom-atom pairwise cutoff mask. max_reach = 2·(max(r) + probe) is an
    # overapproximation — any pair outside this can never have one atom's
    # sphere reach the other, so we mask them out up front. We compute d²
    # via |a|² + |b|² - 2·a·b to avoid the (N, N, 3) broadcast or the
    # occasionally-flaky `torch.cdist` kernel on some backends.
    max_reach_sq = (2.0 * (radius.max().item() + probe_radius)) ** 2
    sq_norm = xyz.pow(2).sum(dim=-1)                        # (N,)
    gram = xyz @ xyz.T                                      # (N, N)
    pair_d2 = sq_norm.unsqueeze(0) + sq_norm.unsqueeze(1) - 2.0 * gram
    # Numerical floor: (a-a)^2 can go slightly negative on round-off; clamp.
    pair_d2.clamp_(min=0.0)
    within = pair_d2 < max_reach_sq                         # (N, N) bool
    within.fill_diagonal_(False)

    # Pack neighbor indices per atom into a dense (N, K_max) tensor. For
    # rows with fewer neighbors than K_max, the unused slots store atom
    # index 0 (whatever), flagged by `neighbor_valid`.
    neighbor_count = within.sum(dim=1)                      # (N,)
    K_max = int(neighbor_count.max().item())
    # Sort each row so True (neighbor) entries come first. `stable=True`
    # keeps original atom indexing within the True block.
    sort_key = (~within).to(torch.int8)
    order = sort_key.argsort(dim=1, stable=True)            # (N, N)
    neighbor_idx = order[:, :K_max]                         # (N, K_max)
    neighbor_valid = within.gather(1, neighbor_idx)         # (N, K_max)

    sasa = torch.empty(N, device=device, dtype=dtype)
    four_pi = 4.0 * math.pi

    for start in range(0, N, atom_chunk):
        end = min(start + atom_chunk, N)
        B = end - start

        # Surface points for atoms [start, end):
        # (B, M, 3) = xyz[i] + (r_i + probe) * sphere
        surf = xyz[start:end].unsqueeze(1) + (
            r_plus[start:end].unsqueeze(-1).unsqueeze(-1) * sphere.unsqueeze(0)
        )

        idx_chunk = neighbor_idx[start:end]                 # (B, K_max)
        valid_chunk = neighbor_valid[start:end]             # (B, K_max)

        # Neighbor positions & thresholds for each atom in the chunk.
        n_xyz = xyz[idx_chunk]                              # (B, K_max, 3)
        n_thr = r_plus_sq[idx_chunk]                        # (B, K_max)

        # Surf-to-neighbor distances: (B, M, K_max)
        diff = surf.unsqueeze(2) - n_xyz.unsqueeze(1)
        d2 = diff.pow(2).sum(dim=-1)

        occ = (d2 < n_thr.unsqueeze(1)) & valid_chunk.unsqueeze(1)
        point_occluded = occ.any(dim=-1)                    # (B, M)
        accessible = (~point_occluded).to(dtype).mean(dim=-1)  # (B,)

        sasa[start:end] = four_pi * r_plus[start:end].pow(2) * accessible

    return sasa
