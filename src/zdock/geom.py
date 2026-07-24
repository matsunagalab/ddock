"""Geometry utilities: quaternion rotation, golden-section spiral sphere
sampling, and docking grid generation. All ops are fully vectorized and
accept a `device` argument so they run on CUDA/MPS without changes.

Parity with Julia sources:
  * `rotate` ↔ `docking.jl` `rotate!` (lines 1469–1490): same convention,
    same rotation matrix entries.
  * `golden_section_spiral` ↔ `docking.jl` lines 906–919.
  * `generate_grid` ↔ `train_param-apart.ipynb` cell 4 override (which is
    identical to `docking.jl` `generate_grid` but preferred because the
    training notebook always shadowed the base function).
"""

from __future__ import annotations

import math

import torch

_PI = math.pi
_GOLDEN_INC = _PI * (3.0 - math.sqrt(5.0))


def rotate(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate coordinates (x, y, z) by quaternion q = (q1, q2, q3, q4).

    Matches docking.jl `rotate!` (lines 1469–1490) element-for-element. Note
    that the Julia convention uses q[1..4] with these rotation-matrix entries,
    which is *not* quite the canonical (w, x, y, z) ordering — we reproduce
    it verbatim so numerical outputs match the reference.

    Returns new tensors (non-mutating). Inputs and q must share the same
    device/dtype.
    """
    if q.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {tuple(q.shape)}")

    q1, q2, q3, q4 = q.unbind(0)

    # Julia indices 1..4 → Python 0..3. The formulas below are copied
    # verbatim from docking.jl, with Julia q[1] → q1 etc.
    r1 = 1.0 - 2.0 * q2 * q2 - 2.0 * q3 * q3
    r2 = 2.0 * (q1 * q2 + q3 * q4)
    r3 = 2.0 * (q1 * q3 - q2 * q4)
    r4 = 2.0 * (q1 * q2 - q3 * q4)
    r5 = 1.0 - 2.0 * q1 * q1 - 2.0 * q3 * q3
    r6 = 2.0 * (q2 * q3 + q1 * q4)
    r7 = 2.0 * (q1 * q3 + q2 * q4)
    r8 = 2.0 * (q2 * q3 - q1 * q4)
    r9 = 1.0 - 2.0 * q1 * q1 - 2.0 * q2 * q2

    x_new = r1 * x + r2 * y + r3 * z
    y_new = r4 * x + r5 * y + r6 * z
    z_new = r7 * x + r8 * y + r9 * z
    return x_new, y_new, z_new


def golden_section_spiral(
    n: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an (n, 3) tensor of approximately uniformly-distributed points
    on the unit sphere. Reproduces docking.jl `golden_section_spiral` exactly.

    Fully vectorized — no Python for-loop, so this is instant even for n ≈ 10^5.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    offset = 2.0 / float(n)
    k = torch.arange(n, device=device, dtype=dtype)
    y = k * offset - 1.0 + (offset / 2.0)
    r = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
    # Large k makes phi = k * inc accumulate to ~2300 rad for n=960, which
    # costs float32 accuracy near +/-1 in cos/sin. Normalize to [0, 2π)
    # before the trig so MPS/CUDA float32 matches CPU float64 to ~1e-5.
    phi = torch.remainder(k * _GOLDEN_INC, 2.0 * _PI)
    x_coord = torch.cos(phi) * r
    z_coord = torch.sin(phi) * r
    return torch.stack([x_coord, y, z_coord], dim=1)


def decenter(
    xyz: torch.Tensor,
    mass: torch.Tensor | None = None,
) -> torch.Tensor:
    """Translate coordinates so that their centre of mass is at the origin.

    Mirrors MDToolbox's `decenter!` (mass-weighted if `mass` is given,
    otherwise unweighted). Returns a new tensor; does not mutate input.
    """
    if mass is None:
        com = xyz.mean(dim=0, keepdim=True)
    else:
        w = mass.to(xyz.dtype).unsqueeze(-1)
        com = (w * xyz).sum(dim=0, keepdim=True) / w.sum()
    return xyz - com


def orient(
    xyz: torch.Tensor,
    mass: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate a decentered molecule so that its principal axes of inertia
    align with the Cartesian axes (longest axis → x, shortest → z).

    Port of `MDToolbox.orient!` (src/structure.jl lines 156–205):

      1. Decenter (mass-weighted if provided).
      2. Build the symmetric inertia tensor I (3×3).
      3. `svd(I)` gives singular vectors in *descending* order of singular
         value; Julia takes `F.Vt[end:-1:1, :]` so the row order is
         *smallest first* → smallest inertia ⇒ longest protein axis ⇒ x.
      4. If `det(p_axis) < 0`, flip the first row to keep a right-handed
         basis.
      5. Project the coordinates onto the new basis: `xyz' = p_axis @ xyz.T`.

    Sign ambiguity: the eigenvectors of the inertia tensor are only defined
    up to overall sign per axis, and different SVD backends (CPU LAPACK,
    cuSOLVER on CUDA, MPS's fallback) disagree on which sign to pick. To
    make `orient()` bit-deterministic across devices — and to match Julia
    (which uses CPU LAPACK) — we always run the 3×3 SVD on CPU and move
    the result back. The tensor is tiny, so the host↔device copy is a few
    microseconds regardless of dtype or device.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3), got {tuple(xyz.shape)}")

    centered = decenter(xyz, mass)
    dtype = centered.dtype
    device = centered.device

    if mass is None:
        m = torch.ones(centered.shape[0], device=device, dtype=dtype)
    else:
        m = mass.to(dtype)

    x = centered[:, 0]
    y = centered[:, 1]
    z = centered[:, 2]

    I = torch.empty((3, 3), device=device, dtype=dtype)
    I[0, 0] = (m * (y * y + z * z)).sum()
    I[1, 1] = (m * (x * x + z * z)).sum()
    I[2, 2] = (m * (x * x + y * y)).sum()
    I[0, 1] = -(m * x * y).sum()
    I[1, 0] = I[0, 1]
    I[0, 2] = -(m * x * z).sum()
    I[2, 0] = I[0, 2]
    I[1, 2] = -(m * y * z).sum()
    I[2, 1] = I[1, 2]

    # Always run the 3×3 SVD on CPU: cuSOLVER (CUDA) and the MPS fallback
    # pick different sign conventions from CPU LAPACK, which makes
    # orient() backend-dependent — and diverges from Julia, which also
    # uses CPU LAPACK. Forcing CPU is a 3×3 copy, effectively free.
    U, S, Vt = torch.linalg.svd(I.cpu())
    Vt = Vt.to(device=device, dtype=dtype)
    p_axis = torch.flip(Vt, dims=[0]).contiguous()  # smallest→largest rows

    # Reflection guard (Julia's sole sign check): ensure right-handed.
    if torch.det(p_axis) < 0:
        p_axis[0] = -p_axis[0]

    # Project: new_xyz[atom, :] = p_axis @ centered[atom, :]
    return centered @ p_axis.T


def generate_grid(
    receptor_xyz: torch.Tensor,
    ligand_xyz: torch.Tensor,
    *,
    spacing: float = 1.2,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the Cartesian FFT grid large enough to hold any translated
    receptor–ligand pose, matching train_param-apart.ipynb cell 4.

    Both coordinate tensors are (N, 3) — already decentered/oriented as the
    caller sees fit. The grid axis range is
        [min_receptor - size_ligand - spacing, max_receptor + size_ligand + spacing]
    per axis, where size_ligand is the *x*-extent of the ligand (this matches
    the Julia code even though it's a slight asymmetry across axes).

    Returns (grid_real, grid_imag, x_grid, y_grid, z_grid). Both grids are
    zero-initialized; callers scatter into them.
    """
    if receptor_xyz.ndim != 2 or receptor_xyz.shape[1] != 3:
        raise ValueError(f"receptor_xyz must be (N, 3), got {tuple(receptor_xyz.shape)}")
    if ligand_xyz.ndim != 2 or ligand_xyz.shape[1] != 3:
        raise ValueError(f"ligand_xyz must be (M, 3), got {tuple(ligand_xyz.shape)}")

    if device is None:
        device = receptor_xyz.device
    if dtype is None:
        dtype = receptor_xyz.dtype

    # Julia uses only the x-axis extent of the ligand for padding.
    size_ligand = (ligand_xyz[:, 0].max() - ligand_xyz[:, 0].min()).item()

    def axis(vals: torch.Tensor) -> torch.Tensor:
        lo = vals.min().item() - size_ligand - spacing
        hi = vals.max().item() + size_ligand + spacing
        # Julia `range(lo, hi; step=spacing)` produces lo + i*spacing for
        # i = 0..n-1, where n = floor((hi - lo) / spacing) + 1. We compute n
        # explicitly to avoid torch.arange's half-open endpoint behaviour
        # picking up an extra element due to float round-off.
        n = int((hi - lo) / spacing) + 1
        return lo + spacing * torch.arange(n, device=device, dtype=dtype)

    x_grid = axis(receptor_xyz[:, 0])
    y_grid = axis(receptor_xyz[:, 1])
    z_grid = axis(receptor_xyz[:, 2])

    nx, ny, nz = x_grid.numel(), y_grid.numel(), z_grid.numel()
    grid_real = torch.zeros((nx, ny, nz), device=device, dtype=dtype)
    grid_imag = torch.zeros((nx, ny, nz), device=device, dtype=dtype)
    return grid_real, grid_imag, x_grid, y_grid, z_grid
