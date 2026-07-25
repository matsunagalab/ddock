"""Quaternion grid generators for the FFT docking search.

ZDOCK 3.0.2 uses a specific Euler-angle table at 6° spacing. This
module provides simpler drop-in alternatives that are adequate for
demonstration and experimentation:

  * `random_quaternions` — N uniformly-distributed SO(3) rotations
    via `scipy.spatial.transform.Rotation.random`.
  * `euler_quaternions` — (φ, θ, ψ) ZYZ Euler grid at `deg` spacing.

Exact ZDOCK-table reproduction is a future refinement (see
PORT_PLAN_FFT.md).

**Quaternion convention**: this module returns quaternions in the
format consumed by `geom.rotate` (= `docking.jl::rotate!`), which is
the *inverse* of scipy's active-rotation convention. For a uniform
sampler this is immaterial — the inverted distribution is still
uniform on SO(3). If you need to interoperate with scipy's
`Rotation.apply` outcome-by-outcome, apply the quaternion conjugate
`(x, y, z, w) → (−x, −y, −z, w)`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _ScipyRotation


def random_quaternions(
    n: int,
    *,
    seed: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return `(n, 4)` uniformly-distributed SO(3) quaternions via
    `scipy.spatial.transform.Rotation.random`.

    Output is in `geom.rotate` / Julia `rotate!` convention (see
    module docstring). For uniform sampling the convention choice
    has no effect on the distribution.
    """
    r = _ScipyRotation.random(n, random_state=seed)
    q = r.as_quat()  # (n, 4) scalar-last (x, y, z, w)
    q = torch.as_tensor(q, dtype=dtype)
    if device is not None:
        q = q.to(device)
    return q


def scipy_rotations_to_quaternions(
    r: _ScipyRotation,
    *,
    as_inverse: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Convert a `scipy.spatial.transform.Rotation` (single or batch)
    to `(N, 4)` quaternions for use with `geom.rotate`.

    `as_inverse=True` (default) emits the quaternion conjugate of
    scipy's output so that `geom.rotate(v, q)` produces the same
    rotated vector as `r.apply(v)`. Set to False to pass the raw
    scipy quaternions (equivalent to applying the inverse rotation
    under our convention — fine for uniform samplers but not for
    known oriented rotations).
    """
    q = r.as_quat().reshape(-1, 4)
    if as_inverse:
        q = q.copy()
        q[:, :3] *= -1  # quaternion conjugate = inverse of unit quat
    q_t = torch.as_tensor(np.ascontiguousarray(q), dtype=dtype)
    if device is not None:
        q_t = q_t.to(device)
    return q_t


def kabsch_quaternion(
    ref_xyz: torch.Tensor,              # (N, 3)
    target_xyz: torch.Tensor,           # (N, 3)
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the quaternion that best rotates `ref_xyz` onto
    `target_xyz` in the least-squares (Kabsch) sense, expressed in
    the `geom.rotate` convention.

    Both point sets are first decentered (COM subtracted) so the
    Kabsch alignment is purely rotational. Equivalent to running
    `scipy.spatial.transform.Rotation.align_vectors(target, ref)` on
    the centered sets and then converting via
    `scipy_rotations_to_quaternions(as_inverse=True)`.

    Returns a `(4,)` quaternion. Caller still needs to separately
    supply the translation `target_COM - ref_COM` to produce the
    full pose.
    """
    if ref_xyz.shape != target_xyz.shape:
        raise ValueError(
            f"ref and target must share shape; got {tuple(ref_xyz.shape)} "
            f"vs {tuple(target_xyz.shape)}"
        )
    if ref_xyz.ndim != 2 or ref_xyz.shape[1] != 3:
        raise ValueError(f"inputs must be (N, 3), got {tuple(ref_xyz.shape)}")

    ref_np = ref_xyz.detach().cpu().numpy().astype(np.float64)
    target_np = target_xyz.detach().cpu().numpy().astype(np.float64)
    ref_c = ref_np - ref_np.mean(axis=0)
    target_c = target_np - target_np.mean(axis=0)

    # Rotation that takes ref onto target (i.e., R · ref ≈ target).
    r_align, _ = _ScipyRotation.align_vectors(target_c, ref_c)
    q = scipy_rotations_to_quaternions(
        r_align, as_inverse=True, device=device, dtype=dtype,
    )
    return q.squeeze(0)  # (4,)


def rotation_cone(
    q_center: torch.Tensor,             # (4,) in geom.rotate convention
    n: int,
    *,
    cone_deg: float = 15.0,
    seed: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return `(n, 4)` quaternions sampled within an angular cone of
    radius `cone_deg` degrees around `q_center`.

    Sampling scheme: for each sample, draw a uniform axis on S² and an
    angle θ ~ Uniform(0, cone_deg). Compose δ(axis, θ) · q_center.

    Angle is uniform in [0, cone_deg]. NOTE this over-weights the **centre**,
    not the shell: the Haar density in θ goes like (1 − cos θ), so the
    innermost 5° bin is ~25x over-represented relative to a volume-uniform
    sample of the same cone (measured at cone_deg=25 over 200k samples:
    39,917 vs 1,615). Since this cone is seeded on the native orientation, the
    leak it introduces is correspondingly ~25x more concentrated on the answer
    than "uniform in the cone" would suggest. For a volume-uniform cone, invert
    F(θ) = (θ − sin θ)/(θ_max − sin θ_max) instead.
    """
    if q_center.shape != (4,):
        raise ValueError(f"q_center must be (4,), got {tuple(q_center.shape)}")
    if cone_deg < 0 or cone_deg > 180:
        raise ValueError(f"cone_deg must be in [0, 180], got {cone_deg}")

    if device is None:
        device = q_center.device
    if dtype is None:
        dtype = q_center.dtype

    rng = np.random.default_rng(seed)

    # Random axis uniformly on S² via Gaussian normalization.
    v = rng.standard_normal((n, 3))
    v = v / np.linalg.norm(v, axis=1, keepdims=True)

    # Angle uniform in [0, cone_rad]. θ=0 at i=0 so the exact center
    # is guaranteed to appear — useful for sanity checks.
    cone_rad = math.radians(cone_deg)
    theta = rng.uniform(0.0, cone_rad, size=n)
    if n:
        theta[0] = 0.0  # first sample is exact center
    # n = 0 is the honest "no leak" setting and must not raise.

    half = theta / 2.0
    delta_xyz = v * np.sin(half)[:, None]                # (n, 3)
    delta_w = np.cos(half)                                # (n,)
    delta = np.concatenate([delta_xyz, delta_w[:, None]], axis=1)  # (n, 4)

    # Hamilton quat mul: q = delta · q_center   (both (x, y, z, w)).
    def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return np.stack([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ], axis=-1)

    q_c_np = q_center.detach().cpu().numpy().astype(np.float64)
    q_out = quat_mul(delta, q_c_np)
    q_out = q_out / np.linalg.norm(q_out, axis=1, keepdims=True)

    q_t = torch.as_tensor(q_out, dtype=dtype)
    return q_t.to(device)


def euler_quaternions(
    deg: float = 15.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """ZYZ Euler-angle grid at `deg` spacing, converted to
    `geom.rotate`-compatible quaternions.

    Enumerates (φ, θ, ψ) with φ ∈ [0, 360), θ ∈ [0, 180], ψ ∈ [0, 360).
    Does NOT de-duplicate orientations that coincide at θ = 0/180
    (gimbal-lock); callers that care can unique-ify by the resulting
    rotation matrix. Coverage is denser near the poles — a known
    Euler limitation; `random_quaternions` avoids this.

    Returns `(R, 4)` unit quaternions in the convention used by
    `geom.rotate` (q1..q4 → Julia docking.jl rotate!).
    """
    if deg <= 0.0 or deg > 180.0:
        raise ValueError(f"deg must be in (0, 180], got {deg}")
    rad = math.radians(deg)
    # Number of samples per axis.
    n_phi = int(round(360.0 / deg))
    n_theta = int(round(180.0 / deg)) + 1     # endpoints included
    n_psi = int(round(360.0 / deg))
    phi = torch.linspace(0.0, 2.0 * math.pi, n_phi + 1, dtype=dtype)[:-1]
    theta = torch.linspace(0.0, math.pi, n_theta, dtype=dtype)
    psi = torch.linspace(0.0, 2.0 * math.pi, n_psi + 1, dtype=dtype)[:-1]

    # Cartesian product, shape (R, 3)
    ph, th, ps = torch.meshgrid(phi, theta, psi, indexing="ij")
    ph, th, ps = ph.reshape(-1), th.reshape(-1), ps.reshape(-1)

    # Convert ZYZ Euler to quaternion. Use the convention that matches
    # `geom.rotate`'s q1..q4 decomposition.
    #
    # We assemble q = q_z(ψ) · q_y(θ) · q_z(φ) where q_axis(a) is the
    # half-angle rotation around that axis, then map to (q1, q2, q3, q4)
    # = (x, y, z, w). This is standard Hamilton composition.
    def q_z(a):
        z = torch.zeros_like(a)
        return torch.stack([z, z, torch.sin(a / 2), torch.cos(a / 2)], dim=-1)

    def q_y(a):
        z = torch.zeros_like(a)
        return torch.stack([z, torch.sin(a / 2), z, torch.cos(a / 2)], dim=-1)

    def quat_mul(a, b):
        # Hamilton product, both in (x, y, z, w) convention.
        ax, ay, az, aw = a.unbind(-1)
        bx, by, bz, bw = b.unbind(-1)
        return torch.stack([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ], dim=-1)

    q = quat_mul(q_z(ps), quat_mul(q_y(th), q_z(ph)))
    q = q / q.norm(dim=-1, keepdim=True)
    if device is not None:
        q = q.to(device)
    return q


# ---------------------------------------------------------------------------
# Uniform incremental SO(3) grid via the Hopf fibration
# ---------------------------------------------------------------------------
#
# ZDOCK does not sample rotations at random: "Evenly distributed Euler angles
# are used for the rotational search. The Euler angle sets ... have been
# obtained from Dr. Julie C. Mitchell. An angle set is equivalent to a
# uniformly distributed set of points on a projective sphere, which ensures
# that minimal orientations are required to cover the entire rotational space.
# The angular distance between any orientation and its nearest orientation ...
# is Delta or smaller" (Chen & Weng 2003, Methods).
#
# The published sets are not distributable, so we generate an equivalent grid
# with the method of Yershova, Jain, LaValle & Mitchell, "Generating Uniform
# Incremental Grids on SO(3) Using the Hopf Fibration", IJRR 29(7):801-812
# (2010) — same Mitchell as the ZDOCK acknowledgement. S^3 factors as
# S^2 x S^1 under the Hopf map, so an even grid on SO(3) is the product of an
# even grid on the sphere (HEALPix) and an even grid on the circle:
#
#     q(theta, phi, psi) = ( cos(theta/2) cos(psi/2),
#                            cos(theta/2) sin(psi/2),
#                            sin(theta/2) cos(phi + psi/2),
#                            sin(theta/2) sin(phi + psi/2) )
#
# What matters for docking is the *covering radius* — the largest angle any
# orientation can be from its nearest grid point — because that is what bounds
# how far a rigid ligand must be rotated away from its true pose, and hence how
# much clash penalty the discretisation forces on the correct answer. Use
# :func:`covering_radius_deg` to measure it.


def _healpix_ring_angles(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """Centres ``(theta, phi)`` of all ``12*nside**2`` HEALPix pixels, RING
    scheme. Equal-area pixels, so the centres are an even sphere sample."""
    npix = 12 * nside * nside
    ipix = np.arange(npix, dtype=np.int64)
    ncap = 2 * nside * (nside - 1)
    z = np.empty(npix, dtype=np.float64)
    phi = np.empty(npix, dtype=np.float64)

    north = ipix < ncap
    p = ipix[north] + 1
    i = np.floor(np.sqrt(p / 2.0 - np.sqrt(np.floor(p / 2.0)))).astype(np.int64) + 1
    j = p - 2 * i * (i - 1)
    z[north] = 1.0 - (i * i) / (3.0 * nside * nside)
    phi[north] = (j - 0.5) * (np.pi / 2.0) / i

    south = ipix >= npix - ncap
    p = npix - ipix[south]
    i = np.floor(np.sqrt(p / 2.0 - np.sqrt(np.floor(p / 2.0)))).astype(np.int64) + 1
    j = 4 * i + 1 - (p - 2 * i * (i - 1))
    z[south] = -1.0 + (i * i) / (3.0 * nside * nside)
    phi[south] = (j - 0.5) * (np.pi / 2.0) / i

    eq = ~(north | south)
    ip = ipix[eq] - ncap
    i = (ip // (4 * nside)) + nside
    j = (ip % (4 * nside)) + 1
    fodd = 0.5 * (1 + ((i + nside) % 2))
    z[eq] = (2 * nside - i) * 2.0 / (3.0 * nside)
    phi[eq] = (j - fodd) * (np.pi / 2.0) / nside

    return np.arccos(np.clip(z, -1.0, 1.0)), np.mod(phi, 2.0 * np.pi)


def hopf_quaternions(
    nside: int,
    n_psi: int | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Full Hopf grid: ``12*nside**2 * n_psi`` quaternions, unordered.

    ``n_psi`` defaults to ``6*nside``, which makes the angular resolution along
    the fibre match the resolution on the base sphere (Yershova et al. 2010).
    """
    if n_psi is None:
        n_psi = 6 * nside
    theta, phi = _healpix_ring_angles(nside)
    psi = (np.arange(n_psi, dtype=np.float64) + 0.5) * (2.0 * np.pi / n_psi)

    th = theta[:, None]
    ph = phi[:, None]
    ps = psi[None, :]
    q = np.stack([
        np.broadcast_to(np.cos(th / 2), (theta.size, n_psi)) * np.cos(ps / 2),
        np.broadcast_to(np.cos(th / 2), (theta.size, n_psi)) * np.sin(ps / 2),
        np.broadcast_to(np.sin(th / 2), (theta.size, n_psi)) * np.cos(ph + ps / 2),
        np.broadcast_to(np.sin(th / 2), (theta.size, n_psi)) * np.sin(ph + ps / 2),
    ], axis=-1).reshape(-1, 4)
    out = torch.as_tensor(q, dtype=dtype)
    return out.to(device) if device is not None else out


def so3_geodesic_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise SO(3) angle (deg) between quaternion sets ``(N,4)``/``(M,4)``.

    Uses ``2*arccos(|a.b|)``: the absolute value is what makes ``q`` and ``-q``
    the same rotation, and is why an off-the-shelf Euclidean farthest-point
    routine cannot be used on raw quaternions.
    """
    d = (a @ b.T).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.acos(d))


def farthest_point_order(
    quats: torch.Tensor,
    n: int | None = None,
    *,
    start: int = 0,
    chunk: int = 4096,
) -> torch.Tensor:
    """Greedy max-min ordering, so that **every prefix** is a good covering.

    This is what makes the grid *incremental*: taking the first ``n`` entries
    of the reordered set gives a near-optimal ``n``-point covering, and growing
    ``n`` never moves the points already chosen.
    """
    N = quats.shape[0]
    n = N if n is None else min(n, N)
    order = torch.empty(n, dtype=torch.long, device=quats.device)
    order[0] = start
    dmin = so3_geodesic_deg(quats, quats[start:start + 1]).squeeze(1)
    for k in range(1, n):
        nxt = int(dmin.argmax())
        order[k] = nxt
        d = so3_geodesic_deg(quats, quats[nxt:nxt + 1]).squeeze(1)
        dmin = torch.minimum(dmin, d)
    return order


def covering_radius_deg(
    quats: torch.Tensor,
    *,
    n_probe: int = 200000,
    seed: int = 1,
    chunk: int = 512,
) -> dict:
    """Measure how far a random orientation can be from the grid.

    Returns the mean / median / 95th percentile / max nearest-grid angle over
    ``n_probe`` uniformly random orientations.

    **``max_deg`` is a downward-biased LOWER BOUND on the covering radius**, not
    the covering radius itself: a sample maximum can only underestimate a
    supremum, and the SO(3) volume within eps of the deepest hole shrinks like
    eps^3, so the gap closes only as ``n_probe^(-1/3)`` — 8x more probes to
    halve the error. Measured bias against a 2M-probe + local-ascent reference:
    -9% at n_probe=4000 and -6% at 20000 for a 1944-point Hopf grid (16.8 deg
    and 17.5 deg against a true value of >=18.5 deg). The key is returned as
    ``max_deg_lower_bound``; ``max_deg`` is kept as an alias for older readers.
    ``mean_deg`` and ``p95_deg`` ARE converged at n_probe=20000 (sd 0.02 deg)
    and are safe to quote. Note also that the ``Delta`` Chen & Weng 2003 quote
    is a *guaranteed bound*, so it is not the same kind of number; packing
    efficiency is the like-for-like comparison.

    The default ``seed`` is 1, not 0: ``random_quaternions`` is prefix-stable,
    so probing a ``random_quaternions(..., seed=0)`` grid with ``seed=0`` makes
    the first ``min(N, n_probe)`` probes literally be the grid points and
    reports a covering radius of ~0.
    """
    probe = random_quaternions(n_probe, seed=seed, device=quats.device,
                               dtype=quats.dtype)
    best = torch.empty(n_probe, device=quats.device, dtype=quats.dtype)
    for s in range(0, n_probe, chunk):
        e = min(s + chunk, n_probe)
        best[s:e] = so3_geodesic_deg(probe[s:e], quats).min(dim=1).values
    b = best.double()
    lb = float(b.max())
    return {"n_grid": int(quats.shape[0]), "n_probe": n_probe,
            "mean_deg": float(b.mean()), "median_deg": float(b.median()),
            "p95_deg": float(b.quantile(0.95)),
            "max_deg_lower_bound": lb, "max_deg": lb}
