"""Small, batched quadratic responses for RMS-normalized deformation modes.

The mode coordinates must already have the Euclidean displacement metric:
``||a|| <= radius``.  A non-identity metric must be whitened by the caller.
This module is an inference solver; it does not differentiate through the
eigensystem or the constrained optimum.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResponseResult:
    """Constrained response and its actual quadratic energy change.

    ``gain`` is ``g @ a + strain`` (nonpositive, lower is better), rather than
    a positive magnitude.  ``strain`` is ``0.5 * a @ K @ a``.  ``lagrange``
    uses the convention ``(K + lagrange * I) @ a = -g``.  At zero radius
    and nonzero gradient it is +inf: the degenerate ball has no finite
    multiplier in this convention, although its unique solution is a=0.
    """

    amplitudes: torch.Tensor
    gain: torch.Tensor
    strain: torch.Tensor
    lagrange: torch.Tensor


def _norm(value: torch.Tensor) -> torch.Tensor:
    """Last-axis norm without overflow from squaring large components."""
    scale = value.abs().amax(dim=-1)
    # Preserve an infinite norm if a trial division overflows; dividing inf
    # by inf would instead produce NaN and misclassify an outside trial.
    divisor = torch.where((scale > 0) & torch.isfinite(scale), scale, torch.ones_like(scale))
    return scale * torch.linalg.vector_norm(value / divisor[..., None], dim=-1)


class QuadraticResponse:
    """Cache a small SPD stiffness matrix and solve many trust-region problems.

    Parameters
    ----------
    stiffness:
        Finite, symmetric positive-definite ``(M, M)`` float32/float64 tensor.
        The empty ``(0, 0)`` matrix represents the rigid limit.  Inputs are
        detached and copied, so subsequent caller mutations cannot change the
        cached model.  Symmetry within 16 machine eps times the matrix scale
        is accepted and canonicalized by averaging the two triangles.
    """

    @torch.no_grad()
    def __init__(self, stiffness: torch.Tensor):
        if not isinstance(stiffness, torch.Tensor):
            raise TypeError("stiffness must be a torch.Tensor")
        if stiffness.dtype not in (torch.float32, torch.float64):
            raise TypeError("stiffness must have float32 or float64 dtype")
        if stiffness.ndim != 2 or stiffness.shape[0] != stiffness.shape[1]:
            raise ValueError("stiffness must be a square (M, M) matrix")
        if not bool(torch.isfinite(stiffness).all()):
            raise ValueError("stiffness must be finite")

        self.n_modes = stiffness.shape[0]
        if self.n_modes == 0:
            self._stiffness = stiffness.detach().clone()
            self._eigenvalues = stiffness.new_empty((0,))
            self._eigenvectors = stiffness.new_empty((0, 0))
            return

        scale = stiffness.abs().amax()
        symmetry_tolerance = 16 * torch.finfo(stiffness.dtype).eps * scale
        if bool((stiffness - stiffness.T).abs().amax() > symmetry_tolerance):
            raise ValueError("stiffness must be symmetric")
        # Half each operand first to avoid overflowing their sum.
        self._stiffness = (0.5 * stiffness + 0.5 * stiffness.T).detach().clone()
        self._eigenvalues, self._eigenvectors = torch.linalg.eigh(self._stiffness)
        if not bool(torch.isfinite(self._eigenvalues).all()):
            raise ValueError("stiffness eigendecomposition must be finite")
        if bool((self._eigenvalues <= 0).any()):
            raise ValueError("stiffness must be positive definite")

    @torch.no_grad()
    def solve(
        self,
        gradient: torch.Tensor,
        radius: float | torch.Tensor,
        *,
        max_iter: int = 64,
    ) -> ResponseResult:
        """Minimize ``g @ a + 0.5 * a @ K @ a`` subject to ``||a|| <= radius``.

        ``gradient`` has shape ``(..., M)`` and the cached matrix's dtype and
        device.  ``radius`` is finite, nonnegative, and broadcastable to the
        batch shape ``gradient.shape[:-1]``.  Positive-radius constrained
        problems use ``max_iter`` vectorized bisection steps; 64 is the default
        for both supported dtypes.  The feasible upper endpoint is returned.
        No Jacobian of this solver is exposed; all outputs are detached.

        Nonfinite intermediate/output values produce an explicit error rather
        than a silently invalid response.  The +inf zero-radius multiplier
        described in :class:`ResponseResult` is the sole exception.
        """
        if not isinstance(gradient, torch.Tensor):
            raise TypeError("gradient must be a torch.Tensor")
        if gradient.dtype != self._stiffness.dtype:
            raise TypeError("gradient and stiffness must have the same dtype")
        if gradient.device != self._stiffness.device:
            raise ValueError("gradient and stiffness must be on the same device")
        if gradient.ndim < 1 or gradient.shape[-1] != self.n_modes:
            raise ValueError("gradient must have shape (..., M)")
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError("gradient must be finite")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")

        batch_shape = gradient.shape[:-1]
        radius_tensor = torch.as_tensor(
            radius, dtype=gradient.dtype, device=gradient.device,
        )
        if not bool(torch.isfinite(radius_tensor).all()) or bool((radius_tensor < 0).any()):
            raise ValueError("radius must be finite and nonnegative")
        try:
            radius_tensor = torch.broadcast_to(radius_tensor, batch_shape)
        except RuntimeError as error:
            raise ValueError("radius must broadcast to the gradient batch shape") from error

        zeros = gradient.new_zeros(batch_shape)
        if self.n_modes == 0:
            return ResponseResult(gradient.new_zeros(gradient.shape), zeros, zeros.clone(), zeros.clone())

        projected = gradient @ self._eigenvectors
        projected_norm = _norm(projected)
        if not bool(torch.isfinite(projected).all()) or not bool(torch.isfinite(projected_norm).all()):
            raise FloatingPointError("projected gradient is not representable in this dtype")

        unconstrained = -projected / self._eigenvalues
        positive_radius = radius_tensor > 0
        active = positive_radius & (_norm(unconstrained) > radius_tensor)
        safe_radius = torch.where(positive_radius, radius_tensor, torch.ones_like(radius_tensor))
        # Since k_i > 0, ||h / (k + lambda)|| <= ||h|| / lambda.
        # A small outward margin compensates for floating-point roundoff.
        margin = 1 + 8 * torch.finfo(gradient.dtype).eps
        upper = torch.where(active, (projected_norm / safe_radius) * margin, zeros)
        if not bool(torch.isfinite(upper).all()):
            raise FloatingPointError("trust-region multiplier bound is not representable in this dtype")
        lower = torch.zeros_like(upper)
        if bool(active.any()):
            for _ in range(max_iter):
                midpoint = lower + (upper - lower) * 0.5
                trial = -projected / (self._eigenvalues + midpoint[..., None])
                outside = _norm(trial) > radius_tensor
                lower = torch.where(active & outside, midpoint, lower)
                upper = torch.where(active & ~outside, midpoint, upper)

        denominator = self._eigenvalues + upper[..., None]
        if not bool(torch.isfinite(denominator).all()):
            raise FloatingPointError("shifted stiffness is not representable in this dtype")
        spectral_amplitudes = -projected / denominator
        spectral_amplitudes = torch.where(
            positive_radius[..., None], spectral_amplitudes, torch.zeros_like(spectral_amplitudes),
        )
        amplitudes = spectral_amplitudes @ self._eigenvectors.T
        # The eigenspace back-transform can exceed the boundary by a few ulps.
        # Correct only this roundoff and recompute the ACTUAL objective below.
        amplitude_norm = _norm(amplitudes)
        safe_norm = torch.where(amplitude_norm > 0, amplitude_norm, torch.ones_like(amplitude_norm))
        roundoff_scale = torch.minimum(torch.ones_like(safe_norm), radius_tensor / safe_norm)
        amplitudes = amplitudes * roundoff_scale[..., None]

        strain = 0.5 * (amplitudes * (amplitudes @ self._stiffness)).sum(dim=-1)
        gain = (gradient * amplitudes).sum(dim=-1) + strain
        lagrange = torch.where(
            ~positive_radius & (projected_norm > 0),
            torch.full_like(upper, float("inf")),
            upper,
        )
        if not all(bool(torch.isfinite(value).all()) for value in (amplitudes, strain, gain)):
            raise FloatingPointError("quadratic response is not representable in this dtype")
        return ResponseResult(amplitudes, gain, strain, lagrange)
