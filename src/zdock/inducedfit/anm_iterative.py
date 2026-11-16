"""CPU float64 ANM from neighbor edges and a matrix-free Lanczos operator.

No all-pairs distance array, Cartesian Hessian, or full eigensystem is built.
Known global rigid modes are shifted above a rigorous spectral upper bound;
additional zero modes are explicitly retained during adaptive low-spectrum
extraction, then excluded.  Exhausting the eigenpair budget is an error, never
a dense fallback.  Near-threshold unresolved modes are also errors.
Zero/positive counts are inferred from the converged iterative solves, not
certified by a full spectrum. Residuals certify returned eigenpairs but cannot
prove completeness of a multiply-degenerate eigenspace; compare subspaces on
representative dense reference problems before relying on a new graph family.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh
import torch

from .modes import ANMModel, _check_xyz, _component_count, _positive_scalar


@dataclass(frozen=True)
class ANMEdgeOperator:
    reference: torch.Tensor
    pairs: torch.Tensor
    reference_distances: torch.Tensor
    directions: np.ndarray
    gamma: float
    n_components: int
    spectral_upper_bound: float

    @property
    def dimension(self):
        return 3 * len(self.reference)

    def matmat(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if value.ndim != 2 or value.shape[0] != self.dimension:
            raise ValueError("ANM operator requires a (3*N_CA, K) matrix")
        vectors = value.reshape(len(self.reference), 3, -1)
        pairs = self.pairs.numpy()
        difference = vectors[pairs[:, 0]] - vectors[pairs[:, 1]]
        along = np.einsum("eck,ec->ek", difference, self.directions)
        force = self.gamma * self.directions[:, :, None] * along[:, None, :]
        result = np.zeros_like(vectors)
        np.add.at(result, pairs[:, 0], force)
        np.add.at(result, pairs[:, 1], -force)
        return result.reshape(value.shape)

    def matvec(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if value.shape not in ((self.dimension,), (self.dimension, 1)):
            raise ValueError("ANM operator requires a vector of length 3*N_CA")
        return self.matmat(value.reshape(self.dimension, 1)).reshape(value.shape)

    def as_linear_operator(self) -> LinearOperator:
        return LinearOperator((self.dimension, self.dimension), matvec=self.matvec,
                              rmatvec=self.matvec, matmat=self.matmat, dtype=np.float64)


def build_anm_edge_operator(ca_xyz: torch.Tensor, *, cutoff: float = 12.0,
                            gamma: float = 1.0,
                            reject_disconnected: bool = True) -> ANMEdgeOperator:
    """Same <=cutoff central springs as dense ANM, using a spatial tree."""
    _check_xyz(ca_xyz, "ca_xyz")
    if ca_xyz.dtype != torch.float64 or ca_xyz.device.type != "cpu":
        raise TypeError("Iterative ANM currently requires float64 CPU coordinates")
    cutoff = _positive_scalar(cutoff, "cutoff")
    gamma = _positive_scalar(gamma, "gamma")
    if not isinstance(reject_disconnected, bool):
        raise TypeError("reject_disconnected must be a bool")
    reference = ca_xyz.detach().clone().contiguous()
    # Expand only the spatial-search radius by roundoff; the exact torch
    # distance and <= cutoff test below define which edges actually exist.
    query_radius = cutoff * (1 + 8 * np.finfo(np.float64).eps)
    candidates = cKDTree(reference.numpy()).query_pairs(query_radius, output_type="ndarray")
    if len(candidates):
        candidates = candidates[np.lexsort((candidates[:, 1], candidates[:, 0]))]
    pairs = torch.as_tensor(candidates.copy(), dtype=torch.long).reshape(-1, 2)
    vectors = reference[pairs[:, 1]] - reference[pairs[:, 0]]
    distances = torch.linalg.vector_norm(vectors, dim=-1)
    if bool((distances == 0).any()):
        raise ValueError("coincident CA atoms do not define a central-spring direction")
    if not bool(torch.isfinite(distances).all()):
        raise ValueError("CA distances must be representable in float64")
    keep = distances <= cutoff
    pairs, vectors, distances = pairs[keep], vectors[keep], distances[keep]
    n_components = _component_count(len(reference), pairs)
    if reject_disconnected and n_components != 1:
        raise ValueError(f"ANM network is disconnected ({n_components} components at cutoff={cutoff})")
    directions = (vectors / distances[:, None]).numpy().copy()
    degree = torch.bincount(pairs.reshape(-1), minlength=len(reference))
    # x^T H x <= gamma sum_edges ||x_i-x_j||^2 <= 2 gamma d_max ||x||^2.
    upper = 2 * gamma * int(degree.max())
    return ANMEdgeOperator(reference, pairs, distances, directions, gamma, n_components, upper)


def _rigid_basis(reference: np.ndarray) -> np.ndarray:
    n = len(reference)
    translations = np.tile(np.eye(3), (n, 1)) / math.sqrt(n)
    centered = reference - reference.mean(axis=0)
    scale = np.max(np.abs(centered))
    if scale == 0:
        return translations
    centered = centered / scale
    rotations = np.stack([np.cross(np.broadcast_to(axis, centered.shape), centered).reshape(-1)
                          for axis in np.eye(3)], axis=1)
    left, singular, _ = np.linalg.svd(rotations, full_matrices=False)
    retained = singular > 64 * np.finfo(np.float64).eps * singular.max()
    result = np.concatenate((translations, left[:, retained]), axis=1)
    # Only a (3N, <=6) basis is factored, never a Cartesian Hessian.
    return np.linalg.qr(result, mode="reduced")[0]


def _lanczos(operator, *, k, which, generator, tolerance, max_iterations):
    try:
        values, vectors = eigsh(operator, k=k, which=which,
            v0=generator.standard_normal(operator.shape[0]),
            ncv=min(operator.shape[0], max(2 * k + 1, 20)),
            tol=tolerance, maxiter=max_iterations)
    except ArpackNoConvergence as error:
        raise RuntimeError(f"Iterative ANM Lanczos did not converge ({len(error.eigenvalues)}/{k} eigenpairs); no partial result or dense fallback") from error
    order = np.argsort(values)
    return values[order], vectors[:, order]


def build_anm_iterative(ca_xyz: torch.Tensor, *, n_modes: int,
                        cutoff: float = 12.0, gamma: float = 1.0,
                        seed: int = 0, tolerance: float = 1e-9,
                        max_iterations: int | None = None,
                        zero_eigenvalue_rtol: float | None = None,
                        reject_disconnected: bool = True,
                        allow_rank_reduction: bool = False,
                        max_eigenpairs: int | None = None) -> ANMModel:
    """Return only requested lowest positive ANM modes, with hessian=None.

    ``n_modes=0`` validates geometry and connectivity, but deliberately skips
    spectral analysis: total zero/positive counts are then None.  Otherwise
    the zero threshold uses the dense default (64*eps times largest eigenvalue)
    unless explicitly overridden.  A Ritz residual crossing that threshold
    raises instead of guessing whether a very soft mode is zero.

    At most ``max_eigenpairs`` low Ritz vectors are requested in any solve,
    and at most that many additional zero directions are stored for deflation;
    default min(3N-rigid_rank, max(32,4*n_modes+16)). Highly floppy structures
    can require an explicitly larger budget. Requested-too-many modes raise
    unless allow_rank_reduction=True AND the complete non-rigid low spectrum
    has actually been resolved. No rank reduction conceals budget exhaustion.
    """
    if isinstance(n_modes, bool) or not isinstance(n_modes, int) or n_modes < 0:
        raise ValueError("n_modes must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    tolerance = _positive_scalar(tolerance, "tolerance")
    if tolerance >= 1:
        raise ValueError("tolerance must be smaller than one")
    if max_iterations is not None and (type(max_iterations) is not int or max_iterations < 1):
        raise ValueError("max_iterations must be a positive integer or None")
    if max_eigenpairs is not None and (type(max_eigenpairs) is not int or max_eigenpairs < 1):
        raise ValueError("max_eigenpairs must be a positive integer or None")
    if not isinstance(allow_rank_reduction, bool):
        raise TypeError("allow_rank_reduction must be a bool")
    rtol = 64 * np.finfo(np.float64).eps if zero_eigenvalue_rtol is None else _positive_scalar(zero_eigenvalue_rtol, "zero_eigenvalue_rtol")
    edges = build_anm_edge_operator(ca_xyz, cutoff=cutoff, gamma=gamma,
                                    reject_disconnected=reject_disconnected)
    reference, dimension = edges.reference, edges.dimension
    rigid = _rigid_basis(reference.numpy())
    rigid_count = rigid.shape[1]
    complement = dimension - rigid_count
    diagnostics = {"backend": "matrix_free_lanczos", "n_ca": len(reference),
        "n_edges": len(edges.pairs), "n_components": edges.n_components,
        "n_rigid_modes": rigid_count, "n_zero_modes": None, "n_positive_modes": None,
        "selected_modes": 0, "requested_modes": n_modes, "seed": seed,
        "solver_tolerance": tolerance, "zero_eigenvalue_rtol": rtol,
        "zero_eigenvalue_threshold": None, "spectral_analysis_performed": n_modes > 0,
        "dense_hessian_materialized": False, "all_ca_pairs_materialized": False,
        "full_spectrum_certified": False,
        "multiplicity_caveat": "Zero counts are solver-inferred; small residuals do not certify complete degenerate eigenspaces",
        "rank_reduction_allowed": allow_rank_reduction, "rank_reduced": False}
    if n_modes == 0:
        return ANMModel(reference, edges.pairs, edges.reference_distances, None,
                        reference.new_empty(0), reference.new_empty((0, len(reference), 3)),
                        float(cutoff), float(gamma), diagnostics)
    if n_modes > complement and not allow_rank_reduction:
        raise ValueError(f"requested {n_modes} modes but at most {complement} non-rigid directions exist")
    if not len(edges.pairs):
        if not allow_rank_reduction:
            raise ValueError("requested positive modes but ANM has no springs and no positive modes")
        diagnostics.update(n_zero_modes=dimension, n_positive_modes=0, selected_modes=0,
                            rank_reduced=True, zero_eigenvalue_threshold=0.)
        return ANMModel(reference, edges.pairs, edges.reference_distances, None,
                        reference.new_empty(0), reference.new_empty((0, len(reference), 3)),
                        float(cutoff), float(gamma), diagnostics)
    generator = np.random.default_rng(seed)
    operator = edges.as_linear_operator()
    largest, largest_vectors = _lanczos(operator, k=1, which="LA", generator=generator,
        tolerance=min(tolerance, 1e-12), max_iterations=max_iterations)
    scale = float(largest[0])
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("Iterative ANM could not resolve a positive spectral scale")
    threshold = rtol * scale
    largest_residual = float(np.linalg.norm(edges.matmat(largest_vectors) - largest_vectors * largest[None, :]))
    residual_limit = max(10 * tolerance, 256 * np.finfo(np.float64).eps) * scale
    if largest_residual > residual_limit:
        raise RuntimeError("Iterative ANM largest-eigenvalue residual exceeds tolerance")
    rigid_residual = float(np.linalg.norm(edges.matmat(rigid), axis=0).max())
    if rigid_residual > 256 * np.finfo(np.float64).eps * max(scale, edges.spectral_upper_bound):
        raise RuntimeError("Rigid ANM basis is not a numerical null space")
    penalty = 2 * edges.spectral_upper_bound
    deflated = rigid

    def shifted_matmat(value):
        return edges.matmat(value) + penalty * deflated @ (deflated.T @ value)

    def shifted_matvec(value):
        return shifted_matmat(np.asarray(value).reshape(dimension, 1)).reshape(np.asarray(value).shape)

    shifted = LinearOperator((dimension, dimension), matvec=shifted_matvec,
                              rmatvec=shifted_matvec, matmat=shifted_matmat, dtype=np.float64)
    cap = min(complement, max(32, 4 * n_modes + 16) if max_eigenpairs is None else max_eigenpairs)
    target = min(n_modes, complement)
    if cap < target:
        raise ValueError("max_eigenpairs cannot accommodate the requested modes")
    # E scalar edge constraints imply nullity >= 3N-E, independently of geometry.
    additional_null_lower_bound = max(0, dimension - len(edges.pairs) - rigid_count)
    attempts = []
    additional_zeros = 0
    while True:
        available = dimension - deflated.shape[1]
        count = min(cap, available, max(target + 8, target + additional_null_lower_bound - additional_zeros))
        if count == 0:
            raise RuntimeError("No non-rigid directions remain although the ANM has springs")
        _, vectors = _lanczos(shifted, k=count, which="SA", generator=generator,
            tolerance=tolerance, max_iterations=max_iterations)
        action = edges.matmat(vectors)
        values = np.einsum("ij,ij->j", vectors, action)
        order = np.argsort(values)
        values, vectors, action = values[order], vectors[:, order], action[:, order]
        residuals = np.linalg.norm(action - vectors * values[None, :], axis=0)
        if not np.isfinite(values).all() or not np.isfinite(residuals).all():
            raise RuntimeError("Iterative ANM produced nonfinite eigenpairs")
        if np.any(values < -threshold):
            raise FloatingPointError("ANM Hessian has negative curvature beyond numerical zero tolerance")
        if np.any(residuals > residual_limit):
            raise RuntimeError("Iterative ANM eigenpair residual exceeds tolerance")
        if np.any(np.abs(values - threshold) <= residuals):
            raise RuntimeError("Iterative ANM cannot distinguish a zero from a soft positive mode at the requested threshold; tighten tolerance")
        positive = values > threshold
        zeros = int((~positive).sum())
        attempts.append({"requested_eigenpairs": count, "positive_found": int(positive.sum()),
                         "additional_zero_found": zeros, "already_deflated_extra_zeros": additional_zeros,
                         "maximum_residual": float(residuals.max())})
        if zeros:
            # A single-vector Krylov method can return only part of a multiply
            # degenerate zero eigenspace. Shift found zero directions away,
            # then use a new local seeded start to discover any remaining ones.
            if additional_zeros + zeros > cap:
                raise RuntimeError("Additional ANM zero modes exhaust max_eigenpairs deflation budget; increase it explicitly (no dense fallback)")
            new_zeros = vectors[:, ~positive]
            new_zeros = new_zeros - deflated @ (deflated.T @ new_zeros)
            new_zeros, triangular = np.linalg.qr(new_zeros, mode="reduced")
            if np.any(np.abs(np.diag(triangular)) < .5):
                raise RuntimeError("Additional zero directions overlap an already deflated null space")
            deflated = np.concatenate((deflated, new_zeros), axis=1)
            additional_zeros += zeros
            continue
        if additional_zeros < additional_null_lower_bound:
            raise RuntimeError("Lanczos spectrum misses zero-mode multiplicity required by the edge-count rank bound")
        if int(positive.sum()) < min(target, available):
            raise RuntimeError("The eigenpair budget cannot resolve the requested positive modes")
        break
    n_positive = dimension - rigid_count - additional_zeros
    if n_modes > n_positive and not allow_rank_reduction:
        raise ValueError(f"requested {n_modes} modes but ANM provides only {n_positive} positive modes")
    selected = np.flatnonzero(positive)[:n_modes]
    selected_vectors = vectors[:, selected]
    overlap = float(np.max(np.abs(deflated.T @ selected_vectors))) if len(selected) else 0.
    orthogonality = float(np.max(np.abs(selected_vectors.T @ selected_vectors - np.eye(len(selected))))) if len(selected) else 0.
    if max(overlap, orthogonality) > max(10 * tolerance, 1024 * np.finfo(np.float64).eps):
        raise RuntimeError("Iterative ANM modes fail orthogonality or rigid-null exclusion")
    # Deterministic signs aid serialization; degenerate subspaces remain free
    # to rotate and must be compared by projectors, not column identities.
    for column in range(selected_vectors.shape[1]):
        pivot = np.argmax(np.abs(selected_vectors[:, column]))
        if selected_vectors[pivot, column] < 0:
            selected_vectors[:, column] *= -1
    diagnostics.update({"n_zero_modes": rigid_count + additional_zeros, "n_positive_modes": n_positive,
        "selected_modes": len(selected), "rank_reduced": len(selected) < n_modes,
        "zero_eigenvalue_threshold": threshold, "largest_eigenvalue": scale,
        "largest_eigenvalue_residual": largest_residual, "rigid_residual": rigid_residual,
        "rigid_shift": penalty, "spectral_upper_bound": edges.spectral_upper_bound,
        "max_eigenpairs": cap, "solver_attempts": attempts,
        "selected_eigenpair_residuals": residuals[selected].tolist(),
        "selected_relative_residuals": (residuals[selected] / scale).tolist(),
        "orthogonality_max_abs_error": orthogonality, "rigid_overlap_max_abs": overlap,
        "zero_count_policy": "known global rigid rank plus successive zero-space deflation and repeated low-spectrum solves",
        "additional_zero_modes_deflated": additional_zeros,
        "additional_zero_lower_bound_from_edge_count": additional_null_lower_bound})
    return ANMModel(reference, edges.pairs, edges.reference_distances, None,
        torch.from_numpy(values[selected].copy()),
        torch.from_numpy(selected_vectors.T.copy()).reshape(-1, len(reference), 3),
        float(cutoff), float(gamma), diagnostics)
