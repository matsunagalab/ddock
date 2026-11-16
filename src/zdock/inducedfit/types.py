"""Reference-free inputs and a fixed grid for the induced-fit prototype.

These types do not parse structures, assign charges, or contain a native pose.
Frozen fields prevent rebinding; tensors remain mutable PyTorch objects, so
callers must treat them as read-only. Construction clones caller tensors to
avoid aliases and fingerprints are computed from their current contents.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch


@dataclass(frozen=True, slots=True)
class AtomKey:
    chain: str
    residue_number: int
    insertion_code: str
    residue_name: str
    atom_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.chain, str):
            raise TypeError("chain must be a string")
        if isinstance(self.residue_number, bool) or not isinstance(self.residue_number, int):
            raise TypeError("residue_number must be an integer")
        if not isinstance(self.insertion_code, str) or len(self.insertion_code) > 1:
            raise ValueError("insertion_code must be a string of length zero or one")
        for name in ("residue_name", "atom_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")


def _finite_float_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a real floating-point tensor")
    if value.layout != torch.strided:
        raise TypeError(f"{name} must be a strided tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True, slots=True, eq=False)
class MonomerInput:
    xyz: torch.Tensor
    radii: torch.Tensor
    charges: torch.Tensor
    atom_keys: tuple[AtomKey, ...]

    def __post_init__(self) -> None:
        for name in ("xyz", "radii", "charges"):
            _finite_float_tensor(getattr(self, name), name)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or self.xyz.shape[0] == 0:
            raise ValueError("xyz must have shape (N, 3) with N > 0")
        n = self.xyz.shape[0]
        for name in ("radii", "charges"):
            value = getattr(self, name)
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
            if value.dtype != self.xyz.dtype or value.device != self.xyz.device:
                raise ValueError(f"{name} must share xyz dtype and device")
        if not bool((self.radii > 0).all()):
            raise ValueError("radii must be strictly positive")
        keys = tuple(self.atom_keys)
        if len(keys) != n or any(not isinstance(key, AtomKey) for key in keys):
            raise ValueError("atom_keys must contain one AtomKey per atom")
        if len(set(keys)) != n:
            raise ValueError("atom_keys must be unique within each monomer")
        object.__setattr__(self, "atom_keys", keys)
        for name in ("xyz", "radii", "charges"):
            object.__setattr__(self, name, getattr(self, name).clone())

    @property
    def n_atoms(self) -> int:
        return self.xyz.shape[0]


@dataclass(frozen=True, slots=True, eq=False)
class DockingInput:
    system_id: str
    receptor: MonomerInput
    ligand: MonomerInput

    def __post_init__(self) -> None:
        if not isinstance(self.system_id, str) or not self.system_id.strip():
            raise ValueError("system_id must be a nonempty string")
        if not isinstance(self.receptor, MonomerInput) or not isinstance(self.ligand, MonomerInput):
            raise TypeError("receptor and ligand must be MonomerInput instances")
        if (self.receptor.xyz.dtype != self.ligand.xyz.dtype
                or self.receptor.xyz.device != self.ligand.xyz.device):
            raise ValueError("receptor and ligand must share dtype and device")


@dataclass(frozen=True, slots=True)
class GridSpec:
    """Fixed node coordinates: origin + spacing * (i, j, k), in ij order."""

    shape: tuple[int, int, int]
    spacing: float
    origin: tuple[float, float, float]

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if len(shape) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape):
            raise ValueError("shape must contain three positive integers")
        if (isinstance(self.spacing, bool) or not isinstance(self.spacing, Real)
                or not math.isfinite(self.spacing) or self.spacing <= 0):
            raise ValueError("spacing must be finite and strictly positive")
        origin = tuple(self.origin)
        if len(origin) != 3 or any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) for x in origin):
            raise ValueError("origin must contain three finite numbers")
        if any(not math.isfinite(x + (n - 1) * self.spacing) for x, n in zip(origin, shape)):
            raise ValueError("grid upper bounds must be finite")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "spacing", float(self.spacing))
        object.__setattr__(self, "origin", tuple(float(x) for x in origin))

    @property
    def voxel_volume(self) -> float:
        return self.spacing ** 3

    @property
    def n_points(self) -> int:
        return math.prod(self.shape)

    def points(self, *, device=None, dtype=None) -> torch.Tensor:
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if not dtype.is_floating_point:
            raise TypeError("grid points require a real floating-point dtype")
        axes = [torch.arange(n, device=device, dtype=dtype) * self.spacing + start
                for n, start in zip(self.shape, self.origin)]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)

    def bounds(self, *, device=None, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Closed bounds of the grid nodes, not voxel exterior faces."""
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if not dtype.is_floating_point:
            raise TypeError("grid bounds require a real floating-point dtype")
        lower = torch.tensor(self.origin, device=device, dtype=dtype)
        upper = lower + torch.tensor([n - 1 for n in self.shape], device=device, dtype=dtype) * self.spacing
        return lower, upper

    def contains_support(self, xyz: torch.Tensor, support: float | torch.Tensor) -> bool:
        """Whether each atom's spherical compact support fits inside the nodes."""
        _finite_float_tensor(xyz, "xyz")
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")
        radius = torch.as_tensor(support, device=xyz.device, dtype=xyz.dtype)
        if radius.ndim == 0:
            radius = radius.expand(xyz.shape[0])
        if radius.shape != (xyz.shape[0],) or not bool(torch.isfinite(radius).all()) or bool((radius < 0).any()):
            raise ValueError("support must be finite, nonnegative, and scalar or shape (N,)")
        lower, upper = self.bounds(device=xyz.device, dtype=xyz.dtype)
        return bool(((xyz - radius[:, None] >= lower) & (xyz + radius[:, None] <= upper)).all())
