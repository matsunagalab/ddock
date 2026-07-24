"""Atom-type ID / vdW radius / charge assignment tables, ported from
`docking/docking.jl` (`set_atomtype_id`, `set_radius`, and `get_iface_ij` /
`get_acescore` constants) and `train_param-apart.ipynb` cell 2
(`set_charge`, `get_charge_score`).

Index conventions (Julia 1-based → Python 0-based preserved as lists/dicts
keyed by the 1-based Julia IDs where the algorithm requires it, so that
`iface_flat[12*(j-1)+i]` in Julia maps cleanly to
`iface_flat[12*(j-1)+(i-1)]`'s equivalent in Python).

The tensors returned by `iface_ij()`, `ace_score()`, `charge_score()` are
created on whatever device/dtype the caller requests, so they can be moved
to CUDA/MPS up-front and reused inside hot loops.
"""

from __future__ import annotations

from typing import Iterable

import torch

from ._atomtype_rules import ATOMTYPE_RULES

# ---------------------------------------------------------------------------
# Static constants copied verbatim from the Julia sources.
# ---------------------------------------------------------------------------

# docking.jl `get_acescore()` — 18 floats, Julia index 1 → Python index 0.
_ACE_SCORE: tuple[float, ...] = (
    -0.495,  # ATOM TYPE "N"
    -0.553,  # ATOM TYPE "CA"
    -0.464,  # ATOM TYPE "C"
    -0.079,  # ATOM TYPE "O"
    0.008,   # ATOM TYPE "GCA"
    -0.353,  # ATOM TYPE "CB"
    1.334,   # ATOM TYPE "KNZ"
    1.046,   # ATOM TYPE "KCD"
    0.933,   # ATOM TYPE "DOD"
    0.726,   # ATOM TYPE "RNH"
    0.693,   # ATOM TYPE "NND"
    0.606,   # ATOM TYPE "RNE"
    0.232,   # ATOM TYPE "SOG"
    0.061,   # ATOM TYPE "HNE"
    -0.289,  # ATOM TYPE "YCZ"
    -0.432,  # ATOM TYPE "FCZ"
    -0.987,  # ATOM TYPE "LCD"
    -1.827,  # ATOM TYPE "CSG"
)

# docking.jl `get_iface_ij()` — symmetric 12×12 matrix of pairwise atom-type
# interaction energies. Row/col ordering: R+, Polar, mc, RHK_mc, CG, WY_sc,
# MEW_sc, DE-, DE_mc, ILV_sc, K+, AILMV_mc (see thesis Fig 3.3).
_IFACE_IJ: tuple[tuple[float, ...], ...] = (
    ( 0.678,  0.133, -0.007,  0.727,  0.091, -0.742, -0.625, -0.064, -0.382, -0.302,  1.221,  0.187),
    ( 0.133,  0.180,  0.065,  0.295,  0.057, -0.631, -0.663,  0.612,  0.342, -0.275,  0.604,  0.248),
    (-0.007,  0.065, -0.145,  0.093, -0.265, -1.078, -1.176,  0.636,  0.355, -0.552,  0.483,  0.109),
    ( 0.727,  0.295,  0.093,  0.696, -0.016, -0.735, -0.804,  0.525,  0.144, -0.202,  1.116,  0.477),
    ( 0.091,  0.057, -0.265, -0.016, -0.601, -0.928, -1.046,  0.644,  0.615, -0.573,  0.062, -0.034),
    (-0.742, -0.631, -1.078, -0.735, -0.928, -0.914, -1.696, -0.577, -0.430, -1.495, -0.303, -0.867),
    (-0.625, -0.663, -1.176, -0.804, -1.046, -1.696, -1.938, -0.214,  0.015, -1.771,  0.092, -1.040),
    (-0.064,  0.612,  0.636,  0.525,  0.644, -0.577, -0.214,  1.620,  1.233, -0.001,  0.368,  0.822),
    (-0.382,  0.342,  0.355,  0.144,  0.615, -0.430,  0.015,  1.233,  1.090,  0.050, -0.024,  0.757),
    (-0.302, -0.275, -0.552, -0.202, -0.573, -1.495, -1.771, -0.001,  0.050, -1.606,  0.253, -0.572),
    ( 1.221,  0.604,  0.483,  1.116,  0.062, -0.303,  0.092,  0.368, -0.024,  0.253,  1.884,  0.731),
    ( 0.187,  0.248,  0.109,  0.477, -0.034, -0.867, -1.040,  0.822,  0.757, -0.572,  0.731,  0.399),
)

# train_param-apart.ipynb cell 2 `get_charge_score()` — 11 floats for charge
# TYPE IDs assigned by `set_charge`. Julia index 1 → Python index 0.
_CHARGE_SCORE: tuple[float, ...] = (
    1.0,   # 1 TERMINAL-N
    -1.0,  # 2 TERMINAL-O
    0.5,   # 3 ARG NH
    -0.5,  # 4 GLU OE
    -0.5,  # 5 ASP OD
    1.0,   # 6 LYS NZ
    -0.1,  # 7 PRO N
    0.0,   # 8 CA
    0.0,   # 9 C
    -0.5,  # 10 O
    0.5,   # 11 N
)

# docking.jl `set_radius` — vdW radii keyed by first letter of the atom name.
_VDW_RADIUS: dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
}


# ---------------------------------------------------------------------------
# Tensor accessors (device/dtype-aware).
# ---------------------------------------------------------------------------


def ace_score(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.tensor(_ACE_SCORE, device=device, dtype=dtype)


def iface_ij(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    flat: bool = False,
) -> torch.Tensor:
    """Return the 12×12 IFACE interaction matrix (or its 144-vector
    column-major flattening if `flat=True`, matching Julia `reshape(M, :)`)."""
    t = torch.tensor(_IFACE_IJ, device=device, dtype=dtype)
    if flat:
        # Julia `reshape(M, :)` flattens in column-major order; Torch default
        # is row-major (C order). Take transpose-then-view to match Julia.
        t = t.T.contiguous().view(-1)
    return t


def charge_score(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.tensor(_CHARGE_SCORE, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Atom-type classification (set_atomtype_id / set_radius / set_charge ports).
# ---------------------------------------------------------------------------

# docking.jl `set_atomtype_id` is a ~370-line if/elseif ladder over
# (resname, atomname). We extract the rules programmatically via
# `tools/extract_atomtype_rules.py` into `_atomtype_rules.py`. A key
# normalization step in the Julia source: "OXT" atom names are rewritten to
# "O" before the ladder (docking.jl lines 452–457), so our rule table never
# sees "OXT".
#
# Build a lookup dict once at import time so per-atom classification is O(1).
_ATOMTYPE_LUT: dict[tuple[str, str], int] = {
    (r, a): tid for (tid, r, a) in ATOMTYPE_RULES
}


def set_atomtype_id(
    resnames: Iterable[str],
    atomnames: Iterable[str],
) -> torch.Tensor:
    """Return an int64 tensor (N,) of 1-based atom type IDs (1..12).

    Atom names equal to "OXT" are normalized to "O" before classification, to
    match docking.jl lines 452–457. Raises if a (resname, atomname) pair is
    not recognized — docking.jl's final `else` branch throws an error too, so
    this matches behaviour.
    """
    resnames = list(resnames)
    atomnames = list(atomnames)
    if len(resnames) != len(atomnames):
        raise ValueError("resnames and atomnames must have equal length")

    n = len(atomnames)
    out = torch.zeros(n, dtype=torch.int64)
    for i, (r, a) in enumerate(zip(resnames, atomnames)):
        a_norm = "O" if a == "OXT" else a
        tid = _ATOMTYPE_LUT.get((r, a_norm))
        if tid is None:
            raise ValueError(
                f"failed to assign atom type for (resname={r!r}, atomname={a!r})"
            )
        out[i] = tid
    return out


def set_radius(
    atomnames: Iterable[str],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a float tensor (N,) of vdW radii.

    The element is determined by the first non-digit character of the PDB
    atom name, so e.g. "OH" → O, "NH1" → N, "CH2" → C, "1HG1" → H. This
    matches `docking_canonical.jl`'s B8-fixed version; the original
    `docking.jl` used `match(r"H.*", name)` which mis-classified names like
    "OH" / "NH1" as hydrogen.
    """
    radii = []
    for a in atomnames:
        if not a:
            raise ValueError("empty atom name")
        # Skip leading digits (PDB atom names like "1HG1" or "2HB").
        j = 0
        while j < len(a) and a[j].isdigit():
            j += 1
        if j == len(a):
            raise ValueError(f"atom name has no element letter: {a!r}")
        element = a[j]
        if element not in _VDW_RADIUS:
            raise ValueError(f"unknown element {element!r} for atom name {a!r}")
        radii.append(_VDW_RADIUS[element])
    return torch.tensor(radii, device=device, dtype=dtype)


def partial_charge_per_atom(
    charge_ids: torch.Tensor,
    charge_score_lut: torch.Tensor,
) -> torch.Tensor:
    """Resolve per-atom partial charges from the 11-entry charge-type LUT.

    Intermediate helper used by the Coulombic ELEC path (score.py). Given a
    vector of 1-based charge IDs as produced by `set_charge()` and a
    `charge_score_lut` of shape (11,) such as `get_charge_score()` returns,
    produce a (N,) float tensor of partial charges per atom.

    IDs outside [1, 11] get 0.0 (no charge). This is a stopgap for the full
    CHARMM19 per-atom partial-charge LUT — see PORT_PLAN.md B14.
    """
    if charge_score_lut.shape != (11,):
        raise ValueError(f"expected charge_score_lut of shape (11,), got {tuple(charge_score_lut.shape)}")
    idx = (charge_ids - 1).clamp(0, 10).to(torch.long)
    out = charge_score_lut[idx]
    # Zero out atoms whose charge_id fell outside 1..11
    valid = (charge_ids >= 1) & (charge_ids <= 11)
    return torch.where(valid, out, torch.zeros_like(out))


def set_charge(
    resnames: Iterable[str],
    atomnames: Iterable[str],
) -> torch.Tensor:
    """Return an int64 tensor (N,) of 1-based charge type IDs (1..11) used by
    `get_charge_score()`.

    This mirrors `train_param-apart.ipynb` cell 2 set_charge with the B3 fix
    applied: terminal O ("O" / "OXT") is now correctly detected via atomname
    instead of resname.
    """
    resnames = list(resnames)
    atomnames = list(atomnames)
    if len(resnames) != len(atomnames):
        raise ValueError("resnames and atomnames must have equal length")

    n = len(atomnames)
    out = torch.zeros(n, dtype=torch.int64)
    is_first_n = True
    for i, (r, a) in enumerate(zip(resnames, atomnames)):
        # Check PRO/N *before* the generic `a == "N"` branch: proline's
        # backbone nitrogen is a secondary amine and must get charge id 7
        # regardless of sequence position.
        if r == "PRO" and a == "N":
            out[i] = 7  # PRO N
        elif a == "N":
            if is_first_n:
                out[i] = 1  # TERMINAL-N
                is_first_n = False
            else:
                out[i] = 11  # N
        elif a == "O":
            out[i] = 10  # O
        elif a == "OXT":
            out[i] = 2   # TERMINAL-O
        elif r == "ARG" and a in ("NH1", "NH2"):
            out[i] = 3
        elif r == "GLU" and a in ("OE1", "OE2"):
            out[i] = 4
        elif r == "ASP" and a in ("OD1", "OD2"):
            out[i] = 5
        elif r == "LYS" and a == "NZ":
            out[i] = 6
        else:
            out[i] = 8
    return out
