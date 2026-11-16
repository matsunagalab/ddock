"""Differentiable backbone torsions for an unlinked, single heavy-atom chain.

This map preserves the input bond geometry; it neither repairs a structure nor
assigns a torsional force field.  The first residue's N/CA/C define a fixed
coordinate gauge.  Rotations are right-handed increments about N->CA (phi)
and CA->C (psi), applied using the axes in the current coordinates.

Only standard amino-acid atom names are accepted.  Every residue must have
N, CA, C and O; missing sidechain atoms are permitted.  H atoms, multiple
chains, detected peptide breaks and detected non-peptide inter-residue
covalent contacts are rejected.  AtomKey carries no CONECT/LINK topology:
distance-based checks cannot certify the absence of an unrecorded crosslink.
The input preparation layer must reject known crosslinks independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .types import AtomKey, MonomerInput


_BACKBONE = frozenset(("N", "CA", "C", "O"))
_SIDECHAIN_NAMES = {
    "ALA": "CB", "ARG": "CB CG CD NE CZ NH1 NH2", "ASN": "CB CG OD1 ND2",
    "ASP": "CB CG OD1 OD2", "CYS": "CB SG", "GLN": "CB CG CD OE1 NE2",
    "GLU": "CB CG CD OE1 OE2", "GLY": "", "HIS": "CB CG ND1 CD2 CE1 NE2",
    "ILE": "CB CG1 CG2 CD1", "LEU": "CB CG CD1 CD2", "LYS": "CB CG CD CE NZ",
    "MET": "CB CG SD CE", "PHE": "CB CG CD1 CD2 CE1 CE2 CZ", "PRO": "CB CG CD",
    "SER": "CB OG", "THR": "CB OG1 CG2", "TRP": "CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
    "TYR": "CB CG CD1 CD2 CE1 CE2 CZ OH", "VAL": "CB CG1 CG2",
}
_COVALENT_RADII = {"C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05}


@dataclass(frozen=True, slots=True)
class TorsionJoint:
    kind: str
    residue_index: int
    axis_start: int
    axis_end: int
    moving_indices: tuple[int, ...]


def _residue_identifier(key: AtomKey) -> tuple[str, int, str]:
    return key.chain, key.residue_number, key.insertion_code


def _check_bond(xyz: torch.Tensor, a: int, b: int, lower: float, upper: float, label: str) -> None:
    length = float(torch.linalg.vector_norm(xyz[a] - xyz[b]))
    if not lower <= length <= upper:
        raise ValueError(f"{label}: distance {length:.4f} A outside [{lower}, {upper}]")


def _reject_nonpeptide_links(xyz: torch.Tensor, keys: tuple[AtomKey, ...],
                            residue_order: list[int], peptide_edges: set[tuple[int, int]]) -> None:
    """Conservative geometry screen, not a substitute for an explicit topology.

    Non-peptide inter-residue pairs closer than covalent radii + 0.20 A are
    rejected.  Severe atomic overlaps are rejected by the same check.  This
    also catches ordinary ~2.05 A disulfides without deleting their atoms.
    """
    residue = torch.tensor(residue_order, dtype=torch.long, device=xyz.device)
    covalent = xyz.new_tensor([_COVALENT_RADII[key.atom_name[0]] for key in keys])
    atom_index = torch.arange(len(keys), device=xyz.device)
    # Chunk the pair search: a protein-sized N x N distance matrix is not kept.
    for start in range(0, len(keys), 128):
        stop = min(start + 128, len(keys))
        distance = torch.cdist(xyz[start:stop], xyz)
        limit = covalent[start:stop, None] + covalent[None, :] + 0.20
        possible = ((distance < limit)
                    & (residue[start:stop, None] != residue[None, :])
                    & (atom_index[start:stop, None] < atom_index[None, :]))
        for local_a, b in possible.nonzero().tolist():
            a = start + local_a
            if (a, b) in peptide_edges or (b, a) in peptide_edges:
                continue
            name = "disulfide" if keys[a].atom_name == keys[b].atom_name == "SG" else "crosslink or severe overlap"
            raise ValueError(
                f"Unsupported {name}: {keys[a]} -- {keys[b]} "
                f"at {float(distance[local_a, b]):.4f} A"
            )


@dataclass(frozen=True, slots=True, eq=False)
class TorsionKinematics:
    reference: torch.Tensor
    ca_indices: torch.Tensor
    joints: tuple[TorsionJoint, ...]
    atom_keys: tuple[AtomKey, ...]
    residue_ids: tuple[tuple[str, int, str], ...]
    _moving_masks: tuple[torch.Tensor, ...]

    @classmethod
    def from_monomer(cls, monomer: MonomerInput) -> "TorsionKinematics":
        """Validate an input once and build reference-free joint connectivity.

        Residues are ordered by author residue number and insertion code;
        atom ordering within a residue need not be canonical or contiguous.
        Numbering gaps are allowed only when the actual peptide C--N bond is
        intact.  First phi / last psi are omitted, as are ALL proline phi
        joints, which would otherwise open its N--CD ring bond.
        """
        if not isinstance(monomer, MonomerInput):
            raise TypeError("monomer must be a MonomerInput")
        xyz, keys = monomer.xyz, monomer.atom_keys
        if len({key.chain for key in keys}) != 1:
            raise ValueError("Only a single chain is supported")
        residue_ids = tuple(sorted({_residue_identifier(key) for key in keys}, key=lambda item: (item[1], item[2])))
        order = {identifier: i for i, identifier in enumerate(residue_ids)}
        residue_atoms: list[dict[str, int]] = [{} for _ in residue_ids]
        residue_names: list[str | None] = [None] * len(residue_ids)
        atom_residue = []
        for index, key in enumerate(keys):
            rid = order[_residue_identifier(key)]
            atom_residue.append(rid)
            if key.residue_name not in _SIDECHAIN_NAMES:
                raise ValueError(f"Unsupported residue {key.residue_name!r}; standard amino acids only")
            if residue_names[rid] not in (None, key.residue_name):
                raise ValueError("Conflicting residue names for one chain/residue/insertion identifier")
            residue_names[rid] = key.residue_name
            allowed = _BACKBONE | frozenset(_SIDECHAIN_NAMES[key.residue_name].split()) | {"OXT"}
            if key.atom_name not in allowed:
                raise ValueError(f"Unsupported atom {key.atom_name!r} in {key.residue_name}; heavy atoms only")
            if key.atom_name in residue_atoms[rid]:
                raise ValueError("Duplicate atom name within a residue")
            if key.atom_name == "OXT" and rid != len(residue_ids) - 1:
                raise ValueError("OXT is only supported at the last residue")
            residue_atoms[rid][key.atom_name] = index

        peptide_edges: set[tuple[int, int]] = set()
        for rid, atoms in enumerate(residue_atoms):
            missing = _BACKBONE - atoms.keys()
            if missing:
                raise ValueError(f"Incomplete backbone at {residue_ids[rid]}: missing {sorted(missing)}")
            _check_bond(xyz, atoms["N"], atoms["CA"], 1.10, 1.80, "Invalid N--CA bond")
            _check_bond(xyz, atoms["CA"], atoms["C"], 1.10, 1.90, "Invalid CA--C bond")
            _check_bond(xyz, atoms["C"], atoms["O"], 0.90, 1.65, "Invalid C--O bond")
            if "OXT" in atoms:
                _check_bond(xyz, atoms["C"], atoms["OXT"], 0.90, 1.65, "Invalid C--OXT bond")
            triangle = torch.linalg.cross(xyz[atoms["N"]] - xyz[atoms["CA"]],
                                          xyz[atoms["C"]] - xyz[atoms["CA"]])
            if float(torch.linalg.vector_norm(triangle)) < 1e-4:
                raise ValueError("Collinear N/CA/C backbone cannot define a torsion frame")
            if rid + 1 < len(residue_atoms):
                following = residue_atoms[rid + 1]
                if "N" not in following:
                    raise ValueError(f"Incomplete backbone at {residue_ids[rid + 1]}: missing N")
                edge = (atoms["C"], following["N"])
                _check_bond(xyz, *edge, 1.10, 1.65, "Peptide break or invalid C--N bond")
                peptide_edges.add(edge)
        _reject_nonpeptide_links(xyz, keys, atom_residue, peptide_edges)

        joints = []
        for rid, atoms in enumerate(residue_atoms):
            if rid > 0 and residue_names[rid] != "PRO":
                moving = tuple(i for i, atom_rid in enumerate(atom_residue)
                               if atom_rid > rid or (atom_rid == rid and keys[i].atom_name not in ("N", "CA")))
                joints.append(TorsionJoint("phi", rid, atoms["N"], atoms["CA"], moving))
            if rid < len(residue_atoms) - 1:
                moving = tuple(i for i, atom_rid in enumerate(atom_residue)
                               if atom_rid > rid or (atom_rid == rid and keys[i].atom_name in ("O", "OXT")))
                joints.append(TorsionJoint("psi", rid, atoms["CA"], atoms["C"], moving))
        masks = []
        for joint in joints:
            mask = torch.zeros(len(keys), dtype=torch.bool, device=xyz.device)
            mask[list(joint.moving_indices)] = True
            masks.append(mask[:, None])
        return cls(xyz.clone(), torch.tensor([atoms["CA"] for atoms in residue_atoms],
                                            dtype=torch.long, device=xyz.device),
                   tuple(joints), keys, residue_ids, tuple(masks))

    @property
    def n_torsions(self) -> int:
        return len(self.joints)

    def validate_angles(self, delta_angles: torch.Tensor) -> None:
        """External-input validation; call OUTSIDE jacrev/jacfwd/hessian/vmap.

        ``coordinates`` intentionally does no tensor-value-dependent Python
        branching so that nested torch.func transforms remain supported.
        """
        self._validate_angle_metadata(delta_angles)
        if not bool(torch.isfinite(delta_angles).all()):
            raise ValueError("delta_angles must be finite")

    def _validate_angle_metadata(self, delta_angles: torch.Tensor) -> None:
        if not isinstance(delta_angles, torch.Tensor) or delta_angles.shape != (self.n_torsions,):
            raise ValueError(f"delta_angles must have shape ({self.n_torsions},)")
        if delta_angles.dtype != self.reference.dtype or delta_angles.device != self.reference.device:
            raise ValueError("delta_angles must share reference dtype and device")

    def coordinates(self, delta_angles: torch.Tensor) -> torch.Tensor:
        """Return (N,3) coordinates for a vector of torsion increments in radians.

        The map is differentiable through all axes and previous rotations.
        It preserves bond lengths/angles and peptide omega of the supported
        connectivity, but does not enforce Ramachandran or clash validity.
        Nonfinite angles propagate; use ``validate_angles`` at input boundaries.
        """
        self._validate_angle_metadata(delta_angles)
        coordinates = self.reference.clone()
        for index, (joint, moving) in enumerate(zip(self.joints, self._moving_masks)):
            origin = coordinates[joint.axis_start]
            axis = coordinates[joint.axis_end] - origin
            axis = axis / torch.linalg.vector_norm(axis)
            relative = coordinates - origin
            cosine, sine = torch.cos(delta_angles[index]), torch.sin(delta_angles[index])
            cross = torch.linalg.cross(axis.expand_as(relative), relative, dim=-1)
            along = (relative * axis).sum(-1, keepdim=True) * axis
            rotated = origin + cosine * relative + sine * cross + (1.0 - cosine) * along
            coordinates = torch.where(moving, rotated, coordinates)
        return coordinates
