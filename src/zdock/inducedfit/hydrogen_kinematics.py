"""Backbone torsions on a prepared protein including force-field hydrogens.

Chemistry preparation supplies atom identities and the covalent bond graph.
This module never assigns or modifies charges.  The heavy-atom torsion model
validates the supported backbone and selects joints; cutting each joint bond
in the *full* graph determines which hydrogens must follow the rotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .kinematics import TorsionJoint, TorsionKinematics
from .types import MonomerInput


class HydrogenatedProtein(Protocol):
    monomer: MonomerInput
    heavy_monomer: MonomerInput
    heavy_indices: torch.Tensor
    bonds: tuple[tuple[int, int], ...]
    provenance: dict


def _component(
    adjacency: list[set[int]],
    start: int,
    cut: tuple[int, int] | None = None,
) -> set[int]:
    reached, pending = {start}, [start]
    while pending:
        atom = pending.pop()
        for neighbor in adjacency[atom]:
            if cut is not None and ((atom, neighbor) == cut or (neighbor, atom) == cut):
                continue
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    return reached


def _bond_graph(bonds, n_atoms: int) -> tuple[tuple[tuple[int, int], ...], list[set[int]]]:
    if isinstance(bonds, torch.Tensor):
        if bonds.dtype != torch.long or bonds.ndim != 2 or bonds.shape[1] != 2:
            raise ValueError("bonds tensor must be long with shape (E, 2)")
        raw = bonds.detach().cpu().tolist()
    else:
        try:
            raw = list(bonds)
        except TypeError as error:
            raise ValueError("bonds must be an iterable of atom-index pairs") from error
    canonical = set()
    adjacency = [set() for _ in range(n_atoms)]
    for pair in raw:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each bond must contain two atom indices")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in pair):
            raise ValueError("bond indices must be integers")
        first, second = pair
        if first == second or min(first, second) < 0 or max(first, second) >= n_atoms:
            raise ValueError("bond indices must be distinct and in range")
        ordered = tuple(sorted((first, second)))
        if ordered in canonical:
            raise ValueError("duplicate covalent bond")
        canonical.add(ordered)
        adjacency[first].add(second)
        adjacency[second].add(first)
    return tuple(sorted(canonical)), adjacency


@dataclass(frozen=True, slots=True, eq=False)
class HydrogenatedKinematics:
    """A graph-validated full-atom wrapper around the torsion rotation kernel.

    The initial heavy coordinates and their ordering are retained.  Root
    N/CA/C remain fixed as in :class:`TorsionKinematics`; all proline phi
    joints remain omitted.  Bond lengths and angles of the supplied graph
    are preserved, but Ramachandran validity and nonbonded clashes are not
    enforced.  This class requires chemistry-prepared topology: it does not
    infer bonds from atom names or distances.
    """

    _delegate: TorsionKinematics
    heavy_kinematics: TorsionKinematics
    heavy_indices: torch.Tensor
    bonds: tuple[tuple[int, int], ...]
    provenance: dict

    @classmethod
    def from_prepared(cls, prepared: HydrogenatedProtein) -> "HydrogenatedKinematics":
        full, heavy = prepared.monomer, prepared.heavy_monomer
        if not isinstance(full, MonomerInput) or not isinstance(heavy, MonomerInput):
            raise TypeError("prepared monomer and heavy_monomer must be MonomerInput objects")
        if full.xyz.dtype not in (torch.float32, torch.float64):
            raise TypeError("coordinates must have float32 or float64 dtype")
        if full.xyz.dtype != heavy.xyz.dtype or full.xyz.device != heavy.xyz.device:
            raise ValueError("full and heavy monomers must share coordinate dtype and device")
        indices = prepared.heavy_indices
        if not isinstance(indices, torch.Tensor) or indices.dtype != torch.long or indices.shape != (heavy.n_atoms,):
            raise ValueError("heavy_indices must be a long vector in original heavy-atom order")
        if indices.device != full.xyz.device:
            raise ValueError("heavy_indices and coordinates must share the same device")
        if bool((indices < 0).any()) or bool((indices >= full.n_atoms).any()) or len(torch.unique(indices)) != len(indices):
            raise ValueError("heavy_indices must be unique and in range")
        mapped_indices = indices.detach().cpu().tolist()
        if tuple(full.atom_keys[index] for index in mapped_indices) != heavy.atom_keys:
            raise ValueError("heavy atom identities/order disagree with the full-atom mapping")
        tolerance = 32 * torch.finfo(full.xyz.dtype).eps
        if not torch.allclose(full.xyz[indices], heavy.xyz, atol=tolerance, rtol=tolerance):
            raise ValueError("full-atom preparation changed the heavy reference coordinates")

        # This factory already rejects unsupported residues, backbone breaks,
        # detected heavy crosslinks, and phi rotation through proline rings.
        heavy_kinematics = TorsionKinematics.from_monomer(heavy)
        bonds, adjacency = _bond_graph(prepared.bonds, full.n_atoms)
        heavy_set = set(mapped_indices)
        for index, key in enumerate(full.atom_keys):
            if index in heavy_set:
                continue
            # Force-field H names may start with a PDB numeric prefix.
            if not key.atom_name.lstrip("0123456789").startswith("H"):
                raise ValueError("every atom outside heavy_indices must be a hydrogen")
            if len(adjacency[index]) != 1 or next(iter(adjacency[index])) not in heavy_set:
                raise ValueError("every hydrogen must have exactly one bond to a heavy atom")
            parent = next(iter(adjacency[index]))
            parent_key = full.atom_keys[parent]
            identity = lambda atom: (atom.chain, atom.residue_number, atom.insertion_code, atom.residue_name)
            if identity(key) != identity(parent_key):
                raise ValueError("hydrogen and its bonded heavy atom must belong to the same residue")
        if len(_component(adjacency, 0)) != full.n_atoms:
            raise ValueError("full-atom covalent graph must be connected")

        joints, masks = [], []
        for heavy_joint in heavy_kinematics.joints:
            axis_start = mapped_indices[heavy_joint.axis_start]
            axis_end = mapped_indices[heavy_joint.axis_end]
            if axis_end not in adjacency[axis_start]:
                raise ValueError(f"missing axis bond for {heavy_joint.kind} at residue {heavy_joint.residue_index}")
            moving = _component(adjacency, axis_end, cut=(axis_start, axis_end))
            if axis_start in moving:
                raise ValueError(f"torsion axis bond lies in a ring or crosslink: {heavy_joint.kind} at residue {heavy_joint.residue_index}")
            expected_heavy = {mapped_indices[index] for index in heavy_joint.moving_indices} | {axis_end}
            if moving & heavy_set != expected_heavy:
                raise ValueError(f"full bond graph disagrees with heavy torsion connectivity: {heavy_joint.kind} at residue {heavy_joint.residue_index}")
            # axis_end is in this component; its own coordinate is on-axis,
            # while its attached H atoms must move with the downstream group.
            ordered = tuple(sorted(moving))
            joints.append(TorsionJoint(heavy_joint.kind, heavy_joint.residue_index, axis_start, axis_end, ordered))
            mask = torch.zeros(full.n_atoms, dtype=torch.bool, device=full.xyz.device)
            mask[list(ordered)] = True
            masks.append(mask[:, None])

        delegate = TorsionKinematics(
            reference=full.xyz.clone(),
            ca_indices=indices[heavy_kinematics.ca_indices].clone(),
            joints=tuple(joints),
            atom_keys=full.atom_keys,
            residue_ids=heavy_kinematics.residue_ids,
            _moving_masks=tuple(masks),
        )
        provenance = {
            "geometry_method": "full_covalent_graph_cut_backbone_torsions",
            "n_heavy_atoms": heavy.n_atoms,
            "n_hydrogens": full.n_atoms - heavy.n_atoms,
            "n_bonds": len(bonds),
            "n_torsions": delegate.n_torsions,
            "chemistry": dict(prepared.provenance),
        }
        return cls(delegate, heavy_kinematics, indices.clone(), bonds, provenance)

    @property
    def reference(self) -> torch.Tensor:
        return self._delegate.reference

    @property
    def ca_indices(self) -> torch.Tensor:
        return self._delegate.ca_indices

    @property
    def n_torsions(self) -> int:
        return self._delegate.n_torsions

    @property
    def joints(self) -> tuple[TorsionJoint, ...]:
        return self._delegate.joints

    @property
    def atom_keys(self):
        return self._delegate.atom_keys

    @property
    def residue_ids(self):
        return self._delegate.residue_ids

    def validate_angles(self, delta_angles: torch.Tensor) -> None:
        self._delegate.validate_angles(delta_angles)

    def coordinates(self, delta_angles: torch.Tensor) -> torch.Tensor:
        """Apply the validated full-atom joint masks using the existing AD map."""
        return self._delegate.coordinates(delta_angles)
