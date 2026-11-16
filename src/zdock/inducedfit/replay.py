"""Replay frozen input-only pilot poses, retaining every prepared hydrogen.

This module reads an inference report and its registered input/export files.
It never reads native structures or accuracy scores. Model regeneration is
checked against the saved stiffness, atom map and exported heavy coordinates.
PDB rounding uncertainty is measured, not silently called an exact replay of
the ligand's unpublished final rigid transform.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from zdock.geom import rotate
from .chemistry import prepare_amber_charges
from .fields import FieldParameters
from .hydrogen_kinematics import HydrogenatedKinematics
from .modes import build_torsion_modes
from .pdb_input import read_pdb_monomer
from .pipeline import centered_protein, load_pair_manifest
from .relax import _rotate_vector
from .types import GridSpec


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_rigid_transform(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return proper row-vector R,t with ``source @ R + t ~= target``."""
    if (source.ndim != 2 or source.shape[1] != 3 or source.shape != target.shape
            or len(source) < 3 or source.dtype not in (torch.float32, torch.float64)
            or source.dtype != target.dtype or source.device != target.device):
        raise ValueError("Rigid fit requires matching (N>=3,3) float tensors on one dtype/device")
    if not bool(torch.isfinite(source).all() and torch.isfinite(target).all()):
        raise ValueError("Rigid fit coordinates must be finite")
    source_center, target_center = source.mean(0), target.mean(0)
    x, y = source - source_center, target - target_center
    left, singular, right = torch.linalg.svd(x.T @ y)
    threshold = 64 * torch.finfo(source.dtype).eps * singular.max()
    if int((singular > threshold).sum()) < 2:
        raise ValueError("Collinear/coincident heavy atoms cannot determine a rigid transform")
    correction = torch.eye(3, dtype=source.dtype, device=source.device)
    correction[-1, -1] = torch.det(left @ right)
    rotation = left @ correction @ right
    return rotation, target_center - source_center @ rotation


def _coordinate_error(actual: torch.Tensor, exported: torch.Tensor) -> dict:
    delta = actual - exported
    return {"maximum_atom_error_angstrom": float(torch.linalg.vector_norm(delta, dim=-1).max()),
            "rms_atom_error_angstrom": float(delta.square().sum(-1).mean().sqrt()),
            "maximum_component_error_angstrom": float(delta.abs().max())}


@dataclass(frozen=True, slots=True, eq=False)
class FrozenPose:
    arm: str
    selection_rank: int
    final_rank: int | None
    amplitudes: torch.Tensor
    receptor_xyz: torch.Tensor
    ligand_xyz: torch.Tensor
    heavy_export_errors: dict
    strain: float
    stored_energy: float
    stage: str = "final"


@dataclass(frozen=True, slots=True, eq=False)
class ReplayContext:
    report_path: Path
    report: dict
    inputs: dict
    proteins: dict
    centers: dict
    model: object
    grid: GridSpec
    parameters: FieldParameters
    quaternions: torch.Tensor
    receptor: object
    ligand: object
    source_audit: dict

    @torch.no_grad()
    def initial_pose(self, arm: str, row: dict, *, export_tolerance: float = 1e-3) -> FrozenPose:
        """Replay the unrelaxed pose from the recorded quaternion/translation."""
        if arm not in self.report["arms"] or row not in self.report["arms"][arm].get("candidates", []):
            raise ValueError("Candidate must belong to this frozen report and arm")
        if not 0 < export_tolerance <= 1e-3:
            raise ValueError("Export tolerance must be positive and no looser than 0.001 A")
        quaternion = self.quaternions[row["rotation_index"]]
        receptor_xyz = self.receptor.xyz
        ligand_xyz = torch.stack(rotate(*self.ligand.xyz.unbind(-1), quaternion), dim=-1)
        ligand_xyz = ligand_xyz + ligand_xyz.new_tensor(row["translation"])
        errors = {}
        for side, xyz in (("receptor", receptor_xyz), ("ligand", ligand_xyz)):
            path = (self.report_path.parent / row["initial_files"][side]).resolve()
            if not path.is_relative_to(self.report_path.parent):
                raise ValueError("Export path escapes the frozen run directory")
            parsed = read_pdb_monomer(path, chain=self.inputs[side].monomer.atom_keys[0].chain,
                                       charge_policy="zero_diagnostic")
            if parsed.monomer.atom_keys != self.inputs[side].monomer.atom_keys:
                raise ValueError(f"{side} initial export changed atom identities or ordering")
            target = parsed.monomer.xyz - self.centers["receptor"]
            errors[side] = {"path": str(path), "sha256": _sha256(path),
                            **_coordinate_error(xyz[self.proteins[side].heavy_indices], target)}
            if errors[side]["maximum_atom_error_angstrom"] > export_tolerance:
                raise ValueError(f"{side} initial replay disagrees with heavy export: {errors[side]}")
        errors["tolerance_angstrom"] = export_tolerance
        errors["ligand_transform_policy"] = "recorded initial quaternion and translation, applied to all atoms"
        return FrozenPose(arm, row["selection_rank"], row.get("final_rank"),
                          receptor_xyz.new_zeros(self.model.n_modes), receptor_xyz, ligand_xyz,
                          errors, 0., float(row["selected_rigid_energy"]), stage="initial")

    @torch.no_grad()
    def final_pose(self, arm: str, row: dict, *, export_tolerance: float = 1e-3) -> FrozenPose:
        """Reconstruct a final pose in the centered receptor input frame.

        Receptor deformation comes from saved amplitudes. The ligand uses
        saved rigid increments when present; older reports require a rigid
        fit to the complete heavy export. Both paths move all prepared atoms,
        including hydrogen coordinates.
        """
        if arm not in self.report["arms"] or row not in self.report["arms"][arm].get("candidates", []):
            raise ValueError("Candidate must belong to this frozen report and arm")
        if "final_rank" not in row or "files" not in row:
            raise ValueError("Candidate lacks a final ranked export")
        if not 0 < export_tolerance <= 1e-3:
            raise ValueError("Export tolerance must be positive and no looser than 0.001 A")
        amplitudes = self.receptor.xyz.new_tensor(row["amplitudes"])
        if amplitudes.shape != (self.model.n_modes,) or not bool(torch.isfinite(amplitudes).all()):
            raise ValueError("Saved amplitudes are invalid")
        receptor_xyz = self.model.coordinates(amplitudes)
        exports, errors = {}, {}
        for side in ("receptor", "ligand"):
            path = (self.report_path.parent / row["files"][side]).resolve()
            if not path.is_relative_to(self.report_path.parent):
                raise ValueError("Export path escapes the frozen run directory")
            parsed = read_pdb_monomer(path, chain=self.inputs[side].monomer.atom_keys[0].chain,
                                       charge_policy="zero_diagnostic")
            if parsed.monomer.atom_keys != self.inputs[side].monomer.atom_keys:
                raise ValueError(f"{side} export changed atom identities or ordering")
            exports[side] = parsed.monomer.xyz - self.centers["receptor"]
            errors[side] = {"path": str(path), "sha256": _sha256(path)}
        increment_keys = {"rotation_increment", "translation_increment"}
        supplied = increment_keys.intersection(row)
        if supplied and supplied != increment_keys:
            raise ValueError("Final rigid replay needs both rotation and translation increments")
        if supplied:
            omega = self.ligand.xyz.new_tensor(row["rotation_increment"])
            delta = self.ligand.xyz.new_tensor(row["translation_increment"])
            if omega.shape != (3,) or delta.shape != (3,) or not bool(torch.isfinite(omega).all() and torch.isfinite(delta).all()):
                raise ValueError("Final rigid increments must be finite 3-vectors")
            rotated = torch.stack(rotate(*self.ligand.xyz.unbind(-1), self.quaternions[row["rotation_index"]]), dim=-1)
            center = rotated.mean(0)
            ligand_xyz = _rotate_vector(rotated - center, omega) + center
            ligand_xyz = ligand_xyz + ligand_xyz.new_tensor(row["translation"]) + delta
            transform_policy = "saved initial quaternion and translation plus final increments, applied to all atoms"
        else:
            rotation, translation = fit_rigid_transform(self.proteins["ligand"].heavy_monomer.xyz,
                                                         exports["ligand"])
            ligand_xyz = self.ligand.xyz @ rotation + translation
            transform_policy = "proper Kabsch fit to rounded heavy export; all H follow the same rigid transform"
        for side, xyz in (("receptor", receptor_xyz), ("ligand", ligand_xyz)):
            errors[side].update(_coordinate_error(xyz[self.proteins[side].heavy_indices], exports[side]))
            if errors[side]["maximum_atom_error_angstrom"] > export_tolerance:
                raise ValueError(f"{side} replay disagrees with heavy export: {errors[side]}")
        strain = float(self.model.deformation_energy(amplitudes))
        if abs(strain - row["strain"]) > 1e-8 * max(1., abs(row["strain"])):
            raise ValueError("Replayed deformation prior differs from the stored value")
        errors["tolerance_angstrom"] = export_tolerance
        errors["ligand_transform_policy"] = transform_policy
        return FrozenPose(arm, row["selection_rank"], row["final_rank"], amplitudes,
                          receptor_xyz, ligand_xyz, errors, strain, float(row["energy"]))


def load_replay(report_path: str | Path) -> ReplayContext:
    """Load a completed input-only run; require exact input hashes and maps.

    Current versus recorded code hashes are logged, not assumed equal: changing
    an entry script need not invalidate frozen coordinates. The recomputed
    stiffness, force-field fingerprints, centers and heavy exports are checked.
    """
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    if report.get("status") not in {"completed", "partial_failure"}:
        raise ValueError("Only completed inference reports can be replayed")
    repo = Path(__file__).resolve().parents[3]
    config = report["configuration"]
    manifest_path = Path(config["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = repo / manifest_path
    if _sha256(manifest_path) != report["input_manifest_sha256"]:
        raise ValueError("Input manifest SHA256 differs from the frozen run")
    manifest, inputs = load_pair_manifest(manifest_path)
    if manifest != report["input"]:
        raise ValueError("Resolved input manifest differs from the frozen report")
    proteins, centers = {}, {}
    for side in ("receptor", "ligand"):
        charged = prepare_amber_charges(inputs[side], pH=config["ph"], seed=config["seed"])
        saved = report["preparation"][side]
        for key in ("openmm_version", "forcefield_xml_sha256", "hydrogens_xml_sha256", "residues_xml_sha256", "modeller_source_sha256"):
            if charged.provenance.get(key) != saved["chemistry"].get(key):
                raise ValueError(f"{side} chemistry provenance changed: {key}")
        proteins[side], centers[side] = centered_protein(charged)
        if proteins[side].heavy_indices.tolist() != saved["heavy_to_all_atom_indices"]:
            raise ValueError(f"{side} heavy-to-all-atom mapping changed")
        if not torch.allclose(centers[side], centers[side].new_tensor(saved["center"]), atol=1e-12, rtol=0):
            raise ValueError(f"{side} input center changed")
    receptor, ligand = proteins["receptor"].monomer, proteins["ligand"].monomer
    model = build_torsion_modes(HydrogenatedKinematics.from_prepared(proteins["receptor"]),
                                n_modes=config["modes"], fit_relative_damping=config["fit_damping"],
                                jacobian_backend=config.get("jacobian_backend", "dense"),
                                jvp_chunk_size=config.get("jvp_chunk_size", 8),
                                anm_backend=config.get("anm_backend", "dense"),
                                anm_seed=config.get("anm_seed", 0),
                                anm_tolerance=config.get("anm_tolerance", 1e-9))
    if not torch.allclose(model.stiffness, model.stiffness.new_tensor(report["stiffness"]), atol=1e-9, rtol=1e-7):
        raise ValueError("Recomputed stiffness differs from the frozen model")
    source_audit = {"report_sha256": _sha256(report_path), "manifest_sha256": _sha256(manifest_path),
                    "source_files": {}, "native_access": False,
                    "hydrogen_replay_assumption": "Regenerated using the same input, seed, OpenMM version, force-field and Modeller source hashes; original run did not save a full-H coordinate hash"}
    for name, recorded in report.get("source_sha256", {}).items():
        path = (repo / name).resolve()
        if not path.is_relative_to(repo):
            raise ValueError("Recorded code path escapes the repository")
        current = _sha256(path) if path.is_file() else None
        source_audit["source_files"][name] = {"recorded": recorded, "current": current, "changed": current != recorded}
    return ReplayContext(report_path, report, inputs, proteins, centers, model,
                          GridSpec(**report["grid"]), FieldParameters(**report["parameters"]),
                          receptor.xyz.new_tensor(report["quaternions"]), receptor, ligand, source_audit)
