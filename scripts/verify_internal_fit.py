#!/usr/bin/env python3
"""Connect PDB-derived torsion modes, FFT selection and nonlinear refinement.

This is a small fragment plus a synthetic two-carbon probe, with electrostatics
disabled explicitly. It checks computation/geometry, not protein docking accuracy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import torch

from zdock.inducedfit.fields import FieldParameters, build_fields, physical_energy
from zdock.inducedfit.kinematics import TorsionKinematics
from zdock.inducedfit.modes import build_torsion_modes
from zdock.inducedfit.pdb_input import read_pdb_monomer, validate_torsion_input
from zdock.inducedfit.relax import relax_pose
from zdock.inducedfit.search import prepare_response_receptor, search_response
from zdock.inducedfit.types import AtomKey, GridSpec, MonomerInput
from verify_internal_modes import geometry, topology_indices


def run(args):
    prepared = read_pdb_monomer(args.pdb, chain=args.chain, charge_policy="zero_diagnostic")
    validate_torsion_input(prepared)
    all_input = prepared.monomer
    residue_ids = sorted({(k.residue_number, k.insertion_code) for k in all_input.atom_keys})[:args.residues]
    indices = [i for i, k in enumerate(all_input.atom_keys) if (k.residue_number, k.insertion_code) in residue_ids]
    center = all_input.xyz[indices].mean(0)
    receptor = MonomerInput(all_input.xyz[indices] - center, all_input.radii[indices], all_input.charges[indices],
                             tuple(all_input.atom_keys[i] for i in indices))
    kinematics = TorsionKinematics.from_monomer(receptor)
    model = build_torsion_modes(kinematics, n_modes=2, fit_relative_damping=1e-3)
    anchor = receptor.xyz[receptor.xyz[:, 0].argmax()] + receptor.xyz.new_tensor([2.5, 0., 0.])
    probe_xyz = anchor + receptor.xyz.new_tensor([[0., -.6, 0.], [0., .6, 0.]])
    ligand = MonomerInput(probe_xyz, receptor.xyz.new_full((2,), 1.7), receptor.xyz.new_zeros(2),
                           (AtomKey("P", 1, "", "SYN", "C1"), AtomKey("P", 1, "", "SYN", "C2")))
    parameters = FieldParameters(surface_width=.5, repulsion_weight=3., charge_weight=0.,
                                  screening_switch_start=2., screening_cutoff=3.)
    all_xyz = torch.cat((receptor.xyz, ligand.xyz))
    spacing = 1.5
    # Complete smooth-field support plus room for local motion, fixed once.
    support = float((torch.cat((receptor.radii, ligand.radii)).max() + parameters.surface_width)
                    / math.sqrt(2 * math.log(2)) * parameters.occupancy_cutoff_sigma)
    lower = torch.floor((all_xyz.amin(0) - support - 2.) / spacing) * spacing
    upper = torch.ceil((all_xyz.amax(0) + support + 2.) / spacing) * spacing
    grid = GridSpec(tuple(int(n) for n in torch.round((upper - lower) / spacing).tolist()), spacing, tuple(lower.tolist()))
    # Add the endpoint node; GridSpec counts nodes, not intervals.
    grid = GridSpec(tuple(n + 1 for n in grid.shape), spacing, grid.origin)
    cached = prepare_response_receptor(receptor, grid, parameters, model.basis, model.stiffness)
    effective_radius = cached.bounded_radius(.1, .5 if args.bound_linear_atom_motion else None)
    quaternion = receptor.xyz.new_tensor([0., 0., 0., 1.])
    reference_geometry = geometry(receptor.xyz, topology_indices(receptor))
    arms = []

    def verify_arm(selection):
        candidates = search_response(cached, ligand, quaternion[None], radius=effective_radius,
                                      top_k=1, peaks_per_rotation=4, selection=selection)
        if len(candidates) != 1:
            raise RuntimeError("no valid selected pose")
        candidate = candidates[0]
        ligand_fields = build_fields(ligand.xyz + candidate.translation, ligand.radii, ligand.charges, grid, parameters)

        def interaction(a):
            fields = build_fields(model.coordinates(a), receptor.radii, receptor.charges, grid, parameters)
            return physical_energy(fields, ligand_fields, grid, parameters, direct=True)

        zero = receptor.xyz.new_zeros(model.n_modes)
        rigid_direct = float(interaction(zero))
        h = 1e-5
        fd = torch.stack([(interaction(h * direction) - interaction(-h * direction)) / (2 * h)
                          for direction in torch.eye(model.n_modes, dtype=zero.dtype)])
        predicted_from_fd = float(fd @ candidate.amplitudes + .5 * candidate.amplitudes @ model.stiffness @ candidate.amplitudes)
        proposed_xyz = model.coordinates(candidate.amplitudes)
        proposed_displacement = proposed_xyz - receptor.xyz
        proposed_ca_rmsd = float(proposed_displacement[model.ca_indices].square().sum(-1).mean().sqrt())
        proposed_atom_displacement = float(proposed_displacement.norm(dim=-1).max())
        proposed_within_limits = proposed_ca_rmsd <= effective_radius + 1e-10 and proposed_atom_displacement <= .5 + 1e-10
        try:
            proposed_gain = float(interaction(candidate.amplitudes) + model.deformation_energy(candidate.amplitudes)) - rigid_direct
            if not math.isfinite(proposed_gain):
                raise FloatingPointError("nonfinite proposed-step energy")
            proposed_energy_state = "evaluated"
        except ValueError as error:
            proposed_gain = None
            proposed_energy_state = str(error)
        # Both selection arms start from zero deformation, separating selection
        # from a response-amplitude warm start.
        refined = relax_pose(receptor, ligand, model, grid, parameters, quaternion, candidate.translation,
                              radius=effective_radius, max_iterations=args.iterations, max_evaluations=100,
                              max_translation=.5, max_rotation=.15, max_atom_displacement=.5)
        actual_geometry = geometry(refined.receptor_xyz, topology_indices(receptor))
        bond_error = float((actual_geometry[0] - reference_geometry[0]).abs().max())
        angle_error = float((actual_geometry[1] - reference_geometry[1]).abs().max())
        wd = actual_geometry[2] - reference_geometry[2]
        omega_error = float(torch.atan2(wd.sin(), wd.cos()).abs().max())
        row = {
            "selection": selection, "translation": candidate.translation.tolist(),
            "selected_score": candidate.score, "selected_rigid_energy": candidate.rigid_energy,
            "predicted_gain": candidate.predicted_gain, "predicted_gain_from_nonlinear_FD": predicted_from_fd,
            "fft_energy_error": abs(candidate.rigid_energy - rigid_direct),
            "gain_FD_error": abs(candidate.predicted_gain - predicted_from_fd),
            "proposed_actual_gain": proposed_gain, "proposed_energy_state": proposed_energy_state,
            "proposed_CA_RMSD": proposed_ca_rmsd,
            "proposed_maximum_atom_displacement": proposed_atom_displacement,
            "proposed_within_refinement_limits": proposed_within_limits,
            "initial_energy": refined.energy_history[0], "final_energy": refined.energy,
            "status": refined.status, "accepted_steps": refined.accepted_steps,
            "energy_evaluations": refined.energy_evaluations, "energy_history": refined.energy_history,
            "final_amplitudes": refined.amplitudes.tolist(), "CA_RMSD": refined.ca_rmsd,
            "maximum_atom_displacement": refined.maximum_atom_displacement, "strain": refined.strain,
            "bond_length_error": bond_error, "bond_angle_error_radians": angle_error,
            "omega_error_radians": omega_error, "receptor_xyz": refined.receptor_xyz.tolist(),
            "ligand_xyz": refined.ligand_xyz.tolist(),
        }
        if not all(math.isfinite(v) for v in row.values() if isinstance(v, (float, int))):
            raise FloatingPointError("nonfinite integration diagnostic")
        row["passed"] = (row['fft_energy_error'] < 1e-8 and row['gain_FD_error'] < 1e-6
                         and max(bond_error, angle_error, omega_error) < 1e-9
                         and row['CA_RMSD'] <= effective_radius + 1e-10
                         and row['maximum_atom_displacement'] <= .5 + 1e-10
                         and row['final_energy'] <= row['initial_energy'] + 1e-10)
        return row

    for selection in ("rigid", "response"):
        try:
            arms.append(verify_arm(selection))
        except Exception as error:
            arms.append({"selection": selection, "passed": False, "status": "exception",
                         "error_type": type(error).__name__, "error": str(error)})
    return {
        "status": "passed" if all(a['passed'] for a in arms) else "numerical_failure",
        "input_sha256": prepared.source_sha256, "input_provenance": prepared.provenance,
        "selected_residues": residue_ids, "fragment_center": center.tolist(),
        "n_receptor_atoms": len(receptor.xyz), "grid": asdict(grid), "parameters": asdict(parameters),
        "mode_diagnostics": model.diagnostics, "arms": arms,
        "requested_radius": .1, "effective_radius": effective_radius,
        "maximum_atom_motion_per_unit_amplitude": cached.maximum_atom_motion_per_unit_amplitude,
        "bound_linear_atom_motion": args.bound_linear_atom_motion,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdb', type=Path, required=True)
    parser.add_argument('--chain', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--residues', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--bound-linear-atom-motion', action='store_true',
                        help='Conservatively bound predicted per-atom motion and use the same radius in both refinements')
    args = parser.parse_args()
    if args.residues < 3 or args.iterations < 1:
        parser.error('at least three residues and one iteration required')
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    paths = list((root/'src/zdock/inducedfit').glob('*.py')) + [root/'src/zdock/geom.py',
             Path(__file__).resolve(), root/'scripts/verify_internal_modes.py']
    report = {
        "experiment": "internal_coordinate_FFT_and_refinement_integration_v1",
        "scope": "input-derived fragment and synthetic carbon probe; zero diagnostic charges; no docking accuracy benchmark",
        "command": sys.argv, "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "device": "cpu", "dtype": "float64", "threads": 1,
        "commit": subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, capture_output=True, check=True).stdout.strip(),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    try:
        report.update(run(args))
    except Exception as error:
        report.update(status='exception', error_type=type(error).__name__, error=str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ('status','error','n_receptor_atoms','grid') if k in report}, indent=2))
    print(f'Saved {args.out}')
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
