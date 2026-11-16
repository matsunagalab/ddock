#!/usr/bin/env python3
"""Post hoc reference-free check of frozen R2 response proposals.

This diagnostic never changes predictions, selects a checkpoint, or reads a
native complex.  It restores the same apo preparation/model/fields and measures
the actual energy along each already-selected response direction.  Linear
amplitude bounds are not interpreted as finite-coordinate guarantees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import torch

from zdock.geom import rotate
from zdock.inducedfit.chemistry import prepare_amber_charges
from zdock.inducedfit.fields import FieldParameters, build_fields, ligand_channels, receptor_channels
from zdock.inducedfit.hydrogen_kinematics import HydrogenatedKinematics
from zdock.inducedfit.modes import build_torsion_modes
from zdock.inducedfit.pipeline import centered_protein, load_pair_manifest
from zdock.inducedfit.types import GridSpec


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(args, output, save):
    root = Path(__file__).resolve().parents[1]
    source = json.loads(args.report.read_text())
    if source.get('status') not in ('completed', 'partial_failure') or source.get('stage') != 'complete':
        raise ValueError('the source run must finish before this post hoc diagnostic starts')
    source_hash = sha256(args.report)
    output.update(source_report=str(args.report.resolve()), source_report_sha256=source_hash)
    if source.get('device') != 'cpu' or source.get('dtype') != 'float64':
        raise ValueError('this diagnostic currently reproduces CPU float64 reports only')
    for relative, expected in source['source_sha256'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path.suffix != '.py':
            raise ValueError('invalid source-code path in the source report')
        if sha256(path) != expected:
            raise ValueError(f'source changed since the frozen run: {relative}')
    output['verified_source_sha256'] = source['source_sha256']
    config = source['configuration']
    torch.set_num_threads(config['threads'])
    manifest_path = args.manifest or Path(config['manifest'])
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if sha256(manifest_path) != source['input_manifest_sha256']:
        raise ValueError('input manifest changed since the frozen run')
    manifest, inputs = load_pair_manifest(manifest_path)
    if manifest != source['input']:
        raise ValueError('manifest contents differ from the frozen run')
    output.update(input=manifest, input_manifest_sha256=source['input_manifest_sha256'])
    proteins = {}
    for side in ('receptor', 'ligand'):
        output['stage'] = f'prepare_{side}'
        save()
        charged = prepare_amber_charges(inputs[side], pH=config['ph'], seed=config['seed'])
        old = source['preparation'][side]['chemistry']
        for key in ('input_sha256', 'charge_model', 'forcefield_xml_sha256', 'openmm_version',
                    'hydrogens_xml_sha256', 'residues_xml_sha256', 'modeller_source_sha256',
                    'pH', 'seed', 'variants', 'n_atoms', 'radii_angstrom'):
            if charged.provenance[key] != old[key]:
                raise ValueError(f'{side} chemistry provenance changed: {key}')
        proteins[side], center = centered_protein(charged)
        expected_center = center.new_tensor(source['preparation'][side]['center'])
        if not torch.allclose(center, expected_center, atol=1e-12, rtol=1e-12):
            raise ValueError(f'{side} centering changed')
        if charged.heavy_indices.tolist() != source['preparation'][side]['heavy_to_all_atom_indices']:
            raise ValueError(f'{side} heavy-atom mapping changed')
    output['stage'] = 'restore_model'
    save()
    model = build_torsion_modes(
        HydrogenatedKinematics.from_prepared(proteins['receptor']),
        n_modes=config['modes'], fit_relative_damping=config['fit_damping'],
    )
    expected_stiffness = model.stiffness.new_tensor(source['stiffness'])
    if not torch.allclose(model.stiffness, expected_stiffness, atol=1e-9, rtol=1e-10):
        raise ValueError('mode stiffness differs from the frozen run')
    grid = GridSpec(tuple(source['grid']['shape']), source['grid']['spacing'], tuple(source['grid']['origin']))
    parameters = FieldParameters(**source['parameters'])
    receptor, ligand = proteins['receptor'].monomer, proteins['ligand'].monomer
    radius = source['effective_radius']
    atom_limit = config['max_atom_displacement']
    quaternions = receptor.xyz.new_tensor(source['quaternions'])
    output.update(parameters=source['parameters'], grid=source['grid'], effective_radius=radius,
                  max_atom_displacement=atom_limit, mode_diagnostics=model.diagnostics,
                  scales=[0.25, 0.5, 1.0], candidates=[])
    candidates = source['arms']['R2']['candidates']
    if not candidates:
        raise ValueError('the source run contains no frozen R2 candidates')
    with torch.no_grad():
        reference_fields = build_fields(receptor.xyz, receptor.radii, receptor.charges, grid, parameters)
        reference_channels = receptor_channels(reference_fields, grid, parameters)
        for candidate in candidates:
            index = candidate['selection_rank']
            output['stage'] = f'candidate_{index}'
            save()
            quaternion = quaternions[candidate['rotation_index']]
            translation = receptor.xyz.new_tensor(candidate['translation'])
            amplitude = receptor.xyz.new_tensor(candidate['predicted_amplitudes'])
            if amplitude.shape != (model.n_modes,) or not bool(torch.isfinite(amplitude).all()):
                raise ValueError(f'invalid frozen amplitudes for candidate {index}')
            ligand_xyz = torch.stack(rotate(*ligand.xyz.unbind(-1), quaternion), dim=-1) + translation
            lig_fields = build_fields(ligand_xyz, ligand.radii, ligand.charges, grid, parameters)
            lig_channels = ligand_channels(lig_fields)
            baseline_energy = float((reference_channels * lig_channels).sum())
            baseline_error = baseline_energy - candidate['selected_rigid_energy']
            if not math.isfinite(baseline_energy) or not math.isclose(
                baseline_energy, candidate['selected_rigid_energy'], abs_tol=1e-7, rel_tol=1e-9,
            ):
                raise ValueError(f'candidate {index} initial energy does not reproduce the frozen run: delta={baseline_error}')
            quadratic_strain = float(0.5 * amplitude @ model.stiffness @ amplitude)
            linear_term = candidate['predicted_gain'] - quadratic_strain
            row = {
                'selection_rank': index, 'rotation_index': candidate['rotation_index'],
                'translation': candidate['translation'], 'predicted_amplitudes': candidate['predicted_amplitudes'],
                'selected_score': candidate['selected_score'], 'predicted_full_gain': candidate['predicted_gain'],
                'reproduced_rigid_energy': baseline_energy, 'rigid_energy_difference_from_run': baseline_error,
                'quadratic_strain_at_full_amplitude': quadratic_strain,
                'linear_term_derived_from_frozen_gain': linear_term, 'trials': [],
            }
            output['candidates'].append(row)
            for scale in output['scales']:
                a = scale * amplitude
                coordinates = model.coordinates(a)
                displacement = coordinates - model.reference
                ca_rmsd = float(displacement[model.ca_indices].square().sum(-1).mean().sqrt())
                max_atom = float(torch.linalg.vector_norm(displacement, dim=-1).max())
                amplitude_norm = float(a.norm())
                trial = {
                    'scale': scale, 'amplitude_norm': amplitude_norm, 'ca_rmsd': ca_rmsd,
                    'maximum_atom_displacement': max_atom,
                    'amplitude_bound_ok': amplitude_norm <= radius + 1e-10,
                    'actual_ca_bound_ok': ca_rmsd <= radius + 1e-10,
                    'actual_atom_bound_ok': max_atom <= atom_limit + 1e-10,
                    'predicted_quadratic_gain': scale * linear_term + scale * scale * quadratic_strain,
                }
                trial['geometry_feasible'] = all(trial[key] for key in (
                    'amplitude_bound_ok', 'actual_ca_bound_ok', 'actual_atom_bound_ok',
                ))
                try:
                    fields = build_fields(coordinates, receptor.radii, receptor.charges, grid, parameters)
                    strain = float(model.deformation_energy(a))
                    actual_energy = float((receptor_channels(fields, grid, parameters) * lig_channels).sum()) + strain
                    if not all(math.isfinite(value) for value in (actual_energy, strain, ca_rmsd, max_atom)):
                        raise FloatingPointError('nonfinite diagnostic energy or geometry')
                    actual_gain = actual_energy - baseline_energy
                    trial.update(energy_status='evaluated', actual_energy=actual_energy, actual_strain=strain,
                                 actual_gain=actual_gain,
                                 prediction_error=actual_gain - trial['predicted_quadratic_gain'])
                except Exception as error:
                    status = 'support_failure' if isinstance(error, ValueError) and 'compact support' in str(error) else 'evaluation_failure'
                    trial.update(energy_status=status, error_type=type(error).__name__, error=str(error))
                row['trials'].append(trial)
                save()
            print(json.dumps(row), flush=True)
    if sha256(args.report) != source_hash:
        raise ValueError('source run report changed during the diagnostic')
    output['status'] = 'completed'
    output['stage'] = 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='new diagnostic JSON; never overwrites an existing result')
    parser.add_argument('--manifest', type=Path, help='optional relocated manifest, required to have the same SHA256')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output already exists')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    output = {
        'experiment': 'frozen_response_direction_diagnostic_v1', 'status': 'running',
        'stage': 'validate_frozen_run', 'post_hoc': True,
        'scope': 'same effective energy and strain; fixed rigid poses; no prediction changes or reranking',
        'native_access': 'none; only the inference manifest inputs are opened',
        'selection': 'all frozen R2 candidates; no fit or parameter selection',
        'units': 'angstrom, radians, and effective smooth-field energy units; not kcal/mol',
        'command': sys.argv, 'platform': platform.platform(), 'torch': torch.__version__,
        'diagnostic_source_sha256': sha256(Path(__file__)),
    }

    def save():
        output['wall_seconds'] = time.monotonic() - started
        usage = resource.getrusage(resource.RUSAGE_SELF)
        output['process_peak_rss_bytes'] = usage.ru_maxrss if sys.platform == 'darwin' else usage.ru_maxrss * 1024
        temporary = args.out.with_name(args.out.name + '.tmp')
        temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.out)

    save()
    try:
        diagnose(args, output, save)
    except Exception as error:
        output.update(status='exception', error_type=type(error).__name__, error=str(error))
    save()
    print(json.dumps({key: output[key] for key in ('status', 'stage', 'error', 'wall_seconds') if key in output}, indent=2))
    return 0 if output['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
