#!/usr/bin/env python3
"""Decompose a verified apo prior's stiffness without changing its Hamiltonian.

The exact anchor torque is differentiated by the producer's central finite
difference protocol. This includes both J.T J and coordinate-map curvature.
No native structures, docking labels, new preparation or parameter search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from zdock.inducedfit.chemistry import prepare_amber_charges
from zdock.inducedfit.pipeline import centered_protein, load_pair_manifest
from zdock.inducedfit.physical_prior import load_prepared_physical_prior


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--prior-report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    producer = json.loads(args.prior_report.read_text())
    registration = dict(command=sys.argv, script_sha256=digest(__file__),
        prior_report_sha256=digest(args.prior_report), manifest_sha256=digest(args.manifest),
        native_access=False, parameter_changes=False, device='cpu', dtype='float64',
        torch_threads=1, seed=producer['registration']['configuration']['seed'],
        purpose='descriptive decomposition of the fixed physical prior, not accuracy evaluation',
        formula='K_anchor = k J.T J + k sum_i (x_i-x_input_i) Hessian_theta(x_i); K_ff = K_total-K_anchor',
        energy_unit='kJ/mol', amplitude_unit='CA RMS displacement in angstrom',
        selection='all eight previously registered modes; no selection by outcome')
    save(args.out/'registration.json', registration)
    report = dict(status='running', registration=registration)
    save(args.out/'report.json', report)
    started = time.monotonic()
    prior = None
    try:
        if registration['manifest_sha256'] != producer['registration']['input_manifest_sha256']:
            raise ValueError('Manifest differs from verified prior input')
        _, inputs = load_pair_manifest(args.manifest)
        protein, _ = centered_protein(prepare_amber_charges(inputs['receptor'], pH=7., seed=registration['seed']))
        parameters = producer['registration']['configuration']
        prior = load_prepared_physical_prior(protein, args.prior_report,
            anchor_k=parameters['anchor_k'], stationarity_tolerance=parameters['stationarity_tolerance'],
            max_heavy_displacement=parameters['max_heavy_displacement'])
        with np.load(args.prior_report.parent/'arrays.npz', allow_pickle=False) as archive:
            directions = torch.from_numpy(archive['angle_basis'].copy())
            coordinate_basis = torch.from_numpy(archive['basis'].copy())
            total = torch.from_numpy(archive['stiffness'].copy())
        m = directions.shape[1]
        if total.shape != (m, m) or coordinate_basis.shape != (m, len(protein.monomer.xyz), 3):
            raise ValueError('Saved physical modes have inconsistent shapes')
        anchor_evaluations = 0

        def anchor_energy(theta):
            delta = prior.kinematics.coordinates(theta)-prior.anchor_reference
            return .5*prior.anchor_k*delta.square().sum()

        def anchor_torque(theta):
            nonlocal anchor_evaluations
            variable = theta.detach().clone().requires_grad_(True)
            gradient, = torch.autograd.grad(anchor_energy(variable), variable)
            anchor_evaluations += 1
            return gradient.detach()

        def stiffness(step):
            columns = []
            for direction in directions.T:
                norm = float(direction.norm())
                offset = direction*(step/norm)
                hv = (anchor_torque(prior.theta0+offset)-anchor_torque(prior.theta0-offset))*(norm/(2*step))
                columns.append(directions.T @ hv)
            return torch.stack(columns, dim=1)

        step = producer['mode_diagnostics']['stiffness_fd_step_rad']
        coarse, raw = stiffness(step), stiffness(step/2)
        relative_change = float((coarse-raw).norm()/raw.norm())
        relative_asymmetry = float((raw-raw.T).norm()/raw.norm())
        if max(relative_change, relative_asymmetry) > producer['mode_diagnostics']['stiffness_fd_relative_tolerance']:
            raise ValueError('Anchor stiffness finite differences are unresolved')
        anchor = .5*(raw+raw.T)
        flat = coordinate_basis.reshape(m, -1)
        gauss_newton = prior.anchor_k*(flat @ flat.T)
        map_curvature = anchor-gauss_newton
        forcefield = total-anchor
        eigenvalues, eigenvectors = torch.linalg.eigh(total)
        eigen_anchor = torch.diag(eigenvectors.T @ anchor @ eigenvectors)
        eigen_ff = torch.diag(eigenvectors.T @ forcefield @ eigenvectors)
        eigen_gn = torch.diag(eigenvectors.T @ gauss_newton @ eigenvectors)
        eigen_map = torch.diag(eigenvectors.T @ map_curvature @ eigenvectors)
        # Independent second energy difference along the softest total eigenmode.
        amplitude_step = 1e-4
        probe_direction = directions @ eigenvectors[:, 0]
        energy_fd = float((anchor_energy(prior.theta0+amplitude_step*probe_direction)
            -2*anchor_energy(prior.theta0)+anchor_energy(prior.theta0-amplitude_step*probe_direction))/amplitude_step**2)
        energy_fd_relative = abs(energy_fd-float(eigen_anchor[0]))/abs(float(eigen_anchor[0]))
        if energy_fd_relative > 1e-4:
            raise ValueError('Anchor stiffness failed independent second energy difference')
        rows = [dict(mode=i+1, total=float(eigenvalues[i]), anchor=float(eigen_anchor[i]),
            ff14SB_OBC2=float(eigen_ff[i]), anchor_fraction=float(eigen_anchor[i]/eigenvalues[i]),
            anchor_gauss_newton=float(eigen_gn[i]), anchor_map_curvature=float(eigen_map[i])) for i in range(m)]
        report.update(status='passed', physical_eigenmodes=rows,
            original_basis_diagonal=[dict(mode=i+1, total=float(total[i,i]), anchor=float(anchor[i,i]),
                ff14SB_OBC2=float(forcefield[i,i]), anchor_fraction=float(anchor[i,i]/total[i,i])) for i in range(m)],
            trace_anchor_fraction=float(torch.trace(anchor)/torch.trace(total)),
            trace_total=float(torch.trace(total)), trace_anchor=float(torch.trace(anchor)),
            trace_forcefield=float(torch.trace(forcefield)),
            forcefield_projected_eigenvalues=torch.linalg.eigvalsh(forcefield).tolist(),
            anchor_projected_eigenvalues=torch.linalg.eigvalsh(anchor).tolist(),
            anchor_map_curvature_relative_norm=float(map_curvature.norm()/anchor.norm()),
            finite_difference=dict(torsion_step_rad=step, stiffness_relative_change=relative_change,
                stiffness_relative_asymmetry=relative_asymmetry, independent_amplitude_step=amplitude_step,
                independent_second_energy_difference=energy_fd,
                independent_relative_error=energy_fd_relative),
            matrices={key:matrix.tolist() for key,matrix in [('total',total),('anchor',anchor),
                ('ff14SB_OBC2',forcefield),('anchor_gauss_newton',gauss_newton),('anchor_map_curvature',map_curvature)]},
            anchor_torque_evaluations=anchor_evaluations,
            restore_validation=prior.diagnostics['restore_validation'])
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        if prior is not None:
            report['openmm_evaluations'] = prior._context.evaluations
            prior.close()
        report['wall_seconds'] = time.monotonic()-started
        save(args.out/'report.json', report)
    print(json.dumps({k:report[k] for k in ['status','trace_anchor_fraction','wall_seconds','error'] if k in report}, indent=2))
    return 0 if report['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
