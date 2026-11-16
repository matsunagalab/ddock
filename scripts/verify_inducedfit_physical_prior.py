#!/usr/bin/env python3
"""Input-only stationary ff14SB/OBC2 prior and mode-curvature verification.

Generated JSON/NPZ stay outside git. Native structures and accuracy labels are
not accepted. Failed preparation and unresolved curvature are retained rather
than silently retried under changed parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch

from zdock.inducedfit.chemistry import prepare_amber_charges
from zdock.inducedfit.pipeline import load_pair_manifest, centered_protein
from zdock.inducedfit.modes import build_torsion_modes
from zdock.inducedfit.physical_prior import PhysicalPrior, build_physical_modes, POLISH_PROTOCOL


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--modes', type=int, default=8)
    parser.add_argument('--anchor-k', type=float, default=10.)
    parser.add_argument('--max-iterations', type=int, default=1000)
    parser.add_argument('--max-evaluations', type=int, default=5000)
    parser.add_argument('--stationarity-tolerance', type=float, default=1e-3)
    parser.add_argument('--max-heavy-displacement', type=float, default=1.)
    parser.add_argument('--preconditioner', choices=['diagonal', 'anchor_metric'], default='anchor_metric')
    parser.add_argument('--polish', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fd-step', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1 or args.modes < 1:
        parser.error('thread and mode counts must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    repo = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), *[repo/'src/zdock/inducedfit'/name for name in
        ['physical_prior.py', 'atomistic_score.py', 'hydrogen_kinematics.py', 'kinematics.py', 'modes.py',
         'chemistry.py', 'pipeline.py', 'pdb_input.py', 'types.py']]]
    registered = dict(command=sys.argv, configuration={key:str(value) if isinstance(value,Path) else value
        for key,value in vars(args).items()}, input_manifest_sha256=digest(args.manifest),
        source_sha256={str(path.relative_to(repo)):digest(path) for path in paths}, native_access=False,
        purpose='stationarity, finite differences and common physical prior; no docking accuracy',
        loss=None, checkpoint=None, platform=platform.platform(), python=platform.python_version(),
        numpy=np.__version__, torch=torch.__version__, device='cpu', dtype='float64',
        thread_policy='torch.set_num_threads; NumPy/SciPy BLAS not independently controlled')
    registered['polishing_protocol'] = POLISH_PROTOCOL if args.polish else None
    save(args.out/'registration.json', registered)
    report = dict(status='running', stage='chemistry', registration=registered)
    save(args.out/'report.json', report)
    started = time.monotonic()
    prior = None
    try:
        manifest, inputs = load_pair_manifest(args.manifest)
        protein, center = centered_protein(prepare_amber_charges(inputs['receptor'], pH=7., seed=args.seed))
        report.update(input=manifest, center=center.tolist(), chemistry=protein.provenance, stage='apo_preparation')
        prior = PhysicalPrior(protein, anchor_k=args.anchor_k)
        report['preparation'] = prior.diagnostics
        save(args.out/'report.json', report)
        prior.prepare(max_iterations=args.max_iterations, max_evaluations=args.max_evaluations,
            stationarity_tolerance=args.stationarity_tolerance, max_heavy_displacement=args.max_heavy_displacement,
            preconditioner=args.preconditioner, polish=args.polish,
            progress=lambda _: save(args.out/'report.json', report))
        report['stage'] = 'mode_directions'
        save(args.out/'report.json', report)
        proposal = build_torsion_modes(prior.prepared_kinematics, n_modes=args.modes,
                                      jacobian_backend='chunked', jvp_chunk_size=8)
        report['proposal_diagnostics'] = proposal.diagnostics
        report['stage'] = 'physical_stiffness'
        save(args.out/'report.json', report)
        model = build_physical_modes(prior, proposal.angle_basis, fd_step=args.fd_step)
        report['mode_diagnostics'] = model.diagnostics
        generator = torch.Generator(device='cpu').manual_seed(args.seed)
        probe = torch.randn(model.n_modes, generator=generator, dtype=torch.float64)
        probe = .01 * probe/probe.norm()
        variable = probe.clone().requires_grad_(True)
        gradient, = torch.autograd.grad(model.deformation_energy(variable), variable)
        step = 1e-5
        numerical = []
        for i in range(model.n_modes):
            offset = torch.zeros_like(probe); offset[i] = step
            numerical.append(float((model.deformation_energy(probe+offset)-model.deformation_energy(probe-offset))/(2*step)))
        numeric = torch.tensor(numerical, dtype=torch.float64)
        max_error = float((gradient-numeric).abs().max())
        relative = float((gradient-numeric).norm())/max(float(gradient.norm()), 1.)
        report['probe'] = dict(amplitudes=probe.tolist(), true_strain=float(model.deformation_energy(probe)),
            quadratic_strain=float(.5*probe @ model.stiffness @ probe), gradient=gradient.tolist(),
            finite_difference_gradient=numerical, gradient_maximum_abs_error=max_error,
            gradient_relative_error=relative, finite_difference_step=step)
        if relative > 1e-4 or max_error > 1e-2:
            raise ValueError('Physical nonlinear mode gradient failed finite-difference verification')
        np.savez(args.out/'arrays.npz', theta0=prior.theta0.numpy(), reference=prior.reference.numpy(),
            angle_basis=model.angle_basis.numpy(), basis=model.basis.numpy(), stiffness=model.stiffness.numpy())
        report['arrays_sha256'] = digest(args.out/'arrays.npz')
        report.update(status='passed', stage='complete')
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        if prior is not None:
            report['openmm_evaluations'] = prior._context.evaluations
            prior.close()
        report['wall_seconds'] = time.monotonic()-started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['process_peak_rss_bytes'] = int(rss if sys.platform=='darwin' else 1024*rss)
        save(args.out/'report.json', report)
    print(json.dumps({key:report[key] for key in ['status','stage','wall_seconds','error'] if key in report}, indent=2))
    return 0 if report['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
