#!/usr/bin/env python3
"""Native-free quadratic oracle for a shared refinement line search.

Use the already verified 8x8 physical K as a numerical scale, synthetic
translation/interaction terms, and identical Armijo/project/budget policies.
This tests an optimizer proposal, not docking accuracy or modified physics.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(hessian, linear, constant, stiffness, method, *, iterations=2, budget=24):
    m = len(stiffness)
    z = np.zeros(3+m)
    evaluations = 0
    accepted = []

    def evaluate(values):
        nonlocal evaluations
        evaluations += 1
        return .5*values@hessian@values+linear@values+constant, hessian@values+linear

    def project(values):
        result = values.copy()
        for block, limit in [(slice(0, 3), 4.), (slice(3, None), .5)]:
            norm = np.linalg.norm(result[block])
            if norm > limit:
                result[block] *= limit/norm
        return result

    current, gradient = evaluate(z)
    history = [float(current)]
    for iteration in range(iterations):
        if evaluations >= budget:
            break
        direction = -gradient.copy()
        if method == 'rigid':
            direction[3:] = 0.
        elif method == 'fixed_point_one':
            direction[3:] *= .1
        elif method == 'inverse_K':
            direction[3:] = -np.linalg.solve(stiffness, gradient[3:])
        else:
            raise ValueError(method)
        step = 1.
        improved = False
        while evaluations < budget and step >= 2.**-24:
            trial = project(z+step*direction)
            delta = trial-z
            if np.linalg.norm(delta) < 1e-12:
                break
            proposed, _ = evaluate(trial)
            if proposed <= current+1e-4*(gradient@delta) and proposed < current:
                accepted.append(dict(step=step, translation_step=float(np.linalg.norm(delta[:3])),
                    amplitude_step=float(np.linalg.norm(delta[3:])), energy=float(proposed)))
                z, current, improved = trial, proposed, True
                history.append(float(current))
                break
            step *= .5
        if not improved:
            break
        if iteration+1 < iterations and evaluations < budget:
            current, gradient = evaluate(z)
    return dict(energy=float(current), translation=z[:3].tolist(), amplitude=z[3:].tolist(),
        translation_norm=float(np.linalg.norm(z[:3])), amplitude_norm=float(np.linalg.norm(z[3:])),
        gradient_norm=float(np.linalg.norm(hessian@z+linear)), evaluations=evaluations,
        accepted_steps=accepted, energy_history=history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    producer = json.loads(args.prior_report.read_text())
    archive = args.prior_report.parent/'arrays.npz'
    if producer['status'] != 'passed' or digest(archive) != producer['arrays_sha256']:
        raise ValueError('A passed, checksum-verified physical stiffness artifact is required')
    registration = dict(command=sys.argv, script_sha256=digest(__file__),
        prior_report_sha256=digest(args.prior_report), arrays_sha256=digest(archive),
        native_access=False, physical_parameter_changes=False, production_source_changes=False,
        synthetic_oracle_only=True, seed=0, dtype='float64', iterations=2, evaluations=24,
        armijo=1e-4, translation_bound=4., amplitude_bound=.5, couplings=[0., .3, .5],
        translation_weights=1., current_amplitude_weights=.1,
        proposed_amplitude_direction='-solve(K, total_amplitude_gradient)',
        objective='0.5 ||t-target||^2 + 0.5 a.T K a + b.T a + t.T C a',
        auxiliary_convergence_check=dict(iterations=100, evaluations=500))
    (args.out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    with np.load(archive, allow_pickle=False) as arrays:
        stiffness = arrays['stiffness'].copy()
    eigenvalues, eigenvectors = np.linalg.eigh(stiffness)
    if eigenvalues[0] <= 0:
        raise ValueError('No clipping or regularization: K must already be positive definite')
    root = (eigenvectors*np.sqrt(eigenvalues))@eigenvectors.T
    inverse_root = (eigenvectors/np.sqrt(eigenvalues))@eigenvectors.T
    m = len(stiffness)
    rng = np.random.default_rng(0)
    orthogonal = np.linalg.qr(rng.standard_normal((m, 3)))[0].T
    b = rng.standard_normal(m); b *= 10./np.linalg.norm(b)
    target = np.array([2., -1., .5])
    results = []
    for coupling in registration['couplings']:
        cross = coupling*orthogonal@root
        hessian = np.block([[np.eye(3), cross], [cross.T, stiffness]])
        linear = np.concatenate([-target, b])
        constant = .5*(target@target)
        optimum = -np.linalg.solve(hessian, linear)
        oracle_energy = float(.5*optimum@hessian@optimum+linear@optimum+constant)
        if np.linalg.norm(optimum[:3]) >= 4. or np.linalg.norm(optimum[3:]) >= .5:
            raise ValueError('Oracle leaves fixed bounds')
        row = dict(coupling=coupling, oracle_energy=oracle_energy,
            oracle_translation=optimum[:3].tolist(), oracle_amplitude=optimum[3:].tolist())
        for method in ['rigid', 'fixed_point_one', 'inverse_K']:
            outcome = run(hessian, linear, constant, stiffness, method)
            outcome['gap_to_flexible_oracle'] = outcome['energy']-oracle_energy
            if any(y >= x for x, y in zip(outcome['energy_history'], outcome['energy_history'][1:])):
                raise ValueError('Nonmonotonic accepted oracle step')
            row[method] = outcome
        transforms = {'fixed_point_one':np.diag([1.]*3+[np.sqrt(.1)]*m),
            'inverse_K':np.block([[np.eye(3),np.zeros((3,m))],[np.zeros((m,3)),inverse_root]])}
        row['preconditioned_condition_numbers'] = {}
        for method, transform in transforms.items():
            spectrum = np.linalg.eigvalsh(transform.T@hessian@transform)
            row['preconditioned_condition_numbers'][method] = float(spectrum[-1]/spectrum[0])
        converged = run(hessian, linear, constant, stiffness, 'inverse_K', iterations=100, budget=500)
        row['inverse_K_convergence_check'] = dict(energy_gap=converged['energy']-oracle_energy,
            evaluations=converged['evaluations'], accepted_steps=len(converged['accepted_steps']))
        if abs(converged['energy']-oracle_energy) > 1e-9:
            raise ValueError('K-preconditioned oracle did not reach the analytic optimum')
        results.append(row)
    report = dict(status='passed', registration=registration,
        stiffness_eigenvalues=eigenvalues.tolist(), cases=results)
    (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','cases':len(results)}, indent=2))


if __name__ == '__main__':
    main()
