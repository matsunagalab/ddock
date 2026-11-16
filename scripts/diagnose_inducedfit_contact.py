#!/usr/bin/env python3
"""Native-free single-atom contact curves in continuous cylindrical space.

Compare fixed shape energies 3*CC-SS and 3*CC-(CS+SC), using exactly the
current field Gaussian and C2 taper.  This tests radial contact consistency;
it is not a binding benchmark, a force-field fit, or a parameter search.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

from scipy.optimize import minimize_scalar
import torch

import zdock.inducedfit.fields as fields_module
from zdock.inducedfit.fields import FieldParameters, _gaussian


# Prespecified radii in angstrom; no parameter is selected from the results.
PAIRS = (('C-C', 1.70, 1.70), ('C-O', 1.70, 1.52),
         ('O-O', 1.52, 1.52), ('H-O', 1.20, 1.52))
MODELS = ('core_core_minus_shell_shell', 'core_core_minus_core_shell')


class CylindricalPair:
    """Integrate 2*pi*rho*d(rho)*dz with both atoms on the z axis.

    The integration rectangle contains the receptor's full compact sphere;
    every product is exactly zero outside that sphere, for any separation.
    All outer rectangle endpoints are outside support, so the ordinary sum
    equals the composite trapezoidal rule (rho=0 also has zero measure).
    """

    def __init__(self, radius_r, radius_l, spacing, parameters):
        self.radius_r = radius_r
        self.radius_l = radius_l
        self.spacing = spacing
        self.parameters = parameters
        self.factor = math.sqrt(2. * math.log(2.))
        support_r = (radius_r + parameters.surface_width) / self.factor * parameters.occupancy_cutoff_sigma
        support_l = (radius_l + parameters.surface_width) / self.factor * parameters.occupancy_cutoff_sigma
        self.no_overlap_distance = support_r + support_l
        extent = math.ceil(support_r / spacing) + 1
        self.z = torch.arange(-extent, extent + 1, dtype=torch.float64)[:, None] * spacing
        rho = torch.arange(extent + 1, dtype=torch.float64)[None, :] * spacing
        self.rho2 = rho.square()
        core, shell = self.atom_fields(self.z.square() + self.rho2, radius_r)
        measure = 2. * math.pi * rho * spacing**2 / parameters.reference_volume
        self.weighted_core = (core * measure).flatten()
        self.weighted_shell = (shell * measure).flatten()
        self.n_points = core.numel()
        self.cache = {}

    def atom_fields(self, squared_distance, radius):
        p = self.parameters
        gaussian = _gaussian(squared_distance, radius / self.factor,
                             p.occupancy_switch_start, p.occupancy_cutoff_sigma)
        expanded_gaussian = _gaussian(squared_distance, (radius + p.surface_width) / self.factor,
                                      p.occupancy_switch_start, p.occupancy_cutoff_sigma)
        core = -torch.expm1(-gaussian)
        return core, -torch.expm1(-expanded_gaussian) - core

    @torch.inference_mode()
    def evaluate(self, distance):
        distance = float(distance)
        if not math.isfinite(distance) or distance < 0:
            raise ValueError('separation must be finite and nonnegative')
        if distance not in self.cache:
            core, shell = self.atom_fields((self.z - distance).square() + self.rho2, self.radius_l)
            core, shell = core.flatten(), shell.flatten()
            cc = float(torch.dot(self.weighted_core, core))
            ss = float(torch.dot(self.weighted_shell, shell))
            cs = float(torch.dot(self.weighted_core, shell) + torch.dot(self.weighted_shell, core))
            values = (3. * cc - ss, 3. * cc - cs)
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError('nonfinite quadrature energy')
            self.cache[distance] = (*values, cc, ss, cs)
        return self.cache[distance]


def continuous_minimum(quadrature, distances, curve, model_index):
    """Refine all strict coarse local minima, retaining both endpoints too."""
    options = [(curve[0][model_index], distances[0]),
               (curve[-1][model_index], distances[-1])]
    optimizations = []
    for i in range(1, len(distances) - 1):
        left, center, right = (curve[j][model_index] for j in (i - 1, i, i + 1))
        if center <= left and center <= right and (center < left or center < right):
            result = minimize_scalar(
                lambda distance: quadrature.evaluate(distance)[model_index],
                bounds=(distances[i - 1], distances[i + 1]), method='bounded',
                options={'xatol': 1e-5, 'maxiter': 100},
            )
            optimizations.append({'success': bool(result.success), 'nfev': result.nfev,
                                  'message': str(result.message)})
            if not result.success:
                raise RuntimeError(f'contact minimization failed: {result.message}')
            options.append((float(result.fun), float(result.x)))
    energy, distance = min(options)
    return distance, energy, optimizations


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='new compact summary CSV')
    parser.add_argument('--curves', type=Path, required=True, help='new full curve CSV')
    parser.add_argument('--metadata', type=Path, required=True, help='new provenance/cost JSON')
    parser.add_argument('--spacing', type=float, nargs='+', default=[.05, .025])
    parser.add_argument('--distance-step', type=float, default=.25)
    args = parser.parse_args()
    if not all(math.isfinite(h) and h > 0 for h in [*args.spacing, args.distance_step]):
        parser.error('quadrature and distance spacings must be positive finite values')
    if len(set(args.spacing)) != len(args.spacing):
        parser.error('quadrature spacings must be unique')
    if len({args.out.resolve(), args.curves.resolve(), args.metadata.resolve()}) != 3:
        parser.error('output paths must differ')
    if any(path.exists() for path in (args.out, args.curves, args.metadata)):
        parser.error('existing outputs are never overwritten')
    torch.set_num_threads(1)
    parameters = FieldParameters(surface_width=.5, repulsion_weight=3., charge_weight=0.)
    parameters.validate()
    started = time.monotonic()
    sources = [Path(__file__).resolve(), Path(fields_module.__file__).resolve()]
    metadata = {
        'status': 'running', 'experiment': 'single_atom_radial_contact_consistency_v1',
        'purpose': 'native-free physical consistency, no accuracy or coefficient selection',
        'native_or_dockq_access': False, 'seed': None, 'charges': 0.,
        'coefficients': {'core_repulsion': 3., 'attraction': 1.},
        'energy_units': 'effective epsilon; integral/reference_volume with angstrom coordinates',
        'formulas': {'old': '3*integral(C_R*C_L)-integral(S_R*S_L)',
                     'candidate': '3*integral(C_R*C_L)-integral(C_R*S_L+S_R*C_L)'},
        'parameters': asdict(parameters), 'pairs': PAIRS, 'spacings': args.spacing,
        'distance_step': args.distance_step, 'minimum_xatol': 1e-5,
        'quadrature': 'composite trapezoidal, cylindrical 2*pi*rho, full receptor support',
        'device': 'cpu', 'dtype': 'float64', 'threads': 1,
        'platform': platform.platform(), 'processor': platform.processor(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'command': sys.argv,
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        'runs': [], 'limitations': [
            'Single-atom radial functions do not test molecular occupancy saturation or docking accuracy.',
            'Radius sums are a geometric comparator, not fitted equilibrium bond/contact lengths.',
            'No discrete 3D voxel-grid/FFT error or electrostatics is measured.',
            'Bounded minimization refines coarse local minima; convergence is assessed across spacings.',
        ],
    }
    summary, curves, results = [], [], {}
    try:
        for pair, radius_r, radius_l in PAIRS:
            for spacing in args.spacing:
                run_started = time.monotonic()
                quadrature = CylindricalPair(radius_r, radius_l, spacing, parameters)
                count = math.ceil(quadrature.no_overlap_distance / args.distance_step)
                distances = [i * args.distance_step for i in range(count + 1)]
                curve = [quadrature.evaluate(distance) for distance in distances]
                for distance, values in zip(distances, curve):
                    curves.append(dict(pair=pair, spacing_A=spacing, distance_A=distance,
                        old_energy=values[0], candidate_energy=values[1],
                        core_core_integral=values[2], shell_shell_integral=values[3],
                        cross_core_shell_integral=values[4]))
                results[pair, spacing] = curve
                optimization_records = {}
                for index, model in enumerate(MODELS):
                    distance, energy, details = continuous_minimum(quadrature, distances, curve, index)
                    optimization_records[model] = details
                    summary.append(dict(pair=pair, radius_R_A=radius_r, radius_L_A=radius_l,
                        radius_sum_A=radius_r + radius_l, spacing_A=spacing, model=model,
                        minimum_distance_A=distance, offset_from_radius_sum_A=distance-radius_r-radius_l,
                        minimum_energy=energy, energy_at_radius_sum=quadrature.evaluate(radius_r+radius_l)[index],
                        energy_at_zero=curve[0][index], energy_at_no_overlap=curve[-1][index],
                        minimum_at_boundary=distance in (distances[0], distances[-1]),
                        quadrature_points=quadrature.n_points,
                        max_curve_difference_from_finest=0., minimum_distance_difference_from_finest_A=0.))
                record = {'pair': pair, 'spacing': spacing, 'quadrature_points': quadrature.n_points,
                          'distinct_distance_evaluations': len(quadrature.cache),
                          'seconds': time.monotonic()-run_started, 'optimizations': optimization_records}
                metadata['runs'].append(record)
                print(json.dumps({key: record[key] for key in ('pair', 'spacing', 'seconds')}) , flush=True)
        finest = min(args.spacing)
        for row in summary:
            index = MODELS.index(row['model'])
            fine_row = next(item for item in summary if item['pair'] == row['pair']
                            and item['spacing_A'] == finest and item['model'] == row['model'])
            row['minimum_distance_difference_from_finest_A'] = row['minimum_distance_A'] - fine_row['minimum_distance_A']
            row['max_curve_difference_from_finest'] = max(abs(a[index]-b[index]) for a, b in zip(
                results[row['pair'], row['spacing_A']], results[row['pair'], finest]))
        write_csv(args.out, summary)
        write_csv(args.curves, curves)
        metadata.update(status='completed', summary=str(args.out), curves=str(args.curves))
    except Exception as error:
        metadata.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        metadata['wall_seconds'] = time.monotonic()-started
        usage = resource.getrusage(resource.RUSAGE_SELF)
        metadata['process_peak_rss_bytes'] = usage.ru_maxrss if sys.platform == 'darwin' else usage.ru_maxrss*1024
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': metadata['status'], 'wall_seconds': metadata['wall_seconds'],
                      'summary': str(args.out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
