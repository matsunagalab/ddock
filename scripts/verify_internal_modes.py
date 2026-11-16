#!/usr/bin/env python3
"""Verify internal-coordinate modes on a single experimental monomer.

No complex/native pose, docking success label or force-field charge is used.
Finite amplitude norm is deliberately compared with actual CA RMSD: matching
the infinitesimal metric does not guarantee an equally small finite motion.
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
import time

import torch

from zdock.inducedfit.kinematics import TorsionKinematics
from zdock.inducedfit.modes import build_torsion_modes
from zdock.inducedfit.pdb_input import read_pdb_monomer, validate_torsion_input, write_pdb_monomer


def topology_indices(monomer):
    residues = {}
    for i, key in enumerate(monomer.atom_keys):
        residues.setdefault((key.chain, key.residue_number, key.insertion_code), {})[key.atom_name] = i
    rows = [residues[key] for key in sorted(residues)]
    bonds, angles, omegas = [], [], []
    for row in rows:
        bonds.extend([(row['N'], row['CA']), (row['CA'], row['C']), (row['C'], row['O'])])
        angles.extend([(row['N'], row['CA'], row['C']), (row['CA'], row['C'], row['O'])])
    for row, following in zip(rows, rows[1:]):
        bonds.append((row['C'], following['N']))
        angles.extend([(row['CA'], row['C'], following['N']), (row['O'], row['C'], following['N']),
                       (row['C'], following['N'], following['CA'])])
        omegas.append((row['CA'], row['C'], following['N'], following['CA']))
    return [torch.tensor(v, dtype=torch.long, device=monomer.xyz.device) for v in (bonds, angles, omegas)]


def geometry(xyz, indices):
    bonds, angles, omegas = indices
    lengths = torch.linalg.vector_norm(xyz[bonds[:, 0]] - xyz[bonds[:, 1]], dim=-1)
    left, right = xyz[angles[:, 0]] - xyz[angles[:, 1]], xyz[angles[:, 2]] - xyz[angles[:, 1]]
    cosine = (left * right).sum(-1) / (left.norm(dim=-1) * right.norm(dim=-1))
    bond_angles = cosine.clamp(-1., 1.).acos()
    p0, p1, p2, p3 = [xyz[omegas[:, k]] for k in range(4)]
    axis = p2 - p1
    axis = axis / axis.norm(dim=-1, keepdim=True)
    v, w = p0 - p1, p3 - p2
    v = v - (v * axis).sum(-1, keepdim=True) * axis
    w = w - (w * axis).sum(-1, keepdim=True) * axis
    omega = torch.atan2((torch.linalg.cross(axis, v, dim=-1) * w).sum(-1), (v * w).sum(-1))
    return lengths, bond_angles, omega


def run(args):
    prepared = read_pdb_monomer(args.pdb, chain=args.chain, charge_policy="zero_diagnostic")
    validate_torsion_input(prepared)
    inputs = prepared.monomer
    started = time.perf_counter()
    kinematics = TorsionKinematics.from_monomer(inputs)
    model = build_torsion_modes(kinematics, n_modes=args.modes, cutoff=12., gamma=1.,
                                relative_regularization=1e-4, fit_relative_damping=args.fit_relative_damping)
    build_seconds = time.perf_counter() - started
    indices = topology_indices(inputs)
    reference_geometry = geometry(inputs.xyz, indices)
    zero = inputs.xyz.new_zeros(model.n_modes)
    reference_gradient = torch.func.grad(model.deformation_energy)(zero)
    prior_gradient_norm = float(reference_gradient.norm())
    if not math.isfinite(prior_gradient_norm):
        raise FloatingPointError("nonfinite reference prior gradient")
    records = []
    for seed in args.seeds:
        generator = torch.Generator().manual_seed(seed)
        direction = torch.randn(model.n_modes, generator=generator, dtype=inputs.xyz.dtype)
        direction /= direction.norm()
        linear_direction = torch.einsum("m,mnc->nc", direction, model.basis)
        h = 1e-5
        fd = (model.coordinates(h * direction) - model.coordinates(-h * direction)) / (2 * h)
        relative_jvp_error = float((fd - linear_direction).norm() / linear_direction.norm())
        for amplitude in args.amplitudes:
            a = direction * amplitude
            xyz = model.coordinates(a)
            measured = geometry(xyz, indices)
            errors = [float((measured[k] - reference_geometry[k]).abs().max()) for k in (0, 1)]
            omega_delta = measured[2] - reference_geometry[2]
            omega_error = float(torch.atan2(omega_delta.sin(), omega_delta.cos()).abs().max())
            displacement = xyz - inputs.xyz
            actual_rmsd = float(displacement[model.ca_indices].square().sum(-1).mean().sqrt())
            maximum_atom = float(displacement.norm(dim=-1).max())
            nonlinear_prior = float(model.deformation_energy(a))
            harmonic_prior = float(.5 * a @ model.stiffness @ a)
            row = {
                "seed": seed, "amplitude_norm": amplitude, "amplitudes": a.tolist(),
                "actual_CA_RMSD": actual_rmsd, "maximum_atom_displacement": maximum_atom,
                "maximum_torsion_increment": float((model.angle_basis @ a).abs().max()),
                "bond_length_max_error": errors[0], "bond_angle_max_error_radians": errors[1],
                "omega_max_error_radians": omega_error, "basis_fd_relative_error": relative_jvp_error,
                "nonlinear_prior": nonlinear_prior, "harmonic_prior": harmonic_prior,
                "prior_absolute_approximation_error": abs(nonlinear_prior - harmonic_prior),
                "actual_RMSD_exceeds_amplitude": actual_rmsd > amplitude + 1e-10,
            }
            numeric = [v for v in row.values() if isinstance(v, (float, int))]
            if not all(math.isfinite(v) for v in numeric):
                raise FloatingPointError("nonfinite internal-coordinate diagnostic")
            row["geometry_pass"] = max(errors + [omega_error]) < 1e-9 and relative_jvp_error < 1e-5
            records.append(row)
            if seed == args.seeds[0] and amplitude == args.amplitudes[0]:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                write_pdb_monomer(args.out.with_suffix(".pdb"), prepared, xyz)
    return {
        "status": "passed" if all(r["geometry_pass"] for r in records) and prior_gradient_norm < 1e-8 else "numerical_failure",
        "finite_response_summary": {
            "scope": "diagnostics only; amplitude norm is not a finite-displacement guarantee",
            "actual_RMSD_exceeds_amplitude_count": sum(r["actual_RMSD_exceeds_amplitude"] for r in records),
            "maximum_atom_displacement": max(r["maximum_atom_displacement"] for r in records),
            "maximum_torsion_increment": max(r["maximum_torsion_increment"] for r in records),
        },
        "input_sha256": prepared.source_sha256, "input_provenance": prepared.provenance,
        "n_atoms": len(inputs.xyz), "n_CA": len(model.ca_indices), "n_torsions": kinematics.n_torsions,
        "n_modes": model.n_modes, "build_seconds": build_seconds,
        "mode_diagnostics": model.diagnostics, "stiffness": model.stiffness.tolist(),
        "reference_prior_gradient_norm": prior_gradient_norm, "trials": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--fit-relative-damping", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[.1, .5])
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1 or args.modes < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("positive threads/modes and unique seeds are required")
    if any(not math.isfinite(a) or a <= 0 for a in args.amplitudes):
        parser.error("amplitudes must be finite and positive")
    if not math.isfinite(args.fit_relative_damping) or args.fit_relative_damping < 0:
        parser.error("fit-relative-damping must be finite and nonnegative")
    torch.set_num_threads(args.threads)
    root = Path(__file__).resolve().parents[1]
    paths = list((root / 'src/zdock/inducedfit').glob('*.py')) + [Path(__file__).resolve()]
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    report = {
        "experiment": "single_monomer_internal_coordinate_verification_v1",
        "scope": "geometry and local prior only; no physical charges, complex labels or docking benchmark",
        "command": sys.argv, "commit": commit,
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "device": "cpu", "dtype": "float64", "threads": torch.get_num_threads(),
        "parameters": {"modes": args.modes, "fit_relative_damping": args.fit_relative_damping,
                       "seeds": args.seeds, "amplitudes": args.amplitudes,
                       "ANM_cutoff": 12., "gamma": 1., "relative_regularization": 1e-4},
    }
    try:
        report.update(run(args))
    except Exception as error:
        report.update(status="exception", error_type=type(error).__name__, error=str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ("status", "n_atoms", "n_CA", "n_torsions", "n_modes", "error") if k in report}, indent=2))
    print(f"Saved {args.out}")
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
