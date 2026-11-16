#!/usr/bin/env python3
"""Reproduce numerical checks of the induced-fit FFT prototype on synthetic atoms.

This is not a protein docking benchmark. It uses supplied Cartesian modes and
effective charges, and checks a proposed deformation, not a fully relaxed pose.
Every requested seed is retained, including exceptions and null/worse steps.
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

from zdock.geom import rotate
from zdock.inducedfit.fields import FieldParameters, build_fields, physical_energy
from zdock.inducedfit.prepare import input_manifest_record
from zdock.inducedfit.search import prepare_response_receptor, search_response
from zdock.inducedfit.types import AtomKey, DockingInput, GridSpec, MonomerInput


def synthetic_system(seed: int, dtype: torch.dtype):
    generator = torch.Generator().manual_seed(seed)

    def monomer(chain, positions, charges):
        xyz = torch.tensor(positions, dtype=dtype)
        xyz = xyz + .03 * torch.randn(xyz.shape, generator=generator, dtype=dtype)
        return MonomerInput(
            xyz, torch.full((len(xyz),), .6, dtype=dtype), torch.tensor(charges, dtype=dtype),
            tuple(AtomKey(chain, i + 1, "", "SYN", "X") for i in range(len(xyz))),
        )

    receptor = monomer("R", [[-1, 0, 0], [1, 0, 0], [0, 1, 0]], [.5, -.3, .2])
    ligand = monomer("L", [[-.5, 0, 0], [.5, 0, 0]], [-.3, .2])
    basis = torch.tensor([
        [[-1., 0, 0], [1., 0, 0], [0., 0, 0]],
        [[0., -1., 0], [0., -1., 0], [0., 2., 0]],
    ], dtype=dtype)
    basis = basis / torch.sqrt(basis.square().sum((1, 2)) / len(receptor.xyz))[:, None, None]
    stiffness = torch.tensor([[4., .3], [.3, 7.]], dtype=dtype)
    quaternions = torch.tensor([[0., 0., 0., 1.], [0., 0., 2**-.5, 2**-.5]], dtype=dtype)
    return DockingInput(f"synthetic_{seed}", receptor, ligand), basis, stiffness, quaternions


def verify_seed(seed, grid, parameters, radius):
    started = time.perf_counter()
    inputs, basis, stiffness, quaternions = synthetic_system(seed, torch.float64)
    receptor, ligand = inputs.receptor, inputs.ligand
    prepared = prepare_response_receptor(receptor, grid, parameters, basis, stiffness)
    prep_seconds = time.perf_counter() - started
    started_search = time.perf_counter()
    candidates = search_response(
        prepared, ligand, quaternions, radius=radius, top_k=3, peaks_per_rotation=2,
    )
    search_seconds = time.perf_counter() - started_search
    details = []
    for candidate in candidates:
        lig_xyz = torch.stack(rotate(*ligand.xyz.unbind(-1), quaternions[candidate.rotation_index]), -1)
        lig_xyz = lig_xyz + candidate.translation
        lig_fields = build_fields(lig_xyz, ligand.radii, ligand.charges, grid, parameters)

        def actual_energy(rec_xyz):
            fields = build_fields(rec_xyz, receptor.radii, receptor.charges, grid, parameters)
            # Independent spatial charge convolution rather than an FFT oracle.
            return physical_energy(fields, lig_fields, grid, parameters, direct=True)

        rigid = actual_energy(receptor.xyz)
        step = 1e-4
        fd_gradient = torch.stack([
            (actual_energy(receptor.xyz + step * mode) - actual_energy(receptor.xyz - step * mode)) / (2 * step)
            for mode in basis
        ])
        amplitudes = candidate.amplitudes
        fd_gain = float(fd_gradient @ amplitudes + .5 * amplitudes @ stiffness @ amplitudes)
        displacement = torch.einsum("m,mnc->nc", amplitudes, basis)
        proposed_energy = actual_energy(receptor.xyz + displacement) + candidate.strain
        realized_gain = float(proposed_energy - rigid)
        detail = {
            "rotation_index": candidate.rotation_index,
            "translation": candidate.translation.tolist(),
            "amplitudes": amplitudes.tolist(),
            "rigid_energy_fft": candidate.rigid_energy,
            "rigid_energy_direct": float(rigid),
            "fft_absolute_error": abs(candidate.rigid_energy - float(rigid)),
            "predicted_gain": candidate.predicted_gain,
            "predicted_gain_from_fd_gradient": fd_gain,
            "gain_fd_absolute_error": abs(candidate.predicted_gain - fd_gain),
            "strain": candidate.strain,
            "realized_gain_of_proposed_step": realized_gain,
            "prediction_absolute_error": abs(realized_gain - candidate.predicted_gain),
            "amplitude_norm": float(torch.linalg.vector_norm(amplitudes)),
            "cartesian_rmsd": float(displacement.square().sum(-1).mean().sqrt()),
            "maximum_atom_displacement": float(torch.linalg.vector_norm(displacement, dim=-1).max()),
        }
        if not all(math.isfinite(value) for value in detail.values() if isinstance(value, (float, int))):
            raise FloatingPointError("nonfinite energy or diagnostic in proposed-step verification")
        details.append(detail)
    if not details:
        raise RuntimeError("No valid candidates")
    passed = (
        max(x["fft_absolute_error"] for x in details) < 1e-9
        and max(x["gain_fd_absolute_error"] for x in details) < 1e-6
        and max(x["amplitude_norm"] for x in details) <= radius + 1e-10
    )
    return {
        "seed": seed, "status": "passed" if passed else "numerical_failure",
        "input_manifest": input_manifest_record(inputs, input_kind="synthetic"),
        "inputs": {side: {name: getattr(getattr(inputs, side), name).tolist()
                           for name in ("xyz", "radii", "charges")} for side in ("receptor", "ligand")},
        "basis": basis.tolist(), "stiffness": stiffness.tolist(), "quaternions": quaternions.tolist(),
        "preparation_seconds": prep_seconds, "search_seconds": search_seconds,
        "total_verification_seconds": time.perf_counter() - started, "candidates": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--radius", type=float, default=.1)
    args = parser.parse_args()
    if args.threads < 1 or not 0 <= args.radius <= .5:
        parser.error("threads must be positive; synthetic-prototype radius must be in [0, 0.5]")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    torch.set_num_threads(args.threads)
    grid = GridSpec((17, 15, 19), 1., (-8., -7., -9.))
    parameters = FieldParameters(
        surface_width=.3, charge_sigma=.45, screening_switch_start=2., screening_cutoff=3.,
        repulsion_weight=3., charge_weight=.2,
    )
    root = Path(__file__).resolve().parents[1]
    source_paths = sorted((root / "src/zdock/inducedfit").glob("*.py")) + [
        root / "src/zdock/geom.py", Path(__file__).resolve(),
    ]
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_paths}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
    results = []
    for seed in args.seeds:
        try:
            results.append(verify_seed(seed, grid, parameters, args.radius))
        except Exception as error:
            results.append({"seed": seed, "status": "exception", "error_type": type(error).__name__, "error": str(error)})
    poses = [candidate for result in results for candidate in result.get("candidates", [])]
    report = {
        "experiment": "synthetic_induced_fit_numerical_verification_v1",
        "scope": "Cartesian point-cloud numerical verification; no protein benchmark, training, or nonlinear relaxation",
        "command": sys.argv, "commit": commit, "source_sha256": source_hashes,
        "platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
        "torch": torch.__version__, "device": "cpu", "dtype": "float64", "threads": torch.get_num_threads(),
        "grid": asdict(grid), "parameters": asdict(parameters), "radius": args.radius, "seeds": args.seeds,
        "systems": results,
        "summary": {
            "requested_systems": len(args.seeds), "passed_systems": sum(x["status"] == "passed" for x in results),
            "checked_candidates": len(poses),
            "max_fft_absolute_error": max((x["fft_absolute_error"] for x in poses), default=None),
            "max_gain_fd_absolute_error": max((x["gain_fd_absolute_error"] for x in poses), default=None),
            "max_prediction_absolute_error": max((x["prediction_absolute_error"] for x in poses), default=None),
            "energy_lowering_steps": sum(x["realized_gain_of_proposed_step"] < 0 for x in poses),
            "nonlowering_steps": sum(x["realized_gain_of_proposed_step"] >= 0 for x in poses),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved {args.out}")
    return 0 if all(result["status"] == "passed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
