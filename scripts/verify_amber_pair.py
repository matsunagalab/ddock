#!/usr/bin/env python3
"""Generate a synthetic complete peptide pair and run the real pair CLI.

This is an integration smoke test, not an apo/holo or docking-accuracy
benchmark.  Both inputs are four-residue ALA-GLY-SER-ALA peptides with explicit
heavy-atom geometry, a terminal OXT, and prescribed nonzero backbone bends.
The child CLI performs actual Amber hydrogen/charge preparation, torsion
modes, FFT selection, constrained refinement and paired heavy-atom exports.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from zdock.inducedfit.pdb_input import read_pdb_monomer
from zdock.inducedfit.types import AtomKey, MonomerInput


def synthetic_peptide(chain: str) -> MonomerInput:
    """A complete, deliberately artificial geometry; no experimental source."""
    sequence = ("ALA", "GLY", "SER", "ALA")
    keys, positions = [], []

    def backbone(index):
        block = index // 2
        if index % 2 == 0:
            n = torch.tensor([5.83 * block, 4.015 * block, 0.], dtype=torch.float64)
            ca = n + n.new_tensor([1.45, 0., 0.])
            c = ca + ca.new_tensor([.55, 1.42, 0.])
        else:
            n = torch.tensor([5.83 * block + 3.32, 4.015 * block + 1.42, 0.], dtype=torch.float64)
            ca = n + n.new_tensor([.53, 1.35, 0.])
            c = ca + ca.new_tensor([1.52, 0., 0.])
        return n, ca, c

    for index, name in enumerate(sequence):
        n, ca, c = backbone(index)
        following_n, _, _ = backbone(index + 1)
        oxygen_direction = (ca - c) / (ca - c).norm() + (following_n - c) / (following_n - c).norm()
        oxygen = c - 1.23 * oxygen_direction / oxygen_direction.norm()
        atoms = {"N": n, "CA": ca, "C": c, "O": oxygen}
        if name != "GLY":
            atoms["CB"] = ca + ca.new_tensor([0., 0., 1.53])
        if name == "SER":
            atoms["OG"] = atoms["CB"] + ca.new_tensor([0., 0., 1.42])
        if index == len(sequence) - 1:
            # Explicit synthetic terminal atom, not missing-atom repair of a
            # supposedly experimental structure.  Complete carboxylate plane.
            terminal_direction = -((ca - c) / (ca - c).norm() + (oxygen - c) / (oxygen - c).norm())
            atoms["OXT"] = c + 1.25 * terminal_direction / terminal_direction.norm()
        for atom, xyz in atoms.items():
            keys.append(AtomKey(chain, index + 1, "", name, atom))
            positions.append(xyz)
    radii = {"C": 1.70, "N": 1.55, "O": 1.52}
    xyz = torch.stack(positions)
    return MonomerInput(xyz, xyz.new_tensor([radii[key.atom_name[0]] for key in keys]),
                        xyz.new_zeros(len(keys)), tuple(keys))


def write_generated_pdb(path: Path, monomer: MonomerInput) -> None:
    lines = ["REMARK 999 SYNTHETIC INTEGRATION INPUT; NOT AN EXPERIMENTAL APO STRUCTURE"]
    for serial, (key, xyz) in enumerate(zip(monomer.atom_keys, monomer.xyz.tolist()), 1):
        x, y, z = xyz
        lines.append(f"ATOM  {serial:5d} {key.atom_name:^4s} {key.residue_name:3s} {key.chain}{key.residue_number:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}{1.:6.2f}{20.:6.2f}          {key.atom_name[0]:>2s}  ")
    path.write_text("\n".join(lines + ["TER", "END"]) + "\n", encoding="ascii")


def generate_inputs(directory: Path) -> tuple[Path, dict]:
    directory.mkdir(parents=True, exist_ok=False)
    angles = {
        "receptor": [.30, -.40, .20, -.25, .35, -.15],
        "ligand": [-.25, .30, -.35, .20, -.20, .25],
    }
    manifest = {"format_version": 1, "system_id": "synthetic_agsa_amber_smoke_v1",
                "input_kind": "synthetic", "split": "development"}
    generated = {}
    for side, chain in (("receptor", "A"), ("ligand", "B")):
        source = synthetic_peptide(chain)
        kinematics = TorsionKinematics.from_monomer(source)
        delta = source.xyz.new_tensor(angles[side])
        xyz = kinematics.coordinates(delta)
        if side == "ligand":
            angle = .7
            rotation = source.xyz.new_tensor([[math.cos(angle), -math.sin(angle), 0.],
                                              [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
            xyz = xyz @ rotation.T + xyz.new_tensor([4., -3., 7.])
        deformed = replace(source, xyz=xyz)
        target = directory / f"{side}.pdb"
        write_generated_pdb(target, deformed)
        # Check the actual rounded file consumed by the child, not only the
        # unrounded generator coordinates.
        parsed = read_pdb_monomer(target, chain=chain, charge_policy="zero_diagnostic")
        parsed_map = TorsionKinematics.from_monomer(parsed.monomer)
        ca = parsed.monomer.xyz[parsed_map.ca_indices]
        singular_values = torch.linalg.svdvals(ca - ca.mean(0))
        if float(singular_values[-1]) < 1e-4:
            raise ValueError(f"Generated {side} CA geometry remains planar/degenerate")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest[side] = {"pdb": target.name, "chain": chain, "sha256": digest}
        generated[side] = {"sequence": "AGSA", "heavy_atom_count": len(parsed.monomer.xyz),
                            "source": "synthetic coordinates in this script", "terminal_OXT": "explicit synthetic geometry",
                            "torsion_increments_radians": angles[side],
                            "CA_centered_singular_values": singular_values.tolist(),
                            "PDB_rounding_max_abs_error_angstrom": float((parsed.monomer.xyz - xyz).abs().max())}
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path, generated


def check_exports(run_directory: Path, child: dict) -> list[dict]:
    for side in ("receptor", "ligand"):
        preparation = child["preparation"][side]
        chemistry = preparation["chemistry"]
        if preparation["heavy_atoms"] != 21 or preparation["all_atoms"] != 41 or chemistry["n_hydrogens"] != 20:
            raise RuntimeError(f"{side} did not retain the expected complete all-H protein")
        if abs(chemistry["net_charge_e"]) > 1e-10 or abs(chemistry["hydrogen_charge_sum_e"]) < .1:
            raise RuntimeError(f"{side} force-field charges were not exercised correctly")
    checks = []
    for arm in ("R0", "R1", "R2"):
        result = child.get("arms", {}).get(arm, {})
        if result.get("status") != "completed" or result.get("exported_count") != 1:
            raise RuntimeError(f"{arm} did not export exactly one completed candidate")
        candidate = result["candidates"][0]
        paths = {side: run_directory / path for side, path in candidate["files"].items()}
        rec = read_pdb_monomer(paths["receptor"], chain="A", charge_policy="zero_diagnostic").monomer
        lig = read_pdb_monomer(paths["ligand"], chain="B", charge_policy="zero_diagnostic").monomer
        complex_rec = read_pdb_monomer(paths["complex"], chain="R", charge_policy="zero_diagnostic").monomer
        complex_lig = read_pdb_monomer(paths["complex"], chain="L", charge_policy="zero_diagnostic").monomer
        if len(rec.xyz) != 21 or len(lig.xyz) != 21:
            raise RuntimeError("Export lost a heavy atom")
        if not torch.equal(rec.xyz, complex_rec.xyz) or not torch.equal(lig.xyz, complex_lig.xyz):
            raise RuntimeError("Separate and combined export coordinates disagree")
        history = candidate["energy_history"]
        if not all(math.isfinite(v) for v in history) or any(b > a + 1e-10 for a, b in zip(history, history[1:])):
            raise RuntimeError(f"{arm} has nonfinite/increasing accepted energy history")
        if candidate["accepted_steps"] < 1:
            raise RuntimeError(f"{arm} did not exercise an accepted nonlinear refinement step")
        if arm == "R0" and candidate["maximum_atom_displacement"] > 1e-10:
            raise RuntimeError("Rigid control unexpectedly changed receptor geometry")
        if arm != "R0" and candidate["ca_rmsd"] <= 1e-8:
            raise RuntimeError(f"{arm} did not exercise internal-coordinate motion")
        checks.append({"arm": arm, "receptor_heavy_atoms": len(rec.xyz), "ligand_heavy_atoms": len(lig.xyz),
                       "combined_coordinate_roundtrip": "exact",
                       "solver_status": candidate["status"], "accepted_steps": candidate["accepted_steps"],
                       "energy_evaluations": candidate["energy_evaluations"], "initial_energy": history[0],
                       "final_energy": candidate["energy"],
                       "export_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory; inputs/results are never overwritten")
    parser.add_argument("--modes", type=int, choices=(1, 2), default=1)
    parser.add_argument("--checkpoint-fields", action="store_true",
                        help="pass optional local field activation checkpointing to the pair CLI")
    parser.add_argument("--timeout", type=float, default=600., help="child CLI walltime limit in seconds")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    script = Path(__file__).resolve()
    report = {"experiment": "synthetic_complete_Amber_pair_CLI_smoke_v1", "status": "running",
              "scope": "synthetic development integration only; no apo or docking-accuracy claim",
              "command": sys.argv, "platform": platform.platform(), "python": platform.python_version(),
              "torch": torch.__version__, "device": "cpu", "dtype": "float64",
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "generator_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
    started = time.monotonic()
    try:
        manifest, generated = generate_inputs(args.out / "inputs")
        report["generated_inputs"] = generated
        output = args.out / "run"
        command = [sys.executable, str(root / "scripts/run_induced_fit_pair.py"),
                   "--manifest", str(manifest.resolve()), "--out", str(output.resolve()),
                   "--arms", "R0", "R1", "R2", "--rotations", "1", "--top-k", "1",
                   "--modes", str(args.modes), "--iterations", "1", "--max-evaluations", "6", "--spacing", "2"]
        if args.checkpoint_fields:
            command.append("--checkpoint-fields")
        report["child_command"] = command
        report["checkpoint_fields"] = args.checkpoint_fields
        report["child_timeout_seconds"] = args.timeout
        child_started = time.monotonic()
        try:
            child = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=args.timeout)
            stdout, stderr = child.stdout, child.stderr
            report["child_returncode"] = child.returncode
        except subprocess.TimeoutExpired as error:
            stdout, stderr = error.stdout or "", error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            report["child_returncode"] = "timeout"
        report["child_wall_seconds"] = time.monotonic() - child_started
        (args.out / "child.stdout.txt").write_text(stdout)
        (args.out / "child.stderr.txt").write_text(stderr)
        child_report = output / "report.json"
        if child_report.is_file():
            child_result = json.loads(child_report.read_text())
            report["child_status"] = child_result.get("status")
            report["child_report_sha256"] = hashlib.sha256(child_report.read_bytes()).hexdigest()
        else:
            child_result = {}
        if report["child_returncode"] != 0 or child_result.get("status") != "completed":
            raise RuntimeError(f"Pair CLI did not complete: returncode={report['child_returncode']}, status={child_result.get('status')}")
        report["export_checks"] = check_exports(output, child_result)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    report["wall_seconds"] = time.monotonic() - started
    target = args.out / "verification.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "error", "child_status", "child_wall_seconds", "wall_seconds") if key in report}, indent=2))
    print(f"Saved {target}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
