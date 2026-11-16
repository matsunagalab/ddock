#!/usr/bin/env python3
"""Registered input-only, approximate field-response compression experiment.

Independent full8/rank4/rank2 searches precede one separate frozen-union
diagnostic process. This is not a docking-accuracy benchmark: native structures
and accuracy scores are neither arguments nor inputs. No local refinement,
rank tuning, field-weight fitting or successful-case replacement occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback


REPO = Path(__file__).resolve().parents[1]
VARIANTS = {"full8": 8, "rank4": 4, "rank2": 2}
TIMEOUT_SECONDS = 180
MAX_GRID_POINTS = 750_000
CONFIGURATION = {
    "source_modes": 8, "mode_jacobian_backend": "chunked", "jvp_chunk_size": 8,
    "anm_backend": "iterative", "anm_seed": 0, "anm_tolerance": 1e-9,
    "anm_cutoff_angstrom": 12., "anm_gamma": 1., "fit_relative_damping": .001,
    "relative_regularization": .0001, "pH": 7., "hydrogen_seed": 0,
    "compression_oversampling": 2, "compression_seed": 0,
    "rotations": 16, "rotation_seed": 17, "peaks_per_rotation": 4, "top_k": 10,
    "min_peak_distance_angstrom": 3., "radius_angstrom": .5,
    "max_atom_displacement_angstrom": 1.5, "grid_max_translation_margin_angstrom": .5,
    "spacing_angstrom": 2., "min_agreement": .75, "max_verification_trials": 6,
    "threads": 1, "device": "cpu", "dtype": "float64",
    "max_grid_points": MAX_GRID_POINTS, "subprocess_timeout_seconds": TIMEOUT_SECONDS,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(*values):
    digest = hashlib.sha256()
    for value in values:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def pose_key(row):
    """Exact common sampled pose identity, independent of response amplitudes."""
    return (row["rotation_index"], *row["translation"])


def candidate_record(candidate, rank):
    return {"selection_rank": rank, "rotation_index": candidate.rotation_index,
            "translation": candidate.translation.tolist(), "amplitudes": candidate.amplitudes.tolist(),
            "score": candidate.score, "rigid_energy": candidate.rigid_energy,
            "predicted_gain": candidate.predicted_gain, "quadratic_strain": candidate.strain}


def union_records(searches):
    records = {}
    for name in VARIANTS:
        for row in searches[name]["candidates"]:
            key = pose_key(row)
            item = records.setdefault(key, {"rotation_index": row["rotation_index"],
                                            "translation": row["translation"], "selected_by": {}})
            item["selected_by"][name] = {"selection_rank": row["selection_rank"],
                                          "rigid_energy": row["rigid_energy"]}
    return [records[key] for key in sorted(records)]


def overlap_records(searches):
    result = []
    for first, second in itertools.combinations(VARIANTS, 2):
        a = {pose_key(row) for row in searches[first]["candidates"]}
        b = {pose_key(row) for row in searches[second]["candidates"]}
        result.append({"first": first, "second": second, "first_count": len(a), "second_count": len(b),
                       "intersection": len(a & b), "union": len(a | b),
                       "jaccard": len(a & b) / len(a | b) if a or b else None,
                       "intersection_fraction_of_requested_top10": len(a & b) / CONFIGURATION["top_k"]})
    return result


def registered_sources():
    paths = [Path(__file__).resolve(), REPO / "src/zdock/geom.py"]
    # Fingerprint this computation's dependency closure, not unrelated runner,
    # replay, accuracy scorer or documentation edits in the shared checkout.
    modules = ("anm_iterative", "chemistry", "field_compression", "fields", "hydrogen_kinematics",
               "kinematics", "modes", "pdb_input", "pipeline", "prepare", "response", "search",
               "sparse_fields", "types", "verified_response")
    paths += [REPO / "src/zdock/inducedfit" / f"{name}.py" for name in modules]
    return {str(path.relative_to(REPO)): sha256(path) for path in paths}


def check_registration(registration):
    if registration["configuration"] != CONFIGURATION:
        raise ValueError("Registered configuration differs from this script")
    if sha256(registration["manifest_path"]) != registration["input_manifest_sha256"]:
        raise ValueError("Input manifest SHA256 changed after registration")
    for name, digest in registration["source_sha256"].items():
        if sha256(REPO / name) != digest:
            raise ValueError(f"Registered source SHA256 changed: {name}")


def prepare_context(registration, report, checkpoint):
    import torch
    from zdock.inducedfit.chemistry import prepare_amber_charges
    from zdock.inducedfit.fields import FieldParameters
    from zdock.inducedfit.hydrogen_kinematics import HydrogenatedKinematics
    from zdock.inducedfit.modes import build_torsion_modes
    from zdock.inducedfit.pipeline import centered_protein, load_pair_manifest, pair_grid, rotation_sample
    from zdock.inducedfit.prepare import monomer_fingerprint

    check_registration(registration)
    torch.set_num_threads(1)
    report["software"] = {"torch": torch.__version__, "python": platform.python_version()}
    report["stage"] = "chemical_preparation"
    checkpoint()
    started = time.perf_counter()
    manifest, inputs = load_pair_manifest(Path(registration["manifest_path"]))
    proteins = {}
    report["input"] = manifest
    report["preparation"] = {}
    for side in ("receptor", "ligand"):
        prepared = prepare_amber_charges(inputs[side], pH=7., seed=0)
        proteins[side], center = centered_protein(prepared)
        report["preparation"][side] = {"chemistry": prepared.provenance,
            "all_atoms": proteins[side].monomer.n_atoms, "center": center.tolist(),
            "full_monomer_sha256": monomer_fingerprint(proteins[side].monomer)}
        checkpoint()
    report["chemistry_seconds"] = time.perf_counter() - started
    report["stage_peak_rss_bytes"]["after_chemistry"] = peak_rss_bytes()
    report["stage"] = "mode_construction"
    checkpoint()
    started = time.perf_counter()
    base = build_torsion_modes(HydrogenatedKinematics.from_prepared(proteins["receptor"]), n_modes=8,
        cutoff=12., gamma=1., fit_relative_damping=.001, relative_regularization=.0001,
        jacobian_backend="chunked", jvp_chunk_size=8, anm_backend="iterative", anm_seed=0, anm_tolerance=1e-9)
    report["mode_seconds"] = time.perf_counter() - started
    report["mode_diagnostics"] = base.diagnostics
    report["base_model_sha256"] = tensor_hash(base.angle_basis, base.basis, base.stiffness)
    report["stage_peak_rss_bytes"]["after_modes"] = peak_rss_bytes()
    receptor, ligand = proteins["receptor"].monomer, proteins["ligand"].monomer
    parameters = FieldParameters(surface_width=.5, repulsion_weight=3., charge_weight=1.,
                                 atom_chunk_size=32, point_chunk_size=2048, backend="sparse")
    grid = pair_grid(receptor, ligand, parameters, spacing=2., max_atom_displacement=1.5, max_translation=.5)
    coefficient = float(torch.linalg.matrix_norm(base.basis.permute(1, 2, 0), ord=2).max())
    radius = min(.5, 1.5 / coefficient) if coefficient else .5
    if not math.isfinite(radius) or not 0 < radius <= .5:
        raise ValueError("Nonpositive or invalid common baseline radius")
    report.update(parameters=asdict(parameters), grid=asdict(grid), effective_radius=radius,
                  maximum_atom_motion_per_unit_amplitude=coefficient,
                  radius_policy="same radius bounded using full eight-mode basis; never enlarged for a compressed subspace")
    quaternions = rotation_sample(16, 17)
    report["quaternions"] = quaternions.tolist()
    checkpoint()
    if grid.n_points > MAX_GRID_POINTS:
        raise ValueError(f"grid has {grid.n_points} points, exceeds fixed {MAX_GRID_POINTS} cap")
    return receptor, ligand, base, grid, parameters, radius, quaternions


def search_worker(name, registration, report, checkpoint):
    import torch
    from zdock.inducedfit.field_compression import compress_response_fields
    from zdock.inducedfit.search import prepare_response_receptor, search_response

    receptor, ligand, base, grid, parameters, radius, quaternions = prepare_context(registration, report, checkpoint)
    report["stage"] = "field_compression"
    checkpoint()
    started = time.perf_counter()
    if name == "full8":
        model = base
        transform = torch.eye(8, dtype=torch.float64)
        report["compression_diagnostics"] = {"method": "none", "theoretical_sketch_bytes": 0,
                                             "theoretical_projected_matrix_bytes": 0}
    else:
        model = compress_response_fields(receptor, base, grid, parameters,
                                          rank=VARIANTS[name], oversampling=2, seed=0)
        transform = model.transform
        report["compression_diagnostics"] = model.diagnostics
    report["compression_seconds"] = time.perf_counter() - started
    report["stage_peak_rss_bytes"]["after_compression"] = peak_rss_bytes()
    report.update(transform=transform.tolist(), stiffness=model.stiffness.tolist(),
                  model_basis_sha256=tensor_hash(model.basis), quaternions=quaternions.tolist())
    report["stage"] = "response_spectra"
    checkpoint()
    started = time.perf_counter()
    prepared = prepare_response_receptor(receptor, grid, parameters, model.basis, model.stiffness)
    report["spectra_preparation_seconds"] = time.perf_counter() - started
    derivative_bytes = sum(value.numel() * value.element_size() for value in prepared.derivative_spectra)
    rigid_bytes = prepared.rigid_spectrum.numel() * prepared.rigid_spectrum.element_size()
    report["retained_fft_bytes"] = {"rigid": rigid_bytes, "derivatives": derivative_bytes,
                                     "total": rigid_bytes + derivative_bytes}
    report["stage_peak_rss_bytes"]["after_spectra"] = peak_rss_bytes()
    report["stage"] = "response_search"
    checkpoint()
    started = time.perf_counter()
    # The common radius is already bounded by the full basis. Do not apply a
    # variant-specific atom bound that would change this registered comparison.
    candidates = search_response(prepared, ligand, quaternions, radius=radius, top_k=10,
                                  peaks_per_rotation=4, min_peak_distance=3., selection="response")
    report["search_seconds"] = time.perf_counter() - started
    report["candidates"] = [candidate_record(candidate, rank) for rank, candidate in enumerate(candidates, 1)]
    report["returned_candidates"] = len(candidates)
    report["candidate_shortfall"] = max(0, 10 - len(candidates))
    report["stage_peak_rss_bytes"]["after_search"] = peak_rss_bytes()
    report.update(status="completed", stage="complete")


def assert_same_inputs(searches, reference):
    for name, search in searches.items():
        if search["status"] != "completed":
            raise ValueError(f"{name} search did not complete")
        for key in ("base_model_sha256", "grid", "parameters", "effective_radius", "quaternions"):
            # GridSpec dataclass tuples become arrays in the saved JSON.
            # Compare their serialization semantics, not tuple/list identity.
            if json.dumps(search[key], sort_keys=True, allow_nan=False) != json.dumps(reference[key], sort_keys=True, allow_nan=False):
                raise ValueError(f"{name} differs from common {key}")
        for side in ("receptor", "ligand"):
            if search["preparation"][side]["full_monomer_sha256"] != reference["preparation"][side]["full_monomer_sha256"]:
                raise ValueError(f"{name} regenerated {side} differs from search input")


def union_worker(root, registration, report, checkpoint):
    import torch
    from zdock.geom import rotate
    from zdock.inducedfit.field_compression import CompressedModeModel
    from zdock.inducedfit.fields import build_fields, jvp_receptor_fields, ligand_channels, receptor_channels
    from zdock.inducedfit.response import QuadraticResponse
    from zdock.inducedfit.search import ResponseCandidate
    from zdock.inducedfit.verified_response import NonlinearResponseVerifier

    searches = {name: json.loads((root / name / "report.json").read_text()) for name in VARIANTS}
    receptor, ligand, base, grid, parameters, radius, quaternions = prepare_context(registration, report, checkpoint)
    assert_same_inputs(searches, report)
    report["search_report_sha256"] = {name: sha256(root / name / "report.json") for name in VARIANTS}
    union = union_records(searches)
    if not union:
        raise ValueError("All registered searches returned an empty union")
    report.update(union_poses=union, planned_pose_count=len(union), planned_variant_pose_evaluations=3 * len(union),
                  top10_overlaps=overlap_records(searches), rows=[], stage="union_derivative_fields")
    checkpoint()
    models, transforms, solvers, derivatives, verifiers = {}, {}, {}, {}, {}
    start = time.perf_counter()
    for name in VARIANTS:
        transform = receptor.xyz.new_tensor(searches[name]["transform"])
        identity = torch.eye(VARIANTS[name], dtype=receptor.xyz.dtype)
        if transform.shape != (8, VARIANTS[name]) or not torch.allclose(transform.T @ transform, identity, atol=1e-7, rtol=1e-7):
            raise ValueError(f"{name} transform changed the original amplitude-ball metric")
        transforms[name] = transform
        if name == "full8":
            model = base
        else:
            basis = torch.einsum("mr,mnc->rnc", transform, base.basis)
            stiffness = transform.T @ base.stiffness @ transform
            stiffness = .5 * (stiffness + stiffness.T)
            model = CompressedModeModel(base, transform, basis, stiffness, searches[name]["compression_diagnostics"])
        if tensor_hash(model.basis) != searches[name]["model_basis_sha256"]:
            raise ValueError(f"Reconstructed {name} basis differs from search")
        ca_basis = model.basis[:, model.ca_indices].reshape(model.n_modes, -1)
        if not torch.allclose(ca_basis @ ca_basis.T / len(model.ca_indices), identity, atol=1e-7, rtol=1e-7):
            raise ValueError(f"{name} reconstructed CA displacement metric is not identity")
        torch.testing.assert_close(model.stiffness, receptor.xyz.new_tensor(searches[name]["stiffness"]), atol=1e-11, rtol=1e-11)
        models[name], solvers[name] = model, QuadraticResponse(model.stiffness)
        # These are actual composite-direction JVPs, not algebraically defined
        # projections of full g, so projection parity is independently tested.
        derivatives[name] = []
        for direction in model.basis:
            _, tangent = jvp_receptor_fields(receptor.xyz, receptor.radii, receptor.charges, direction, grid, parameters)
            derivatives[name].append(tangent.detach())
        verifiers[name] = NonlinearResponseVerifier(receptor, ligand, model, grid, parameters,
            radius=radius, max_atom_displacement=1.5, min_agreement=.75, max_trials=6)
        report["prepared_union_variants"] = list(models)
        checkpoint()
    report["union_field_preparation_seconds"] = time.perf_counter() - start
    report["retained_union_derivative_field_bytes"] = sum(value.numel() * value.element_size()
                                                        for values in derivatives.values() for value in values)
    report["stage_peak_rss_bytes"]["after_union_fields"] = peak_rss_bytes()
    rigid_channels = receptor_channels(build_fields(receptor.xyz, receptor.radii, receptor.charges, grid, parameters), grid, parameters)
    report["stage"] = "union_frozen_pose_evaluation"
    checkpoint()
    start = time.perf_counter()
    with torch.no_grad():
        for index, pose in enumerate(union, 1):
            quaternion = quaternions[pose["rotation_index"]]
            translation = receptor.xyz.new_tensor(pose["translation"])
            xyz = torch.stack(rotate(*ligand.xyz.unbind(-1), quaternion), dim=-1) + translation
            ligand_values = ligand_channels(build_fields(xyz, ligand.radii, ligand.charges, grid, parameters))
            rigid = float((rigid_channels * ligand_values).sum())
            rigid_errors = {name: rigid - selected["rigid_energy"] for name, selected in pose["selected_by"].items()}
            if not all(math.isclose(rigid, selected["rigid_energy"], abs_tol=1e-7, rel_tol=1e-9)
                       for selected in pose["selected_by"].values()):
                raise ValueError("Union rigid energy differs from a selected FFT pose")
            gradients = {name: torch.stack([(derivative * ligand_values).sum() for derivative in values])
                         for name, values in derivatives.items()}
            full = gradients["full8"]
            solved = {name: solvers[name].solve(gradient, radius) for name, gradient in gradients.items()}
            row = {"pose_index": index, **pose, "rigid_energy": rigid, "rigid_search_errors": rigid_errors, "variants": {}}
            report["rows"].append(row)
            for name in VARIANTS:
                gradient, transform, response = gradients[name], transforms[name], solved[name]
                expected = transform.T @ full
                projection_error = float((gradient - expected).norm())
                if not torch.allclose(gradient, expected, atol=1e-7, rtol=1e-9):
                    raise ValueError(f"{name} actual field gradient differs from projected gradient")
                gain_loss = float(response.gain - solved["full8"].gain)
                if gain_loss < -1e-7 * max(1., abs(float(solved["full8"].gain))):
                    raise ValueError("Compressed constrained response improved beyond the full feasible set")
                candidate = ResponseCandidate(pose["rotation_index"], translation, response.amplitudes,
                    rigid + float(response.gain), rigid, float(response.gain), float(response.strain))
                checked = verifiers[name].verify(candidate, quaternion)
                row["variants"][name] = {"gradient": gradient.tolist(), "expected_projected_gradient": expected.tolist(),
                    "gradient_projection_error_norm": projection_error,
                    "gradient_loss_in_original_amplitudes_norm": float((full - transform @ gradient).norm()),
                    "gradient_loss_relative_norm": float((full - transform @ gradient).norm() / full.norm()) if float(full.norm()) else 0.,
                    "quadratic_gain": float(response.gain), "quadratic_strain": float(response.strain),
                    "quadratic_amplitudes": response.amplitudes.tolist(), "quadratic_gain_loss_vs_full": gain_loss,
                    "quadratic_gain_loss_fraction": gain_loss / abs(float(solved["full8"].gain)) if float(solved["full8"].gain) else 0.,
                    "verified_energy": checked.score, "verified_gain": checked.verified_gain,
                    "verified_prediction_at_accepted_scale": checked.predicted_gain,
                    "verified_strain": checked.strain, "verified_amplitudes": checked.amplitudes.tolist(),
                    "accepted_scale": checked.accepted_scale, "fallback": checked.fallback,
                    "rigid_verifier_error": checked.rigid_energy_error, "trials": list(checked.trials)}
                checkpoint()
            print(f"union pose {index}/{len(union)} completed", flush=True)
    report["union_evaluation_seconds"] = time.perf_counter() - start
    report["stage_peak_rss_bytes"]["after_union_evaluation"] = peak_rss_bytes()
    report["summary"] = {}
    for name in VARIANTS:
        rows = [row["variants"][name] for row in report["rows"]]
        losses = [row["quadratic_gain_loss_vs_full"] for row in rows]
        report["summary"][name] = {"evaluated_pose_count": len(rows),
            "accepted_count": sum(not row["fallback"] for row in rows), "fallback_count": sum(row["fallback"] for row in rows),
            "maximum_gradient_projection_error_norm": max(row["gradient_projection_error_norm"] for row in rows),
            "mean_gradient_loss_relative_norm": sum(row["gradient_loss_relative_norm"] for row in rows) / len(rows),
            "mean_quadratic_gain_loss": sum(losses) / len(losses), "maximum_quadratic_gain_loss": max(losses),
            "mean_quadratic_gain_loss_fraction": sum(row["quadratic_gain_loss_fraction"] for row in rows) / len(rows),
            "mean_verified_gain": sum(row["verified_gain"] for row in rows) / len(rows)}
    report.update(status="completed", stage="complete")


def child_main(root, name, stage):
    path = root / name / "report.json" if stage == "search" else root / "union" / "report.json"
    path.parent.mkdir(parents=True, exist_ok=False)
    registration = json.loads((root / "registration.json").read_text())
    started = time.perf_counter()
    report = {"format_version": 1, "status": "running", "stage": "starting", "variant": name,
              "native_access": False, "registration_sha256": sha256(root / "registration.json"),
              "stage_peak_rss_bytes": {}, "memory_note": "RSS is process high-water including imports, chemistry, modes and allocator; stage increases are not isolated workspace sizes"}

    def checkpoint():
        report["elapsed_seconds_so_far"] = time.perf_counter() - started
        report["peak_rss_bytes_so_far"] = peak_rss_bytes()
        save(path, report)

    checkpoint()
    try:
        if stage == "search":
            search_worker(name, registration, report, checkpoint)
        else:
            union_worker(root, registration, report, checkpoint)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_rss_bytes"] = peak_rss_bytes()
        checkpoint()


def run_child(root, name, stage):
    label = name if stage == "search" else "union"
    command = [sys.executable, str(Path(__file__).resolve()), "--out", str(root),
               "--child-stage", stage, "--variant", name]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    started = time.perf_counter()
    timed_out = False
    with (root / f"{label}.log").open("w") as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
                                    env=environment, timeout=TIMEOUT_SECONDS, check=False)
            return_code = result.returncode
        except subprocess.TimeoutExpired:
            return_code, timed_out = None, True
    path = root / label / "report.json"
    if path.exists():
        report = json.loads(path.read_text())
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {"status": "failed", "stage": "before_first_checkpoint"}
    report["parent_process"] = {"command": command, "return_code": return_code, "timed_out": timed_out,
                                 "elapsed_seconds": time.perf_counter() - started, "timeout_seconds": TIMEOUT_SECONDS}
    if timed_out:
        report["status"] = "timed_out"
    elif return_code != 0 and report.get("status") != "failed":
        report.update(status="failed", error="Child process exited unsuccessfully")
    elif report.get("status") != "completed":
        report.update(status="failed", error=report.get("error", "Child exited without a completed report"))
    save(path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=REPO / "docs/anm_memory_v1.md")
    parser.add_argument("--child-stage", choices=("search", "union"), help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=tuple(VARIANTS), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.out.resolve()
    if args.child_stage:
        if args.child_stage == "search" and args.variant is None:
            parser.error("search child requires --variant")
        child_main(root, args.variant, args.child_stage)
        return
    if args.manifest is None:
        parser.error("--manifest is required")
    if not args.manifest.is_file() or not args.protocol.is_file():
        parser.error("manifest and protocol must exist")
    root.mkdir(parents=True, exist_ok=False)
    registration = {"format_version": 1, "configuration": CONFIGURATION,
        "manifest_path": str(args.manifest.resolve()), "input_manifest_sha256": sha256(args.manifest),
        "protocol_path": str(args.protocol.resolve()), "protocol_sha256": sha256(args.protocol),
        "source_sha256": registered_sources(), "command": sys.argv, "platform": platform.platform(),
        "native_access": False, "scope": "development input-only numerical/candidate diagnostic; no global docking accuracy",
        "failure_policy": "fixed three variants, no replacements; union stage skipped if any search fails",
        "union_order": "all selected pose keys sorted by rotation index then translation, independent of score"}
    save(root / "registration.json", registration)
    summary = {"format_version": 1, "status": "running", "native_access": False,
               "searches": {}, "union": {"status": "pending"}}
    save(root / "report.json", summary)
    searches = {}
    for name in VARIANTS:
        searches[name] = run_child(root, name, "search")
        summary["searches"][name] = {"status": searches[name]["status"], "report": f"{name}/report.json"}
        save(root / "report.json", summary)
    if all(item["status"] == "completed" for item in searches.values()):
        union = run_child(root, "full8", "union")
        summary["union"] = {"status": union["status"], "report": "union/report.json"}
        summary["status"] = "completed" if union["status"] == "completed" else "partial_failure"
    else:
        summary.update(status="partial_failure", union={"status": "skipped", "reason": "At least one registered search failed; no outcome-based replacement"})
    summary["search_success_count"] = sum(row["status"] == "completed" for row in searches.values())
    summary["search_attempt_count"] = len(VARIANTS)
    save(root / "report.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
