#!/usr/bin/env python3
"""Input-only quadrature audit of a frozen induced-fit report, without optimization.

Four preregistered lattices: spacing 2/1 A, phase 0/half cell on every axis.
All use one physical cube containing all complete compact molecular supports.
This evaluates physical_energy at the actual frozen coordinates. It does not
run a global correlation search; charge convolution uses the existing padded
linear FFT convolution, not the prohibitively large spatial oracle.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback

import torch

from zdock.inducedfit.fields import (build_fields, ligand_channels, physical_energy,
                                     receptor_channels, support_radii)
from zdock.inducedfit.replay import load_replay
from zdock.inducedfit.types import GridSpec


def tensor_hash(*values):
    digest = hashlib.sha256()
    for value in values:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def common_grids(poses, receptor_radii, ligand_radii, parameters):
    """Construct a common half-open cube with a fixed 4 A exterior margin.

    Every node, for every phase, lies within this SAME physical cube; compact
    support is fully within even the most restrictive lattice node bounds.
    The phase-zero 2 A origin is a multiple of 2 A in the replay frame.
    """
    lower, upper = [], []
    for pose in poses:
        for xyz, radii in ((pose.receptor_xyz, receptor_radii), (pose.ligand_xyz, ligand_radii)):
            support = support_radii(radii, parameters)[:, None]
            lower.append((xyz - support).amin(0))
            upper.append((xyz + support).amax(0))
    support_lower = torch.stack(lower).amin(0)
    support_upper = torch.stack(upper).amax(0)
    origin = torch.floor(support_lower / 2.) * 2. - 4.
    length = float(torch.ceil((support_upper + 4. - origin).max() / 2.) * 2.)
    grids = {}
    for spacing, phase in itertools.product((2., 1.), (0., .5)):
        count = round(length / spacing)
        grid = GridSpec((count, count, count), spacing,
                        tuple(float(x) + phase * spacing for x in origin))
        lo, hi = grid.bounds(dtype=receptor_radii.dtype, device=receptor_radii.device)
        if not bool((lo <= support_lower).all() and (hi >= support_upper).all()):
            raise ValueError("Common cube failed complete-support validation")
        grids[f"h{spacing:g}_phase{phase:g}"] = grid
    return grids, {"cube_lower_angstrom": origin.tolist(), "cube_upper_angstrom": (origin + length).tolist(),
                   "cube_side_angstrom": length, "cube_convention": "half-open; all sample nodes inside fixed cube",
                   "molecular_support_lower_angstrom": support_lower.tolist(),
                   "molecular_support_upper_angstrom": support_upper.tolist(),
                   "minimum_prescribed_exterior_margin_angstrom": 4.}


def average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        for index in order[start:stop]:
            ranks[index] = (start + stop - 1) / 2. + 1.
        start = stop
    return ranks


def rank_comparison(rows, first, second):
    a = [row["scores"][first]["total"] for row in rows]
    b = [row["scores"][second]["total"] for row in rows]
    ar, br = average_ranks(a), average_ranks(b)
    am, bm = sum(ar) / len(ar), sum(br) / len(br)
    denominator = math.sqrt(sum((x - am) ** 2 for x in ar) * sum((x - bm) ** 2 for x in br))
    rho = sum((x - am) * (y - bm) for x, y in zip(ar, br)) / denominator if denominator else None
    inverted, comparable, tied = 0, 0, 0
    for i, j in itertools.combinations(range(len(rows)), 2):
        product = (a[i] - a[j]) * (b[i] - b[j])
        if product == 0:
            tied += 1
        else:
            comparable += 1
            inverted += product < 0
    return {"first": first, "second": second, "n": len(rows), "spearman_average_exact_ties": rho,
            "inverted_pairs": inverted, "comparable_nontied_pairs": comparable,
            "pairs_tied_in_either": tied,
            "maximum_absolute_total_change": max(abs(x - y) for x, y in zip(a, b)),
            "mean_absolute_total_change": sum(abs(x - y) for x, y in zip(a, b)) / len(rows),
            "first_top1": rows[min(range(len(a)), key=lambda i: (a[i], rows[i]["id"]))]["id"],
            "second_top1": rows[min(range(len(b)), key=lambda i: (b[i], rows[i]["id"]))]["id"]}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", default=1, type=int)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    output = {"format_version": 1, "status": "preparing", "native_access": False,
              "question": "Are coarse physical energies/ranks stable to grid refinement and half-cell phase?",
              "run_report": str(args.run_report.resolve()), "command": sys.argv,
              "hardware": {"platform": platform.platform(), "torch": torch.__version__,
                           "device": "cpu", "dtype": "float64", "threads": args.threads},
              "timing_caveat": "Diagnostic elapsed times; other research jobs may run concurrently, not a speed benchmark",
              "scores_unit": "effective epsilon, not kcal/mol", "optimization": False,
              "rows": []}
    save(args.out / "report.json", output)
    try:
        context = load_replay(args.run_report)
        params = replace(context.parameters, backend="sparse")
        poses = []
        for arm, record in context.report["arms"].items():
            for row in record["candidates"]:
                pose = context.final_pose(arm, row)
                poses.append(pose)
                output["rows"].append({"id": f"{arm}_selection{pose.selection_rank}", "arm": arm,
                    "selection_rank": pose.selection_rank, "original_final_rank": pose.final_rank,
                    "coordinate_sha256": tensor_hash(pose.receptor_xyz, pose.ligand_xyz),
                    "heavy_export_errors": pose.heavy_export_errors, "strain": pose.strain,
                    "original_total_energy": pose.stored_energy, "scores": {}})
        if not poses:
            raise ValueError("No final poses to audit")
        grids, cube = common_grids(poses, context.receptor.radii, context.ligand.radii, params)
        registration = {"format_version": 1, "native_access": False,
                        "source_audit": context.source_audit, "parameters": asdict(params),
                        "common_physical_cube": cube, "grids": {k: asdict(v) for k, v in grids.items()},
                        "pose_ids_and_hashes": [{"id": row["id"], "sha256": row["coordinate_sha256"]} for row in output["rows"]],
                        "selection_rule": "Every final candidate in every arm of the fixed report; no coordinate optimization or accuracy access",
                        "energy_method": "physical_energy at frozen coordinates; sparse compact fields; padded linear FFT charge convolution; no global correlation",
                        "planned_comparisons": "All six lattice pairs, global and each arm; rank correlation, inversions, top1 and component changes",
                        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        # Persist BEFORE the first energy: cube/configs/pose set depend only on inputs/exports.
        save(args.out / "registration.json", registration)
        output.update(status="running", registration_sha256=hashlib.sha256((args.out / "registration.json").read_bytes()).hexdigest())
        save(args.out / "report.json", output)
        with torch.no_grad():
            for name, grid in grids.items():
                cache = {}
                for pose, row in zip(poses, output["rows"]):
                    digest = row["coordinate_sha256"]
                    item_start = time.perf_counter()
                    if digest in cache:
                        row["scores"][name] = {**cache[digest], "total": cache[digest]["physical"] + pose.strain,
                                               "strain": pose.strain, "cache_reused": True,
                                               "elapsed_seconds": time.perf_counter() - item_start}
                    else:
                        rec = build_fields(pose.receptor_xyz, context.receptor.radii, context.receptor.charges, grid, params)
                        lig = build_fields(pose.ligand_xyz, context.ligand.radii, context.ligand.charges, grid, params)
                        physical = float(physical_energy(rec, lig, grid, params))
                        components = (receptor_channels(rec, grid, params) * ligand_channels(lig)).sum((1, 2, 3)).tolist()
                        if not math.isfinite(physical) or not math.isclose(sum(components), physical, rel_tol=1e-10, abs_tol=1e-8):
                            raise FloatingPointError("Nonfinite energy or inconsistent component decomposition")
                        item = {"repulsion": components[0], "surface": components[1], "charge": components[2],
                                "physical": physical, "total": physical + pose.strain,
                                "strain": pose.strain, "cache_reused": False,
                                "elapsed_seconds": time.perf_counter() - item_start}
                        row["scores"][name] = item
                        cache[digest] = item
                        del rec, lig
                    save(args.out / "report.json", output)
                    print(json.dumps({"grid": name, "pose": row["id"], **row["scores"][name]}), flush=True)
        output["comparisons"] = {}
        groups = {"all": output["rows"], **{arm: [row for row in output["rows"] if row["arm"] == arm]
                                          for arm in context.report["arms"]}}
        for group, rows in groups.items():
            output["comparisons"][group] = [rank_comparison(rows, a, b) for a, b in itertools.combinations(grids, 2)]
        output["phase_zero_coarse_vs_original"] = {
            "maximum_absolute_total_change": max(abs(row["scores"]["h2_phase0"]["total"] - row["original_total_energy"]) for row in output["rows"]),
            "interpretation": "Includes ligand-transform reconstruction from rounded PDB exports and regenerated H; not a new optimization"}
        output["status"] = "completed"
    except BaseException as error:
        output.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        output["elapsed_seconds"] = time.perf_counter() - start
        save(args.out / "report.json", output)


if __name__ == "__main__":
    main()
