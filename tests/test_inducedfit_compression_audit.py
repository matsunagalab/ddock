"""Experiment orchestration: frozen unions, comparable inputs and failures."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("compression_audit", Path(__file__).resolve().parents[1] / "scripts/audit_inducedfit_compression.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def searches():
    def row(rotation, translation, rank):
        return {"rotation_index": rotation, "translation": translation, "selection_rank": rank,
                "rigid_energy": -10. + rank}
    return {"full8": {"candidates": [row(2, [2., 0., 0.], 1), row(0, [0., 0., 0.], 2)]},
            "rank4": {"candidates": [row(0, [0., 0., 0.], 1), row(1, [0., 2., 0.], 2)]},
            "rank2": {"candidates": [row(2, [2., 0., 0.], 1)]}}


def test_union_retains_all_variants_and_sorts_without_scores():
    union = audit.union_records(searches())
    assert [row["rotation_index"] for row in union] == [0, 1, 2]
    assert set(union[0]["selected_by"]) == {"full8", "rank4"}
    assert union[0]["selected_by"]["full8"]["selection_rank"] == 2
    assert len(union) == 3


def test_overlap_shortfalls_do_not_shrink_requested_top10_denominator():
    rows = audit.overlap_records(searches())
    assert rows[0]["intersection"] == 1 and rows[0]["union"] == 3
    assert rows[0]["intersection_fraction_of_requested_top10"] == .1
    assert rows[0]["jaccard"] == pytest.approx(1 / 3)


def test_child_timeout_preserves_partial_rows_and_last_stage(tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        path = tmp_path / "full8" / "report.json"
        path.parent.mkdir()
        audit.save(path, {"status": "running", "stage": "response_search", "rows": [{"example": 1}]})
        assert kwargs["timeout"] == 180
        raise subprocess.TimeoutExpired(command, 180)
    monkeypatch.setattr(audit.subprocess, "run", timeout)
    report = audit.run_child(tmp_path, "full8", "search")
    assert report["status"] == "timed_out"
    assert report["stage"] == "response_search" and report["rows"] == [{"example": 1}]
    assert report["parent_process"]["return_code"] is None


def test_child_crash_before_report_remains_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=-9))
    report = audit.run_child(tmp_path, "rank2", "search")
    assert report["status"] == "failed" and report["stage"] == "before_first_checkpoint"
    assert report["parent_process"]["return_code"] == -9
    assert not report["parent_process"]["timed_out"]


def test_registration_rejects_input_drift_before_calculation(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    registration = {"configuration": audit.CONFIGURATION, "manifest_path": str(manifest),
                    "input_manifest_sha256": audit.sha256(manifest), "source_sha256": {}}
    audit.check_registration(registration)
    manifest.write_text('{"modified":true}')
    with pytest.raises(ValueError, match="manifest SHA256"):
        audit.check_registration(registration)


def test_union_rejects_changed_mode_gauge_or_hydrogen_coordinates():
    reference = {"base_model_sha256": "mode", "grid": {}, "parameters": {}, "effective_radius": .3, "quaternions": [[0., 0., 0., 1.]],
                 "preparation": {side: {"full_monomer_sha256": side} for side in ("receptor", "ligand")}}
    search = {**copy.deepcopy(reference), "status": "completed"}
    audit.assert_same_inputs({"full8": search}, reference)
    search["base_model_sha256"] = "other gauge"
    with pytest.raises(ValueError, match="base_model_sha256"):
        audit.assert_same_inputs({"full8": search}, reference)
    search["base_model_sha256"] = "mode"
    search["preparation"]["ligand"]["full_monomer_sha256"] = "changed H"
    with pytest.raises(ValueError, match="ligand differs"):
        audit.assert_same_inputs({"full8": search}, reference)


def test_union_compares_json_grid_arrays_with_live_dataclass_tuples():
    reference = {"base_model_sha256": "mode", "grid": {"shape": (83, 83, 83), "origin": (-82., -82., -82.), "spacing": 2.},
                 "parameters": {}, "effective_radius": .3, "quaternions": [[0., 0., 0., 1.]],
                 "preparation": {side: {"full_monomer_sha256": side} for side in ("receptor", "ligand")}}
    search = json.loads(json.dumps({**reference, "status": "completed"}))
    audit.assert_same_inputs({"full8": search}, reference)
    search["grid"]["origin"][0] += 1.
    with pytest.raises(ValueError, match="common grid"):
        audit.assert_same_inputs({"full8": search}, reference)
