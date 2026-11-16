"""Evaluation contracts independent of DockQ installation and docking outputs."""

import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("pilot_scorer", Path(__file__).parents[1] / "scripts/score_inducedfit_pilot.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def test_rank_provenance_and_failed_candidates_survive_report_loading(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"status": "partial_failure", "configuration": {"top_k": 4}, "arms": {"R2": {
        "status": "partial_failure", "candidates": [
            {"selection_rank": 1, "final_rank": 2, "initial_files": {"complex": "selected1.pdb"}, "files": {"complex": "final1.pdb"}},
            {"selection_rank": 2, "final_rank": 1, "initial_files": {"complex": "selected2.pdb"}, "files": {"complex": "final2.pdb"}},
            {"selection_rank": 3, "status": "exception", "error": "OOM", "initial_files": {"complex": "selected3.pdb"}},
        ]}}}))
    rows, arms = scorer.models_from_report(path)
    assert arms["R2"]["requested_candidates"] == 4
    assert len(rows) == 8
    top_final = next(row for row in rows if row["stage"] == "final" and row["rank"] == 1)
    assert top_final["selection_rank"] == 2
    assert top_final["path"] == str(tmp_path / "final2.pdb")
    failed = next(row for row in rows if row["stage"] == "final" and row["selection_rank"] == 3)
    assert failed["path"] is None and failed["predictor_error"] == "OOM"
    assert sum(row["predictor_status"] == "candidate_shortfall" for row in rows if "predictor_status" in row) == 2


def test_summary_separates_initial_and_final_and_does_not_rerank_by_accuracy():
    rows = [
        {"arm": "R2", "stage": "initial", "rank": 1, "status": "scored", "metrics": {"DockQ": .1}},
        {"arm": "R2", "stage": "initial", "rank": 2, "status": "scored", "metrics": {"DockQ": .8}},
        {"arm": "R2", "stage": "final", "rank": 2, "status": "scored", "metrics": {"DockQ": .95}},
        {"arm": "R2", "stage": "final", "rank": 1, "status": "missing_prediction"},
    ]
    summary = scorer.summarize(rows)
    assert summary["R2_initial"]["top1_DockQ"] == .1
    assert summary["R2_initial"]["oracle_at_5_DockQ"] == .8
    assert summary["R2_final"]["top1_DockQ"] == 0
    assert summary["R2_final"]["top1_scored"] is False
    assert summary["R2_final"]["oracle_at_5_DockQ"] == .95
    assert summary["R2_final"]["failed_or_missing_candidates"] == 1


def _raw_result():
    return {"best_mapping": {"A": "R", "D": "L"}, "best_result": {
        "AD": {"DockQ": .23, "iRMSD": 2.0, "LRMSD": 3.0, "fnat": .1}}}


def test_official_chain_direction_and_metric_finiteness_are_enforced():
    raw = _raw_result()
    assert scorer.checked_metrics(raw, "RL:AD")["dockq_quality"] == "acceptable"
    raw["best_mapping"] = {"R": "A", "L": "D"}
    with pytest.raises(ValueError, match="mapping"):
        scorer.checked_metrics(raw, "RL:AD")
    raw = _raw_result()
    raw["best_result"]["AD"]["DockQ"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        scorer.checked_metrics(raw, "RL:AD")


@pytest.mark.parametrize("mapping", ["*:AD", "RL:AA", "R:L", "RL:AD:EF"])
def test_mapping_cannot_search_native_assignments(mapping):
    with pytest.raises(ValueError):
        scorer.parse_mapping(mapping)


def test_running_predictor_report_is_not_scored(tmp_path):
    path = tmp_path / "running.json"
    path.write_text(json.dumps({"status": "running", "arms": {}}))
    with pytest.raises(ValueError, match="finished"):
        scorer.models_from_report(path)
