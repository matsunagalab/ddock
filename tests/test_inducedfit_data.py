"""Input/reference separation, provenance invalidation, and grid conventions."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields

import pytest
import torch

from zdock.inducedfit.prepare import (
    build_cache_metadata,
    config_fingerprint,
    input_fingerprint,
    input_manifest_record,
    read_cache_metadata,
    validate_input_manifest,
    write_cache_metadata,
)
from zdock.inducedfit.types import AtomKey, DockingInput, GridSpec, MonomerInput


def _monomer(*, offset=0.0, charge=0.1, residue_number=1):
    return MonomerInput(
        xyz=torch.tensor([[offset, 0.0, 0.0], [1.0 + offset, 1.0, 0.0]], dtype=torch.float64),
        radii=torch.tensor([1.5, 1.7], dtype=torch.float64),
        charges=torch.tensor([charge, -0.1], dtype=torch.float64),
        atom_keys=(AtomKey("A", residue_number, "", "GLY", "CA"),
                   AtomKey("A", residue_number, "", "GLY", "C")),
    )


def _input(*, system_id="synthetic", **kwargs):
    return DockingInput(system_id, _monomer(**kwargs), _monomer(offset=2.0))


def _metadata(inp=None, **kwargs):
    config = {"input_kind": "synthetic", "prep_config": {"version": 1},
              "core_config": {"spacing": 1.2}, "mode_config": {"n_modes": 4}}
    config.update(kwargs)
    return build_cache_metadata(_input() if inp is None else inp, **config)


def test_input_has_no_reference_fields_and_rejects_added_answers():
    inp = _input()
    assert {field.name for field in fields(inp)} == {"system_id", "receptor", "ligand"}
    assert {field.name for field in fields(inp.receptor)} == {"xyz", "radii", "charges", "atom_keys"}
    with pytest.raises(TypeError):
        DockingInput("x", inp.receptor, inp.ligand, native_lig=inp.ligand.xyz)
    with pytest.raises(FrozenInstanceError):
        inp.system_id = "changed"
    assert not hasattr(inp, "__dict__")
    assert not hasattr(inp.receptor, "__dict__")


def test_constructor_owns_tensor_storage_and_preserves_gradients():
    m = _monomer()
    xyz = m.xyz.clone().requires_grad_()
    owned = MonomerInput(xyz, m.radii, m.charges, m.atom_keys)
    owned.xyz.square().sum().backward()
    torch.testing.assert_close(xyz.grad, 2 * xyz)
    with torch.no_grad():
        xyz.add_(7)
    torch.testing.assert_close(owned.xyz, m.xyz)


@pytest.mark.parametrize("field,value", [
    ("xyz", torch.zeros(2, 2, dtype=torch.float64)),
    ("xyz", torch.tensor([[float("nan"), 0, 0], [0, 0, 0]], dtype=torch.float64)),
    ("radii", torch.tensor([0.0, 1.0], dtype=torch.float64)),
    ("charges", torch.zeros(2, dtype=torch.float32)),
    ("charges", torch.tensor([float("inf"), 0.0], dtype=torch.float64)),
])
def test_bad_input_rejected_before_scoring(field, value):
    m = _monomer()
    values = {name: getattr(m, name) for name in ("xyz", "radii", "charges", "atom_keys")}
    values[field] = value
    with pytest.raises((ValueError, TypeError)):
        MonomerInput(**values)


def test_duplicate_atom_identity_rejected():
    m = _monomer()
    with pytest.raises(ValueError, match="unique"):
        MonomerInput(m.xyz, m.radii, m.charges, (m.atom_keys[0], m.atom_keys[0]))


def test_grid_ij_order_boundaries_and_support():
    grid = GridSpec((2, 3, 2), 0.5, (-1, 2, 4))
    expected = torch.tensor([
        [-1, 2, 4], [-1, 2, 4.5], [-1, 2.5, 4], [-1, 2.5, 4.5],
        [-1, 3, 4], [-1, 3, 4.5], [-0.5, 2, 4], [-0.5, 2, 4.5],
        [-0.5, 2.5, 4], [-0.5, 2.5, 4.5], [-0.5, 3, 4], [-0.5, 3, 4.5],
    ], dtype=torch.float64)
    torch.testing.assert_close(grid.points(dtype=torch.float64), expected)
    assert grid.n_points == 12
    assert grid.voxel_volume == 0.125
    center = torch.tensor([[-0.75, 2.5, 4.25]], dtype=torch.float64)
    assert grid.contains_support(center, 0.25)
    assert not grid.contains_support(center, 0.2501)
    assert not grid.contains_support(torch.tensor([[0.0, 2.5, 4.25]]), 0)


@pytest.mark.parametrize("shape,spacing,origin", [
    ((2, 0, 3), 1, (0, 0, 0)), ((2, True, 3), 1, (0, 0, 0)),
    ((2, 2, 3), -1, (0, 0, 0)), ((2, 2, 3), float("nan"), (0, 0, 0)),
    ((2, 2, 3), 1, (0, float("inf"), 0)),
])
def test_invalid_grid_rejected(shape, spacing, origin):
    with pytest.raises(ValueError):
        GridSpec(shape, spacing, origin)


def test_fingerprint_changes_for_content_identity_dtype_and_configuration():
    base = input_fingerprint(_input())
    assert base == input_fingerprint(_input())
    for changed in (_input(system_id="other"), _input(offset=0.001),
                    _input(charge=0.2), _input(residue_number=2)):
        assert input_fingerprint(changed) != base
    assert config_fingerprint({"a": 1, "b": [2, 3]}) == config_fingerprint({"b": (2, 3), "a": 1})
    initial = _metadata()["cache_key"]
    for changed in (_metadata(input_kind="apo"), _metadata(prep_config={"version": 2}),
                    _metadata(core_config={"spacing": 0.9}), _metadata(mode_config={"n_modes": 8})):
        assert changed["cache_key"] != initial
    with pytest.raises(ValueError, match="finite"):
        config_fingerprint({"bad": float("nan")})
    with pytest.raises(TypeError):
        config_fingerprint({"bad": torch.tensor(1.0)})


@pytest.mark.parametrize("field", ["native_lig", "q_star", "t_star", "DockQ", "native_path"])
def test_manifest_rejects_reference_fields_top_level_and_nested(field):
    record = input_manifest_record(_input(), input_kind="synthetic")
    record[field] = "answer"
    with pytest.raises(ValueError, match="unrecognized"):
        validate_input_manifest(record)
    record = input_manifest_record(_input(), input_kind="synthetic")
    record["receptor"][field] = "answer"
    with pytest.raises(ValueError, match="unrecognized"):
        validate_input_manifest(record)


def test_metadata_misses_are_safe_and_do_not_reuse_wrong_inputs(tmp_path):
    path = tmp_path / "metadata.json"
    meta = _metadata()
    assert read_cache_metadata(path, expected=meta) is None
    write_cache_metadata(path, meta)
    assert read_cache_metadata(path, expected=meta) == meta
    assert read_cache_metadata(path, expected=_metadata(_input(offset=0.1))) is None
    assert read_cache_metadata(path, expected=_metadata(mode_config={"n_modes": 8})) is None
    assert list(tmp_path.iterdir()) == [path]
    path.write_text('{"format_version":', encoding="utf-8")
    assert read_cache_metadata(path, expected=meta) is None
    bad = deepcopy(meta)
    bad["format_version"] = 0
    import json
    path.write_text(json.dumps(bad), encoding="utf-8")
    assert read_cache_metadata(path, expected=meta) is None
    bad = deepcopy(meta)
    bad["input"]["system_id"] = "different"
    with pytest.raises(ValueError, match="integrity"):
        write_cache_metadata(path, bad)
