"""Protect input isolation and the coordinate frame of exported predictions."""
import hashlib
import json

import pytest
import torch

from zdock.inducedfit.fields import FieldParameters, build_fields
from zdock.inducedfit.pdb_input import read_pdb_monomer
from zdock.inducedfit.pipeline import export_heavy_pair, load_pair_manifest, pair_grid, rotation_sample
from zdock.inducedfit.types import AtomKey, MonomerInput
from test_inducedfit_pdb_input import _atom


def pair(tmp_path):
    pdb = tmp_path / 'input.pdb'
    pdb.write_text(_atom(xyz=(101., 202., 303.)) + '\n')
    item = {'pdb': pdb.name, 'chain': 'A', 'sha256': hashlib.sha256(pdb.read_bytes()).hexdigest()}
    record = {'format_version': 1, 'system_id': 'smoke', 'input_kind': 'synthetic',
              'split': 'development', 'receptor': item.copy(), 'ligand': item.copy()}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(record))
    return path, record


def test_manifest_hash_and_nested_native_rejection(tmp_path):
    path, record = pair(tmp_path)
    _, loaded = load_pair_manifest(path)
    assert loaded['ligand'].monomer.n_atoms == 1
    record['receptor']['native'] = 'do_not_read.pdb'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='unrecognized'):
        load_pair_manifest(path)
    del record['receptor']['native']
    record['ligand']['sha256'] = '0' * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        load_pair_manifest(path)


def test_manifest_rejects_top_level_labels(tmp_path):
    path, record = pair(tmp_path)
    record['dockq'] = .9
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='unrecognized'):
        load_pair_manifest(path)


def test_export_both_partners_in_receptor_frame(tmp_path):
    path, _ = pair(tmp_path)
    _, loaded = load_pair_manifest(path)
    center = torch.tensor([10., 20., 30.], dtype=torch.float64)
    rec = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
    lig = torch.tensor([[4., 0., 0.]], dtype=torch.float64)
    out = tmp_path / 'prediction'
    paths = export_heavy_pair(out, loaded['receptor'], loaded['ligand'], rec, lig, center)
    for name, chain, expected in [('receptor', 'R', rec), ('ligand', 'L', lig)]:
        standalone = read_pdb_monomer(paths[name], chain='A', charge_policy='zero_diagnostic')
        combined = read_pdb_monomer(paths['complex'], chain=chain, charge_policy='zero_diagnostic')
        torch.testing.assert_close(standalone.monomer.xyz, expected + center)
        torch.testing.assert_close(combined.monomer.xyz, expected + center)
    with pytest.raises(FileExistsError):
        export_heavy_pair(out, loaded['receptor'], loaded['ligand'], rec, lig, center)


def test_rotation_sample_reproducible_and_normalized_without_global_rng():
    state = torch.random.get_rng_state()
    a = rotation_sample(20, 42)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(a, rotation_sample(20, 42))
    assert not torch.equal(a, rotation_sample(20, 43))
    torch.testing.assert_close(a.norm(dim=-1), torch.ones(20, dtype=torch.float64))
    torch.testing.assert_close(a[0], torch.tensor([0., 0., 0., 1.], dtype=torch.float64))


def test_pair_grid_holds_rotations_and_compact_support():
    xyz = torch.tensor([[1., -1., 0.], [-1., 1., 0.]], dtype=torch.float64)
    m = MonomerInput(xyz, torch.full((2,), 1.7, dtype=xyz.dtype), torch.zeros(2, dtype=xyz.dtype),
                     (AtomKey('A', 1, '', 'ALA', 'CA'), AtomKey('A', 2, '', 'ALA', 'CA')))
    parameters = FieldParameters()
    grid = pair_grid(m, m, parameters, spacing=2., max_atom_displacement=.5, max_translation=.5)
    from zdock.geom import rotate
    for q in rotation_sample(3, 0):
        rotated = torch.stack(rotate(*xyz.unbind(-1), q), dim=-1)
        build_fields(rotated + xyz.new_tensor([4., 0., 0.]), m.radii, m.charges, grid, parameters)
