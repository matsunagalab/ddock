"""Frozen replay must retain atom identity, proper rigid motions and input hashes."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from zdock.inducedfit.fields import FieldParameters
from zdock.inducedfit.pipeline import export_heavy_pair
from zdock.inducedfit.replay import ReplayContext, fit_rigid_transform, load_replay
from zdock.inducedfit.types import AtomKey, GridSpec, MonomerInput
from zdock.inducedfit.pdb_input import read_pdb_monomer, write_pdb_monomer
from test_inducedfit_pdb_input import _atom


def test_rigid_fit_proper_motion_and_reflection():
    source = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]], dtype=torch.float64)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=source.dtype)
    target = source @ rotation + source.new_tensor([4., 5., -6.])
    r, t = fit_rigid_transform(source, target)
    torch.testing.assert_close(source @ r + t, target)
    torch.testing.assert_close(torch.det(r), r.new_tensor(1.))
    mirrored = source * source.new_tensor([-1., 1., 1.])
    r, t = fit_rigid_transform(source, mirrored)
    assert torch.det(r) > .999999
    assert (source @ r + t - mirrored).square().sum() > .1


def test_rigid_fit_rejects_ambiguous_or_nonfinite_atoms():
    line = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], dtype=torch.float64)
    with pytest.raises(ValueError, match='Collinear'):
        fit_rigid_transform(line, line)
    bad = line.clone()
    bad[1, 1] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        fit_rigid_transform(line, bad)


def make_replay(tmp_path):
    raw = tmp_path / 'source.pdb'
    coordinates = [(0., 0., 0.), (1., 0., 0.), (0., 2., 0.), (0., 0., 3.)]
    names = ['N', 'CA', 'C', 'O']
    raw.write_text('\n'.join(_atom(serial=i + 1, name=name, xyz=xyz, element=name[0])
                             for i, (name, xyz) in enumerate(zip(names, coordinates))) + '\n')
    prepared = read_pdb_monomer(raw, chain='A', charge_policy='zero_diagnostic')
    heavy = prepared.monomer
    full = MonomerInput(torch.cat([heavy.xyz, heavy.xyz.new_tensor([[.1, .2, .3]])]),
                        torch.cat([heavy.radii, heavy.radii.new_tensor([1.2])]),
                        torch.zeros(5, dtype=heavy.xyz.dtype),
                        heavy.atom_keys + (AtomKey('A', 1, '', 'ALA', 'H'),))
    proteins = {side: SimpleNamespace(monomer=full, heavy_monomer=heavy, heavy_indices=torch.arange(4))
                for side in ('receptor', 'ligand')}
    model = SimpleNamespace(n_modes=1, coordinates=lambda a: full.xyz + a[0] * full.xyz.new_tensor([.1, 0., 0.]),
                            deformation_energy=lambda a: .5 * a.square().sum())
    center = full.xyz.new_tensor([10., 20., 30.])
    translation = full.xyz.new_tensor([4., 5., 6.])
    final_receptor = model.coordinates(full.xyz.new_tensor([.2]))
    final_ligand = full.xyz + translation
    files = export_heavy_pair(tmp_path / 'final', prepared, prepared, final_receptor[:4], final_ligand[:4], center)
    initial_files = export_heavy_pair(tmp_path / 'initial', prepared, prepared, full.xyz[:4], final_ligand[:4], center)
    relative = lambda items: {k: str(Path(v).relative_to(tmp_path)) for k, v in items.items()}
    row = {'selection_rank': 1, 'final_rank': 1, 'rotation_index': 0, 'translation': translation.tolist(),
           'amplitudes': [.2], 'strain': .02, 'energy': -2., 'selected_rigid_energy': -1.,
           'files': relative(files), 'initial_files': relative(initial_files)}
    report = {'arms': {'R': {'candidates': [row]}}}
    return ReplayContext(tmp_path / 'report.json', report, {side: prepared for side in proteins}, proteins,
                         {side: center for side in proteins}, model, GridSpec((5, 5, 5), 1., (-2., -2., -2.)),
                         FieldParameters(), full.xyz.new_tensor([[0., 0., 0., 1.]]), full, full, {}), row


def test_replay_restores_full_hydrogens_and_receptor_frame(tmp_path):
    context, row = make_replay(tmp_path)
    pose = context.final_pose('R', row)
    assert len(pose.ligand_xyz) == 5
    torch.testing.assert_close(pose.ligand_xyz, context.ligand.xyz + context.ligand.xyz.new_tensor(row['translation']))
    torch.testing.assert_close(pose.receptor_xyz, context.model.coordinates(context.receptor.xyz.new_tensor([.2])))
    initial = context.initial_pose('R', row)
    torch.testing.assert_close(initial.receptor_xyz, context.receptor.xyz)
    assert initial.stage == 'initial' and initial.strain == 0.
    row.update(rotation_increment=[0., 0., 0.], translation_increment=[0., 0., 0.])
    exact = context.final_pose('R', row)
    torch.testing.assert_close(exact.ligand_xyz, pose.ligand_xyz)
    assert 'saved initial' in exact.heavy_export_errors['ligand_transform_policy']
    del row['translation_increment']
    with pytest.raises(ValueError, match='both rotation'):
        context.final_pose('R', row)


def test_replay_rejects_foreign_candidate_identity_drift_and_path_escape(tmp_path):
    context, row = make_replay(tmp_path)
    with pytest.raises(ValueError, match='belong'):
        context.final_pose('other', row)
    row['files']['ligand'] = '../outside.pdb'
    with pytest.raises(ValueError, match='escapes'):
        context.final_pose('R', row)
    row['files']['ligand'] = 'final/ligand.pdb'
    path = tmp_path / row['files']['ligand']
    path.write_text(path.read_text().replace(' ALA ', ' GLY '))
    with pytest.raises(ValueError, match='identities'):
        context.final_pose('R', row)


def test_exact_increment_rotation_uses_all_atom_centroid(tmp_path):
    context, row = make_replay(tmp_path)
    xyz = context.ligand.xyz
    center = xyz.mean(0)
    rotation = xyz.new_tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    delta = xyz.new_tensor([.4, -.5, .6])
    expected = (xyz - center) @ rotation + center + xyz.new_tensor(row['translation']) + delta
    row.update(rotation_increment=[0., 0., math.pi / 2], translation_increment=delta.tolist())
    write_pdb_monomer(tmp_path / row['files']['ligand'], context.inputs['ligand'],
                       xyz=expected[:4] + context.centers['receptor'])
    pose = context.final_pose('R', row)
    torch.testing.assert_close(pose.ligand_xyz, expected)
    assert pose.heavy_export_errors['ligand']['maximum_atom_error_angstrom'] < .001


def test_manifest_drift_fails_before_chemistry(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'status': 'completed', 'configuration': {'manifest': str(manifest)},
                                  'input_manifest_sha256': '0' * 64}))
    monkeypatch.setattr('zdock.inducedfit.replay.prepare_amber_charges', lambda *a, **k: pytest.fail('chemistry must not run'))
    with pytest.raises(ValueError, match='manifest SHA256'):
        load_replay(report)
