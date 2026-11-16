"""The frozen grid audit must compare a common complete-support domain."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from zdock.inducedfit.fields import FieldParameters, support_radii

spec = importlib.util.spec_from_file_location('grid_audit', Path(__file__).resolve().parents[1] / 'scripts/audit_inducedfit_grid.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_fixed_cube_contains_support_for_every_spacing_phase():
    xyz = torch.tensor([[-2.3, 0., 3.1], [4.4, -5.2, 0.]], dtype=torch.float64)
    radii = xyz.new_tensor([1.7, 1.5])
    poses = [SimpleNamespace(receptor_xyz=xyz, ligand_xyz=xyz + 10.),
             SimpleNamespace(receptor_xyz=xyz + .2, ligand_xyz=xyz - 6.)]
    grids, region = audit.common_grids(poses, radii, radii, FieldParameters())
    assert len(grids) == 4
    cube_lo = xyz.new_tensor(region['cube_lower_angstrom'])
    cube_hi = xyz.new_tensor(region['cube_upper_angstrom'])
    for grid in grids.values():
        lower, upper = grid.bounds(dtype=xyz.dtype)
        assert (lower >= cube_lo).all() and (upper < cube_hi).all()
        for pose in poses:
            for points in (pose.receptor_xyz, pose.ligand_xyz):
                support = support_radii(radii, FieldParameters())[:, None]
                assert (points - support >= lower).all() and (points + support <= upper).all()


def test_rank_comparison_reports_reversal_and_ignores_ties():
    rows = [{'id': str(i), 'scores': {'a': {'total': x}, 'b': {'total': y}}}
            for i, (x, y) in enumerate([(1., 3.), (1., 3.), (2., 2.), (3., 1.)])]
    comparison = audit.rank_comparison(rows, 'a', 'b')
    assert comparison['spearman_average_exact_ties'] == -1.
    assert comparison['inverted_pairs'] == comparison['comparable_nontied_pairs'] == 5
    assert comparison['pairs_tied_in_either'] == 1
