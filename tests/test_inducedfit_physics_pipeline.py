"""Small continuous-objective integration checks, without native structures.

The analytic prior is a stationary SPD quadratic in CA-normalized amplitudes.
The intermolecular objective is the production screened charge potential. A
nonlinear spectator atom deliberately violates the tangent displacement bound.
"""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from zdock.inducedfit.physics_pipeline import (
    geometry, mixed_pool, pose_key, prepare_typed_cache, refine_physical_pose,
    typed_pair_grid, verify_physical_pool,
)
from zdock.inducedfit.typed_fft import build_probe_types
from zdock.inducedfit.typed_potential import (
    TypedParameters, TypedPotentialConfig, direct_pair_energy,
)


DTYPE = torch.float64
CONFIG = TypedPotentialConfig()
QUATERNIONS = torch.tensor([[0., 0., 0., 1.]], dtype=DTYPE)
LIGAND = torch.zeros((1, 3), dtype=DTYPE)


def parameters(charges):
    charges = torch.tensor(charges, dtype=DTYPE)
    return TypedParameters(torch.zeros_like(charges), torch.zeros_like(charges), charges)


class TinyQuadraticPrior:
    """Two CA-normalized tangents; one direction has a long atom lever arm."""

    def __init__(self, nonlinear=0.):
        self.reference = torch.tensor([[0., 0., 0.], [0., 30., 0.]], dtype=DTYPE)
        self.ca_indices = torch.tensor([0])
        self.basis = torch.zeros((2, 2, 3), dtype=DTYPE)
        self.basis[0, 0, 0], self.basis[1, 0, 1] = 1., 1.
        self.basis[0, 1, 0] = 10.
        self.stiffness = torch.eye(2, dtype=DTYPE)
        self.n_modes = 2
        self.diagnostics = {'baseline_projected_gradient': [0., 0.]}
        self.nonlinear = nonlinear
        self.curvature = torch.tensor([[0., 0., 0.], [1., 0., 0.]], dtype=DTYPE)

    def coordinates(self, amplitudes):
        return (self.reference + torch.einsum('m,mnc->nc', amplitudes, self.basis)
                + self.nonlinear * amplitudes[1].square() * self.curvature)

    def deformation_energy(self, amplitudes):
        return .5 * amplitudes @ self.stiffness @ amplitudes


def row(y, score=0., **extra):
    return dict(rotation_index=0, translation=[0., y, 0.],
                coarse_score=score, coarse_rigid_energy=score + .5, **extra)


def verify(pool, model, **kwargs):
    return verify_physical_pool(pool, QUATERNIONS, LIGAND, model,
                                parameters([1., 0.]), parameters([-1.]), CONFIG, **kwargs)


@pytest.mark.parametrize('interpolation', ['cardinal', 'mass'])
def test_small_pair_grid_contains_full_radial_support_for_both_interpolations(interpolation):
    # Without an explicit cutoff margin the automatic grid was only [-8, 8].
    # Cardinal interpolation rejected that grid, while mass mode silently
    # sampled only its truncated radial support. Both now get a complete grid.
    config = replace(CONFIG, grid_interpolation=interpolation)
    receptor = torch.zeros((1, 3), dtype=DTYPE)
    ligand_parameters = parameters([-1.])
    grid = typed_pair_grid(receptor, LIGAND, potential_cutoff=config.cutoff)
    assert grid.contains_support(receptor, config.cutoff)
    cache = prepare_typed_cache(receptor, parameters([1.]),
        build_probe_types(ligand_parameters), grid, config)
    assert bool(torch.isfinite(cache.rigid).all())


def test_mixed_pool_preserves_raw_peaks_and_fixed_source_quotas():
    raw = {'rigid': [row(i, -100. + i, source='rigid') for i in range(5)],
           'response': [row(i + 10, -200. + i, source='response') for i in range(5)]}
    original = deepcopy(raw)
    pool = mixed_pool(raw, 5)
    assert [p['source'] for p in pool].count('rigid') == 3
    assert [p['source'] for p in pool].count('response') == 2
    assert len(set(map(pose_key, pool))) == 5
    assert raw == original
    assert all(candidate is not p for p in pool for candidate in raw['rigid'] + raw['response'])
    assert len(raw['rigid']) == len(raw['response']) == 5


def test_mixed_pool_deduplicates_and_deterministically_backfills():
    raw = {'rigid': [row(0, -4), row(1, -3), row(2, -2)],
           'response': [row(0, -10), row(10, -9), row(11, -8)]}
    original = deepcopy(raw)
    pool = mixed_pool(raw, 4)
    assert [p['translation'][1] for p in pool] == [0, 1, 10, 2]
    assert pool[0]['coarse_score'] == -4
    assert len(set(map(pose_key, pool))) == 4
    assert mixed_pool(raw, 4) == pool
    assert raw == original


def test_verified_ranking_uses_continuous_energy_instead_of_coarse_scores():
    model = TinyQuadraticPrior()
    pool = [row(8., -1e6, coarse_amplitudes=[1e3, 1e3]),
            row(4., 1e6, coarse_amplitudes=[-1e3, -1e3])]
    original = deepcopy(pool)
    result = verify(pool, model, rigid=True)
    assert [p['translation'][1] for p in result] == [4., 8.]
    assert [p['coarse_score'] for p in result] == [1e6, -1e6]
    for candidate in result:
        direct = direct_pair_energy(model.reference,
            LIGAND + torch.tensor(candidate['translation'], dtype=DTYPE),
            parameters([1., 0.]), parameters([-1.]), CONFIG)
        assert candidate['verified_energy'] == pytest.approx(float(direct), abs=1e-13)
        assert candidate['rigid_energy'] == candidate['verified_energy']
        assert candidate['verified_amplitudes'] == [0., 0.]
        assert candidate['verified_gain'] == 0.
    assert pool == original


def test_atom_constraint_recovers_valid_motion_lost_by_conservative_ball():
    model = TinyQuadraticPrior()
    atom = verify([row(4.)], model, constraint='atom', radius=.5, max_atom=.2)[0]
    ball = verify([row(4.)], model, constraint='ball', radius=.5, max_atom=.2)[0]
    assert not atom['fallback'] and not ball['fallback']
    assert atom['proposed_amplitudes'][1] == pytest.approx(.2, abs=1e-8)
    assert ball['proposed_amplitudes'][1] == pytest.approx(.02, abs=1e-10)
    assert atom['verified_energy'] < ball['verified_energy'] - .1
    assert atom['solver']['finite_geometry_verified'] is False
    for result in (atom, ball):
        amplitude = torch.tensor(result['verified_amplitudes'], dtype=DTYPE)
        feasible, _, atom_displacement = geometry(model, amplitude, .5, .2)
        assert feasible and atom_displacement <= .2 + 1e-9
        assert result['verified_gain'] == pytest.approx(
            result['verified_energy'] - result['rigid_energy'], abs=1e-13)


def test_nonlinear_atom_motion_requires_geometry_verification_and_shrink():
    model = TinyQuadraticPrior(nonlinear=40.)
    result = verify([row(4.)], model, radius=.5, max_atom=.2)[0]
    proposal = torch.tensor(result['proposed_amplitudes'], dtype=DTYPE)
    tangent_displacement = torch.einsum('m,mnc->nc', proposal, model.basis).norm(dim=-1).max()
    assert float(tangent_displacement) <= .2 + 1e-9
    assert result['trials'][0]['maximum_atom_displacement'] > .2
    assert result['trials'][0]['geometry_feasible'] is False
    assert 'energy' not in result['trials'][0]
    assert not result['fallback'] and result['accepted_scale'] < 1.
    assert geometry(model, torch.tensor(result['verified_amplitudes'], dtype=DTYPE), .5, .2)[0]
    assert result['verified_gain'] < 0


def test_no_feasible_nonlinear_trial_falls_back_to_the_exact_rigid_pose():
    model = TinyQuadraticPrior(nonlinear=1e8)
    result = verify([row(4.)], model, radius=.5, max_atom=.2)[0]
    assert len(result['trials']) == 6
    assert all(not trial['geometry_feasible'] for trial in result['trials'])
    assert result['fallback']
    assert result['verified_amplitudes'] == [0., 0.]
    assert result['verified_gain'] == 0.
    assert result['verified_energy'] == result['rigid_energy']


def test_feasible_tangent_gain_is_rejected_when_true_prior_energy_increases():
    class QuarticPrior(TinyQuadraticPrior):
        def deformation_energy(self, amplitudes):
            # The gradient and Hessian at zero still exactly match the model.
            return super().deformation_energy(amplitudes) + 1e8 * amplitudes.pow(4).sum()

    result = verify([row(4.)], QuarticPrior(), radius=.5, max_atom=.2)[0]
    assert result['quadratic_gain'] < 0.
    assert all(trial['geometry_feasible'] for trial in result['trials'])
    assert all(trial['predicted_gain'] < 0. < trial['actual_gain'] for trial in result['trials'])
    assert result['fallback']
    assert result['verified_energy'] == result['rigid_energy']
    assert result['verified_amplitudes'] == [0., 0.]


@pytest.mark.parametrize('flexible', [False, True])
def test_refinement_starts_at_zero_and_decreases_same_energy_with_true_geometry(flexible):
    model = TinyQuadraticPrior(nonlinear=40.)
    pose = row(4., -1e5, coarse_amplitudes=[100., 100.],
               verified_amplitudes=[.1, .1], verified_energy=-1e10)
    record, receptor, ligand = refine_physical_pose(pose, QUATERNIONS[0], LIGAND,
        model, parameters([1., 0.]), parameters([-1.]), CONFIG,
        flexible=flexible, radius=.15, max_atom=.1, max_translation=.15,
        max_rotation=.08, max_iterations=6, max_evaluations=32)
    initial_energy = direct_pair_energy(model.reference, LIGAND + LIGAND.new_tensor([0., 4., 0.]),
                                        parameters([1., 0.]), parameters([-1.]), CONFIG)
    assert record['energy_history'][0] == pytest.approx(float(initial_energy), abs=1e-13)
    assert record['accepted_steps'] > 0
    assert all(after < before for before, after in zip(record['energy_history'], record['energy_history'][1:]))
    amplitude = torch.tensor(record['amplitudes'], dtype=DTYPE)
    actual_energy = direct_pair_energy(receptor, ligand, parameters([1., 0.]), parameters([-1.]), CONFIG)
    actual_energy += model.deformation_energy(amplitude)
    assert record['energy'] == pytest.approx(float(actual_energy), abs=1e-13)
    assert record['energy'] < float(initial_energy)
    feasible, ca_rmsd, maximum_atom = geometry(model, amplitude, .15, .1)
    assert feasible
    assert record['ca_rmsd'] == pytest.approx(ca_rmsd)
    assert record['maximum_atom_displacement'] == pytest.approx(maximum_atom)
    assert float(torch.tensor(record['translation_increment']).norm()) <= .15 + 1e-8
    assert float(torch.tensor(record['rotation_increment']).norm()) <= .08 + 1e-8
    assert record['energy_evaluations'] <= 32
    torch.testing.assert_close(receptor, model.coordinates(amplitude))
    if not flexible:
        torch.testing.assert_close(amplitude, torch.zeros_like(amplitude))


def test_zero_refinement_budget_returns_zero_amplitudes_despite_proposal():
    model = TinyQuadraticPrior()
    record, receptor, _ = refine_physical_pose(
        row(4., coarse_amplitudes=[100., 100.], verified_amplitudes=[.1, .1]),
        QUATERNIONS[0], LIGAND, model, parameters([1., 0.]), parameters([-1.]), CONFIG,
        max_iterations=0)
    assert record['amplitudes'] == [0., 0.]
    assert record['accepted_steps'] == 0
    assert record['energy_evaluations'] == 1
    torch.testing.assert_close(receptor, model.reference)


def test_ball_ablation_remains_conservative_during_continuous_refinement():
    model = TinyQuadraticPrior()
    records = {}
    for constraint in ('ball', 'atom'):
        records[constraint], _, _ = refine_physical_pose(
            row(4.), QUATERNIONS[0], LIGAND, model,
            parameters([1., 0.]), parameters([-1.]), CONFIG,
            constraint=constraint, max_atom=.2, radius=.5, max_translation=0.,
            max_rotation=0., max_iterations=4, max_evaluations=32)
    assert records['ball']['amplitude_radius'] == pytest.approx(.02)
    assert torch.tensor(records['ball']['amplitudes']).norm() <= .020001
    assert records['atom']['amplitudes'][1] > .02
    assert records['atom']['energy'] < records['ball']['energy']


def test_stiff_prior_preconditioning_does_not_freeze_rigid_translation():
    model = TinyQuadraticPrior()
    model.stiffness = torch.diag(torch.tensor([2e4, 7e4], dtype=DTYPE))
    records = {}
    for preconditioner in ('fixed', 'physical'):
        records[preconditioner], _, _ = refine_physical_pose(
            row(4.), QUATERNIONS[0], LIGAND, model,
            parameters([1., 0.]), parameters([-1.]), CONFIG,
            preconditioner=preconditioner, max_translation=2.,
            max_iterations=2, max_evaluations=24)
    translation = {key:torch.tensor(value['translation_increment']).norm()
                   for key,value in records.items()}
    assert translation['physical'] > 10 * translation['fixed']
    assert records['physical']['energy'] < records['fixed']['energy']
    for record in records.values():
        assert all(after < before for before,after in
                   zip(record['energy_history'], record['energy_history'][1:]))
        assert record['energy_evaluations'] <= 24
