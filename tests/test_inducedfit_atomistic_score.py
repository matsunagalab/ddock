"""Real OpenMM endpoint identities, not native-pose accuracy assertions."""
from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from zdock.inducedfit.atomistic_score import AtomisticScorer, PROTOCOL, _topology
from zdock.inducedfit.chemistry import prepare_amber_charges
from zdock.inducedfit.hydrogen_kinematics import HydrogenatedKinematics
from zdock.inducedfit.pdb_input import read_pdb_monomer
from test_inducedfit_chemistry import make_dipeptide_pdb


@pytest.fixture(scope="module")
def protein(tmp_path_factory):
    pytest.importorskip("openmm")
    path = make_dipeptide_pdb(tmp_path_factory.mktemp("atomistic") / "gly_ala.pdb")
    return prepare_amber_charges(read_pdb_monomer(path, chain="A", charge_policy="zero_diagnostic"), seed=23)


@pytest.fixture(scope="module")
def scorer(protein):
    with AtomisticScorer.from_proteins(protein, protein) as result:
        yield result


def pose(protein):
    return protein.monomer.xyz.clone(), protein.monomer.xyz + protein.monomer.xyz.new_tensor([1., 2., 7.])


def test_fixed_model_has_complete_parameter_provenance_and_unmodified_input(scorer, protein):
    before = protein.monomer.xyz.clone()
    rec, lig = pose(protein)
    result = scorer.score(rec, lig)
    assert PROTOCOL["implicit_solvent"] == "implicit/obc2.xml"
    assert scorer.provenance["complex_monomer_particle_parameters_exact"]
    assert scorer.provenance["original_charges_validated"]
    assert len(scorer.provenance["system_xml_sha256"]) == 3
    assert torch.equal(protein.monomer.xyz, before)
    assert result.receptor_strain.total == 0.
    assert result.ligand_strain.total == pytest.approx(0., abs=1e-9)
    assert result.interaction.bonded == pytest.approx(0., abs=1e-9)
    assert abs(result.interaction.gbsa) > .01
    assert result.endpoint_total == pytest.approx(result.interaction.total, abs=1e-9)
    assert result.clashes["severe_heavy_lj_pairs"] == 0


def test_coulomb_and_lj_cross_interactions_match_independent_pair_sum(scorer, protein):
    rec, lig = pose(protein)
    result = scorer.score(rec, lig)
    p, q = scorer._particle_parameters
    distance_nm = np.linalg.norm(rec.numpy()[:, None] - lig.numpy()[None], axis=-1) / 10
    sigma = (p[:, 1, None] + q[None, :, 1]) / 2
    epsilon = np.sqrt(p[:, 2, None] * q[None, :, 2])
    ratio6 = (sigma / distance_nm)**6
    lj = np.sum(4 * epsilon * (ratio6**2 - ratio6))
    coulomb = np.sum(138.93545764438198 * p[:, 0, None] * q[None, :, 0] / distance_nm)
    assert result.interaction.vdw == pytest.approx(lj, abs=1e-9)
    assert result.interaction.coulomb == pytest.approx(coulomb, abs=1e-9)


def test_split_groups_sum_to_untouched_official_forcefield(scorer, protein):
    import openmm
    from openmm import app, unit
    rec, lig = pose(protein)
    forcefield = app.ForceField("amber14/protein.ff14SB.xml", "implicit/obc2.xml")
    system = forcefield.createSystem(_topology((protein, protein), app), nonbondedMethod=app.NoCutoff,
        constraints=None, rigidWater=False, removeCMMotion=False, soluteDielectric=1.,
        solventDielectric=78.5, implicitSolventKappa=0./unit.nanometer, sasaMethod="ACE")
    integrator = openmm.VerletIntegrator(.001 * unit.picoseconds)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(torch.cat((rec, lig)).numpy() * unit.angstrom)
    total = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert scorer.score(rec, lig).complex_energy.total == pytest.approx(total, abs=1e-9)
    del context, integrator


def test_rigid_invariance_and_partner_exchange(scorer, protein):
    rec, lig = pose(protein)
    original = scorer.score(rec, lig)
    theta = .63
    rotation = rec.new_tensor([[math.cos(theta), -math.sin(theta), 0],
                              [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    shifted = scorer.score(rec @ rotation.T + 42., lig @ rotation.T + 42.)
    swapped = scorer.score(lig, rec)
    for term in ("bonded", "vdw", "coulomb", "gbsa", "total"):
        assert getattr(original.interaction, term) == pytest.approx(getattr(shifted.interaction, term), abs=1e-8)
        assert getattr(original.interaction, term) == pytest.approx(getattr(swapped.interaction, term), abs=1e-8)
    assert shifted.receptor_strain.total == pytest.approx(0., abs=1e-8)
    assert shifted.ligand_strain.total == pytest.approx(0., abs=1e-8)


def test_nonlinear_strain_and_endpoint_subtraction_identity(scorer, protein):
    kinematics = HydrogenatedKinematics.from_prepared(protein)
    angles = protein.monomer.xyz.new_full((kinematics.n_torsions,), .12)
    deformed = kinematics.coordinates(angles)
    _, lig = pose(protein)
    result = scorer.score(deformed, lig)
    assert abs(result.receptor_strain.total) > .01
    assert result.endpoint_total == pytest.approx(
        result.complex_energy.total - sum(e.total for e in scorer.reference_energies), abs=1e-9)
    assert result.interaction.bonded == pytest.approx(0., abs=1e-9)
    assert result.receptor_strain.bonded != 0  # torsional energy is allowed to change


def test_clashes_are_reported_not_softened(scorer, protein):
    rec = protein.monomer.xyz
    result = scorer.score(rec, rec + .15)
    assert result.clashes["severe_heavy_lj_pairs"] >= len(protein.heavy_indices)
    assert result.clashes["minimum_heavy_distance_angstrom"] < .3
    assert result.interaction.vdw > 1e6


def test_invalid_coordinates_and_provenance_fail_explicitly(scorer, protein):
    rec, lig = pose(protein)
    with pytest.raises(ValueError, match="full-atom coordinates"):
        scorer.score(rec[protein.heavy_indices], lig)
    with pytest.raises(ValueError, match="finite full-atom"):
        scorer.score(rec * float("nan"), lig)
    bad = replace(protein, provenance={**protein.provenance, "forcefield_xml_sha256": {"amber14/protein.ff14SB.xml": "0"*64}})
    with pytest.raises(ValueError, match="SHA256"):
        AtomisticScorer.from_proteins(bad, protein)
    wrong = replace(protein, monomer=replace(protein.monomer, charges=protein.monomer.charges + .1))
    with pytest.raises(ValueError, match="charges differ"):
        AtomisticScorer.from_proteins(wrong, protein)


def test_closed_context_cannot_return_a_stale_score(protein):
    local = AtomisticScorer.from_proteins(protein, protein)
    local.close()
    with pytest.raises(RuntimeError, match="closed"):
        local.score(*pose(protein))
