"""Fixed, reference-free Amber ff14SB/OBC2 endpoint energy diagnostic.

All energies are kJ/mol.  This is *not* a binding free energy: no entropy,
ensemble averaging, protonation changes, minimization, or fitted weights are
included.  H atoms and covalent topology come from the original ChargedProtein,
never a rounded pose PDB.  Interactions use SAME-coordinate subtraction; strain
is reported separately against each prepared apo monomer.  Effective ddock
field radii/energies do not enter this model.

The single protocol is fixed in PROTOCOL (OpenMM's OBC2/ACE, Amber igb=5):
https://docs.openmm.org/latest/userguide/application/02_running_sims.html#implicit-solvent
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import math
from pathlib import Path

import numpy as np
import torch

from .chemistry import ChargedProtein, FORCEFIELD_FILE, _forcefield_sources, _load_openmm


PROTOCOL = {
    "version": "ff14sb_obc2_ace_endpoint_v1",
    "forcefield": FORCEFIELD_FILE,
    "implicit_solvent": "implicit/obc2.xml",
    "sasa_method": "ACE",
    "solute_dielectric": 1.0,
    "solvent_dielectric": 78.5,
    "kappa_per_nm": 0.0,
    "nonbonded_method": "NoCutoff",
    "platform": "Reference",
    "energy_unit": "kJ/mol",
    "coordinate_unit": "angstrom",
    "minimization": False,
    "ranking": "interaction.total; endpoint_total is a separate unweighted diagnostic",
    "interaction_definition": "E_AB(x_A,x_B)-E_A(x_A)-E_B(x_B)",
    "strain_definition": "E_A(x_A)-E_A(apo_A), E_B(x_B)-E_B(apo_B)",
    "endpoint_definition": "interaction.total+receptor_strain.total+ligand_strain.total",
    "clash_definition": "intermolecular r < 0.75 * 2^(1/6) * mixed_Amber_sigma, positive mixed epsilon only",
    "clash_ratio_threshold": 0.75,
    "limitations": "single structure potential energy; no entropy or ensemble averaging; fixed input protonation; protein-only input policy inherited",
}


@dataclass(frozen=True, slots=True)
class EnergyComponents:
    bonded: float
    vdw: float
    coulomb: float
    gbsa: float
    total: float

    @classmethod
    def from_terms(cls, values):
        values = tuple(float(x) for x in values)
        return cls(*values, math.fsum(values))

    def subtract(self, *others):
        return EnergyComponents.from_terms(
            getattr(self, name) - math.fsum(getattr(other, name) for other in others)
            for name in ("bonded", "vdw", "coulomb", "gbsa")
        )


@dataclass(frozen=True, slots=True)
class AtomisticScore:
    interaction: EnergyComponents
    receptor_strain: EnergyComponents
    ligand_strain: EnergyComponents
    endpoint_total: float
    complex_energy: EnergyComponents
    receptor_energy: EnergyComponents
    ligand_energy: EnergyComponents
    clashes: dict

    def to_dict(self):
        return asdict(self)


def _topology(proteins, app):
    """Rebuild only the original bonds, preserving all-atom particle order."""
    topology = app.Topology()
    for side, protein in enumerate(proteins):
        chain = topology.addChain(id=str(side))
        residues = {}
        atoms = []
        previous = None
        for key in protein.monomer.atom_keys:
            rid = (key.chain, key.residue_number, key.insertion_code, key.residue_name)
            if rid != previous:
                if rid in residues:
                    raise ValueError("Prepared all-atom residues must be contiguous")
                residues[rid] = topology.addResidue(key.residue_name, chain,
                    id=str(key.residue_number), insertionCode=key.insertion_code)
            symbol = key.atom_name[0]
            if symbol not in {"C", "N", "O", "S", "H"}:
                raise ValueError(f"Unsupported prepared protein element for {key}")
            atoms.append(topology.addAtom(key.atom_name, app.element.get_by_symbol(symbol), residues[rid]))
            previous = rid
        seen = set()
        for a, b in protein.bonds:
            pair = tuple(sorted((int(a), int(b))))
            if pair in seen or pair[0] == pair[1] or pair[0] < 0 or pair[1] >= len(atoms):
                raise ValueError("Invalid or duplicate prepared covalent bond")
            seen.add(pair)
            topology.addBond(atoms[pair[0]], atoms[pair[1]])
    return topology


def _particle_parameters(system, openmm, unit):
    nonbonded = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    gbsa = [f for f in system.getForces() if isinstance(f, openmm.CustomGBForce)]
    if len(nonbonded) != 1 or len(gbsa) != 1:
        raise ValueError("Expected exactly one Amber NonbondedForce and one OBC2 CustomGBForce")
    nb, gb = nonbonded[0], gbsa[0]
    n = system.getNumParticles()
    if nb.getNumParticles() != n or gb.getNumParticles() != n:
        raise ValueError("Incomplete implicit-solvent particle parameters")
    parameters = np.asarray([
        [q.value_in_unit(unit.elementary_charge), sigma.value_in_unit(unit.nanometer),
         epsilon.value_in_unit(unit.kilojoule_per_mole)]
        for q, sigma, epsilon in (nb.getParticleParameters(i) for i in range(n))
    ], dtype=np.float64)
    gb_parameters = np.asarray([list(gb.getParticleParameters(i)) for i in range(n)], dtype=np.float64)
    if not np.isfinite(parameters).all() or not np.isfinite(gb_parameters).all():
        raise ValueError("Nonfinite force-field parameters")
    if not np.allclose(parameters[:, 0], gb_parameters[:, 0], rtol=0, atol=1e-12):
        raise ValueError("OBC2 and Amber particle charges disagree")
    return parameters, gb_parameters


def _split_force_groups(system, openmm, unit):
    """Replace the single NB force by its exactly additive LJ/Coulomb parts."""
    bonded_types = (openmm.HarmonicBondForce, openmm.HarmonicAngleForce, openmm.PeriodicTorsionForce)
    nb_index = None
    for index, force in enumerate(system.getForces()):
        if isinstance(force, bonded_types):
            force.setForceGroup(0)
        elif isinstance(force, openmm.CustomGBForce):
            force.setForceGroup(3)
        elif isinstance(force, openmm.NonbondedForce):
            nb_index = index
        else:
            raise ValueError(f"Unsupported force in fixed Amber/OBC2 diagnostic: {type(force).__name__}")
    if nb_index is None:
        raise ValueError("Missing Amber NonbondedForce")
    original = system.getForce(nb_index)
    if original.getNonbondedMethod() != openmm.NonbondedForce.NoCutoff:
        raise ValueError("Fixed endpoint diagnostic requires NoCutoff")
    if original.getNumGlobalParameters() or original.getNumParticleParameterOffsets() or original.getNumExceptionParameterOffsets():
        raise ValueError("Parameterized NonbondedForce offsets are unsupported")
    copies = [openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(original)) for _ in range(2)]
    lj, electrostatic = copies
    for i in range(original.getNumParticles()):
        q, sigma, epsilon = original.getParticleParameters(i)
        lj.setParticleParameters(i, 0 * unit.elementary_charge, sigma, epsilon)
        electrostatic.setParticleParameters(i, q, sigma, 0 * unit.kilojoule_per_mole)
    for i in range(original.getNumExceptions()):
        a, b, qprod, sigma, epsilon = original.getExceptionParameters(i)
        lj.setExceptionParameters(i, a, b, 0 * unit.elementary_charge**2, sigma, epsilon)
        electrostatic.setExceptionParameters(i, a, b, qprod, sigma, 0 * unit.kilojoule_per_mole)
    system.removeForce(nb_index)
    for group, force in enumerate(copies, 1):
        force.setForceGroup(group)
        system.addForce(force)


class _EnergyContext:
    def __init__(self, system, openmm, unit):
        self.unit = unit
        self.n_atoms = system.getNumParticles()
        self.integrator = openmm.VerletIntegrator(.001 * unit.picoseconds)
        self.context = openmm.Context(system, self.integrator, openmm.Platform.getPlatformByName("Reference"))

    def energy(self, xyz):
        if self.context is None:
            raise RuntimeError("Atomistic scorer has been closed")
        array = torch.as_tensor(xyz).detach().to(device="cpu", dtype=torch.float64).numpy()
        if array.shape != (self.n_atoms, 3) or not np.isfinite(array).all():
            raise ValueError(f"Expected finite full-atom coordinates with shape ({self.n_atoms}, 3)")
        self.context.setPositions(array * self.unit.angstrom)
        values = [self.context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
                  .value_in_unit(self.unit.kilojoule_per_mole) for i in range(4)]
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("Nonfinite Amber/OBC2 energy; overlapping or invalid coordinates")
        return EnergyComponents.from_terms(values)

    def close(self):
        self.context = None
        self.integrator = None


class AtomisticScorer:
    """Cached CPU Reference contexts; not thread safe; no coordinate changes.

    Construct with ``from_proteins`` and reuse ``score`` for frozen poses.  Close
    when done (also supports a context manager).  Full H coordinates are required.
    """

    @classmethod
    def from_proteins(cls, receptor: ChargedProtein, ligand: ChargedProtein):
        openmm, app, unit = _load_openmm()
        data = Path(app.__file__).parent / "data"
        ff_path = data / FORCEFIELD_FILE
        implicit_path = data / PROTOCOL["implicit_solvent"]
        hashes = _forcefield_sources(ff_path)
        current_digest = hashlib.sha256(ff_path.read_bytes()).hexdigest()
        for name, protein in (("receptor", receptor), ("ligand", ligand)):
            if protein.provenance.get("forcefield_file") != FORCEFIELD_FILE:
                raise ValueError(f"{name} was not prepared with the fixed ff14SB model")
            source_hashes = protein.provenance.get("forcefield_xml_sha256", {})
            matching = [digest for path, digest in source_hashes.items()
                        if str(path).endswith(FORCEFIELD_FILE)]
            if matching != [current_digest] or sorted(source_hashes.values()) != sorted(hashes.values()):
                raise ValueError(f"{name} force-field source SHA256 differs from preparation")
            if protein.provenance.get("openmm_version") != openmm.__version__:
                raise ValueError(f"{name} OpenMM version differs from preparation")
        forcefield = app.ForceField(str(ff_path), str(implicit_path))
        systems, parameters, gb_parameters = [], [], []
        for proteins in ((receptor,), (ligand,), (receptor, ligand)):
            topology = _topology(proteins, app)
            system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff,
                constraints=None, rigidWater=False, removeCMMotion=False,
                soluteDielectric=1.0, solventDielectric=78.5,
                implicitSolventKappa=0.0 / unit.nanometer, sasaMethod="ACE")
            p, gb = _particle_parameters(system, openmm, unit)
            expected = torch.cat([protein.monomer.charges for protein in proteins]).detach().cpu().double().numpy()
            charge_atol = max(1e-12, 4 * max(torch.finfo(protein.monomer.charges.dtype).eps for protein in proteins))
            if not np.allclose(p[:, 0], expected, rtol=0, atol=charge_atol):
                raise ValueError("Reconstructed Amber charges differ from original full-H preparation")
            systems.append(system)
            parameters.append(p)
            gb_parameters.append(gb)
        if not np.array_equal(parameters[2], np.concatenate(parameters[:2])) or not np.array_equal(gb_parameters[2], np.concatenate(gb_parameters[:2])):
            raise ValueError("Complex and monomer force-field particle parameters disagree")
        instance = cls()
        instance.receptor, instance.ligand = receptor, ligand
        instance._particle_parameters = parameters[:2]
        instance._contexts = []
        try:
            serialized = [openmm.XmlSerializer.serialize(system) for system in systems]
            for system in systems:
                _split_force_groups(system, openmm, unit)
                instance._contexts.append(_EnergyContext(system, openmm, unit))
            instance.reference_energies = (
                instance._contexts[0].energy(receptor.monomer.xyz),
                instance._contexts[1].energy(ligand.monomer.xyz),
            )
        except Exception:
            instance.close()
            raise
        from openmm.app.internal import customgbforces
        instance.provenance = {
            "protocol": dict(PROTOCOL), "openmm_version": openmm.__version__,
            "forcefield_xml_sha256": {**hashes, **_forcefield_sources(implicit_path)},
            "customgbforces_source_sha256": hashlib.sha256(Path(inspect.getfile(customgbforces)).read_bytes()).hexdigest(),
            "system_xml_sha256": {key: hashlib.sha256(value.encode()).hexdigest()
                                  for key, value in zip(("receptor", "ligand", "complex"), serialized)},
            "input_provenance": {"receptor": receptor.provenance, "ligand": ligand.provenance},
            "reference_energies": {"receptor": asdict(instance.reference_energies[0]),
                                   "ligand": asdict(instance.reference_energies[1])},
            "original_charges_validated": True, "complex_monomer_particle_parameters_exact": True,
        }
        return instance

    def _clashes(self, receptor_xyz, ligand_xyz):
        rec = torch.as_tensor(receptor_xyz).detach().cpu().double()
        lig = torch.as_tensor(ligand_xyz).detach().cpu().double()
        rec_p, lig_p = (torch.from_numpy(p) for p in self._particle_parameters)
        rec_heavy = torch.zeros(len(rec), dtype=torch.bool)
        lig_heavy = torch.zeros(len(lig), dtype=torch.bool)
        rec_heavy[self.receptor.heavy_indices.cpu()] = True
        lig_heavy[self.ligand.heavy_indices.cpu()] = True
        count = heavy_count = 0
        minimum = minimum_heavy = math.inf
        minimum_ratio = math.inf
        for start in range(0, len(rec), 256):
            distance = torch.cdist(rec[start:start + 256], lig, compute_mode="donot_use_mm_for_euclid_dist")
            minimum = min(minimum, float(distance.min()))
            heavy = rec_heavy[start:start + 256, None] & lig_heavy[None, :]
            if bool(heavy.any()):
                minimum_heavy = min(minimum_heavy, float(distance[heavy].min()))
            sigma = (rec_p[start:start + 256, 1, None] + lig_p[None, :, 1]) * 5.  # nm mean -> A
            positive = (rec_p[start:start + 256, 2, None] > 0) & (lig_p[None, :, 2] > 0) & (sigma > 0)
            ratio = distance / (2**(1/6) * sigma)
            if bool(positive.any()):
                minimum_ratio = min(minimum_ratio, float(ratio[positive].min()))
            clash = positive & (ratio < PROTOCOL["clash_ratio_threshold"])
            count += int(clash.sum())
            heavy_count += int((clash & heavy).sum())
        return {"minimum_distance_angstrom": minimum, "minimum_heavy_distance_angstrom": minimum_heavy,
                "minimum_lj_distance_ratio": minimum_ratio, "severe_lj_pairs": count,
                "severe_heavy_lj_pairs": heavy_count,
                "threshold_ratio": PROTOCOL["clash_ratio_threshold"]}

    def score(self, receptor_xyz, ligand_xyz) -> AtomisticScore:
        rec = self._contexts[0].energy(receptor_xyz)
        lig = self._contexts[1].energy(ligand_xyz)
        both = self._contexts[2].energy(torch.cat((torch.as_tensor(receptor_xyz), torch.as_tensor(ligand_xyz))))
        interaction = both.subtract(rec, lig)
        strains = (rec.subtract(self.reference_energies[0]), lig.subtract(self.reference_energies[1]))
        return AtomisticScore(interaction, *strains,
            math.fsum((interaction.total, strains[0].total, strains[1].total)), both, rec, lig,
            self._clashes(receptor_xyz, ligand_xyz))

    def close(self):
        for context in self._contexts:
            context.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
