"""Optional Amber ff14SB charges with every added hydrogen retained.

This is a force-field *preparation* adapter, not an Amber energy backend.
Only OpenMM's hydrogen-placement minimization is used; pre-existing heavy
atoms are fixed and checked.  Missing heavy atoms, caps, links and disulfides
are not repaired.  Geometry radii used by ddock remain separate from Amber LJ.

API reference: https://docs.openmm.org/latest/api-python/generated/
openmm.app.modeller.Modeller.html#openmm.app.modeller.Modeller.addHydrogens
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import math
from pathlib import Path
import random
import threading
import xml.etree.ElementTree as ET

import torch

from .kinematics import TorsionKinematics
from .pdb_input import PreparedPDB, validate_torsion_input
from .types import AtomKey, MonomerInput


FORCEFIELD_FILE = "amber14/protein.ff14SB.xml"
GEOMETRIC_RADII_ANGSTROM = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20}
_HYDROGEN_PREPARATION_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True, eq=False)
class ChargedProtein:
    monomer: MonomerInput
    heavy_monomer: MonomerInput
    heavy_indices: torch.Tensor
    bonds: tuple[tuple[int, int], ...]
    provenance: dict


def _load_openmm():
    try:
        openmm = importlib.import_module("openmm")
        app = importlib.import_module("openmm.app")
        unit = importlib.import_module("openmm.unit")
    except ImportError as error:
        raise ImportError("Amber charge preparation requires the optional OpenMM chemistry dependency") from error
    return openmm, app, unit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forcefield_sources(path: Path) -> dict[str, str]:
    """Hash the fixed XML and any includes; do not silently change FF files."""
    sources = {}

    def visit(current):
        current = current.resolve()
        if str(current) in sources:
            return
        sources[str(current)] = _sha256(current)
        for include in ET.parse(current).getroot().findall("Include"):
            child = current.parent / include.attrib["file"]
            if not child.is_file():
                raise ValueError(f"Cannot resolve force-field include {child}")
            visit(child)

    visit(path)
    return sources


def _require_complete_heavy_atoms(prepared: PreparedPDB, ff_path: Path):
    root = ET.parse(ff_path).getroot()
    elements = {node.attrib["name"]: node.attrib["element"] for node in root.findall("./AtomTypes/Type")}
    templates = {node.attrib["name"]: node for node in root.findall("./Residues/Residue")}
    residues: dict[tuple[str, int, str, str], list[int]] = {}
    for index, key in enumerate(prepared.monomer.atom_keys):
        residues.setdefault((key.chain, key.residue_number, key.insertion_code, key.residue_name), []).append(index)
    ordered = sorted(residues, key=lambda key: (key[0], key[1], key[2]))
    for position, residue_id in enumerate(ordered):
        template_name = "HIE" if residue_id[3] == "HIS" else residue_id[3]
        if template_name not in templates:
            raise ValueError(f"No standard ff14SB heavy-atom template for {residue_id}")
        expected = {atom.attrib["name"] for atom in templates[template_name].findall("Atom")
                    if elements[atom.attrib["type"]] != "H"}
        if position == len(ordered) - 1:
            expected.add("OXT")
        actual = {prepared.monomer.atom_keys[i].atom_name for i in residues[residue_id]}
        if expected != actual:
            raise ValueError(
                f"Incomplete or unsupported heavy atoms at {residue_id}: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}; "
                "heavy atoms are never added or repaired"
            )
    return ordered, residues


def prepare_amber_charges(prepared: PreparedPDB, *, pH: float = 7.0, seed: int = 0) -> ChargedProtein:
    """Prepare a complete single chain with fixed heavy atoms and all H charges.

    The returned ``monomer`` is the ONLY full electrostatic input.  The
    ``heavy_monomer`` retains original heavy order and its individual heavy
    charges solely for geometry/kinematics: summing or scoring it alone drops
    hydrogen charge and is NOT an equivalent electrostatic representation.

    The Reference platform and Python hydrogen-placement RNG seed are fixed.
    Python RNG state is restored; calls through this adapter are serialized.
    Reproducibility is checked for a pinned OpenMM version, not promised across
    platforms/versions.  No simulation or whole-protein minimization is run.
    """
    if not isinstance(prepared, PreparedPDB):
        raise TypeError("prepared must be a PreparedPDB retaining topology provenance")
    if isinstance(pH, bool) or not math.isfinite(pH) or not 0 <= pH <= 14:
        raise ValueError("pH must be finite and between 0 and 14")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    validate_torsion_input(prepared)
    # Includes backbone continuity and geometric disulfide/crosslink screening.
    TorsionKinematics.from_monomer(prepared.monomer)
    openmm, app, unit = _load_openmm()
    data_dir = Path(app.__file__).resolve().parent / "data"
    ff_path = data_dir / FORCEFIELD_FILE
    ff_sources = _forcefield_sources(ff_path)
    ordered, residue_indices = _require_complete_heavy_atoms(prepared, ff_path)
    original = prepared.monomer
    topology = app.Topology()
    chain = topology.addChain(id=ordered[0][0])
    topology_order = []
    for residue_id in ordered:
        _, number, insertion_code, name = residue_id
        residue = topology.addResidue(name, chain, id=str(number), insertionCode=insertion_code)
        for index in residue_indices[residue_id]:
            key = original.atom_keys[index]
            element = app.element.get_by_symbol(key.atom_name[0])
            topology.addAtom(key.atom_name, element, residue, id=str(index + 1))
            topology_order.append(index)
    topology.createStandardBonds()
    # Do NOT call createDisulfideBonds: unsupported links must not be inferred
    # and silently made part of the model.
    positions = [openmm.Vec3(*row) for row in original.xyz[topology_order].detach().cpu().double().tolist()] * unit.angstrom
    forcefield = app.ForceField(str(ff_path))
    modeller = app.Modeller(topology, positions)
    platform = openmm.Platform.getPlatformByName("Reference")
    with _HYDROGEN_PREPARATION_LOCK:
        random_state = random.getstate()
        random.seed(seed)
        try:
            variants = modeller.addHydrogens(forcefield, pH=float(pH), platform=platform)
        except Exception as error:
            raise ValueError(f"OpenMM addHydrogens failed without heavy-atom repair: {error}") from error
        finally:
            random.setstate(random_state)
    try:
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff,
                                          constraints=None, rigidWater=False, removeCMMotion=False)
    except Exception as error:
        raise ValueError(f"OpenMM ff14SB parameter assignment failed: {error}") from error
    atoms = list(modeller.topology.atoms())
    if system.getNumParticles() != len(atoms) or any(atom.element is None for atom in atoms):
        raise ValueError("Extra particles/virtual sites are unsupported; atom/particle map is not bijective")
    forces = [force for force in system.getForces() if isinstance(force, openmm.NonbondedForce)]
    if len(forces) != 1 or forces[0].getNumParticles() != len(atoms):
        raise ValueError("Expected exactly one NonbondedForce with one charge per atom")
    particle_charges = [float(forces[0].getParticleParameters(i)[0].value_in_unit(unit.elementary_charge))
                        for i in range(len(atoms))]
    keys, elements = [], []
    for atom in atoms:
        residue = atom.residue
        keys.append(AtomKey(residue.chain.id, int(residue.id), residue.insertionCode or "",
                            residue.name, atom.name))
        symbol = atom.element.symbol
        if symbol not in GEOMETRIC_RADII_ANGSTROM:
            raise ValueError(f"Unsupported prepared element {symbol}")
        elements.append(symbol)
    if len(set(keys)) != len(keys):
        raise ValueError("Prepared full-atom AtomKey mapping is not unique")
    full_indices = {key: index for index, key in enumerate(keys)}
    heavy_keys = {key for key, element in zip(keys, elements) if element != "H"}
    if heavy_keys != set(original.atom_keys):
        raise ValueError("Hydrogen addition changed the heavy-atom identity set")
    heavy_indices = torch.tensor([full_indices[key] for key in original.atom_keys],
                                  dtype=torch.long, device=original.xyz.device)
    openmm_xyz = torch.tensor(modeller.positions.value_in_unit(unit.angstrom), dtype=torch.float64)
    if not bool(torch.isfinite(openmm_xyz).all()):
        raise ValueError("OpenMM returned nonfinite atom positions")
    # Check in double precision BEFORE casting to a possibly float32 input.
    # Otherwise rounding could hide movement of an existing heavy atom.
    raw_heavy_change = float((openmm_xyz[heavy_indices.cpu()] - original.xyz.detach().cpu().double()).abs().max())
    # A->nm->A conversion can lose a few double-precision ulps.  Only after
    # checking that change is numerical roundoff do we restore exact input
    # bits, so downstream geometry identity is literal rather than approximate.
    roundoff_tolerance = max(1e-10, 16 * torch.finfo(torch.float64).eps
                             * max(1., float(original.xyz.abs().max())))
    if raw_heavy_change > roundoff_tolerance:
        raise ValueError(f"OpenMM moved existing heavy atoms by {raw_heavy_change:.6g} A")
    xyz = openmm_xyz.to(dtype=original.xyz.dtype, device=original.xyz.device)
    xyz[heavy_indices] = original.xyz
    charges = original.xyz.new_tensor(particle_charges)
    radii = original.xyz.new_tensor([GEOMETRIC_RADII_ANGSTROM[symbol] for symbol in elements])
    full = MonomerInput(xyz, radii, charges, tuple(keys))
    heavy = MonomerInput(original.xyz, radii[heavy_indices], charges[heavy_indices], original.atom_keys)
    bonds = tuple(sorted({tuple(sorted((a.index, b.index))) for a, b in modeller.topology.bonds()}))
    degree = [0] * len(atoms)
    for a, b in bonds:
        degree[a] += 1
        degree[b] += 1
    if any(degree[i] != 1 for i, element in enumerate(elements) if element == "H"):
        raise ValueError("Every retained hydrogen must have exactly one covalent parent")
    net_charge = math.fsum(particle_charges)
    if not math.isfinite(net_charge) or abs(net_charge - round(net_charge)) > 1e-5:
        raise ValueError(f"Prepared isolated protein has a noninteger net charge: {net_charge}")
    residue_variants = [{"chain": residue_id[0], "residue_number": residue_id[1],
                         "insertion_code": residue_id[2], "residue_name": residue_id[3], "variant": variant}
                        for residue_id, variant in zip(ordered, variants, strict=True)]
    provenance = {
        "chemistry_version": 1, "input_sha256": prepared.source_sha256,
        "input_provenance": prepared.provenance,
        "charge_model": "OpenMM Amber ff14SB NonbondedForce particle charges in elementary_charge",
        "forcefield_file": FORCEFIELD_FILE, "forcefield_xml_sha256": ff_sources,
        "openmm_version": str(openmm.__version__), "platform": "Reference",
        "hydrogens_xml_sha256": _sha256(data_dir / "hydrogens.xml"),
        "residues_xml_sha256": _sha256(data_dir / "residues.xml"),
        "modeller_source_sha256": _sha256(Path(importlib.import_module("openmm.app.modeller").__file__)),
        "pH": float(pH), "seed": seed, "seed_policy": "Python random state saved, seeded and restored around addHydrogens",
        "variants": residue_variants, "hydrogen_policy": "add_missing_H_with_fixed_heavy_atoms_keep_all_H_and_charges",
        "geometry_relaxation": "OpenMM hydrogen-placement minimization only; existing heavy atoms fixed",
        "heavy_coordinate_roundtrip_max_abs_change_angstrom": raw_heavy_change,
        "heavy_coordinate_roundtrip_tolerance_angstrom": roundoff_tolerance,
        "heavy_coordinate_output_max_abs_change_angstrom": float((full.xyz[heavy_indices] - original.xyz).abs().max()),
        "heavy_coordinate_roundoff_policy": "verify roundoff then restore exact input heavy-coordinate bits",
        "n_heavy_atoms": len(original.xyz), "n_hydrogens": elements.count("H"), "n_atoms": len(atoms),
        "net_charge_e": net_charge, "heavy_charge_sum_e": math.fsum(particle_charges[i] for i in heavy_indices.cpu().tolist()),
        "hydrogen_charge_sum_e": math.fsum(q for q, element in zip(particle_charges, elements) if element == "H"),
        "heavy_subset_role": "geometry_only_not_a_complete_electrostatic_model",
        "radii_angstrom": GEOMETRIC_RADII_ANGSTROM.copy(), "radii_role": "geometric_effective_field_not_Amber_LJ_sigma",
        "system_definition": "NoCutoff, constraints=None, rigidWater=False, removeCMMotion=False; parameter extraction only",
        "heavy_atom_mapping": [{"original_heavy_index": i, "full_atom_index": int(index), "atom_key": asdict(key)}
                               for i, (index, key) in enumerate(zip(heavy_indices.cpu().tolist(), original.atom_keys))],
        "reference_documentation": "https://docs.openmm.org/latest/api-python/generated/openmm.app.modeller.Modeller.html",
    }
    return ChargedProtein(full, heavy, heavy_indices, bonds, provenance)
