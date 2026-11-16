"""Reference-free file input, common search geometry and evaluation export.

The manifest deliberately cannot contain a native path or benchmark labels.
It is an input contract, not proof that the selected structures are truly apo;
that provenance must be audited separately before an accuracy experiment.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import torch

from .pdb_input import read_pdb_monomer, write_pdb_monomer
from .types import GridSpec, MonomerInput


def load_pair_manifest(path: Path) -> tuple[dict, dict]:
    path = Path(path)
    record = json.loads(path.read_text())
    if not isinstance(record, dict) or set(record) != {
        "format_version", "system_id", "input_kind", "split", "receptor", "ligand"
    }:
        raise ValueError("pair manifest has missing or unrecognized fields")
    if type(record['format_version']) is not int or record['format_version'] != 1:
        raise ValueError("unsupported pair manifest version")
    for name in ('system_id', 'input_kind', 'split'):
        if not isinstance(record[name], str) or not record[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    loaded = {}
    for side in ('receptor', 'ligand'):
        item = record[side]
        if not isinstance(item, dict) or set(item) != {'pdb', 'chain', 'sha256'}:
            raise ValueError(f"{side} has missing or unrecognized fields")
        if not isinstance(item['pdb'], str) or not item['pdb']:
            raise ValueError(f"{side}.pdb must be a nonempty path")
        if not isinstance(item['chain'], str) or len(item['chain']) != 1:
            raise ValueError(f"{side}.chain must be one PDB character")
        digest = item['sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f"{side}.sha256 must be a lowercase SHA256")
        source = (path.parent / item['pdb']).resolve()
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"{side} source SHA256 mismatch")
        # Zero values are temporary geometry input; production fields are built
        # only after explicit all-hydrogen Amber preparation in the runner.
        loaded[side] = read_pdb_monomer(source, chain=item['chain'], charge_policy='zero_diagnostic')
    return record, loaded


def centered_protein(protein):
    """Translate all atoms and heavy geometry by the same heavy-atom centroid."""
    center = protein.heavy_monomer.xyz.mean(0)
    def moved(m):
        return MonomerInput(m.xyz - center, m.radii, m.charges, m.atom_keys)
    return replace(protein, monomer=moved(protein.monomer),
                   heavy_monomer=moved(protein.heavy_monomer)), center


def rotation_sample(count: int, seed: int, *, dtype=torch.float64) -> torch.Tensor:
    """Seeded uniform S3 samples; identity is included explicitly as sample 0.

    A small sample is an integration smoke test, not global angular coverage.
    """
    if type(count) is not int or count < 1:
        raise ValueError('rotation count must be positive')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    q = torch.randn((count, 4), generator=generator, dtype=dtype)
    q /= q.norm(dim=-1, keepdim=True)
    q[0] = q.new_tensor([0., 0., 0., 1.])
    return q


def pair_grid(receptor, ligand, parameters, *, spacing: float,
              max_atom_displacement: float, max_translation: float) -> GridSpec:
    """Fixed cubic grid accommodating every ligand rotation and contact pose.

    Bounding spheres are deliberately conservative. Support and motion margin
    are defined from the two inputs, without a reference interface or crop.
    """
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError('spacing must be positive and finite')
    if any(not math.isfinite(x) or x < 0 for x in (max_atom_displacement, max_translation)):
        raise ValueError('motion limits must be nonnegative and finite')
    radii = torch.cat((receptor.radii, ligand.radii))
    support = max(float((radii.max() + parameters.surface_width)
                        / math.sqrt(2 * math.log(2)) * parameters.occupancy_cutoff_sigma),
                  parameters.charge_sigma * parameters.charge_cutoff_sigma)
    rec_extent = float(receptor.xyz.norm(dim=-1).max())
    lig_extent = float(ligand.xyz.norm(dim=-1).max())
    # A ligand center can reach the receptor surface in any direction while
    # its far side and full compact field remain inside the grid.
    extent = rec_extent + 2 * lig_extent + support + max_atom_displacement + max_translation + 4.
    half_cells = math.ceil(extent / spacing)
    return GridSpec((2 * half_cells + 1,) * 3, spacing, (-half_cells * spacing,) * 3)


def export_heavy_pair(directory: Path, receptor_input, ligand_input,
                      receptor_xyz: torch.Tensor, ligand_xyz: torch.Tensor,
                      receptor_center: torch.Tensor) -> dict:
    """Export both partners in the receptor input frame; combined chains R/L.

    xyz is in the centered docking frame, in original heavy input atom order.
    In particular, never add the ligand's original centroid to docked positions.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    paths = {}
    lines = []
    serial = 1
    for side, chain, prepared, xyz in (
        ('receptor', 'R', receptor_input, receptor_xyz),
        ('ligand', 'L', ligand_input, ligand_xyz),
    ):
        target = directory / f'{side}.pdb'
        write_pdb_monomer(target, prepared, xyz=xyz + receptor_center)
        paths[side] = str(target)
        for line in target.read_text().splitlines():
            if line.startswith('ATOM  '):
                if serial > 99999:
                    raise ValueError('combined PDB exceeds atom serial capacity')
                lines.append(f'{line[:6]}{serial:5d}{line[11:21]}{chain}{line[22:]}')
                serial += 1
        lines.append('TER')
    combined = directory / 'complex.pdb'
    combined.write_text('\n'.join(lines + ['END', '']))
    paths['complex'] = str(combined)
    return paths
