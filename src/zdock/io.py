"""Extended PDB (`*.pdb.ms`) reader for ZDOCK benchmark inputs.

ZDOCK / MDToolbox's `*.pdb.ms` format uses standard PDB columns 1-54 for the
ATOM header and coordinates, then appends 4 or 5 whitespace-separated extra
fields:

    ATOM      1  N   GLN     1      14.376  -5.731  -7.538  2     1 1.63    1PPI -0.15
                                                              ^     ^   ^      ^     ^
                                                             occ  temp rad  ACEtype charge

- Receptor files (`*_r_u.pdb.ms`) carry the ACE atom-type tag (e.g. `1PPI`).
- Ligand files (`*_l_u.pdb.ms`) and `complex.*.pdb` omit the tag and only
  carry occupancy, tempfactor, radius, charge.

Only `atomname` / `resname` / xyz are required downstream: per-atom radius,
atomtype id, and charge are re-derived via `zdock.atomtypes.*` to stay in
lock-step with the Julia reference. The extra columns are exposed for
diagnostic / comparison work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class PdbAtoms:
    """Parsed atom table from a `*.pdb.ms` or ZDOCK complex PDB file."""
    atomname: list[str]
    resname: list[str]
    chain: list[str]
    resseq: np.ndarray        # (N,) int64
    xyz: np.ndarray           # (N, 3) float64
    occupancy: np.ndarray     # (N,) int64
    tempfactor: np.ndarray    # (N,) int64
    radius_col: np.ndarray    # (N,) float32 — per-atom radius from PDB
    ace_type_col: list[str | None]  # ACE atom-type tag, None for ligand rows
    charge_col: np.ndarray    # (N,) float32 — per-atom partial charge

    def __len__(self) -> int:
        return len(self.atomname)


def parse_pdb_ms(path: str | Path) -> PdbAtoms:
    """Parse an extended PDB file (`*.pdb.ms` or a ZDOCK `complex.*.pdb`).

    Only `ATOM` / `HETATM` records are read; everything else is skipped. The
    columns up to index 54 are standard PDB fixed-width. The tail (>=54) is
    whitespace-split and must have either 4 or 5 tokens.

    Raises ValueError on an unrecognized tail length so corrupt inputs fail
    loudly rather than silently feeding garbage into the rest of the pipeline.
    """
    atomname: list[str] = []
    resname: list[str] = []
    chain: list[str] = []
    resseq: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    occupancy: list[int] = []
    tempfactor: list[int] = []
    radius_col: list[float] = []
    ace_type_col: list[str | None] = []
    charge_col: list[float] = []

    with open(path, "r") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            # Right-pad short lines so fixed-width slicing cannot IndexError.
            if len(line) < 54:
                raise ValueError(
                    f"{path}:{lineno}: ATOM record truncated "
                    f"(length {len(line)}, need >=54)"
                )
            atomname.append(line[12:16].strip())
            resname.append(line[17:20].strip())
            chain.append(line[21:22].strip())
            resseq.append(int(line[22:26]))
            xs.append(float(line[30:38]))
            ys.append(float(line[38:46]))
            zs.append(float(line[46:54]))

            tail = line[54:].split()
            if len(tail) == 5:
                occupancy.append(int(tail[0]))
                tempfactor.append(int(tail[1]))
                radius_col.append(float(tail[2]))
                ace_type_col.append(tail[3])
                charge_col.append(float(tail[4]))
            elif len(tail) == 4:
                occupancy.append(int(tail[0]))
                tempfactor.append(int(tail[1]))
                radius_col.append(float(tail[2]))
                ace_type_col.append(None)
                charge_col.append(float(tail[3]))
            else:
                raise ValueError(
                    f"{path}:{lineno}: expected 4 or 5 tail tokens, got {len(tail)}: {tail!r}"
                )

    if not atomname:
        raise ValueError(f"{path}: no ATOM/HETATM records found")

    xyz = np.stack([np.asarray(xs), np.asarray(ys), np.asarray(zs)], axis=-1)
    return PdbAtoms(
        atomname=atomname,
        resname=resname,
        chain=chain,
        resseq=np.asarray(resseq, dtype=np.int64),
        xyz=xyz,
        occupancy=np.asarray(occupancy, dtype=np.int64),
        tempfactor=np.asarray(tempfactor, dtype=np.int64),
        radius_col=np.asarray(radius_col, dtype=np.float32),
        ace_type_col=ace_type_col,
        charge_col=np.asarray(charge_col, dtype=np.float32),
    )


def parse_pdb_ms_many(paths: Iterable[str | Path]) -> list[PdbAtoms]:
    """Convenience wrapper that parses a sequence of files. Useful when
    loading a list of `complex.*.pdb` decoys."""
    return [parse_pdb_ms(p) for p in paths]
