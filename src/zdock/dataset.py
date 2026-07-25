"""Self-contained decoy-generation and labeling pipeline.

This is the missing data-plumbing layer referenced (but never shipped) by
``data.py``'s docstring. It builds a training/eval dataset **without any
external ZDOCK binary**: the repository's own FFT pose search
(:func:`zdock.search.docking_search`) proposes candidate poses, and the
differentiable DockQ / RMSD metrics (:mod:`zdock.dockq`) label them
against the native (bound) placement.

Frame convention (critical — everything must live in one frame):

* ``rec_dec = rec_xyz_raw - rec_com``           (receptor decentered)
* ``lig_ref = orient(lig_xyz_raw, iface_mass)`` (what the FFT search rotates)
* ``native_lig = lig_xyz_raw - rec_com``        (native pose in rec_dec frame)
* a decoy is ``rotate(lig_ref, q) + t`` in the same rec_dec frame.

Atom ordering is preserved across ``lig_ref`` and ``native_lig`` (both are
rigid transforms of ``lig_xyz_raw``), so per-atom RMSD / DockQ are
well-defined.

Positives are guaranteed to exist in the candidate set by mixing:

1. global FFT search decoys (realistic ZDOCK-style candidates), and
2. a near-native rotation cone around the analytic native orientation
   ``q*`` (so the set always contains acceptable/medium/high poses).

This matches the decoy recipe in the research proposal (global random +
native-neighborhood; hard negatives are mined later, during training).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .atomtypes import (
    _ATOMTYPE_LUT,
    _VDW_RADIUS,
    charge_score as default_charge_score,
    iface_ij,
    set_atomtype_id,
    set_charge,
    set_radius,
)
from .dockq import dockq_batch, ligand_rmsd_to_native
from .geom import decenter, orient
from .io import parse_pdb_ms
from .rotation_grid import kabsch_quaternion, random_quaternions, rotation_cone
from .sasa import compute_sasa
from .search import _rotate_batch, docking_search


@dataclass
class SimpleAtoms:
    """Minimal atom table (only the fields feature-derivation needs)."""

    atomname: list[str]
    resname: list[str]
    xyz: np.ndarray            # (N, 3) float64


def _element_of(atomname: str) -> str | None:
    """First non-digit character of a PDB atom name (its element letter)."""
    for ch in atomname:
        if not ch.isdigit():
            return ch
    return None


def parse_pdb_plain(path: str | Path) -> SimpleAtoms:
    """Parse a standard PDB (e.g. DB5.5 ``*_r_b.pdb``), keeping only the
    first model's heavy protein atoms that our atom-type / radius tables
    recognise. Hydrogens, HETATM, waters, alternate conformers (altLoc
    other than blank/'A'), and any ``(resname, atomname)`` pair outside the
    ZDOCK atom-type LUT are dropped. Returns the surviving atoms plus a
    drop count via the ``n_dropped`` attribute on the result.
    """
    atomname: list[str] = []
    resname: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    n_dropped = 0
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue
            alt = line[16]
            if alt not in (" ", "A"):
                n_dropped += 1
                continue
            a = line[12:16].strip()
            r = line[17:20].strip()
            a_norm = "O" if a == "OXT" else a
            elem = _element_of(a)
            if (r, a_norm) not in _ATOMTYPE_LUT or elem not in _VDW_RADIUS:
                n_dropped += 1
                continue
            atomname.append(a)
            resname.append(r)
            xs.append(float(line[30:38]))
            ys.append(float(line[38:46]))
            zs.append(float(line[46:54]))
    if not atomname:
        raise ValueError(f"{path}: no usable protein ATOM records found")
    out = SimpleAtoms(
        atomname=atomname,
        resname=resname,
        xyz=np.stack([np.asarray(xs), np.asarray(ys), np.asarray(zs)], axis=-1),
    )
    out.n_dropped = n_dropped  # type: ignore[attr-defined]
    return out


@dataclass
class PreparedProtein:
    """Everything needed to score / search / label one complex."""

    name: str
    # receptor (decentered)
    rec_xyz: torch.Tensor          # (N_rec, 3)
    rec_radius: torch.Tensor
    rec_sasa: torch.Tensor
    rec_atomtype_id: torch.Tensor
    rec_charge_id: torch.Tensor
    # ligand reference (oriented+decentered — the FFT-search input)
    lig_ref: torch.Tensor          # (N_lig, 3)
    lig_radius: torch.Tensor
    lig_sasa: torch.Tensor
    lig_atomtype_id: torch.Tensor
    lig_charge_id: torch.Tensor
    # native ligand placement in the receptor-decentered frame
    native_lig: torch.Tensor       # (N_lig, 3)
    # analytic native pose (q*, t*) that maps lig_ref -> native_lig
    q_star: torch.Tensor           # (4,)
    t_star: torch.Tensor           # (3,)

    #: every field that holds a tensor — used by :meth:`to` / :meth:`state_dict`
    TENSOR_FIELDS = (
        "rec_xyz", "rec_radius", "rec_sasa", "rec_atomtype_id", "rec_charge_id",
        "lig_ref", "lig_radius", "lig_sasa", "lig_atomtype_id", "lig_charge_id",
        "native_lig", "q_star", "t_star",
    )

    @property
    def n_rec(self) -> int:
        return int(self.rec_xyz.shape[0])

    @property
    def n_lig(self) -> int:
        return int(self.lig_ref.shape[0])

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> "PreparedProtein":
        """Return a copy with every tensor field moved to ``device``.

        Integer id fields keep their integer dtype; only floating-point
        fields follow ``dtype``. Streaming pipelines use this to hold the
        whole corpus on CPU (or on disk) and page one complex onto the
        accelerator at a time.
        """
        moved = {}
        for f in self.TENSOR_FIELDS:
            t = getattr(self, f)
            want = dtype if (dtype is not None and t.is_floating_point()) else None
            moved[f] = t.to(device=device, dtype=want, non_blocking=non_blocking)
        return PreparedProtein(name=self.name, **moved)

    def cpu(self) -> "PreparedProtein":
        return self.to("cpu")

    def state_dict(self) -> dict:
        """Plain ``{name, tensors...}`` dict for ``torch.save`` disk caching."""
        d = {"name": self.name}
        for f in self.TENSOR_FIELDS:
            d[f] = getattr(self, f)
        return d

    @classmethod
    def from_state_dict(cls, d: dict) -> "PreparedProtein":
        return cls(name=d["name"], **{f: d[f] for f in cls.TENSOR_FIELDS})


def _atoms_to_features(atoms, device, dtype):
    xyz = torch.as_tensor(atoms.xyz, device=device, dtype=dtype)
    atomtype_id = set_atomtype_id(atoms.resname, atoms.atomname).to(device)
    charge_id = set_charge(atoms.resname, atoms.atomname).to(device)
    radius = set_radius(atoms.atomname, device=device, dtype=dtype)
    sasa = compute_sasa(xyz, radius)
    return xyz, radius, sasa, atomtype_id, charge_id


def prepare_protein_from_pdbms(
    name: str,
    rec_path: str | Path,
    lig_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PreparedProtein:
    """Read receptor/ligand ``*.pdb.ms`` files and prepare a protein."""
    return prepare_protein(
        name, parse_pdb_ms(rec_path), parse_pdb_ms(lig_path),
        device=device, dtype=dtype,
    )


def prepare_protein_from_pdb(
    name: str,
    rec_path: str | Path,
    lig_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PreparedProtein:
    """Read receptor/ligand standard ``*.pdb`` files (e.g. DB5.5 bound
    constituents) and prepare a protein."""
    return prepare_protein(
        name, parse_pdb_plain(rec_path), parse_pdb_plain(lig_path),
        device=device, dtype=dtype,
    )


def prepare_protein(
    name: str,
    rec_atoms,
    lig_atoms,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PreparedProtein:
    """Derive all features and reference frames from parsed atom tables
    (anything exposing ``.xyz`` / ``.resname`` / ``.atomname``)."""
    rec_xyz_raw, rec_radius, rec_sasa, rec_atomtype_id, rec_charge_id = (
        _atoms_to_features(rec_atoms, device, dtype)
    )
    lig_xyz_raw, lig_radius, lig_sasa, lig_atomtype_id, lig_charge_id = (
        _atoms_to_features(lig_atoms, device, dtype)
    )

    rec_com = rec_xyz_raw.mean(dim=0, keepdim=True)
    rec_dec = rec_xyz_raw - rec_com

    # Ligand reference frame = orient() with the same iface-based inertia
    # weights docking_score_elec / docking_search use internally.
    iface_mat = iface_ij(device=device, dtype=dtype)
    lig_mass = iface_mat[lig_atomtype_id - 1, 0]
    lig_ref = orient(lig_xyz_raw, mass=lig_mass)

    # Native ligand placement in the rec_dec frame (atom order preserved).
    native_lig = lig_xyz_raw - rec_com

    # Analytic native pose that carries lig_ref onto native_lig:
    #   rotate(lig_ref, q*) + t* == native_lig
    # kabsch_quaternion decenters both sides, so q* aligns the orientations;
    # t* then matches the (unweighted) centroids.
    q_star = kabsch_quaternion(lig_ref, native_lig, device=device, dtype=dtype)
    rotated = _rotate_batch(lig_ref, q_star.unsqueeze(0))[0]
    t_star = (native_lig - rotated).mean(dim=0)

    return PreparedProtein(
        name=name,
        rec_xyz=rec_dec,
        rec_radius=rec_radius,
        rec_sasa=rec_sasa,
        rec_atomtype_id=rec_atomtype_id,
        rec_charge_id=rec_charge_id,
        lig_ref=lig_ref,
        lig_radius=lig_radius,
        lig_sasa=lig_sasa,
        lig_atomtype_id=lig_atomtype_id,
        lig_charge_id=lig_charge_id,
        native_lig=native_lig,
        q_star=q_star,
        t_star=t_star,
    )


def generate_decoys(
    prot: PreparedProtein,
    *,
    alpha: torch.Tensor,
    iface_ij_flat: torch.Tensor,
    beta: torch.Tensor,
    charge_score_lut: torch.Tensor,
    n_random_rot: int = 3000,
    n_cone: int = 400,
    cone_deg: float = 25.0,
    ntop: int = 2000,
    spacing: float = 3.0,
    rot_chunk_size: int = 32,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the FFT search + near-native cone and return candidate poses.

    Returns ``(poses, scores)`` where ``poses`` is ``(F, N_lig, 3)`` in the
    receptor-decentered frame and ``scores`` is ``(F,)`` — the raw docking
    score assigned by the FFT search with the given parameters.
    """
    device = prot.rec_xyz.device
    dtype = prot.rec_xyz.dtype

    q_random = random_quaternions(n_random_rot, seed=seed, device=device, dtype=dtype)
    q_cone = rotation_cone(prot.q_star, n_cone, cone_deg=cone_deg, seed=seed,
                           device=device, dtype=dtype)
    quats = torch.cat([q_random, q_cone], dim=0)

    result = docking_search(
        prot.rec_xyz, prot.rec_radius, prot.rec_sasa,
        prot.rec_atomtype_id, prot.rec_charge_id,
        prot.lig_ref, prot.lig_radius, prot.lig_sasa,
        prot.lig_atomtype_id, prot.lig_charge_id,
        quats,
        alpha=alpha, iface_ij_flat=iface_ij_flat, beta=beta,
        charge_score_lut=charge_score_lut,
        spacing=spacing, ntop=ntop, rot_chunk_size=rot_chunk_size,
    )
    fft_poses = (
        _rotate_batch(prot.lig_ref, quats[result.quat_indices])
        + result.translations.unsqueeze(1)
    )

    # Explicit near-native poses at t* to guarantee strong positives exist
    # (the FFT translation grid may not land exactly on the native offset).
    near_poses = (
        _rotate_batch(prot.lig_ref, q_cone) + prot.t_star.unsqueeze(0).unsqueeze(0)
    )

    poses = torch.cat([fft_poses, near_poses], dim=0)
    fft_scores = result.scores
    return poses, fft_scores


def label_decoys(
    prot: PreparedProtein,
    poses: torch.Tensor,
    *,
    contact_cutoff: float = 5.0,
    iface_cutoff: float = 8.0,
    pose_chunk: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-pose (RMSD, DockQ) against the native placement.

    DockQ builds a dense ``(F, N_rec, N_lig)`` pairwise-distance tensor, so
    for realistic protein sizes we must chunk over poses to keep peak
    memory bounded.
    """
    rmsd = ligand_rmsd_to_native(prot.native_lig, poses)
    dockq_parts = []
    for s in range(0, poses.shape[0], pose_chunk):
        comp = dockq_batch(
            prot.rec_xyz, poses[s:s + pose_chunk], prot.native_lig,
            contact_cutoff=contact_cutoff, iface_cutoff=iface_cutoff,
        )
        dockq_parts.append(comp.dockq)
    return rmsd, torch.cat(dockq_parts, dim=0)
