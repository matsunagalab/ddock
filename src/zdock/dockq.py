"""DockQ-v2 quality metric for 2-chain protein-protein poses.

DockQ (Basu & Wallner 2016; Mirabello & Wallner 2024 v2) combines three
components into a single [0, 1] quality score:

    DockQ = (1/3) · ( Fnat
                    + 1 / (1 + (iRMSD / 1.5)²)
                    + 1 / (1 + (LRMSD / 8.5)²) )

- `Fnat`: fraction of native atom-pair contacts preserved in the pose
  (native contact = atoms within 5 Å). High Fnat ⇒ interface chemistry
  matches the reference.
- `iRMSD`: RMSD of "interface atoms" between pose and reference. An
  atom is in the interface if its nearest opposing-chain atom is
  within 8 Å in the reference. In our setup the receptor is fixed
  across all poses (single-receptor search), so the conventional
  Kabsch alignment step is the identity and we drop it — see the
  assertion inside `dockq_batch`.
- `LRMSD`: whole-ligand RMSD between pose and reference. Since the
  receptor frame is shared, this reduces to a direct RMSD.

The 2024 v2 refresh introduces multi-interface aggregation for >2
chain complexes; for the 2-chain dimers in BM4 the formula is
identical to v1 on the single interface, and this module implements
that single-interface case exactly.

For atom-level Fnat / iRMSD we deliberately use atomic distances (not
residue-level Cα or backbone subsets) because this codebase stores
atom arrays without residue labels. This is an **approximation** of
the CAPRI-canonical metric; it preserves the ranking property (better
pose ⇒ higher DockQ) but absolute values are not reproducible against
published residue-level DockQ numbers without adding residue/backbone
metadata to the dataset.

All distance computations are batched on GPU. For F poses and
(N_rec, N_lig) atoms, the per-pose cost is a single
(F, N_rec, N_lig) pairwise distance tensor per quantity; native
contacts are computed once (F-independent).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


_DOCKQ_CONTACT_CUTOFF = 5.0   # Å, native contact definition
_DOCKQ_IFACE_CUTOFF = 8.0     # Å, interface atom definition
_DOCKQ_IRMSD_SCALE = 1.5      # Å
_DOCKQ_LRMSD_SCALE = 8.5      # Å


@dataclass
class DockQComponents:
    """Per-pose DockQ and its three components.

    Shape conventions:
        fnat, i_rmsd, l_rmsd, dockq: (F,) float tensors, same dtype /
        device as the input pose coordinates.
    """
    fnat: torch.Tensor
    i_rmsd: torch.Tensor
    l_rmsd: torch.Tensor
    dockq: torch.Tensor


def native_contacts(
    rec_xyz: torch.Tensor,             # (N_rec, 3)
    native_lig_xyz: torch.Tensor,      # (N_lig, 3)
    *,
    cutoff: float = _DOCKQ_CONTACT_CUTOFF,
) -> torch.Tensor:
    """Return a boolean (N_rec, N_lig) mask — True where the atom pair
    is within `cutoff` in the reference.

    This is the "native contact set" used to compute Fnat for arbitrary
    poses against the same reference.
    """
    if rec_xyz.ndim != 2 or rec_xyz.shape[1] != 3:
        raise ValueError(f"rec_xyz must be (N_rec, 3), got {tuple(rec_xyz.shape)}")
    if native_lig_xyz.ndim != 2 or native_lig_xyz.shape[1] != 3:
        raise ValueError(
            f"native_lig_xyz must be (N_lig, 3), got {tuple(native_lig_xyz.shape)}"
        )
    d2 = torch.cdist(rec_xyz, native_lig_xyz)    # (N_rec, N_lig)
    return d2 < cutoff


def interface_atom_masks(
    rec_xyz: torch.Tensor,             # (N_rec, 3)
    native_lig_xyz: torch.Tensor,      # (N_lig, 3)
    *,
    cutoff: float = _DOCKQ_IFACE_CUTOFF,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (rec_mask, lig_mask) of boolean tensors.

    - `rec_mask[i]`: True if rec atom i has any native-lig atom within
      `cutoff`.
    - `lig_mask[j]`: True if native-lig atom j has any rec atom within
      `cutoff`.

    Same cutoff for both sides; can be tightened if needed by the
    caller.
    """
    d = torch.cdist(rec_xyz, native_lig_xyz)      # (N_rec, N_lig)
    contact = d < cutoff
    return contact.any(dim=1), contact.any(dim=0)


def dockq_batch(
    rec_xyz: torch.Tensor,                     # (N_rec, 3)
    lig_xyz_batch: torch.Tensor,               # (F, N_lig, 3)
    native_lig_xyz: torch.Tensor,              # (N_lig, 3)
    *,
    contact_cutoff: float = _DOCKQ_CONTACT_CUTOFF,
    iface_cutoff: float = _DOCKQ_IFACE_CUTOFF,
) -> DockQComponents:
    """Batched DockQ over F poses sharing a single receptor.

    **Receptor-frame assumption**: because we do single-receptor FFT
    search, the receptor coordinates are shared across every pose and
    the reference. The conventional Kabsch alignment of receptor
    atoms between pose and reference is therefore the identity — we
    skip it. If you pass poses from a workflow that allows the
    receptor to move, this function will under-report iRMSD.

    Inputs:
        rec_xyz: (N_rec, 3) receptor atoms, decentered or raw — must
            share frame with all poses and with native_lig_xyz.
        lig_xyz_batch: (F, N_lig, 3) ligand poses in the same frame.
        native_lig_xyz: (N_lig, 3) reference ("pseudo-native") ligand
            coords in the same frame.

    Returns `DockQComponents(fnat, i_rmsd, l_rmsd, dockq)`, each (F,).
    """
    if lig_xyz_batch.ndim != 3 or lig_xyz_batch.shape[-1] != 3:
        raise ValueError(
            f"lig_xyz_batch must be (F, N_lig, 3), got {tuple(lig_xyz_batch.shape)}"
        )
    if rec_xyz.device != lig_xyz_batch.device:
        raise ValueError(
            f"rec_xyz on {rec_xyz.device}, lig_xyz_batch on "
            f"{lig_xyz_batch.device} — must match"
        )
    if native_lig_xyz.shape != lig_xyz_batch.shape[1:]:
        raise ValueError(
            f"native_lig_xyz {tuple(native_lig_xyz.shape)} must match "
            f"lig_xyz_batch[0] {tuple(lig_xyz_batch.shape[1:])}"
        )

    F = lig_xyz_batch.shape[0]
    device = lig_xyz_batch.device
    dtype = lig_xyz_batch.dtype

    # ---- native contact set (computed once) -----------------------------
    native_contact_mask = native_contacts(
        rec_xyz, native_lig_xyz, cutoff=contact_cutoff,
    )                                               # (N_rec, N_lig) bool
    n_native = native_contact_mask.sum().to(dtype)
    # Fnat denominator; fall back to 1 to avoid div-by-zero for complexes
    # with no detectable native contacts (would only happen with wildly
    # wrong cutoffs or a broken pseudo-native). The dockq score itself
    # then becomes iRMSD + LRMSD only.
    n_native_safe = torch.where(
        n_native > 0, n_native, torch.ones_like(n_native),
    )

    # ---- per-pose pose contacts → Fnat ----------------------------------
    # Pairwise distances between receptor (fixed) and every pose ligand.
    # rec_xyz: (N_rec, 3); lig_xyz_batch: (F, N_lig, 3)
    # Use torch.cdist in batched form: cdist broadcasts a leading batch
    # dim on its inputs if both have them, so expand rec_xyz.
    d_pose = torch.cdist(
        rec_xyz.unsqueeze(0).expand(F, -1, -1),
        lig_xyz_batch,
    )                                               # (F, N_rec, N_lig)
    pose_contact_mask = d_pose < contact_cutoff     # (F, N_rec, N_lig)
    intersect = pose_contact_mask & native_contact_mask.unsqueeze(0)
    fnat = intersect.sum(dim=(-2, -1)).to(dtype) / n_native_safe  # (F,)
    if n_native.item() == 0:
        fnat = torch.zeros_like(fnat)

    # ---- interface atom masks (from native reference) -------------------
    rec_iface, lig_iface = interface_atom_masks(
        rec_xyz, native_lig_xyz, cutoff=iface_cutoff,
    )
    # Interface RMSD: receptor fixed ⇒ rec atoms contribute 0; only
    # ligand interface atoms contribute per pose.
    #   iRMSD² = mean_over_iface_lig(||pose_lig_i − native_lig_i||²)
    # If no interface atoms (shouldn't happen for bound complexes),
    # fall back to LRMSD so DockQ stays well-defined.
    lig_iface_count = lig_iface.sum()
    if lig_iface_count.item() > 0:
        lig_iface_idx = lig_iface.nonzero(as_tuple=True)[0]
        diff = lig_xyz_batch[:, lig_iface_idx, :] - native_lig_xyz[lig_iface_idx]
        i_rmsd = diff.pow(2).sum(-1).mean(-1).sqrt()
    else:
        diff = lig_xyz_batch - native_lig_xyz
        i_rmsd = diff.pow(2).sum(-1).mean(-1).sqrt()

    # ---- ligand RMSD (whole ligand, shared-receptor frame) --------------
    l_rmsd = (lig_xyz_batch - native_lig_xyz).pow(2).sum(-1).mean(-1).sqrt()

    # ---- DockQ --------------------------------------------------------
    scale_i = torch.tensor(_DOCKQ_IRMSD_SCALE, dtype=dtype, device=device)
    scale_l = torch.tensor(_DOCKQ_LRMSD_SCALE, dtype=dtype, device=device)
    term_i = 1.0 / (1.0 + (i_rmsd / scale_i).pow(2))
    term_l = 1.0 / (1.0 + (l_rmsd / scale_l).pow(2))
    dockq = (fnat + term_i + term_l) / 3.0

    return DockQComponents(
        fnat=fnat, i_rmsd=i_rmsd, l_rmsd=l_rmsd, dockq=dockq,
    )


def ligand_rmsd_to_native(
    native_lig_xyz: torch.Tensor,      # (N_lig, 3)
    cand_lig_xyz: torch.Tensor,        # (F, N_lig, 3) or (N_lig, 3)
    *,
    atom_mask: torch.Tensor | None = None,   # (N_lig,) bool subset
) -> torch.Tensor:
    """Ligand RMSD of candidate pose(s) against the native placement.

    Because the whole pipeline is single-receptor FFT search, the receptor
    frame is shared between the native and every candidate, so there is no
    Kabsch superposition — this is a direct per-atom RMSD (the same L-RMSD
    that `dockq_batch` reports). Everything that computes "RMSD from the h5"
    routes through here: `native_lig_xyz = h5[prot]["lig_xyz_native"]` and
    `cand_lig_xyz` is any set of candidate poses in the same raw frame /
    atom ordering (e.g. the FFT decoys from `05_fft_generate_decoys.py`).

    Args:
        native_lig_xyz: (N_lig, 3) native ligand coords (raw `_l_u`).
        cand_lig_xyz: (F, N_lig, 3) batch of candidate poses, or a single
            (N_lig, 3) pose.
        atom_mask: optional (N_lig,) boolean mask to restrict the RMSD to a
            subset of atoms (e.g. interface or Cα atoms). Default: all atoms.

    Returns:
        (F,) RMSD tensor, or a 0-d scalar if `cand_lig_xyz` is a single pose.
    """
    if native_lig_xyz.ndim != 2 or native_lig_xyz.shape[1] != 3:
        raise ValueError(
            f"native_lig_xyz must be (N_lig, 3), got {tuple(native_lig_xyz.shape)}"
        )
    single = cand_lig_xyz.ndim == 2
    cand = cand_lig_xyz.unsqueeze(0) if single else cand_lig_xyz
    if cand.ndim != 3 or cand.shape[-1] != 3:
        raise ValueError(
            f"cand_lig_xyz must be (F, N_lig, 3) or (N_lig, 3), got "
            f"{tuple(cand_lig_xyz.shape)}"
        )
    if cand.shape[1] != native_lig_xyz.shape[0]:
        raise ValueError(
            f"atom-count mismatch: native has {native_lig_xyz.shape[0]} atoms, "
            f"candidates have {cand.shape[1]} — they must share atom ordering"
        )
    native = native_lig_xyz.to(device=cand.device, dtype=cand.dtype)
    diff = cand - native                             # (F, N_lig, 3)
    if atom_mask is not None:
        idx = atom_mask.to(cand.device).nonzero(as_tuple=True)[0]
        diff = diff[:, idx, :]
    rmsd = diff.pow(2).sum(-1).mean(-1).sqrt()       # (F,)
    return rmsd[0] if single else rmsd


# ---------------------------------------------------------------------------
# CAPRI quality tiers, for reporting / stratified decoy curation.
# ---------------------------------------------------------------------------

CAPRI_INCORRECT = 0.0    # DockQ < 0.23
CAPRI_ACCEPTABLE = 0.23
CAPRI_MEDIUM = 0.49
CAPRI_HIGH = 0.80


def capri_tier(dockq: torch.Tensor) -> torch.Tensor:
    """Map DockQ to CAPRI tier (0..3) elementwise:
        0: incorrect  (DockQ < 0.23)
        1: acceptable (0.23 ≤ DockQ < 0.49)
        2: medium     (0.49 ≤ DockQ < 0.80)
        3: high       (DockQ ≥ 0.80)
    """
    tier = torch.zeros_like(dockq, dtype=torch.long)
    tier = torch.where(dockq >= CAPRI_ACCEPTABLE, tier + 1, tier)
    tier = torch.where(dockq >= CAPRI_MEDIUM, tier + 1, tier)
    tier = torch.where(dockq >= CAPRI_HIGH, tier + 1, tier)
    return tier
