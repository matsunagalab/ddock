"""Loader for the consolidated multi-protein training h5.

Schema (see `scripts/build_training_dataset.py` for the writer):

    /<PROTEIN>/
        rec_xyz          (N_rec, 3)       float32
        rec_radius       (N_rec,)         float32
        rec_sasa         (N_rec,)         float32
        rec_atomtype_id  (N_rec,)         int64
        rec_charge_id    (N_rec,)         int64
        lig_xyz          (F, N_lig, 3)    float32
        lig_radius       (N_lig,)         float32
        lig_sasa         (N_lig,)         float32
        lig_atomtype_id  (N_lig,)         int64
        lig_charge_id    (N_lig,)         int64
        rmsd             (F,)             float32    (optional)
        hit_mask         (F,)             bool       (optional if rmsd present)

Root attrs:
        rmsd_threshold_angstrom          float   (threshold used at build time)
        zdock_benchmark                  str
        created_at                       ISO-8601 timestamp
        git_commit                       str

If `rmsd` is present, the caller can override `rmsd_threshold_angstrom` at
load time to recompute `hit_mask = rmsd <= threshold` without rebuilding.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import torch

from .train import ProteinInputs


def list_proteins(h5_path: str | Path) -> list[str]:
    """Return the protein group names present in the consolidated file."""
    with h5py.File(h5_path, "r") as f:
        return sorted(k for k in f.keys() if isinstance(f[k], h5py.Group))


def load_training_dataset(
    h5_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    protein_names: list[str] | None = None,
    rmsd_threshold_angstrom: float | None = None,
    max_poses: int | None = None,
) -> list[ProteinInputs]:
    """Read the consolidated h5 and materialize one `ProteinInputs` per
    selected protein.

    Args:
        h5_path: path to the consolidated file written by
            `scripts/build_training_dataset.py`.
        device / dtype: torch destination for float tensors. Integer ID
            tensors are always int64.
        protein_names: optional whitelist (in this order). If None, every
            group in the file is loaded in sorted order.
        rmsd_threshold_angstrom: if given, override the stored threshold and
            recompute `hit_mask` from the group's `rmsd` field. Raises
            ValueError for any group missing `rmsd` in that case.
        max_poses: optional cap on the F dimension applied at h5 read time.
            Slices `lig_xyz`, `rmsd`, and `hit_mask` to `[:max_poses]` so
            the full 54,000-pose trajectory never reaches device memory.
            Poses in the h5 are stored in ZDOCK raw-score order (highest
            first), so `max_poses=K` is identical to taking the ZDOCK
            top-K — the same semantics as `examples/02_train.py:
            cap_poses`. `None` (default) reads every pose.

    Returns:
        List of `ProteinInputs`, one per selected group.
    """
    h5_path = Path(h5_path)
    out: list[ProteinInputs] = []

    # Slice spec for F-dim datasets. `slice(None)` == `[:]`, i.e. full read.
    f_slice: slice = slice(None) if max_poses is None else slice(0, max_poses)

    with h5py.File(h5_path, "r") as f:
        names = protein_names if protein_names is not None else sorted(
            k for k in f.keys() if isinstance(f[k], h5py.Group)
        )
        for name in names:
            if name not in f:
                raise KeyError(f"{h5_path}: protein group {name!r} not found")
            g = f[name]
            if not isinstance(g, h5py.Group):
                raise TypeError(f"{h5_path}[{name}]: not a group")

            # F-invariant float tensors.
            def _f(key: str) -> torch.Tensor:
                return torch.as_tensor(g[key][()], dtype=dtype, device=device)

            # F-invariant integer tensors.
            def _i(key: str) -> torch.Tensor:
                return torch.as_tensor(g[key][()], dtype=torch.int64, device=device)

            # F-dim float tensor: apply pose slice at h5 read.
            def _f_poses(key: str) -> torch.Tensor:
                return torch.as_tensor(g[key][f_slice], dtype=dtype, device=device)

            rmsd_ds = g.get("rmsd")
            rmsd = (
                torch.as_tensor(rmsd_ds[f_slice], dtype=dtype, device=device)
                if rmsd_ds is not None else None
            )

            if rmsd_threshold_angstrom is not None:
                if rmsd is None:
                    raise ValueError(
                        f"{h5_path}[{name}]: rmsd dataset missing; cannot "
                        f"recompute hit_mask at threshold {rmsd_threshold_angstrom}"
                    )
                hit_mask = (rmsd <= rmsd_threshold_angstrom)
            else:
                hit_mask_ds = g.get("hit_mask")
                if hit_mask_ds is None:
                    if rmsd is None:
                        raise ValueError(
                            f"{h5_path}[{name}]: neither hit_mask nor rmsd present"
                        )
                    # Fall back to the file-level default threshold attr.
                    default_threshold = float(
                        f.attrs.get("rmsd_threshold_angstrom", 2.5)
                    )
                    hit_mask = (rmsd <= default_threshold)
                else:
                    hit_mask = torch.as_tensor(
                        hit_mask_ds[f_slice], dtype=torch.bool, device=device,
                    )

            out.append(
                ProteinInputs(
                    rec_xyz=_f("rec_xyz"),
                    rec_radius=_f("rec_radius"),
                    rec_sasa=_f("rec_sasa"),
                    rec_atomtype_id=_i("rec_atomtype_id"),
                    rec_charge_id=_i("rec_charge_id"),
                    lig_xyz=_f_poses("lig_xyz"),
                    lig_radius=_f("lig_radius"),
                    lig_sasa=_f("lig_sasa"),
                    lig_atomtype_id=_i("lig_atomtype_id"),
                    lig_charge_id=_i("lig_charge_id"),
                    hit_mask=hit_mask,
                    rmsd=rmsd,
                )
            )
    return out
