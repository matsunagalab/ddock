"""Content-addressed CPU/disk cache for :class:`~zdock.dataset.PreparedProtein`.

Featurizing a complex (parse PDB → atom types → radii → SASA → orient →
Kabsch native pose) is deterministic and identical across training seeds, so
the scaling-law experiment pays it once and then pages complexes in from disk.

PINDER system ids look like
``7bgl__DB1_P0A1N8--7bgl__DA1_A0A745A2I3`` — long, and containing ``/``-free
but filesystem-unfriendly characters in general — so the cache file name is a
SHA-1 digest of the id rather than the id itself. The id is stored *inside* the
payload and verified on load, which also makes an (astronomically unlikely)
digest collision a loud failure instead of silent data corruption.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

from .dataset import PreparedProtein

#: bumped whenever the on-disk payload layout changes
CACHE_FORMAT = 1


def cache_key(pid: str) -> str:
    """Filesystem-safe, collision-resistant key for a PINDER system id."""
    return hashlib.sha1(pid.encode("utf-8")).hexdigest()[:20]


def cache_path(cache_dir: str | Path, pid: str) -> Path:
    """``<cache_dir>/<ab>/<key>.pt`` — two-level fan-out keeps directory
    listings small when caching thousands of complexes."""
    key = cache_key(pid)
    return Path(cache_dir) / key[:2] / f"{key}.pt"


def save_prepared(cache_dir: str | Path, prot: PreparedProtein) -> Path:
    """Atomically write ``prot`` (moved to CPU) into the cache."""
    path = cache_path(cache_dir, prot.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"format": CACHE_FORMAT, "id": prot.name,
            "state": prot.cpu().state_dict()}
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    torch.save(blob, tmp)
    os.replace(tmp, path)
    return path


def has_prepared(cache_dir: str | Path, pid: str) -> bool:
    return cache_path(cache_dir, pid).exists()


def load_prepared(
    cache_dir: str | Path,
    pid: str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> PreparedProtein | None:
    """Return the cached protein on ``device``, or ``None`` if absent.

    A truncated / half-written file (possible if a worker was killed) is
    treated as a cache miss rather than raising.
    """
    path = cache_path(cache_dir, pid)
    if not path.exists():
        return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 — corrupt/partial file ⇒ regenerate
        return None
    if blob.get("format") != CACHE_FORMAT:
        return None
    if blob.get("id") != pid:
        raise RuntimeError(
            f"prep cache collision: {path} holds {blob.get('id')!r}, asked {pid!r}"
        )
    prot = PreparedProtein.from_state_dict(blob["state"])
    if device != "cpu" or dtype is not None:
        prot = prot.to(device, dtype=dtype)
    return prot
