"""Input provenance and JSON-only cache metadata for the synthetic prototype.

No structure parsing, protonation, native structures, or tensor deserialization
is performed here. Fingerprints cover the supplied tensors and atom identities;
they are not hashes of an original PDB file. A future file reader must additionally
record source-file hashes and chemical preparation rules.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping

import torch

from .types import DockingInput, MonomerInput

FORMAT_VERSION = 1


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("configuration numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("configuration keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"configuration must be JSON data, got {type(value).__name__}")


def canonical_json_bytes(value) -> bytes:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def config_fingerprint(config: Mapping) -> str:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    return hashlib.sha256(canonical_json_bytes(config)).hexdigest()


def _hash_chunk(digest, data: bytes) -> None:
    # Length prefixes avoid ambiguous concatenations of variable-size fields.
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def monomer_fingerprint(monomer: MonomerInput) -> str:
    digest = hashlib.sha256()
    _hash_chunk(digest, canonical_json_bytes([asdict(key) for key in monomer.atom_keys]))
    for name in ("xyz", "radii", "charges"):
        tensor = getattr(monomer, name).detach().cpu().contiguous()
        _hash_chunk(digest, canonical_json_bytes({"name": name, "shape": list(tensor.shape),
                                                "dtype": str(tensor.dtype)}))
        # uint8 also works for bfloat16, which numpy cannot represent directly.
        _hash_chunk(digest, tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def input_fingerprint(docking_input: DockingInput) -> str:
    return config_fingerprint({"system_id": docking_input.system_id,
                               "receptor": monomer_fingerprint(docking_input.receptor),
                               "ligand": monomer_fingerprint(docking_input.ligand)})


def _digest(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def input_manifest_record(docking_input: DockingInput, *, input_kind: str) -> dict:
    """Create a reference-free provenance record; this is not a file loader."""
    record = {"format_version": FORMAT_VERSION, "system_id": docking_input.system_id,
              "input_kind": input_kind, "input_sha256": input_fingerprint(docking_input)}
    for side in ("receptor", "ligand"):
        monomer = getattr(docking_input, side)
        record[side] = {"input_sha256": monomer_fingerprint(monomer),
                        "atom_count": monomer.n_atoms,
                        "chain_ids": sorted({key.chain for key in monomer.atom_keys})}
    validate_input_manifest(record)
    return record


def validate_input_manifest(record: Mapping) -> None:
    """Strict allowlists reject native/label fields, including nested additions."""
    required = {"format_version", "system_id", "input_kind", "input_sha256", "receptor", "ligand"}
    if not isinstance(record, Mapping) or set(record) != required:
        raise ValueError("input manifest has missing or unrecognized fields")
    if type(record["format_version"]) is not int or record["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported input manifest format_version")
    for key in ("system_id", "input_kind"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if not _digest(record["input_sha256"]):
        raise ValueError("input_sha256 must be a lowercase SHA256 digest")
    for side in ("receptor", "ligand"):
        item = record[side]
        if not isinstance(item, Mapping) or set(item) != {"input_sha256", "atom_count", "chain_ids"}:
            raise ValueError(f"{side} has missing or unrecognized fields")
        if not _digest(item["input_sha256"]) or type(item["atom_count"]) is not int or item["atom_count"] <= 0:
            raise ValueError(f"{side} must have valid hash and positive atom_count")
        chains = item["chain_ids"]
        if (not isinstance(chains, list) or not chains or any(not isinstance(c, str) for c in chains)
                or len(set(chains)) != len(chains)):
            raise ValueError(f"{side}.chain_ids must be unique strings")


def build_cache_metadata(docking_input: DockingInput, *, input_kind: str,
                         prep_config: Mapping, core_config: Mapping, mode_config: Mapping) -> dict:
    metadata = {"format_version": FORMAT_VERSION,
                "input": input_manifest_record(docking_input, input_kind=input_kind),
                "prep_sha256": config_fingerprint(prep_config),
                "core_sha256": config_fingerprint(core_config),
                "mode_sha256": config_fingerprint(mode_config)}
    metadata["cache_key"] = config_fingerprint(metadata)
    return metadata


def _validate_cache_metadata(metadata: Mapping) -> None:
    required = {"format_version", "input", "prep_sha256", "core_sha256", "mode_sha256", "cache_key"}
    if not isinstance(metadata, Mapping) or set(metadata) != required:
        raise ValueError("invalid cache metadata fields")
    if type(metadata["format_version"]) is not int or metadata["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported cache format_version")
    validate_input_manifest(metadata["input"])
    if any(not _digest(metadata[key]) for key in ("prep_sha256", "core_sha256", "mode_sha256", "cache_key")):
        raise ValueError("invalid cache metadata digest")
    if metadata["cache_key"] != config_fingerprint({k: v for k, v in metadata.items() if k != "cache_key"}):
        raise ValueError("cache metadata integrity mismatch")


def write_cache_metadata(path: str | Path, metadata: Mapping) -> None:
    """Atomically replace only a JSON metadata file, never a tensor pickle."""
    _validate_cache_metadata(metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            tmp = Path(handle.name)
            handle.write(canonical_json_bytes(metadata) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def read_cache_metadata(path: str | Path, *, expected: Mapping) -> dict | None:
    """Absent, corrupt, unsupported, or mismatched metadata is a cache miss."""
    _validate_cache_metadata(expected)
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        _validate_cache_metadata(metadata)
    except (OSError, ValueError, TypeError):
        return None
    return metadata if metadata == expected else None
