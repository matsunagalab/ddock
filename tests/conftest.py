"""pytest fixtures for reading Julia reference outputs (HDF5) and selecting
device/dtype.

Reference files live under `docking/tests/refs/1KXQ/phaseN_<name>.h5`. They
are generated once by `julia tests/julia_ref/generate_refs.jl`.

The device fixture honours the `ZDOCK_DEVICE` environment variable:

    pytest -q                        # -> cpu, float64
    ZDOCK_DEVICE=cuda pytest -q      # -> cuda, float32
    ZDOCK_DEVICE=mps  pytest -q      # -> mps,  float32

Tolerances are relaxed when running in float32; see `tol` fixture.
"""

from __future__ import annotations

import os

# macOS-specific: PyTorch and h5py both bundle libomp, which triggers the
# infamous "Initializing libomp.dylib, but found libomp.dylib already
# initialized" abort. Setting this env var before torch import allows both
# runtimes to coexist. Harmless on Linux/Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# PyTorch + MKL + LibOMP all fight over thread pools on macOS Python 3.14,
# producing segfaults on non-trivial tensor ops. Forcing a single thread
# avoids the crash while losing a little CPU parallelism; GPU/MPS paths are
# unaffected.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

# Location of Julia-generated reference outputs (bundled under tests/data).
REFS = Path(__file__).resolve().parent / "data" / "refs" / "1KXQ"


@pytest.fixture(scope="session")
def refs_root() -> Path:
    if not REFS.exists():
        pytest.skip(
            f"Reference outputs missing at {REFS}. "
            "Run `cd ../docking && julia tests/julia_ref/generate_refs.jl` first."
        )
    return REFS


@pytest.fixture(scope="session")
def device() -> torch.device:
    name = os.environ.get("ZDOCK_DEVICE", "cpu").lower()
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("ZDOCK_DEVICE=cuda requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("ZDOCK_DEVICE=mps requested but MPS is not available")
    return torch.device(name)


@pytest.fixture(scope="session")
def dtype(device: torch.device) -> torch.dtype:
    # float64 on CPU for exact matching; float32 on accelerators (MPS has no
    # float64, CUDA is fastest in float32).
    return torch.float64 if device.type == "cpu" else torch.float32


@pytest.fixture(scope="session")
def tol(dtype: torch.dtype) -> dict:
    """Default (atol, rtol) per dtype. Individual tests may override."""
    if dtype == torch.float64:
        return dict(atol=1e-10, rtol=1e-8)
    return dict(atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------- helpers


def load_h5(path: Path) -> dict:
    """Load every dataset in an HDF5 file into a dict of numpy arrays /
    python scalars. String datasets are decoded to `list[str]`."""
    out: dict = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            ds = f[key]
            data = ds[()]
            if isinstance(data, np.ndarray) and data.dtype.kind in ("O", "S"):
                # h5py stores Julia's String[] as variable-length bytes
                data = [s.decode("utf-8") if isinstance(s, bytes) else s for s in data]
            out[key] = data
    return out


@pytest.fixture(scope="session")
def load_ref(refs_root: Path):
    """Returns a function that loads refs_root / f"{phase}_{name}.h5" and
    hands back a dict of arrays."""

    def _load(phase: str, name: str) -> dict:
        path = refs_root / f"{phase}_{name}.h5"
        if not path.exists():
            pytest.skip(f"reference file missing: {path}")
        return load_h5(path)

    return _load


def to_tensor(
    array: np.ndarray, *, device: torch.device, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Convert a numpy array from HDF5 into a tensor on the target device.

    For floating-point arrays, cast to the target dtype. Integer / bool arrays
    are preserved as-is (still moved to device).
    """
    t = torch.as_tensor(np.asarray(array))
    if dtype is not None and t.is_floating_point():
        t = t.to(dtype)
    return t.to(device)
