# ddock — Differentiable ZDOCK in PyTorch

`ddock` is a pure-PyTorch, fully differentiable reimplementation of the
[ZDOCK](https://zdock.wenglab.org/) protein–protein docking scoring
function. Every scoring term is autograd-safe, so the **156 learnable
ZDOCK parameters** (α, the 12×12 IFACE matrix, and the 11 charge-type
weights) can be optimized end-to-end with any PyTorch optimizer.

The importable Python package is `zdock`.

## Highlights

- **Pure PyTorch** — NumPy, h5py and SciPy are only used for I/O and
  reference comparisons.
- **Same code runs on macOS (MPS), Linux (CUDA) and CPU** out of the box.
- **Differentiable scoring** (`zdock.score.docking_score_elec`) — batched
  re-scoring of candidate poses with autograd through all 156 parameters.
- **FFT-based pose search** (`zdock.search.docking_search`) — evaluates
  every translation of the ligand at a fixed rotation in one batched FFT,
  the same math as the upstream ZDOCK binary but end-to-end autograd-safe.
- **Physically correct Coulombic electrostatics** by default (Chen 2002/2003),
  with a `legacy` mode available for bit-exact thesis reproduction.
- **Training utilities** (`zdock.train`) — Adam optimization with several
  interchangeable ranking / regression losses (split-MSE, ListNet on RMSD
  or DockQ, hard-negative margin).

## The scoring model

For F candidate poses, the score of pose `f` is

```
score[f] = α · S_SC[f]  +  S_IFACE[f]  +  β · S_ELEC[f]
```

| Term       | Name                    | Captures                                     |
|------------|-------------------------|----------------------------------------------|
| `S_SC`     | Shape complementarity   | How well the molecular surfaces fit together |
| `S_IFACE`  | Interface statistics    | Atom-type pairwise contact preferences       |
| `S_ELEC`   | Electrostatic energy    | Coulombic charge–charge interaction          |
| α, β       | Weight scalars          | Relative importance of SC vs ELEC            |

**Learnable parameters:** α (1) + `iface_ij` (12×12 = 144) +
`charge_score` (11) = **156 total**. β is a function input but is held
fixed (default 3.0) during training because it is scale-redundant with
`charge_score`.

Atoms are scattered onto a 3D grid (default spacing 3 Å) so scoring
reduces to GPU-friendly grid inner products; receptor grids are built
once and only the ligand side is recomputed per pose.

## Installation

With [uv](https://docs.astral.sh/uv/) — this is all you need:

```bash
uv sync
```

That single command creates the virtual environment, installs the right
PyTorch build for your platform, and installs `ddock` itself. **No extra
flags, no manual PyTorch install** — it works out of the box on:

- **macOS (Apple Silicon / MPS)** → the PyPI `torch` wheel (MPS support baked in)
- **Linux (NVIDIA / CUDA 12.4)** → the `torch ...+cu124` wheel from PyTorch's own index

The routing is pinned in `uv.lock`, so the same lockfile resolves
correctly on both platforms. Run anything with `uv run`, e.g.:

```bash
uv sync
uv run pytest -q                       # run the test suite
uv run python -c "import torch; print(torch.cuda.is_available() or torch.backends.mps.is_available())"
```

<details>
<summary>Alternatives (pip / CPU-only Linux)</summary>

Plain pip (requires Python ≥ 3.10, PyTorch ≥ 2.1):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
```

CPU-only Linux (skip the CUDA wheel):

```bash
uv sync --index https://pypi.org/simple --reinstall-package torch
```

</details>

## Quick start — re-scoring poses

```python
import torch
from zdock.score import docking_score_elec
from zdock.atomtypes import iface_ij, charge_score

# rec_* : receptor atoms (N_rec,)     lig_* : ligand atoms (N_lig,)
# lig_xyz has shape (F, N_lig, 3): F candidate poses.
scores = docking_score_elec(
    rec_xyz, rec_radius, rec_sasa, rec_atomtype_id, rec_charge_id,
    lig_xyz, lig_radius, lig_sasa, lig_atomtype_id, lig_charge_id,
    alpha=torch.tensor(1.0),
    iface_ij_flat=iface_ij().flatten(),   # (144,) learnable
    beta=torch.tensor(3.0),
    charge_score=charge_score(),          # (11,) learnable
)                                          # -> (F,) differentiable scores
```

Because every term is differentiable, `scores.sum().backward()` yields
gradients w.r.t. `alpha`, `iface_ij_flat` and `charge_score`.

## Package layout

```
src/zdock/
├── score.py          # docking_score_elec — the differentiable scoring core
├── search.py         # FFT-based pose search (docking_search)
├── spread.py         # grid scatter primitives (SC / IFACE / Coulomb)
├── sasa.py           # solvent-accessible surface area
├── geom.py           # rotations, grid generation, orient/decenter
├── rotation_grid.py  # quaternion sampling + Kabsch alignment
├── atomtypes.py      # 12 atom types, 11 charge types, default parameters
├── train.py          # Adam training loop + loss functions
├── data.py           # HDF5 training-dataset loader
├── dockq.py          # differentiable DockQ approximation (eval metric)
├── io.py             # ZDOCK .pdb.ms structure reader
└── zdock_output.py   # ZDOCK .out parser + pose regeneration
```

## Tests

The suite validates numerical agreement against reference outputs bundled
under `tests/data/` (a small 1KXQ example):

```bash
uv run pytest -q          # or: pytest -q
```

Set the device explicitly with `ZDOCK_DEVICE`:

```bash
ZDOCK_DEVICE=mps  pytest -q     # Apple Silicon
ZDOCK_DEVICE=cuda pytest -q     # NVIDIA GPU
```

CPU runs in float64 for exact matching; accelerators use float32 with
relaxed tolerances. Long-running training/benchmark tests are marked
`slow` and skipped by default (`pytest -m slow` to opt in).

## License

MIT — see [LICENSE](LICENSE).
