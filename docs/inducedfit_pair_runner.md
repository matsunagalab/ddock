# Experimental induced-fit pair runner

This runner connects complete, standard, crosslink-free single-chain PDB inputs
to Amber ff14SB charge assignment, hydrogen-aware torsion modes, FFT candidate
selection, constrained refinement and heavy-atom prediction export. It does not
read a native complex. Docking accuracy and HDOCK superiority are unestablished.

## Installation and input

```bash
uv sync --extra chemistry
```

Create a JSON file with exactly these keys. Paths are relative to the manifest
(absolute paths also work); replace each digest with the SHA256 of that source
PDB. `input_kind` and `split` are provenance declarations, not automatically
verified benchmark membership. A proper dataset audit must establish them.

```json
{
  "format_version": 1,
  "system_id": "development_pair",
  "input_kind": "apo",
  "split": "development",
  "receptor": {"pdb": "../inputs/receptor.pdb", "chain": "A", "sha256": "REPLACE_WITH_SHA256"},
  "ligand": {"pdb": "../inputs/ligand.pdb", "chain": "A", "sha256": "REPLACE_WITH_SHA256"}
}
```

Extra fields (including native paths or labels) are rejected. PDB content hashes
are checked before parsing. Missing side-chain atoms, missing terminal OXT,
chain breaks, explicit links and disulfides fail preparation. This is deliberately
a restricted pilot; preparation failures belong in the registered denominator.
It is not appropriate to select only successful predictions for evaluation.

## Running and outputs

First verify preparation. Every output directory must be new.

```bash
.venv/bin/python scripts/run_induced_fit_pair.py \
  --manifest data/inducedfit/manifests/development_pair.json \
  --out data/inducedfit/runs/development_pair_prepare --prepare-only
.venv/bin/python scripts/run_induced_fit_pair.py \
  --manifest data/inducedfit/manifests/development_pair.json \
  --out data/inducedfit/runs/development_pair_smoke \
  --arms R0 R1 R2 --rotations 4 --top-k 1 --modes 2 --iterations 2 --seed 0
```

The small default budgets test integration, not docking performance. In
particular, four random rotations do not cover SO(3), and 2 Å spacing is coarse
relative to the 0.6 Å charge Gaussian. Grid convergence and an adequate fixed
angular grid must precede accuracy claims. All parameters, sampled rotations,
source hashes, chemistry versions/protonation choices, heavy-to-full atom maps,
failures, intermediate energies and resource usage are saved in `report.json`.
Partial reports are saved before expensive stages. An interrupted process may
leave `status=running`; this must be counted as interrupted, not completed.

For protein-sized inputs, `--field-backend sparse` evaluates the same compact
fields on atom-local stencils and scatters them onto the full grid. It retains
the same field formula, charge normalization and FFT correlation; dense/sparse
values, gradients and cell-boundary behavior are tested. The default `dense`
backend remains an independent reference. `--max-grid-points` limits molecular
grid size, not total FFT/gradient memory.

| Arm | Candidate selection | Local refinement |
| --- | --- | --- |
| R0 | Rigid field score | Rigid pose |
| R1 | Rigid field score | Pose and receptor torsions |
| R2 | Constrained quadratic response score | Pose and receptor torsions |
| R2v | Actual nonlinear energy after verifying a larger response pool | Pose and receptor torsions, from zero deformation |

R1 and R2 use the same grid, rotations, candidate count, zero initial deformation,
solver budgets and displacement limits. R0/R1 reuse identical rigid candidates.
Shared preparation and search times are reported separately: per-arm elapsed
times alone are not independent end-to-end runtimes. The ligand stays internally
rigid. Every refinement includes all hydrogens in the fields and displacement
check; atom positions follow the topology-aware torsion map.

Use `--arms R1 R2 R2v --response-pool-size 50` to test verified selection.
R2v checks all response proposals before TopK, shrinking each amplitude from
1 to 1/32 until actual energy decreases, actual/predicted gain is at least
0.75, and true CA/all-atom displacement and field-support bounds pass. A proposal
with no accepted scale falls back to its rigid energy. Nonfinite values and
failure to reproduce the rigid energy fail the arm. R2 and R2v share the raw
pool when both are requested, and the extra verification time is reported.
This is not an equal-total-time comparison. Both still refine from zero
amplitudes; `verified_*/` saves the deformation used only for candidate selection.
The report retains every verification trial and final rigid pose increments.

Add `--checkpoint-fields` to recompute field chunks during local reverse-mode
gradients instead of retaining their large intermediate tensors. FFT search and
its JVP preprocessing retain the normal path. On the documented 41-atom synthetic
pair (three CPU runs per setting), median peak RSS decreased 64.46% at a 19.88%
wall-time cost, with identical final energies and PDB hashes. Protein/GPU scaling
is unmeasured. Reproduce with:

```bash
.venv/bin/python scripts/benchmark_inducedfit_checkpoint.py \
  --out data/inducedfit/verification/new_checkpoint_comparison --repeats 3
```

`R*/candidate_*/{receptor,ligand,complex}.pdb` contain original heavy atom
identities and residue numbers. Both partners are in the receptor input frame.
The combined file uses chains R and L. Hydrogens are omitted only at export for
evaluation, not during scoring. File names retain selection rank; `final_rank`
in the report gives ranking by the final energy. There is no guarantee that
refinement reaches convergence within the budget.

## Physical scope and next checks

The charges are Amber ff14SB elementary charges. The field energy is still an
effective smooth shape/screened-charge model, with geometric radii and CA-spring
strain, not Amber energy in kcal/mol. Ramachandran, internal nonbonded penalties,
side-chain rotamers, explicit solvent and antibody loop closure are not included.
Hydrogen placement keeps heavy atoms fixed and verifies their identity/position;
input hydrogens are replaced under the logged pH/seed/protonation policy.

Use a separate evaluation process for native structures, an official scorer
with native-vs-native sanity checks, and report preparation/search/export/scorer
failures separately. Keep every registered system in the denominator. The
existing holo PINDER scoring wrapper is not an apo evaluation adapter. This
runner's energy differences are not DockQ differences or evidence against HDOCK.

The first [registered real apo pilot](apo_pilot_v1.md) completed on one
barnase/barstar pair using explicitly declared protein-only chemistry, sparse
fields and checkpointing. All 30 initial/final models scored below DockQ 0.23.
See the [per-candidate results](apo_pilot_v1_results.csv) and research log §9.18
for negative results, ranking errors and the response-gain diagnostic. This is
a completed execution test, not a successful docking benchmark.

## Frozen-pose physical diagnostics

These separate tools replay all original hydrogens and evaluate frozen poses
without reading native coordinates or optimizing against a reference:

```bash
.venv/bin/python scripts/audit_inducedfit_grid.py \
  --run-report data/inducedfit/apo_pilot_v1/pilot_run/report.json \
  --out data/inducedfit/new_grid_audit
.venv/bin/python scripts/audit_inducedfit_atomistic.py \
  --report data/inducedfit/apo_pilot_v1/pilot_run/report.json \
  --out data/inducedfit/new_atomistic_audit.json
```

The grid audit compares 2/1 Å spacing and zero/half-cell phase at the same
frozen coordinates and common physical extent. The atomistic audit uses
ff14SB + OBC2/ACE, reporting same-coordinate interaction energy and monomer
strain separately in kJ/mol. Its primary interaction ranking and separate
unweighted endpoint ranking are saved; neither is a binding free energy.
These diagnostics do not modify the runner's final smooth-energy ranking.
Replay verifies input hashes, chemistry, atom maps, amplitudes and heavy exports;
older reports without final pose increments require a rigid fit to rounded
heavy coordinates, whose error is retained.

## Optional ANM memory backends

ANM already supplies the torsion directions. These options change how those
directions are computed while retaining the network, torsion map and prior:

```bash
# Add to the pair-runner command; both defaults remain dense.
--jacobian-backend chunked --jvp-chunk-size 8 \
--anm-backend iterative --anm-seed 0 --anm-tolerance 1e-9
```

The chunked path retains the CA Jacobian and evaluates full-atom JVPs only for
the adopted directions. The iterative path computes Hessian-vector products
from network edges and extracts low modes without allocating a dense Hessian.
It currently requires CPU float64 and fails explicitly on unresolved null
directions, ambiguous eigenvalues or nonconvergence. Solver-inferred nullity is
not a certification of the complete spectrum or degenerate eigenspaces.
Reports record these options and replay uses the same backend.

On the registered 108-CA / 1700-atom apo receptor, chunked JVP alone reduced
median process peak RSS by 14.25% and increased mode-build time by 22.87%.
Adding iterative ANM reduced RSS by only 6.51% relative to fully dense here.
For a synthetic 1000-CA ANM alone, iterative RSS was 56.76% lower and runtime
1.99 times higher. These are three-process medians for preprocessing on one CPU,
not end-to-end docking speedups; the dense defaults remain unchanged. See
[registered conditions](anm_memory_v1.md), the three result CSVs beside it,
and research log §9.20.

`scripts/audit_inducedfit_compression.py` separately evaluates randomized
response-field compression. Reducing the number of response FFTs restricts the
deformation subspace, so it is an approximation rather than an interchangeable
backend. It is not enabled in the production pair runner and has no measured
DockQ or HDOCK advantage.
The input-only apo audit reduced retained FFT bytes by 44.44% / 66.67% for
eight to four / two directions, but lost 28.93% / 54.45% of quadratic response
gain on average over twelve union poses. Construction-inclusive timing did
not improve consistently. See [all search attempts](anm_compression_v1_search.csv),
[union results including the first audit failure](anm_compression_v1_union.csv),
and research log §9.20. Timing used one PyTorch thread; NumPy/SciPy BLAS thread
counts were not independently verified.
