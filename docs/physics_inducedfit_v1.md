# Physical induced-fit implementation and pilot v1

Registered before numerical experiments, 2026-11-16; starting commit 9b78062.
Implement improvement items 1–4. No native structures enter preparation,
direction construction, FFT search, verification or refinement.

**Completed status:** all four items are implemented and the five-arm real-apo
development pilot finished. Accuracy did not improve: final best-of-five DockQ
was about0.049557 in all arms, versus0.201863 in the historical R1 pilot (changed
preparation/scoring, not a single-component comparison). All arms selected the
same Top5 and no deformation constraint was active. See research log§9.22 and
the [machine handoff](HANDOFF_2026-11-16.md). CUDA support is not implemented.

## Fixed physical definitions

- Extract ff14SB q, sigma (Å), epsilon (kJ/mol) from the prepared topology.
- Cross-partner score: Lorentz–Berthelot LJ, C2 Taylor continuation in
  u=(r/sigma)^2 below r=0.9sigma; screened Coulomb with dielectric 78.5,
  kappa=0.1/Å and a finite C2 core at 0.5 Å. C2 switch at 10/12 Å.
  Units are kJ/mol throughout; no old effective-epsilon shape or CA spring
  is added. This screened pair score plus monomer OBC2 strain is a declared
  hybrid potential, not a complete Amber complex or binding free energy.
- Receptor strain: ff14SB/OBC2/ACE plus an explicit 10 kJ/mol/Å² Cartesian
  anchor on all original atoms. Apo-only torsion minimization is shared by
  all arms. Max iterations1000/evaluations5000, final max torque1e-3
  kJ/mol/rad, max heavy displacement1 Å. Failure remains in the denominator;
  no force subtraction, negative-strain clipping or Hessian eigenvalue clipping.
- Build ANM/torsion directions at that prepared reference, CA-normalize and
  measure stiffness from finite differences of the actual torsional force.
- Atomwise response keeps ||a||<=0.5 Å and ||B_i a||<=1.5 Å. A conservative
  ball using the same bounds is the constraint ablation. Actual nonlinear
  CA/atom bounds and energy are always checked (six half-scales; gain
  agreement>=0.75); numerical failures are errors, not successful fallbacks.
- Adaptation: predicted contacts within8 Å, four representatives selected
  deterministically from rigid candidates, four original ANM directions plus
  up to four local H-orthogonal response directions. An equally sized seeded
  random local control is also run. Record rank shortfalls, active masks,
  force calls/HVPs, locality and construction costs. No native contact masks.

## Numerical gates

All91 nonzero LJ type pairs plus zero-LJ, finite overlaps, C2 boundaries,
force/JVP differences and FFT versus same-grid sums. Small pair fixtures use
spacing1/.5/.25 Å and zero/half-cell phases. Retain failed convergence gates.
If 2 Å does not resolve contact, explicitly use it only as a coarse proposer;
actual candidate verification and refinement use the continuous pair energy.
This does not repair candidate recall automatically; preserve all raw peaks.

The first smoke used projected Armijo descent with a fixed diagonal coordinate
preconditioner (.01 for rotation radians,1 for translation Å,.1 for CA-amplitude
Å), Armijo1e-4, at most24 halvings, and true nonlinear geometry checks.
Because stationarity is finite, the tiny negative prior tolerance is explicitly
1e-6+2*abs(g_prior(0) dot a) kJ/mol; energies are never clipped. The recorded
baseline gradient is also included in the quadratic response model.

Physical prior tests check OpenMM forces, Å/nm conversion, stationary reference,
nonlinear energy and stiffness. Constraint tests use analytic anisotropic
fixtures and independent feasibility/KKT checks. Adaptive tests check local
support, new-subspace independence, stiffness, reproducible random controls
and negative curvature/no-contact failures.

## Real apo development pilot

Same hash-checked apo_pilot_v1/inference_manifest.json (1A2P B / 1A19 A).
This already-used system is development, not held-out validation/test.
CPU float64, PyTorch1 thread; record NumPy/SciPy threadpool settings if available.
No training loss/checkpoints or accuracy-based parameter choice.

First run a four-rotation integration smoke (Top1, pool8, two refinement steps).
Then fixed128 rotations (identity+seed0), four peaks/rotation, mixed pool50
(25 rigid+25 response with deterministic duplicate/backfill rule), Top5,
eight local steps/64 evaluations, max rigid rotation0.15 rad/translation4 Å
(amended before real-apo inference; original registration was0.5 Å).
Common input-derived cubic grid, 2 Å coarse proposal, full assignment support.
Record memory estimate and fail rather than exceed the configured cache budget.

Arms: typed rigid; physical8 with conservative ball; physical8 with atomwise
constraints; four ANM+four adaptive directions; four ANM+four random local
directions. Each flexible arm verifies its full pool before selecting Top5
and starts refinement at zero deformation. Ball/atom arms share the same
optimistic first-pass pool. Adaptive/random arms perform a genuine second FFT
pass using their new fixed bases; preserve a rigid quota. Same final mode count
is required for a claimed equal-rank comparison; otherwise report arm failure.

Export initial/final heavy PDBs and full coordinate/basis/prior artifacts into
ignored data. Freeze inference first, then run official DockQ in its separate
environment with native-vs-native sanity and all missing/failed rows retained.
Compare Top1/best5 to each other and contextual old pilot results, noting changed
preparation/scoring; do not claim a clean single-component comparison with the
old pilot. OBC2 endpoint/clash diagnostics are distinct from the primary ranking.

All failures, changes justified by numerical bugs, actual commands, hardware,
times and numerical/accuracy results go into EXPERIMENT_REPORT.md §9.22.
Independent multi-system apo and antibody-loop accuracy claims require later
registered datasets; this implementation pilot cannot establish HDOCK superiority.

## Numerical amendments before real-apo inference

The native-free 91 LJ-pair ×8-phase fixture at2 Å spacing found a maximum
sampled radial minimum shift3.67 Å. Five tiny-H-type phases preferred zero
separation. Positive mass interpolation avoids extra-deep wells, but does not
preserve contact geometry. Cardinal coefficient interpolation was also tried
and rejected as the default because it produced spurious deep wells. Keep
mass interpolation for coarse proposals only and raise the common local rigid
translation radius to two cells (4 Å) for all five new arms. This is room for
continuous correction, not a guaranteed error bound or a claim of recovered
candidate recall. No new real-apo DockQ results were available for this change.

The conservative-ball ablation uses its bound during both pool verification
and continuous refinement. The atomwise arms use actual nonlinear per-atom
checks throughout. All arms retain the same actual CA/atom geometry limits.

Apo minimization with a diagonal anchor preconditioner failed the torque gate.
A full fixed anchor metric G=10 JᵀJ+I improved it but stopped at the energy
roundoff floor before reaching1e-3. Register a matrix-free Newton-CG torque
polish, using h=1e-4 finite-difference true HVPs and the same metric, original
physical energy and geometry/stationarity gates. Allow explicitly recorded
energy roundoff1e-7 kJ/mol only when torque residual decreases; no force or
Hessian clipping. Keep the total force-evaluation budget5000 and all failures.
Successful apo-only artifacts may be reused only after input, source, array
and forcefield hashes plus recomputed energy, full torque and geometry pass.

The first smoke completed all five arms, but exposed a numerical conditioning
problem before any new DockQ evaluation: the stiff amplitude block shrank
the common line-search step, including rigid translation. A separate quadratic
oracle with the actual apo K reproduced this behavior without a native target.
For subsequent runs, keep rotation/translation weights .01/1 and use
`d_a=-solve(K,g_a)` for the amplitude block. Cholesky solves use the fixed
physical K as an optimizer metric; the energy is unchanged. Require a negative
projected directional derivative, the same Armijo condition and actual
geometry gates. `--refinement-preconditioner fixed` retains the first smoke's
method; the new default is `physical`.

The first contact basis took404 HVPs with CG119/93/102/78 iterations. Add a
constraint-preserving PCG preconditioner from the same original-input anchor
metric G. First restrict G to the contact torsion mask, then constrain its
inverse action to the already fixed H-orthogonal nullspace. Preserve the
physical residual/curvature gates (recursive residual<=1e-5 and independently
recomputed residual<=5e-5), max128 iterations, seed and random control
distribution. No ridge is added to H or energy. The new default is
`--adaptive-preconditioner anchor`; `none` retains the initial CG. These are
numerical amendments before the128-rotation accuracy run, not parameter
selection using DockQ. Source snapshots preserve the initial implementation.

## Running the experimental path

Install the existing optional `chemistry` dependency and supply the strict
inference manifest described in `inducedfit_pair_runner.md`. First verify the
apo reference (the verifier saves the exact source hashes and coordinates):

```bash
.venv/bin/python scripts/verify_inducedfit_physical_prior.py \
  --manifest data/inducedfit/apo_pilot_v1/inference_manifest.json \
  --out data/inducedfit/physical_prior_v1_metric_polish \
  --modes 8 --anchor-k 10 --max-iterations 1000 --max-evaluations 5000 \
  --stationarity-tolerance .001 --max-heavy-displacement 1 \
  --fd-step .0001 --seed 0 --threads 1 --preconditioner anchor_metric --polish

.venv/bin/python scripts/run_physics_induced_fit_pair.py \
  --manifest data/inducedfit/apo_pilot_v1/inference_manifest.json \
  --prepared-prior data/inducedfit/physical_prior_v1_metric_polish/report.json \
  --out data/inducedfit/physics_inducedfit_pilot_v1
```

Use a fresh output directory for every attempt. Omitting `--prepared-prior`
performs the same preparation in the runner; a failed or stale artifact is
rejected rather than reused. The default five arms are `rigid ball8 atom8
contact8 random8`. Source hashes are checked again at completion. No native
coordinates or DockQ command is accepted by this inference script.

The FFT cache budget (default4GB) bounds retained spectra, not total process
RSS. Three FFT passes, shared preparation, and each arm's verification and
refinement costs are reported separately; the five arms are not equal wall
time. Full raw per-rotation peaks and coordinate/model arrays live only in
ignored `data/`. Every requested TopK row remains in the evaluation denominator,
including an arm that fails before exporting any structures.
