# Existing ANM implementation: memory and response compression study

Registered before execution, 2026-11-16. ANM is already the source of ddock's
deformation directions. This study changes its computation, not its adoption.
It does not fix the separate contact-distance problem measured in apo pilot v2.

## Exact-model computational comparison

Keep the original CA network, cutoff 12 Å, gamma 1, anchored torsion map,
least-squares damping 0.001, CA metric and nonlinear prior regularization
0.0001. Compare dense reference, chunked CA JVP (chunk size 8), and chunked
JVP plus iterative lowest-mode ANM. The iterative path uses CPU float64,
seed 0, eigenvalue tolerance 1e-9; record all solver diagnostics and failures.
No silent dense fallback, no new force field and no native coordinates.

Use the same development apo receptor 1A2P B, 108 CA / 1700 atoms including H,
from the hash-checked apo_pilot_v1 inference manifest. Prepare with ff14SB,
pH 7, seed 0. Request eight modes to exercise more than the prior two-mode run.
Use independent processes, one CPU thread, three interleaved repetitions per
backend. Separate preparation and mode-construction times; record process
peak RSS including Python/Torch/chemistry and compare numerical subspaces,
stiffness and force-driven nonlinear displacements, not eigenvector signs.

Also compare ANM alone on deterministic synthetic CA clouds with 216 and
1000 nodes: first N points of a cubic lattice, 3.8 Å pitch, uniform coordinate
jitter ±0.15 Å with seed 101. These are scaling fixtures, not proteins or a
docking benchmark. Same cutoff/gamma, eight lowest positive modes, three
interleaved independent processes. Report residuals, subspace errors, edge
counts, wall time and RSS; retain failures/timeouts in the denominator.

All comparisons have an explicit 180-second subprocess limit. No learned loss,
checkpoint choice or train/test tuning. Dense/iterative parity should be judged
with subspace and invariant physical quantities at numerical tolerance, not
bitwise equality or mode-column signs. Large speed/memory gains are hypotheses.

## Approximate response-field compression

This is a distinct approximation, not an equivalent numerical backend.
For scoring-channel derivative D and original SPD stiffness K, compress
A = D K^(-1/2) using a Gaussian random range sketch, then a streamed second
pass. Use seed 0, oversampling 2, source modes 8, target ranks 4 and 2. Retain
the original nonlinear coordinate map/prior under the new amplitude transform;
whiten the true CA metric and transform stiffness consistently. Never reset
the original regularization strength after compression.

Use the same apo pair, field parameters and 2 Å input-derived grid as pilot v2.
Use 16 rotations (identity + seed-17 samples), four peaks per rotation and
top 10. Use the full eight-mode basis to set one common bounded radius from
requested 0.5 Å and max atom displacement 1.5 Å. Compare full versus reduced
response on the union of selected rigid poses: rigid energies, projected
gradients, constrained quadratic gains, candidate overlap, and actual
nonlinear verification (agreement 0.75, six scales). No native/DockQ input.

Record actual retained FFT bytes separately from construction workspace,
timings, process RSS, sketch error and loss of response. A small field norm
error does not guarantee equal candidate ranks. This experiment does not select
a new production rank or claim accuracy improvement. Full-rank and reduced-rank
algebra/finite-coordinate invariants must pass small independent tests first.

Results, exact commands, source hashes, failures and justified follow-up work
belong in EXPERIMENT_REPORT.md §9.20. Generated inputs, arrays and logs stay in
ignored data directories.
