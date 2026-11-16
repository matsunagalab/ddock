# Apo pilot v2: verified nonlinear response

Registered 2026-11-16 before this run. This is a development follow-up to v1,
whose DockQ and response errors are already known. It is not a held-out test.
Keep the v1 input manifest, chemical assumptions, seed 0, modes, rotations,
grid, physical weights, local solver and evaluation mapping unchanged.

## Question and fixed intervention

Does checking the actual nonlinear energy and geometry before final TopK avoid
the overstated response bonuses and infeasible proposals observed in v1?
Whether that improves structural accuracy is a separate measured outcome.

- R1: rigid selection, flexible refinement from zero amplitudes.
- R2: approximate response selection, identical flexible refinement from zero.
- R2v: take the top 50 approximate response candidates from the same 128
  rotations and four peaks per rotation, verify all 50, then select top 5 by
  verified energy. Refine from zero amplitudes, exactly as R1 and R2.
- R2 and R2v share the same raw pool. Its first five reproduce the R2 rule.
  Additional pool verification costs are reported explicitly; this is not an
  equal-total-runtime comparison. R0 is omitted because the question concerns
  flexible selection and its v1 control is already recorded.

For each proposal, test amplitude scales 1, 1/2, 1/4, 1/8, 1/16, 1/32 in order.
Accept the first scale satisfying actual CA RMSD and amplitude radius,
maximum all-atom displacement and full field support, with actual energy gain
negative and actual/predicted gain ratio at least 0.75. Otherwise use the rigid
pose and its recomputed energy. Nonfinite values and rigid-energy reproduction
errors are hard failures. The selected score is always an evaluated physical
energy, never an unchecked response bonus. All trials are retained.

The scaled prediction is s*(gain(1) - 0.5*a'K*a) + s^2*0.5*a'K*a. The actual
energy uses the same nonlinear coordinates, smooth interaction fields and
nonlinear CA spring prior as local refinement. This fixes the finite-step
approximation, not the omissions of the smooth physical model.

Verification is a selection operation. The verified deformation is exported
separately; initial evaluation and local refinement start at a=0 to isolate
candidate selection. Initial and final DockQ are scored by the independent
official DockQ 2.1.3 process, RL:AD, allowed mismatches 1. Top1 means the
algorithm's own rank; best5 is explicitly an oracle. No DockQ-driven rerun or
parameter tuning is part of this fixed run.

```bash
.venv/bin/python scripts/run_induced_fit_pair.py \
  --manifest data/inducedfit/apo_pilot_v1/inference_manifest.json \
  --out data/inducedfit/apo_pilot_v2/pilot_run \
  --arms R1 R2 R2v --response-pool-size 50 \
  --field-backend sparse --checkpoint-fields \
  --rotations 128 --top-k 5 --peaks-per-rotation 4 --modes 2 \
  --radius 0.5 --max-atom-displacement 1.5 --max-translation 0.5 \
  --iterations 8 --max-evaluations 64 --spacing 2 \
  --max-grid-points 750000 --seed 0 --threads 1
```

## Independent frozen-pose diagnostics

On v1's already frozen structures, without native input or any pose optimization:

1. Evaluate the same smooth field on 2 and 1 Å grids, at zero and half-cell
   phase shifts, with a common input-derived physical extent and full support.
   Report component changes and rank stability; this does not identify a
   converged grid without a finer-grid study.
2. Evaluate standard Amber ff14SB + OpenMM OBC2/ACE implicit solvent, NoCutoff,
   solute/solvent dielectrics 1/78.5, kappa 0, Reference platform. Reconstruct
   all original hydrogens and full topology. Primary diagnostic is
   E_complex(xR,xL)-E_R(xR)-E_L(xL), in kJ/mol. Report monomer deformation energy
   against apo and their unweighted sum separately. Neither is a binding free
   energy; unminimized experimental inputs can have large strain artifacts.

These diagnostics are reference-free and their protocols are fixed before
joining the existing DockQ results. They are not silently substituted for v2's
registered smooth-energy ranking. Record full outputs, all failures and
interpretation in EXPERIMENT_REPORT.md §9.19.
