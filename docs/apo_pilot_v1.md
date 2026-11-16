# Real apo pilot v1: barnase/barstar

Registered 2026-11-16 before docking. One historical development system,
1BRS, with separately crystallized apo inputs 1A2P chain B and 1A19 chain A.
This small-deformation case tests execution and scoring on real proteins. It is
not an untouched test set, a representative flexible cohort, or a HDOCK comparison.

## Input model fixed before prediction

Use all deposited selected-chain standard heavy ATOM coordinates. Exclude
crystallographic Zn and solvent explicitly; do not invent missing heavy atoms.
The preparation script preserves every retained ATOM line and records removed
records, metal distances, original/derived SHA256 and model assumptions.

1A2P B has 108 resolved residues (864 heavy atoms); its N-terminal Ala1/Gln2
are missing. The observed residue 3 is treated as a charged model N terminus.
Nine deposited atoms have zero occupancy and remain in the input as uncertain
modeled coordinates. The removed Zn contacts Glu60, so its removal changes the
chemical model even though coordinates are fixed. The source paper describes
this Zn as mediating crystal contacts. 1A19 A is free barstar C82A, with 89
residues (719 heavy atoms). These assumptions are limitations, not silent repairs.

Both inputs pass ff14SB preparation: 1,700/1,433 all atoms including H. Use
OpenMM 8.6.1 Reference, pH 7 and seed 0. The inference manifest has no native
path or scoring labels. Source copies and policy are retained under
`data/inducedfit/apo_pilot_v1/`.

## Fixed docking protocol

| Setting | Value |
| --- | --- |
| Arms | R0 rigid selection/rigid refinement; R1 rigid selection/flexible refinement; R2 response selection/flexible refinement |
| Input frame | Independent heavy-atom centering, no native alignment |
| Rotations | Identity + 127 seeded random unit quaternions, seed 0 |
| Candidate budget | Top 5 per arm, 4 peaks/rotation, 3 Å within-rotation suppression |
| Modes | Receptor-only, 2 input ANM/torsion modes; fit damping 0.001 |
| Amplitude radius | Requested 0.5 Å, conservatively reduced by maximum atom motion |
| Nonlinear bounds | CA RMSD within effective radius, every atom ≤1.5 Å |
| Local pose bounds | Translation ≤0.5 Å, rotation ≤0.15 rad |
| Local budget | 8 iterations, maximum 64 energy evaluations per candidate |
| Initial deformation | Zero for R1 and R2; same solver and bounds |
| Grid | Fixed input-only cube, 2 Å spacing; preflight 83³ nodes |
| Field | Same smooth occupancy/surface/charge model; surface width 0.5 Å, repulsion 3, charge 1 |
| Charge kernel | Gaussian σ0.6 Å; screened kernel κ0.1 Å⁻¹, 20 Å cutoff |
| Compute | CPU float64, one PyTorch thread; sparse exact compact-support field builder; local activation checkpointing |
| Memory guard | Maximum 750,000 molecular-grid points; not a process RSS guarantee |
| Stop rule | One fixed run; retain errors/timeouts and do not retune against DockQ |

Sparse field evaluation visits only atom-local grid stencils. It uses the same
compact Gaussian/taper, per-atom charge normalization and full linear FFT
correlation. Dense/sparse values and gradients must pass independent numerical
checks before the pilot. The sparse implementation changes computation, not the
physical energy. Initial and refined structures are exported for all candidates.

```bash
.venv/bin/python scripts/prepare_historical_apo_pilot.py --help
.venv/bin/python scripts/run_induced_fit_pair.py \
  --manifest data/inducedfit/apo_pilot_v1/inference_manifest.json \
  --out data/inducedfit/apo_pilot_v1/pilot_run \
  --arms R0 R1 R2 --field-backend sparse --checkpoint-fields \
  --rotations 128 --top-k 5 --peaks-per-rotation 4 --modes 2 \
  --radius 0.5 --max-atom-displacement 1.5 --max-translation 0.5 \
  --iterations 8 --max-evaluations 64 --spacing 2 \
  --max-grid-points 750000 --seed 0 --threads 1
```

The runtime is a diagnostic, not a controlled throughput benchmark. A small
angular sample and grid coarser than charge σ limit interpretation. No learning,
loss fitting, checkpoint selection or native-guided parameter choice occurs.

## Evaluation fixed before prediction

Use official DockQ 2.1.3 in a separate environment/process, native 1BRS chains
A/D, mapping predicted R/L to A/D. Native-vs-native sanity must be approximately
1 before evaluating predictions. Sequence audit finds one barstar substitution
and two native-side missing residues; fix `--allowed_mismatches 1` and retain the
alignment/matching-residue policy. This is set from sequence identity, not DockQ.

Report per-candidate initial/final DockQ, iRMSD, LRMSD, fnat, status and final
energy rank. Summarize Top1 and best of the five retained candidates, not an
oracle over unexported poses. Include preparation/search/refinement/scorer
failures separately. A small energy decrease need not improve DockQ, and a
failed pilot must not be described as evidence of HDOCK superiority.

Results, deviations and justified next experiments belong in
`EXPERIMENT_REPORT.md` §9.18, preserving all earlier preparation failures.
