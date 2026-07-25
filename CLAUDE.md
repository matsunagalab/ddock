# Project Research Log

This repository is an experimental research project. Keep
`EXPERIMENT_REPORT.md` as the cumulative source of truth for the research.

Whenever you design, implement, run, debug, or interpret an experiment:

- Update `EXPERIMENT_REPORT.md` in the same work session.
- Record the question or hypothesis, rationale, implementation, dataset and
  leakage controls, split, parameters, loss, seed, hardware, and commands.
- Record numerical results, including failed runs, null results, instability,
  skipped samples, OOMs, and other caveats. Do not report only favorable runs.
- Clearly separate measured facts from hypotheses and interpretation.
- Explain comparisons against the relevant baseline and whether model or
  checkpoint selection used train, validation, or test data.
- Preserve prior results and corrections as research history; do not silently
  replace contradictory or invalidated conclusions.
- End each experiment section with its conclusion, limitations, and justified
  next experiments.

Do not commit large generated datasets, caches, checkpoints, or logs. Document
how to reproduce them instead.
