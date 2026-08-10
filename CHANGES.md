# Submission Changes

## Scientific protocol

- Audited the historical code, reports and experiment artifacts before treating any claim as canonical.
- Retired unsupported historical headline claims, including the claimed 76-asset scope and significance statement backed by an empty artifact.
- Added explicit `target_date` provenance and a target-safe chronological Train/Validation/Test split.
- Removed boundary rows whose next-day targets cross into the next partition.
- Added split invariants for chronology, uniqueness, non-overlap and target containment.
- Kept preprocessing leakage-safe by fitting `StandardScaler` on Train only.
- Added reusable Gaussian HMM diagnostics: convergence, iterations, train likelihood per observation, free-parameter count, AIC and BIC.

## Canonical evaluation

- Added `run_canonical.py` with a fixed five-asset universe and fixed 2014–2024 request window.
- Evaluated K=2/3/4 with three fixed seeds per K.
- Locked K and seed using Validation likelihood only, before the reported Test comparison.
- Evaluated four transparent baselines and the selected HMM on the same Test partition.
- Preserved every historical result directory; the canonical run is isolated at `experiments_canonical/canonical_20260810_014841/`.
- Added a programmatic audit, aggregate table and publication figures under `analysis/`.

## Submission artifacts

- Added an expanded Hebrew academic paper in XeLaTeX with theory, methods, results, discussion, limitations, bibliography, tables and figures.
- Added an executed Hebrew notebook with saved outputs and a disabled-by-default full rerun gate.
- Added a reproducible environment file and rebuilt the README around the canonical submission path.
- Added tests for leakage boundaries, HMM parameter formulas, diagnostics and canonical protocol invariants.

## Verified conclusion

The canonical HMM did not outperform the strongest pre-specified baseline in DPA on any of the five assets (three ties, two losses). The report makes no profitability or statistical-significance claim. Hidden states are interpreted only as statistical regimes characterized by return, volatility, occupancy and transition behavior.

## Remaining human-only edits

Before formal submission, fill in the title-page placeholders for student name/number, program, lecturer and date, then rebuild `reports/report.pdf`.
