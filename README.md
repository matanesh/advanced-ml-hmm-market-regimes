# Gaussian HMM Market-Regime Analysis

Final project for **Advanced Methods in Machine Learning**, Afeka Academic College of Engineering.

This repository evaluates whether a multivariate Gaussian Hidden Markov Model can identify interpretable market regimes and improve next-day return-direction prediction relative to transparent baselines. The submission emphasizes chronological evaluation, leakage prevention, validation-only model selection, reproducibility, and cautious reporting.

## Submission files

| Deliverable | Path |
|---|---|
| Compiled paper (Hebrew, PDF) | `reports/report.pdf` |
| Editable LaTeX source | `reports/report.tex` |
| Bibliography | `reports/references.bib` |
| Executed notebook | `HMM_Market_Regimes_Project.ipynb` |
| Canonical runner | `run_canonical.py` |
| Canonical run | `experiments_canonical/canonical_20260810_014841/` |
| Independent artifact audit | `analysis/canonical_findings.md` |
| Scientific audit of historical work | `AUDIT.md` |
| Changes made for submission | `CHANGES.md` |

> Before final submission, fill in the student name/ID, program, lecturer, and submission date placeholders on the PDF title page and rebuild it.

## Canonical experiment

The fixed universe is `SPY`, `GLD`, `TLT`, `BTC-USD`, and `DIS`, requested over 2014-01-01 through 2024-12-31. Four features are used: log return, 20-day rolling volatility, daily range, and log volume change.

The protocol uses:

- chronological Train/Validation/Test partitions (approximately 70%/15%/15%);
- target-safe boundary removal so a next-day target never crosses partitions;
- a scaler fitted on Train only;
- `K ∈ {2,3,4}` and seeds `{42,123,456}`;
- validation log-likelihood for locking K and seed;
- one final evaluation on the held-out Test partition;
- four baselines evaluated on the identical Test observations;
- convergence, iterations, parameter count, AIC, BIC, likelihood and forecast-error diagnostics.

This is a **single held-out split per asset**, not rolling-origin evaluation.

## Main result

The selected HMM achieved mean DPA of **52.81%** across the five assets. It did not outperform the strongest pre-specified baseline on any asset (three ties and two losses). The states remain descriptively useful for identifying differences in volatility, occupancy and persistence, but this run does **not** establish forecasting superiority, statistical significance, or trading profitability.

## Environment setup

Recommended: Python 3.11+ in a clean virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Review the executed notebook

```bash
jupyter lab HMM_Market_Regimes_Project.ipynb
```

The committed notebook loads the canonical artifacts by default and runs without downloading data or retraining. The expensive path is explicitly gated by `RUN_FULL_EXPERIMENTS = False`.

Clean execution check:

```bash
jupyter nbconvert --to notebook --execute \
  HMM_Market_Regimes_Project.ipynb \
  --output /tmp/HMM_verified.ipynb \
  --ExecutePreprocessor.timeout=600 \
  --ExecutePreprocessor.kernel_name=python3
```

## Re-run the canonical experiment

```bash
python run_canonical.py
```

A new timestamped directory is created under `experiments_canonical/`. Market data are downloaded through `yfinance`; because upstream historical data can be revised, a new run may not be bit-for-bit identical to the committed canonical artifact.

## Tests

```bash
python -m pytest tests/ -q
python test_setup.py
python -m py_compile run_canonical.py src/*.py tests/*.py
```

## Build the paper

XeLaTeX and BibTeX are required.

```bash
cd reports
xelatex -interaction=nonstopmode -halt-on-error report.tex
bibtex report
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
```

## Repository status and provenance

- `experiments_canonical/canonical_20260810_014841/` is the sole source for numerical claims in the paper and notebook.
- Earlier exploratory directories are retained for historical context only and must not be mixed with canonical conclusions.
- `AUDIT.md` documents why older headline claims were retired.
- `analysis/canonical_findings.md` documents the machine-readable cross-check and known schema/provenance limitations.

## Scope

Educational and academic use. The project does not provide investment advice or claim a profitable trading strategy.
