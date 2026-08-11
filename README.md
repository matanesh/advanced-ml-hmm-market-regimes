# Gaussian HMM Market-Regime Analysis

Final project for **Advanced Methods in Machine Learning**, Afeka Academic College of Engineering.

This repository evaluates two related questions: whether a multivariate Gaussian Hidden Markov Model improves next-day return-direction prediction relative to transparent baselines, and whether it identifies statistically meaningful latent market regimes even when short-horizon prediction is weak. The submission emphasizes chronological evaluation, leakage prevention, validation-only model selection, reproducibility, posterior uncertainty and cautious reporting.

## Submission files

| Deliverable | Path |
|---|---|
| Compiled paper (Hebrew, PDF) | `reports/report.pdf` |
| Editable LaTeX source | `reports/report.tex` |
| Bibliography | `reports/references.bib` |
| Executed notebook | `HMM_Market_Regimes_Project.ipynb` |
| Canonical runner | `run_canonical.py` |
| Canonical run | `experiments_canonical/canonical_20260810_014841/` |
| Extended robustness runner | `run_extended_analysis.py` |
| Extension design | `RESEARCH_EXTENSION_PLAN.md` |
| Independent artifact audit | `analysis/canonical_findings.md` |
| Scientific audit of historical work | `AUDIT.md` |
| Changes made for submission | `CHANGES.md` |

> The original canonical artifact is intentionally preserved. New robustness outputs are written under `experiments_extended/` and must not silently replace the canonical numerical claims until they have been inspected and incorporated into the notebook/report.

## Canonical experiment

The frozen canonical universe is `SPY`, `GLD`, `TLT`, `BTC-USD`, and `DIS`, requested over 2014-01-01 through 2024-12-31. Four features are used: log return, 20-day rolling volatility, daily range, and log volume change.

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

## Frozen canonical result

The selected HMM achieved mean DPA of **52.81%** across the five canonical assets. It did not outperform the strongest pre-specified baseline on any asset (three ties and two losses). The states remained descriptively useful for identifying differences in volatility, occupancy and persistence, but this run does **not** establish forecasting superiority, statistical significance, or trading profitability.

This negative predictive result motivates the extension below rather than being hidden or replaced.

## Extended regime / robustness analysis

`run_extended_analysis.py` asks whether the HMM is more informative as a **latent market-state estimator** than as a point-forecasting model.

The economically diverse panel is:

- `SPY` — broad U.S. equities and the reference regime series;
- `QQQ` — growth / technology-heavy equities;
- `IWM` — U.S. small caps;
- `TLT` — long-duration U.S. Treasuries;
- `GLD` — gold;
- `HYG` — high-yield corporate credit;
- `BTC-USD` — crypto / high-volatility alternative asset;
- `JPM` — large financial equity;
- `NVDA` — high-beta growth equity.

The extension adds four controlled analyses:

1. **Seed stability.** K=2/3/4 are repeated across five random initializations. Label-invariant Adjusted Rand Index quantifies whether different fits partition the same dates similarly.
2. **Posterior uncertainty.** Past-only posterior state probabilities are converted to normalized entropy and confidence. The analysis checks whether uncertainty is elevated around inferred state transitions and turbulent observations.
3. **Expanding-window robustness.** SPY is evaluated over three chronological walk-forward folds, with model selection repeated independently inside each fold.
4. **Cross-asset regime behavior.** Other assets are summarized conditional on SPY's held-out inferred state, including mean return, volatility, downside frequency, correlation and beta to SPY.

The extension also compares the original hard-state HMM forecast with a **soft-posterior forecast**. The soft version propagates the full current posterior through the transition matrix before taking the expected state-conditional return. It therefore uses HMM uncertainty instead of discarding it through a single hard state assignment.

### Smoke test

The smoke test uses two seeds and skips walk-forward analysis:

```bash
python run_extended_analysis.py --quick
```

### Full extension

```bash
python run_extended_analysis.py
```

Market data are cached in `data_cache/` after a successful download so repeated experiments do not depend unnecessarily on Yahoo Finance availability. To deliberately refresh the cache:

```bash
python run_extended_analysis.py --force-download
```

Every full run creates a timestamped directory under `experiments_extended/` containing candidate diagnostics, seed-stability tables, selected-model state summaries, posterior time series, regime-conditioned prediction metrics, cross-asset analysis, walk-forward results and a provenance manifest.

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

The committed notebook currently loads the frozen canonical artifacts by default and does not download data or retrain models. The new extended artifacts should be incorporated only after a verified full extension run is available.

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

A new timestamped directory is created under `experiments_canonical/`. The frozen canonical artifact remains the source of the original headline result.

## Tests

Offline protocol and extension checks:

```bash
python -m pytest tests/test_protocol_foundation.py tests/test_regime_extension.py -q
```

Complete suite:

```bash
python -m pytest tests/ -q
python test_setup.py
python -m py_compile run_canonical.py run_extended_analysis.py src/*.py tests/*.py
```

Some legacy integration tests may require Yahoo Finance access; a network/rate-limit failure should be distinguished from a model or protocol failure.

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

- `experiments_canonical/canonical_20260810_014841/` is the frozen source for the original canonical claims.
- `experiments_extended/` contains new robustness analyses and is versioned separately.
- Earlier exploratory directories are retained for historical context only and must not be mixed silently with canonical conclusions.
- `AUDIT.md` documents why older headline claims were retired.
- `analysis/canonical_findings.md` documents the machine-readable cross-check and known schema/provenance limitations.

## Scope

Educational and academic use. The project does not provide investment advice or claim a profitable trading strategy. A central hypothesis of the extension is that the HMM may be more valuable for **conditional market-structure representation** than for next-day return prediction; this is treated as a testable empirical claim, not an assumption.
