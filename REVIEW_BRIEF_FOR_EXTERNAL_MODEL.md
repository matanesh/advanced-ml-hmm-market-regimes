# Independent Review Brief

## Review target

Graduate-level final project for the course **Advanced Methods in Machine Learning** at Afeka Academic College of Engineering. The project evaluates multivariate Gaussian HMMs for market-regime representation and next-day direction forecasting.

## Authoritative artifacts

1. `reports/report.pdf` and `reports/report.tex`
2. `HMM_Market_Regimes_Project.ipynb`
3. `run_canonical.py`
4. `src/data.py`, `src/model.py`, `src/evaluation.py`
5. `experiments_canonical/canonical_20260810_014841/`
6. `analysis/canonical_findings.md`
7. `AUDIT.md`

Do not use older experiment directories as evidence for current conclusions.

## Protocol anchors

- Assets: SPY, GLD, TLT, BTC-USD, DIS.
- Requested range: 2014-01-01 through 2024-12-31.
- One chronological Train/Validation/Test split per asset; this is not rolling-origin evaluation.
- Features: log return, 20-day rolling volatility, daily range, log volume change.
- Target: next-day log return with explicit target date.
- Boundary targets are removed; scaler is fitted on Train only.
- K=2/3/4; seeds 42/123/456; full covariance.
- K and seed are selected using Validation log-likelihood only.
- Test DPA/MAE/RMSE/MAPE are reported after selection.

## Canonical result anchors

- All 5 assets completed; all 45 HMM starts converged.
- Selected configurations: SPY K4/456, GLD K4/123, TLT K3/123, BTC-USD K4/123, DIS K4/123.
- Mean HMM DPA: 52.81%.
- Mean strongest-baseline DPA: 53.18%.
- HMM vs strongest baseline: 0 wins, 3 ties, 2 losses.
- No statistical significance or profitability claim is supported.

## Known limitations and disclosed corrections

- Only one held-out Test interval per asset.
- The canonical manifest lacks artifact hashes and populated artifact paths.
- Data are sourced through yfinance and no raw-data snapshot is committed.
- Validation is not refitted into the final model after selection.
- Test likelihood is computed for all starts before selection, although it is not used in selection.
- The implementation named `Discrete Markov Chain` is more precisely a direction-conditioned mean-return baseline.
- Historical significance and broad-universe claims were retired after audit.
- Student/lecturer metadata on the title page remain visible placeholders.

## Required review format

Return findings in three severities:

- **P0:** invalidates a main result, evidence chain, leakage claim, or reproducibility claim.
- **P1:** substantive scientific, implementation, interpretation, or presentation issue that should be fixed before submission.
- **P2:** polish, readability, citation or maintainability improvement.

For each finding, cite an exact file and line/cell/artifact, explain why it matters, and propose a concrete correction. Also check for generic, promotional, repetitive or unsupported academic prose. Do not infer authorship or attempt AI-detector evasion; review academic authenticity through specificity, evidence and transparent limitations.

## Verification commands

```bash
python -m pytest tests/ -q
jupyter nbconvert --to notebook --execute HMM_Market_Regimes_Project.ipynb \
  --output /tmp/HMM_verified.ipynb --ExecutePreprocessor.timeout=600
cd reports
xelatex -interaction=nonstopmode -halt-on-error report.tex
bibtex report
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
```
