# Gaussian HMM Market-Regime Analysis

Final project for **Advanced Methods in Machine Learning**, Afeka Academic College of Engineering.

The project separates two questions that are often conflated in financial ML:

1. Can a Gaussian Hidden Markov Model improve **next-day directional prediction**?
2. Can it identify **stable, interpretable latent market regimes** even when short-horizon prediction is weak?

The final evidence supports the second use more strongly than the first. The HMM is therefore interpreted primarily as a **latent market-state estimator**, not as a guaranteed trading predictor.

## Final submission files

| Deliverable | Path |
|---|---|
| Compiled Hebrew report | `reports/report.pdf` |
| XeLaTeX source | `reports/report.tex` + `reports/sections/` |
| Bibliography | `reports/references.bib` |
| Executed notebook | `HMM_Market_Regimes_Project.ipynb` |
| Notebook builder | `build_final_notebook.py` |
| Extended HMM runner | `run_extended_analysis.py` |
| External-context analysis | `analyze_extended_context.py` |
| Supervised ML baselines | `run_supervised_baselines.py` |
| Final HMM artifacts | `experiments_extended/extended_20260811_121957/` |
| Final supervised artifacts | `experiments_supervised/supervised_20260811_133344/` |
| Historical frozen canonical run | `experiments_canonical/canonical_20260810_014841/` |
| Change log | `CHANGES.md` |
| Final build handoff | `FINAL_SUBMISSION_EXECUTION.md` |

## Data and protocol

The final panel is economically diverse rather than a large redundant list of stocks:

`SPY`, `QQQ`, `IWM`, `TLT`, `GLD`, `HYG`, `BTC-USD`, `JPM`, `NVDA`.

Requested date range: `2014-01-01` through `2024-12-31`.

Features:

- log return;
- 20-day rolling volatility;
- daily range;
- log volume change.

Protocol:

- chronological Train / Validation / Test split, approximately 70% / 15% / 15%;
- target-safe boundary removal so a next-day target never crosses partitions;
- no random time-series shuffle;
- preprocessing fit on Train only;
- HMM `K ∈ {2,3,4}`;
- HMM seeds `{42,123,456,789,2026}`;
- model selection using Validation log-likelihood only;
- Test kept outside model selection.

## Final HMM analyses

The extended experiment adds:

- hard-state and soft-posterior forecasts;
- posterior confidence and normalized entropy;
- seed stability using label-invariant Adjusted Rand Index;
- three expanding walk-forward folds on SPY;
- state-conditioned prediction errors;
- SPY-conditioned cross-asset behavior;
- external post-hoc validation against VIX, which is **not** a training feature;
- empirical state duration versus the first-order HMM geometric-duration implication.

## Supervised ML baselines

The exact same feature set and chronological partitions are used for:

- Logistic Regression;
- Random Forest Classifier;
- HistGradientBoostingClassifier.

These baselines use pre-specified hyperparameters and are not tuned on Test. In addition to directional accuracy, their evaluation includes balanced accuracy, ROC-AUC, log loss and Brier score.

## Main empirical result

Mean directional prediction accuracy across the nine assets is approximately:

| Model | Mean DPA |
|---|---:|
| Naive - train mean | 54.54% |
| Logistic Regression | 54.14% |
| Random Forest | 53.27% |
| Gaussian HMM - soft posterior | 53.06% |
| HistGradientBoosting | 52.49% |
| Gaussian HMM - hard state | 52.06% |

No model demonstrates a stable universal forecasting advantage.

The stronger HMM findings concern regime structure. For SPY, the selected `K=4` model produces states with materially different return, volatility, daily range, drawdown and dwell-time characteristics. Its state partition is highly stable across seeds (mean pairwise ARI ≈ 0.995). Posterior entropy rises sharply around state switches, VIX separates several inferred states despite not being a feature, and cross-asset correlations change with the SPY hidden state.

Therefore the final conclusion is deliberately cautious:

> In this project the Gaussian HMM is more informative as a model of conditional market structure than as a point-forecasting model for the next day.

No profitability or investment-advice claim is made.

## Reinforcement Learning future work

The report contains a dedicated future-work section on **regime-aware Reinforcement Learning**. The proposed experiment would compare an RL policy using raw market features against policies that additionally receive HMM posterior probabilities and posterior entropy. Transaction costs, turnover, multiple seeds and walk-forward evaluation would be mandatory.

No RL result is claimed in the current project because no RL experiment was run.

## Environment

Recommended: Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Build the final notebook

The notebook is deterministically generated from the verified artifacts:

```bash
python build_final_notebook.py
jupyter nbconvert --to notebook --execute \
  HMM_Market_Regimes_Project.ipynb \
  --output /tmp/HMM_verified.ipynb \
  --ExecutePreprocessor.timeout=600 \
  --ExecutePreprocessor.kernel_name=python3
```

The notebook execution itself does not need to download market data or retrain the expensive experiments.

## Re-run the experiments

Extended HMM run:

```bash
python run_extended_analysis.py
```

External context diagnostics:

```bash
python analyze_extended_context.py --run-dir experiments_extended/<run_id>
```

Supervised ML baselines:

```bash
python run_supervised_baselines.py
```

Each new experiment writes a new timestamped directory and does not overwrite the verified final artifacts.

## Tests

```bash
python -m py_compile \
  build_final_notebook.py run_canonical.py run_extended_analysis.py \
  run_supervised_baselines.py analyze_extended_context.py src/*.py tests/*.py

python -m pytest tests/test_protocol_foundation.py tests/test_regime_extension.py -q
python -m pytest tests/ -q
```

Network-dependent legacy tests should be distinguished from deterministic protocol failures.

## Build the paper

```bash
cd reports
xelatex -interaction=nonstopmode -halt-on-error report.tex
bibtex report
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
```

## Provenance

- `experiments_extended/extended_20260811_121957/` is the source for the final HMM robustness and regime claims.
- `experiments_supervised/supervised_20260811_133344/` is the source for the final supervised baseline comparison.
- `experiments_canonical/canonical_20260810_014841/` is preserved as the earlier frozen canonical experiment and historical audit trail.
- Earlier exploratory directories must not be mixed silently with final numerical conclusions.

## Scope

Educational and academic use only. The project does not claim a profitable trading strategy and does not provide investment advice.
