# Scientific and Submission Audit

**Audit date:** 2026-08-09  
**Branch:** `overnight/submission-ready-hmm`  
**Status:** The historical project is useful exploratory work, but its headline results are **not canonical confirmatory evidence**. A corrected, versioned protocol must be run before final conclusions are written.

## Scope and source requirements

The audit checked the original HMM project proposal, the supplied academic-submission standard and checklist, source modules, experiment runners, historical result directories, analysis scripts, README and draft report.

The original research question remains appropriate: whether a Gaussian HMM that learns latent statistical regimes improves next-day directional prediction relative to simple baselines, and how the result changes for K=2, K=3 and K=4.

## What is already sound

- Daily observations are sorted by date.
- The main split uses chronological `iloc` slicing without shuffle.
- `StandardScaler` is fitted on training features and only transforms test features.
- Core features use current/past observations: log return, trailing 20-day volatility, daily range and volume change.
- Gaussian HMM is the central model and K=2/3/4 use the same split, features and covariance family.
- Required baselines exist: train-mean, persistence, moving average and a direction-conditioned observable baseline.
- Test-state decoding uses only a prefix ending at the current test date; it does not smooth with future test observations.
- Historical artifacts contain state summaries, transition matrices and plots that can support exploratory interpretation.

## Critical scientific findings

### 1. One-observation boundary-target leakage

`next_log_return` and `next_close` are created with `shift(-1)` before the chronological split. The final training row therefore has a target taken from the first test observation. The HMM fit itself uses feature columns and is not directly trained on this target, but train-target baselines and downstream classifiers use the contaminated row.

**Required correction:** define a target-safe split that removes the final row of every training/validation segment when its target lies in the following segment. Add assertions for monotonic indexes, non-overlap and target dates.

### 2. Model selection on the final test set

K and the headline “best model” are selected by final-test DPA. Repeated runs for a ticker are also reduced by maximum test DPA. Winning assets are then selected from the same test results for significance analysis. This creates optimistic selection bias.

**Required correction:** freeze a Train/Validation/Test or rolling-origin protocol. Select K, initialization and any context setting only from training/validation criteria. Evaluate the locked choice once on each held-out test fold. Report all pre-specified assets, not only winners.

### 3. Post-selection significance and missing evidence

The McNemar analysis was applied only to test-set winners without correction for multiple comparisons or serial dependence. The implementation calls removed SciPy API `stats.binom_test`; the referenced `analysis/significance_tests.csv` is one byte and contains no results. The report's p-values are therefore not backed by a valid saved artifact.

**Required correction:** withdraw the old significance claim. For the corrected pre-specified universe, report paired effect sizes and block-bootstrap confidence intervals; if formal tests are retained, adjust across the declared family and explain dependence limitations.

### 4. Missing convergence and complexity diagnostics

HMM fitting does not save `model.monitor_.converged`, iteration count or likelihood history. Warnings are suppressed. AIC and BIC are absent, and raw likelihood is not normalized by sequence length.

**Required correction:** save convergence diagnostics, iterations, per-observation log-likelihood, parameter count, AIC and BIC for each K and initialization. Failed/non-converged starts remain in the audit trail rather than disappearing.

### 5. Single-start/local-optimum risk

Most historical models use one seed. Existing NVDA seed sensitivity includes a visibly inferior likelihood for one seed, showing that local optima are material.

**Required correction:** run multiple pre-specified initializations for every K, select the fitted start by training or validation likelihood only, and retain all start diagnostics.

### 6. Metric and protocol inconsistencies

- Direction is `>=0` in the main evaluator but `>0` in robustness/significance scripts.
- HMM runtime includes fit only, whereas baseline runtime includes prediction/evaluation.
- Test likelihood starts from the model's initial distribution while forecasting uses training context.
- The “Discrete Markov Chain” is a conditional-mean predictor by current direction rather than an explicitly saved transition-probability model.
- Forecast timing must be documented as after market close on day t, because day-t High/Low/Volume are features for t+1.

**Required correction:** centralize direction labels and metrics, define comparable fit and end-to-end runtime fields, state likelihood conditioning clearly, and implement/document the observable Markov baseline explicitly.

## Historical artifact audit

### Inventory

- `experiments/`: one old SPY run.
- `experiments_multi_asset/`: 15 historical asset runs, including crypto.
- `experiments_extended/`: 100 run directories, of which 99 are complete, representing 51 unique tickers.
- `analysis/`: exploratory winning-case, robustness and plot artifacts.
- `interpretation/`: three PNG files without independent tabular provenance.
- `model_comparison/`: five short-period comparison CSVs using a different 2020+ estimand.

### Unsupported or inconsistent claims

- The draft report says 76 assets; the completed extended summary contains 51 unique assets. Its model-count totals also sum to 51.
- The extended universe does not include crypto; crypto exists only in the separate historical multi-asset set.
- `summary.csv` has 99 rows and repeated tickers; it is stale append-only output.
- `summary_with_dirname.csv` selects maximum test DPA among repeated runs for some tickers and is therefore exploratory, not confirmatory.
- One run is partial: `run_20260808_184936_SPY_K[2, 3, 4]`.
- One transition JSON is malformed: `run_20260808_185612_MSFT_K[2, 3, 4]/transition_analysis_K4.json`.
- `MODEL_COMPARISON_SUMMARY.md` contradicts its CSV for NVDA.
- `PROJECT_COMPLETED.md` incorrectly labels the project submission-ready.

### Historical evidence policy

Historical results are preserved and may be described as **exploratory provenance**. They must not be overwritten. The old p-value claim is withdrawn. New confirmatory conclusions will come only from a versioned canonical directory with a manifest, fixed dates/universe/protocol, hashes and schema validation.

## Submission gaps

- No Jupyter notebook exists.
- No report source or PDF exists.
- No canonical manifest or frozen input snapshot exists.
- No exact environment lock exists.
- README dates and experiment descriptions are inconsistent.
- Generated CSV/PNG artifacts are ignored by Git, so a clean clone cannot reproduce the claimed report.
- No `CHANGES.md` or lecturer-facing ZIP exists.
- Approval of the external reference article is not documented in the available files; this remains a user/lecturer confirmation item.

## Corrected overnight study design

The corrected study will prioritize a small, pre-specified and interpretable universe rather than another broad search. Proposed core assets represent distinct classes without being selected by historical HMM success:

- `SPY` — broad US equity market.
- `GLD` — gold/alternative defensive exposure.
- `TLT` — long-duration US Treasury exposure.
- `BTC-USD` — cryptocurrency market.
- `DIS` — individual equity retained as an explicitly labeled historical case study, not as confirmatory proof.

Use a fixed end date, a chronological rolling-origin design with non-overlapping test folds, multi-start K=2/3/4 selection within training/validation only, and the same baselines in every fold. Save every asset × fold × model result.

## Original extension priorities

1. **Rolling-origin stability across asset classes** — test whether relative HMM performance and inferred regime structure persist across multiple held-out periods.
2. **State stability across initializations** — align labels post hoc and quantify ARI, occupancy, dwell-time and transition-matrix stability. Semantic names are allowed only when quantitative ordering is stable.
3. Optional only after core gates: **filtered posterior uncertainty** with Brier score/calibration, using past-only probabilities and no threshold tuning on test.

These extensions answer model-reliability questions; they are not trading strategies and will not be presented as evidence of profitability.

## Immediate acceptance gates

1. Tests reproduce and then eliminate the boundary leak.
2. Corrected runner cannot select K from test metrics.
3. Every fitted HMM records seed, convergence, iterations, likelihood, AIC/BIC and runtime.
4. Canonical manifest declares universe, dates, folds, features, baselines, seeds and file hashes.
5. Notebook executes top-to-bottom from saved artifacts with no hidden local dependency.
6. Report claims reconcile exactly with machine-readable outputs.
7. PDF passes page-by-page visual QA.
8. Final ZIP passes isolated extraction/run checks.
