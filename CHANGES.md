# Submission Changes

## Scientific protocol

- Audited historical code, reports and experiment artifacts before treating any claim as canonical.
- Retired unsupported historical headline claims and preserved old runs for provenance only.
- Added explicit `target_date` provenance and a target-safe chronological Train/Validation/Test split.
- Removed boundary rows whose next-day targets cross partitions.
- Added chronology, uniqueness, non-overlap and target-containment invariants.
- Fit preprocessing on Train only and kept Test completely outside model selection.
- Added Gaussian HMM diagnostics: convergence, iterations, likelihood per observation, parameter count, AIC and BIC.

## Extended HMM analysis

- Expanded the economically diverse asset panel to `SPY`, `QQQ`, `IWM`, `TLT`, `GLD`, `HYG`, `BTC-USD`, `JPM`, `NVDA`.
- Evaluated `K=2,3,4` with seeds `42,123,456,789,2026`.
- Added hard-state and soft-posterior HMM forecasts.
- Added posterior confidence and normalized posterior entropy diagnostics.
- Added seed-stability analysis using Adjusted Rand Index.
- Added three expanding walk-forward folds on SPY.
- Added state-conditioned prediction-error analysis.
- Added SPY-conditioned cross-asset return/volatility/correlation analysis.
- Added external post-hoc validation of SPY regimes against VIX, which was not used as a model feature.
- Added empirical dwell-time versus first-order HMM implied-duration diagnostics.
- Verified extended run: `experiments_extended/extended_20260811_121957/`.

## Supervised ML comparison

- Added a controlled next-day direction comparison using the same features and the same chronological splits.
- Added `Logistic Regression`, `Random Forest Classifier`, and `HistGradientBoostingClassifier` with pre-specified hyperparameters.
- Added DPA, balanced accuracy, ROC-AUC, log loss and Brier score.
- Verified that the scaler is fit only on Train and that neither Validation nor Test is used to tune these fixed baselines.
- Verified supervised run: `experiments_supervised/supervised_20260811_133344/`.

## Final empirical interpretation

- No model shows a stable universal forecasting advantage across the nine assets.
- Mean DPA across assets is approximately: Naive train-mean 54.54%, Logistic Regression 54.14%, Discrete Markov Chain 53.90%, Random Forest 53.27%, HMM soft posterior 53.06%, HistGradientBoosting 52.49%, HMM hard state 52.06%, Naive persistence 50.45%, Moving Average 5 50.27%.
- SPY `K=4` regimes are strongly differentiated descriptively by return, volatility, daily range, drawdown and dwell time.
- SPY `K=4` is highly stable across seeds (mean pairwise ARI ≈ 0.995).
- Posterior entropy is much higher on state-switch days than on non-switch days, while being only weakly related to return magnitude or rolling volatility. Because hard switches are defined from the same posterior, this is treated as an internal uncertainty diagnostic rather than independent validation.
- VIX provides partial external support for regime interpretation: SPY states correspond to different VIX environments despite VIX not being a training feature.
- Cross-asset correlations change with the SPY hidden state. This is reported as descriptive evidence only; the smaller states contain relatively few Test observations and no causal or significance claim is made.
- The final write-up now reports all pre-specified simple baselines rather than only the strongest Naive baseline, and explicitly includes MAE/RMSE/price-MAPE for the numeric-return methods.
- Added an explicit SPY `K=2/3/4` table showing Validation log-likelihood, BIC, occupancy and fit time, so model-selection claims are visible rather than merely stated.

## Notebook and report rewrite

- Replaced the old five-asset narrative with a final nine-asset research story centered on the distinction between forecasting and latent regime representation.
- Added `build_final_notebook.py`, which deterministically rebuilds `HMM_Market_Regimes_Project.ipynb` from verified result artifacts with natural Hebrew explanations around all major code/results.
- Added `patch_final_notebook.py` for final academic-review corrections and compact completeness tables; it does not retrain or alter experiment artifacts.
- Reorganized the XeLaTeX report into modular sections under `reports/sections/`.
- Filled the title page with student name, ID, program and submission month.
- Added explicit discussion of class imbalance and why DPA alone can be misleading.
- Added a substantial Future Work section on regime-aware Reinforcement Learning, including a proposed HMM-posterior ablation and a balanced literature discussion. No RL result is claimed because no RL experiment was run.
- Expanded the bibliography with PPO, SAC, FinRL and financial Deep RL studies, including both positive and cautionary empirical results.
- Added final scientific caveats distinguishing internal posterior diagnostics from external validation and clarified that the common 2014-2024 interval is a requested window rather than an identical observed start date for every asset.

## Final build / QA status

A complete reviewed build was executed and committed at `eda3ddc3178be6f3a9513c9a4bfb47a573efa5d5`:

- notebook executed top-to-bottom with 38 cells and all 11 code cells completed without error outputs;
- `26 passed` in the full pytest run;
- report compiled to a 20-page PDF;
- zero TeX errors, missing glyphs, unresolved references or unresolved citations were reported;
- all PDF pages were rendered successfully.

After that build, the final completeness audit found that the headline comparison table omitted three already-executed, pre-specified simple baselines and that MAE/RMSE/MAPE plus the explicit SPY K comparison were not visible enough in the final narrative. The source and notebook patch have now been corrected without running any new experiment. One final notebook execution and XeLaTeX/BibTeX rebuild are therefore required before merge to `main`.
