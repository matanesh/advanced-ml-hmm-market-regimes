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
- Mean DPA across assets is approximately: Naive train-mean 54.54%, Logistic Regression 54.14%, Random Forest 53.27%, HMM soft posterior 53.06%, HistGradientBoosting 52.49%, HMM hard state 52.06%.
- SPY `K=4` regimes are strongly differentiated by return, volatility, daily range, drawdown and dwell time.
- SPY `K=4` is highly stable across seeds (mean pairwise ARI ≈ 0.995).
- Posterior entropy is much higher on state-switch days than on non-switch days, while being only weakly related to return magnitude or rolling volatility.
- VIX provides partial external validation: SPY states correspond to materially different VIX environments despite VIX not being a training feature.
- Cross-asset correlations change with the SPY hidden state, supporting the interpretation of HMM as a latent market-condition estimator rather than only a ticker-specific clustering device.

## Notebook and report rewrite

- Replaced the old five-asset narrative with a final nine-asset research story centered on the distinction between forecasting and latent regime representation.
- Added `build_final_notebook.py`, which deterministically rebuilds `HMM_Market_Regimes_Project.ipynb` from verified result artifacts with natural Hebrew explanations around all major code/results.
- Reorganized the XeLaTeX report into modular sections under `reports/sections/`.
- Filled the title page with student name, ID, program and submission month.
- Added explicit discussion of class imbalance and why DPA alone can be misleading.
- Added a substantial Future Work section on regime-aware Reinforcement Learning, including a proposed HMM-posterior ablation and a balanced literature discussion. No RL result is claimed because no RL experiment was run.
- Expanded the bibliography with PPO, SAC, FinRL and financial Deep RL studies, including both positive and cautionary empirical results.

## Remaining execution/QA before submission

1. Run `python build_final_notebook.py`.
2. Execute the rebuilt notebook top-to-bottom with `nbconvert` and commit the executed notebook.
3. Compile `reports/report.tex` with XeLaTeX/BibTeX.
4. Render and visually inspect every PDF page for RTL/LTR, clipped tables, figures and equations.
5. Run the relevant pytest suite and syntax checks once more.
6. Update README to point to the final extended/supervised artifacts.
7. Merge the final reviewed branch into `main` only after the above checks pass.
