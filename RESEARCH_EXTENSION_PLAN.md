# Research extension plan — HMM market regimes

This document defines the next controlled extension of the project. It intentionally separates **predictive performance** from **latent regime discovery**.

## Core research questions

1. **Prediction:** Does a Gaussian HMM improve next-day directional prediction relative to simple baselines?
2. **Representation:** Does the HMM partition the market into statistically distinct and interpretable regimes?
3. **Regime information:** Do hidden-state identity and posterior uncertainty contain useful information about volatility, drawdowns, persistence, and cross-asset behavior even when next-day DPA is weak?

The project should treat a negative answer to Question 1 as a valid scientific result. The main contribution may instead be that HMM behaves better as a **latent market-state estimator** than as a short-horizon return predictor.

## Asset panel

Use a small, economically diverse panel rather than many redundant stocks:

- **SPY** — broad U.S. equity market; primary reference asset.
- **QQQ** — growth / technology-heavy equities.
- **IWM** — U.S. small-cap equities.
- **TLT** — long-duration U.S. Treasuries.
- **GLD** — gold.
- **HYG** — high-yield corporate credit.
- **BTC-USD** — crypto / high-volatility alternative asset.
- **JPM** — large financial equity.
- **NVDA** — high-beta / growth equity.

Optional additions should have a clear economic role, not merely increase ticker count. Examples: XLE for energy, XLU for defensive utilities, EEM for emerging markets.

## Required regime diagnostics

For each selected K and especially the selected model:

- mean log return by state;
- return volatility by state;
- daily range by state;
- volume-change behavior by state when available;
- occupancy;
- average dwell time;
- transition probabilities;
- drawdown behavior by state;
- posterior state probabilities;
- posterior entropy as an uncertainty measure.

Semantic state labels must be assigned only after these statistics are inspected.

## Posterior uncertainty analysis

For each day, compute posterior probabilities

`P(S_t = k | X_1:T)`

and normalized entropy

`H_t = -sum_k p_tk log(p_tk) / log(K)`.

Questions to test:

- Is entropy higher near hard state switches?
- Are high-entropy periods associated with higher realized volatility or larger absolute returns?
- Are state assignments materially more stable when posterior confidence is high?

A positive result would support the interpretation that the model captures not only regimes but also uncertainty around regime boundaries.

## Stability analysis

A hidden state is useful only if the discovered structure is not a one-off artifact of initialization or one train/test split.

Perform two complementary checks:

1. **Seed stability:** repeat each K over multiple random seeds. Compare likelihood, BIC/AIC, predictive metrics, and state statistics after label alignment.
2. **Time-window stability:** use a small number of chronological walk-forward or expanding-window splits. The purpose is robustness, not hyperparameter mining.

Avoid selecting the best seed ex post. Report the distribution or mean ± standard deviation across runs where appropriate.

## Cross-asset regime analysis

Use SPY as the reference market-regime detector and examine the behavior of other assets conditional on SPY's inferred state.

For each SPY state, summarize for QQQ, IWM, TLT, GLD, HYG, BTC-USD and selected equities:

- mean return;
- volatility;
- correlation with SPY;
- downside frequency;
- average absolute return.

This tests whether the HMM state is economically broader than a simple label attached only to SPY.

A particularly interesting finding would be that a state associated with stressed equity conditions also changes cross-asset correlation structure or relative defensive behavior.

## Prediction conditioned on regime

Evaluate model/baseline errors separately by inferred regime. This can reveal information hidden by aggregate DPA.

Questions:

- Are prediction errors concentrated in high-volatility states?
- Does any method perform disproportionately better or worse in specific regimes?
- Is next-day direction inherently less predictable near high-entropy regime transitions?

Do not use test labels to redesign the model after seeing the result; this is descriptive post-hoc analysis.

## Main conclusion to test

The strongest defensible conclusion, if supported by the data, is:

> The Gaussian HMM is more informative as a model of conditional market structure than as a point-forecasting model. Its hidden states define distinct return/risk environments and its posterior probabilities quantify uncertainty around regime changes, even when next-day directional accuracy does not consistently beat simple baselines.

This conclusion must remain conditional on the empirical results. If the diagnostics do not support it, the report should say so explicitly.

## Execution discipline

- Keep the original canonical run untouched.
- Write all new outputs to a new experiment directory.
- Record exact dates, tickers, seeds, K, features and package versions.
- Fit preprocessing only on training data.
- Use chronological splits only.
- Do not claim trading profitability.
- Do not choose assets or seeds solely because they produce attractive results.
