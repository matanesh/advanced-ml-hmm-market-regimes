# Advanced ML HMM Market Regimes Project

This project implements Gaussian Hidden Markov Models (HMM) for detecting hidden market regimes and predicting next-day direction in financial time series. The work was conducted as part of the "Advanced Methods in Machine Learning" course final project.

## Project Structure

```
advanced-ml-hmm-market-regimes/
├── src/
│   ├── __init__.py
│   ├── data.py          # Data loading, validation, and feature engineering
│   ├── model.py         # HMM training and inference functions
│   └── evaluation.py    # Evaluation metrics and state summarization
├── experiments_multi_asset/   # Directory where experiment results are saved
├── run_experiment.py    # Script to run a single asset experiment
├── run_multi_asset_experiments.py  # Script to run experiments for multiple assets
├── run_remaining_experiments.py    # Script to run experiments for remaining assets
├── test_setup.py        # Script to test the project setup
├── hmm_run_original.py  # Original script (for reference)
�└── README.md
```

## Features

- **Data Source**: Uses yfinance to download historical price data.
- **Features**: Log returns, rolling volatility, daily range, volume change.
- **Model**: Gaussian HMM (via hmmlearn) with Baum-Welch training.
- **Evaluation**: 
  - Directional Prediction Accuracy (DPA)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
  - Log-Likelihood
  - Runtime
- **Baselines**: 
  - Naive (train mean return)
  - Naive (persistence)
  - Moving Average
  - Discrete Markov Chain
- **Regime Interpretation**: After training, hidden states are analyzed for mean return, volatility, and duration.
- **Reproducibility**: Fixed random seed, chronological train/test split, no future information leakage.

## Requirements

- Python 3.11+
- Required packages: yfinance, hmmlearn, scikit-learn, pandas, numpy, matplotlib

Install with:
```bash
pip install yfinance hmmlearn scikit-learn pandas numpy matplotlib
```

## Usage

### Single Asset Experiment

To run an experiment for a single asset (e.g., SPY) and save results:

```bash
python run_experiment.py
```

This will create a directory `experiments/spy_default/` with results.

### Multiple Assets Experiment

To run experiments for a predefined set of assets (as per the project requirements):

```bash
python run_multi_asset_experiments.py
```

This will run experiments for:
- Broad indices/ETFs: SPY, QQQ, IWM
- Large-cap equities: AAPL, MSFT, NVDA, JPM, AMZN
- Sector/defensive/alternative: XLE, GLD, TLT, EFA
- Crypto: BTC-USD, ETH-USD
- Optional: DIA

Results are saved in `experiments_multi_asset/` with subdirectories for each asset.

### Remaining Assets

If you need to run additional assets (e.g., after adding more tickers), use:

```bash
python run_remaining_experiments.py
```

## Experiment Output

For each experiment, the following are saved:
- `config.json`: The configuration used
- `results.csv`: Evaluation metrics for all models (baselines and HMMs)
- `state_table_K{K}.csv`: Summary of hidden states (for each K)
- `transition_matrix_K{K}.csv`: Transition matrix (for each K)
- Plots:
  - `{ticker}_price_over_time.png`
  - `{ticker}_log_returns.png`
  - `{ticker}_rolling_volatility.png`
  - `{ticker}_price_states_K{K}.png`
  - `{ticker}_returns_states_K{K}.png`
  - `{ticker}_transition_matrix_K{K}.png`
  - `{ticker}_state_boxplot_K{K}.png`

## Key Findings

From the experiments conducted:
- For most traditional assets (SPY, QQQ, IWM, AAPL, MSFT, NVDA, JPM, XLE, GLD, TLT, EFA, DIA), the naive baseline (train mean return) achieved the highest DPA.
- For cryptocurrencies (BTC-USD, ETH-USD) and AMZN, Gaussian HMM achieved the highest DPA:
  - BTC-USD: Gaussian HMM K=2 (DPA: 0.5104)
  - ETH-USD: Gaussian HMM K=4 (DPA: 0.5378)
  - AMZN: Gaussian HMM K=4 (DPA: 0.5376)

This suggests that Gaussian HMM may be particularly useful for detecting regimes in more volatile or cryptocurrency markets.

## Notes

- The project does not claim profitable trading strategies; it focuses on regime detection and forecasting comparison.
- All experiments use a chronological train/test split (approximately 70% train, 30% test) with data from 2014-01-01 to present.
- Hidden states are interpreted only after training, using the training data to compute state-dependent statistics.

## License

This project is for educational purposes as part of a university course.
