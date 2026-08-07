"""
Run HMM market regimes experiments for multiple assets.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd
from run_experiment import ExperimentConfig, run_experiment_grid

def main():
    # Define the asset universe as per requirements
    tickers = [
        # Broad indices / ETFs
        "SPY", "QQQ", "IWM",
        # Large-cap equities
        "AAPL", "MSFT", "NVDA", "JPM", "AMZN",
        # Sector / defensive / alternative assets
        "XLE", "GLD", "TLT", "EFA",
        # Crypto
        "BTC-USD", "ETH-USD",
        # Optional
        "DIA"
    ]

    # Create a list of configurations, one for each ticker
    configs = []
    for ticker in tickers:
        config = ExperimentConfig(
            ticker=ticker,
            start_date="2014-01-01",  # Approximately 10 years of data
            end_date=None,            # Up to present
            test_size=0.30,
            feature_cols=["log_return", "rolling_volatility_20", "daily_range", "volume_change"],
            ma_window=5,
            context_window=100,
            k_values=[2, 3, 4],
            covariance_type="full",
            random_state=42,
            n_iter=300,
            tol=1e-4
        )
        configs.append(config)

    # Define the base directory for saving experiments
    base_save_dir = "experiments_multi_asset"

    # Run the grid of experiments
    print(f"Running experiments for {len(configs)} assets...")
    all_results = run_experiment_grid(configs, base_save_dir=base_save_dir)

    # Print a summary of all experiments
    print("\n" + "="*60)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("="*60)
    summary_rows = []
    for result in all_results:
        config = result["config"]
        best_model = result["best_model"]
        # Extract the DPA of the best model from the results DataFrame
        best_dpa = result["results_df"].iloc[0]["DPA_direction_accuracy"]
        summary_rows.append({
            "ticker": config.ticker,
            "best_model": best_model,
            "best_dpa": best_dpa
        })
    
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    
    # Save the summary to a CSV file
    summary_path = os.path.join(base_save_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main()