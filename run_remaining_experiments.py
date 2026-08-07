"""
Run HMM market regimes experiments for the remaining assets.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd
from run_experiment import ExperimentConfig, run_experiment_grid

def main():
    # Define the remaining assets
    tickers = ["BTC-USD", "ETH-USD", "DIA"]

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

    # Print a summary of all experiments (including previous ones)
    print("\n" + "="*60)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("="*60)
    
    # We will read the existing summary if it exists, or create a new one from all experiment directories
    summary_rows = []
    # List all experiment directories
    exp_dirs = [d for d in os.listdir(base_save_dir) if d.startswith('run_')]
    for exp_dir in exp_dirs:
        exp_path = os.path.join(base_save_dir, exp_dir)
        # Try to read the config.json to get the ticker
        config_path = os.path.join(exp_path, 'config.json')
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            ticker = config_dict['ticker']
        else:
            # If config.json doesn't exist, try to extract from directory name
            # This is a fallback
            parts = exp_dir.split('_')
            if len(parts) >= 3:
                ticker = parts[2]
            else:
                ticker = "unknown"
        # Read the results.csv to get the best model and its DPA
        results_path = os.path.join(exp_path, 'results.csv')
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            best_row = df.iloc[0]  # Already sorted by DPA descending
            best_model = best_row['model']
            best_dpa = best_row['DPA_direction_accuracy']
        else:
            best_model = "unknown"
            best_dpa = 0.0
        summary_rows.append({
            "ticker": ticker,
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