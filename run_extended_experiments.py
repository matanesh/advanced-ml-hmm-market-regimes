"""
Run HMM market regimes experiments for an extended asset universe and longer time period.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd
from run_experiment import ExperimentConfig, run_experiment_grid

def main():
    # Define the extended asset universe
    tickers = [
        # Broad indices / ETFs (US)
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", "BND",
        # Broad indices / ETFs (International)
        "EFA", "EEM", "IEUR", "ILF", "EPP", "EWP", "EWL", "EWJ", "EWZ", "INDA",
        # Large-cap equities (US)
        "AAPL", "MSFT", "NVDA", "JPM", "AMZN", "GOOGL", "META", "TSLA", "JNJ",
        "PG", "UNH", "HD", "VZ", "DIS", "NFLX", "CMCSA", "ADBE", "CRM", "ORCL",
        # Sector ETFs
        "XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLB", "XLRE", "XLU",
        # Commodities
        "GLD", "SLV", "USO", "UNG", "DBB", "JJG", "WEAT", "CORN", "SOYB",
        # Bonds
        "TLT", "IEI", "SHY", "LQD", "HYG",
        # International Equity ETFs
        "VWO", "VEA", "IEUR", "EEMA",
        # Crypto
        "BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD", "XRP-USD", "DOT-USD", "DOGE-USD", "AVAX-USD", "MATIC-USD"
    ]

    # Remove BRK.B as it seems delisted/problematic
    # Also note: some of the above might not have data for the full period, but yfinance will handle that.

    # Create a list of configurations, one for each ticker
    configs = []
    for ticker in tickers:
        config = ExperimentConfig(
            ticker=ticker,
            start_date="2004-01-01",  # Extended to 20+ years of data
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
    base_save_dir = "experiments_extended"

    # Run the grid of experiments
    print(f"Running experiments for {len(configs)} assets from 2004-01-01 to present...")
    all_results = run_experiment_grid(configs, base_save_dir=base_save_dir)

    # Print a summary of all experiments
    print("\n" + "="*60)
    print("SUMMARY OF ALL EXPERIMENTS (EXTENDED UNIVERSE)")
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