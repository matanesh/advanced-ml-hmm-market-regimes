"""
Robustness checks for a winning case: SPY (HMM K=3) and NVDA (HMM K=4).
We vary:
  - random seed (for HMM initialization)
  - context window (for past-only state decoding)
  - train/test split ratio
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd

from data import load_asset, validate_data, build_features, chronological_split
from model import train_gaussian_hmm, decode_past_only_states, hmm_next_return_predictions
from evaluation import evaluate_predictions

def run_experiment(ticker, k_val, start_date="2004-01-01", 
                   test_size=0.30, context_window=100, random_seed=42):
    # Load data
    raw_df = load_asset(ticker, start_date, None)
    raw_df = validate_data(raw_df)
    df, price_col = build_features(raw_df)
    train_df, test_df = chronological_split(df, test_size)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feature_cols = ["log_return", "rolling_volatility_20", "daily_range", "volume_change"]
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)
    
    # Train HMM
    model, runtime = train_gaussian_hmm(X_train, n_states=k_val, random_state=random_seed)
    # States
    train_states = model.predict(X_train)
    test_states = decode_past_only_states(model, X_train, X_test, context_window=context_window)
    # Predicted returns (next step)
    pred_returns = hmm_next_return_predictions(model, train_df, train_states, test_states)
    # Convert to direction prediction (1 for up, 0 for down)
    hmm_pred_dir = (pred_returns > 0).astype(int)
    actual_dir = (test_df["next_log_return"] > 0).astype(int)
    # Naive prediction: train mean return
    naive_pred_return = np.full(len(test_df), train_df["next_log_return"].mean())
    naive_pred_dir = (naive_pred_return > 0).astype(int)
    
    # Compute DPA
    dpa_hmm = np.mean(hmm_pred_dir == actual_dir)
    dpa_naive = np.mean(naive_pred_dir == actual_dir)
    
    return {
        "dpa_hmm": dpa_hmm,
        "dpa_naive": dpa_naive,
        "dpa_diff": dpa_hmm - dpa_naive,
        "runtime": runtime,
        "log_likelihood_train": model.score(X_train),
        "log_likelihood_test": model.score(X_test)
    }

def robustness_check(ticker, k_val, param_name, param_values):
    print(f"\n=== Robustness check for {ticker} HMM K={k_val} varying {param_name} ===")
    results = []
    for val in param_values:
        kwargs = {param_name: val}
        try:
            res = run_experiment(ticker, k_val, **kwargs)
            res[param_name] = val
            results.append(res)
            print(f"{param_name}={val}: DPA_HMM={res['dpa_hmm']:.4f}, DPA_naive={res['dpa_naive']:.4f}, diff={res['dpa_diff']:.4f}")
        except Exception as e:
            print(f"Failed for {param_name}={val}: {e}")
    return pd.DataFrame(results)

if __name__ == "__main__":
    # We'll check SPY (K=3) and NVDA (K=4)
    cases = [
        ("SPY", 3),
        ("NVDA", 4),
    ]
    
    all_results = {}
    for ticker, k in cases:
        print(f"\n{'#'*60}")
        print(f"Robustness checks for {ticker} HMM K={k}")
        print('#'*60)
        
        # 1. Vary random seed
        seeds = [42, 123, 456, 789, 999]
        df_seeds = robustness_check(ticker, k, "random_seed", seeds)
        
        # 2. Vary context window
        context_windows = [50, 100, 200, 300]
        df_context = robustness_check(ticker, k, "context_window", context_windows)
        
        # 3. Vary train/test split ratio
        test_sizes = [0.20, 0.30, 0.40]
        df_split = robustness_check(ticker, k, "test_size", test_sizes)
        
        all_results[ticker] = {
            "seeds": df_seeds,
            "context": df_context,
            "split": df_split
        }
        
        # Save each
        df_seeds.to_csv(f"analysis/robustness_{ticker}_seeds.csv", index=False)
        df_context.to_csv(f"analysis/robustness_{ticker}_context.csv", index=False)
        df_split.to_csv(f"analysis/robustness_{ticker}_split.csv", index=False)
    
    # Print summary
    print("\n\n=== Summary of robustness checks ===")
    for ticker, k in cases:
        print(f"\n{ticker} HMM K={k}:")
        seeds_df = all_results[ticker]["seeds"]
        if not seeds_df.empty:
            print(f"  Random seed - DPA_HMM mean: {seeds_df['dpa_hmm'].mean():.4f} (std: {seeds_df['dpa_hmm'].std():.4f})")
            print(f"  Random seed - DPA_diff mean: {seeds_df['dpa_diff'].mean():.4f} (std: {seeds_df['dpa_diff'].std():.4f})")
        context_df = all_results[ticker]["context"]
        if not context_df.empty:
            print(f"  Context window - DPA_HMM mean: {context_df['dpa_hmm'].mean():.4f} (std: {context_df['dpa_hmm'].std():.4f})")
            print(f"  Context window - DPA_diff mean: {context_df['dpa_diff'].mean():.4f} (std: {context_df['dpa_diff'].std():.4f})")
        split_df = all_results[ticker]["split"]
        if not split_df.empty:
            print(f"  Test size - DPA_HMM mean: {split_df['dpa_hmm'].mean():.4f} (std: {split_df['dpa_hmm'].std():.4f})")
            print(f"  Test size - DPA_diff mean: {split_df['dpa_diff'].mean():.4f} (std: {split_df['dpa_diff'].std():.4f})")