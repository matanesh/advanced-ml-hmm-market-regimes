"""
Statistical significance test for DPA improvements of HMM over naive baseline.
Uses McNemar's test on paired predictions.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import json
import scipy.stats as stats

from data import load_asset, validate_data, build_features, chronological_split
from model import train_gaussian_hmm, decode_past_only_states, hmm_next_return_predictions
from evaluation import evaluate_predictions

def mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar's test for two classifiers.
    pred_a, pred_b: binary arrays (0/1) indicating correct prediction.
    Returns chi2, p-value, and the contingency table.
    """
    # Contingency table:
    #               pred_b
    #            correct  wrong
    # pred_a
    # correct      a       b
    # wrong        c       d
    a = np.sum((pred_a == 1) & (pred_b == 1))
    b = np.sum((pred_a == 1) & (pred_b == 0))
    c = np.sum((pred_a == 0) & (pred_b == 1))
    d = np.sum((pred_a == 0) & (pred_b == 0))
    table = [[a, b], [c, d]]
    # Using chi-square correction (Edwards)
    chi2 = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0
    # p-value from chi-square distribution with 1 df
    p = 1 - stats.chi2.cdf(chi2, 1) if (b + c) > 0 else 1.0
    return chi2, p, table

def analyze_case_significance(ticker, k_val, start_date="2004-01-01"):
    print(f"\n=== Significance test for {ticker} HMM K={k_val} ===")
    # Load data
    raw_df = load_asset(ticker, start_date, None)
    raw_df = validate_data(raw_df)
    df, price_col = build_features(raw_df)
    train_df, test_df = chronological_split(df, 0.30)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feature_cols = ["log_return", "rolling_volatility_20", "daily_range", "volume_change"]
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)
    
    # Train HMM
    model, runtime = train_gaussian_hmm(X_train, n_states=k_val, random_state=42)
    # States
    train_states = model.predict(X_train)
    test_states = decode_past_only_states(model, X_train, X_test, context_window=100)
    # Predicted returns (next step)
    pred_returns = hmm_next_return_predictions(model, train_df, train_states, test_states)
    # Convert to direction prediction (1 for up, 0 for down)
    # We predict next log return; direction = 1 if pred > 0 else 0
    hmm_pred_dir = (pred_returns > 0).astype(int)
    # Actual direction
    actual_dir = (test_df["next_log_return"] > 0).astype(int)
    # Naive prediction: train mean return
    naive_pred_return = np.full(len(test_df), train_df["next_log_return"].mean())
    naive_pred_dir = (naive_pred_return > 0).astype(int)
    
    # Correct predictions
    hmm_correct = (hmm_pred_dir == actual_dir).astype(int)
    naive_correct = (naive_pred_dir == actual_dir).astype(int)
    
    # Compute metrics
    n = len(test_df)
    dpa_hmm = np.mean(hmm_correct)
    dpa_naive = np.mean(naive_correct)
    print(f"Test set size: {n}")
    print(f"HMM DPA: {dpa_hmm:.4f}")
    print(f"Naive DPA: {dpa_naive:.4f}")
    print(f"Difference: {dpa_hmm - dpa_naive:.4f}")
    
    # McNemar test
    chi2, p, table = mcnemar_test(actual_dir, hmm_correct, naive_correct)
    print(f"McNemar's chi2: {chi2:.4f}, p-value: {p:.4f}")
    print("Contingency table (rows: HMM correct/wrong; cols: Naive correct/wrong):")
    print(f"[[{table[0][0]}, {table[0][1]}], [{table[1][0]}, {table[1][1]}]]")
    
    # Also binomial sign test (if we ignore ties)
    # n_big = b + c (discordant pairs)
    b = table[0][1]  # HMM correct, Naive wrong
    c = table[1][0]  # HMM wrong, Naive correct
    n_discord = b + c
    if n_discord > 0:
        # Under null, b ~ Binomial(n_discord, 0.5)
        p_exact = stats.binom_test(b, n_discord, p=0.5, alternative='two-sided')
        print(f"Exact binomial test (sign test): b={b}, n={n_discord}, p={p_exact:.4f}")
    else:
        p_exact = None
        print("No discordant pairs.")
    
    return {
        "ticker": ticker,
        "k": k_val,
        "n_test": n,
        "dpa_hmm": dpa_hmm,
        "dpa_naive": dpa_naive,
        "dpa_diff": dpa_hmm - dpa_naive,
        "mcnemar_chi2": chi2,
        "mcnemar_p": p,
        "contingency_table": table,
        "discordant_b": b if n_discord > 0 else None,
        "discordant_c": c if n_discord > 0 else None,
        "sign_test_p": p_exact
    }

if __name__ == "__main__":
    # Winning cases from the extended experiments (where HMM beat naive train mean return)
    cases = [
        ("DIS", 3),
        ("ADBE", 4),
        ("META", 4),
        ("NFLX", 4),
        ("XLP", 4),
        ("JPM", 4),
        ("PG", 4),
        ("NVDA", 4),
        ("SPY", 3),
    ]
    
    all_results = []
    for ticker, k in cases:
        try:
            res = analyze_case_significance(ticker, k)
            all_results.append(res)
            print(f"\nCompleted significance test for {ticker} K={k}\n{'='*60}")
        except Exception as e:
            print(f"Failed to analyze {ticker} K={k}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    summary_df = pd.DataFrame([{
        "ticker": r["ticker"],
        "k": r["k"],
        "n_test": r["n_test"],
        "dpa_hmm": r["dpa_hmm"],
        "dpa_naive": r["dpa_naive"],
        "dpa_diff": r["dpa_diff"],
        "mcnemar_p": r["mcnemar_p"],
        "sign_test_p": r["sign_test_p"]
    } for r in all_results])
    print("\n=== Summary of significance tests ===")
    print(summary_df)
    summary_df.to_csv("analysis/significance_tests.csv", index=False)
    print("Saved summary to analysis/significance_tests.csv")