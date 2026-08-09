"""
Analyze winning HMM cases from the extended experiments.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data import load_asset, validate_data, build_features, chronological_split
from model import train_gaussian_hmm, decode_past_only_states, hmm_next_return_predictions
from evaluation import evaluate_predictions, summarize_states
from analysis import analyze_transitions

def analyze_case(ticker, k_val, start_date="2004-01-01"):
    print(f"=== Analyzing {ticker} with HMM K={k_val} ===")
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
    print(f"Training runtime: {runtime:.2f}s")
    print(f"Log-likelihood train: {model.score(X_train):.2f}")
    print(f"Log-likelihood test: {model.score(X_test):.2f}")
    
    # States
    train_states = model.predict(X_train)
    test_states = decode_past_only_states(model, X_train, X_test, context_window=100)
    
    # State summary
    state_summary = summarize_states(train_df, train_states)
    print("\nState summary (training data):")
    print(state_summary)
    
    # Transition analysis
    state_names = [f"State {i}" for i in range(k_val)]
    trans_analysis = analyze_transitions(model.transmat_, state_names)
    print(f"\nAverage diagonal probability (persistence): {trans_analysis['average_diagonal_probability']:.3f}")
    print(f"Average row entropy: {trans_analysis['average_row_entropy']:.3f}")
    print(f"Spectral gap: {trans_analysis['spectral_gap']:.3f}")
    
    # Predictions and evaluation
    pred_returns = hmm_next_return_predictions(model, train_df, train_states, test_states)
    results = evaluate_predictions(f"HMM K={k_val}", test_df, pred_returns, 
                                   runtime_sec=runtime,
                                   log_likelihood_train=model.score(X_train),
                                   log_likelihood_test=model.score(X_test))
    print(f"\nHMM evaluation: {results}")
    
    # Compare with naive train mean return
    naive_pred = np.full(len(test_df), train_df["next_log_return"].mean())
    naive_results = evaluate_predictions("Naive - train mean return", test_df, naive_pred)
    print(f"Naive evaluation: {naive_results}")
    
    # Save plots for this case
    os.makedirs(f"analysis/{ticker}_K{k_val}", exist_ok=True)
    
    # Plot price with states
    plot_df = pd.concat([train_df, test_df]).copy()
    plot_df[f"state_k{k_val}"] = np.concatenate([train_states, test_states])
    plot_df["split"] = ["train"]*len(train_df) + ["test"]*len(test_df)
    
    plt.figure(figsize=(14, 6))
    sc = plt.scatter(plot_df.index, plot_df[price_col], c=plot_df[f"state_k{k_val}"], s=10, cmap='viridis')
    plt.title(f"{ticker} price colored by HMM states (K={k_val})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.colorbar(sc, label="State")
    plt.tight_layout()
    plt.savefig(f"analysis/{ticker}_K{k_val}/price_states.png", dpi=150)
    plt.close()
    
    # Plot returns with states
    plt.figure(figsize=(14, 4))
    sc = plt.scatter(plot_df.index, plot_df["log_return"], c=plot_df[f"state_k{k_val}"], s=10, cmap='viridis')
    plt.title(f"{ticker} log returns colored by HMM states (K={k_val})")
    plt.xlabel("Date")
    plt.ylabel("Log return")
    plt.colorbar(sc, label="State")
    plt.tight_layout()
    plt.savefig(f"analysis/{ticker}_K{k_val}/returns_states.png", dpi=150)
    plt.close()
    
    # Transition matrix heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(model.transmat_, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=state_names, yticklabels=state_names)
    plt.title(f"Transition matrix - HMM K={k_val}")
    plt.tight_layout()
    plt.savefig(f"analysis/{ticker}_K{k_val}/transition_matrix.png", dpi=150)
    plt.close()
    
    # State characteristics
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].bar(state_summary['state'], state_summary['frequency_%'])
    axes[0, 0].set_title('State Frequencies')
    axes[0, 0].set_xlabel('State')
    axes[0, 0].set_ylabel('Frequency (%)')
    
    axes[0, 1].bar(state_summary['state'], state_summary['mean_daily_return_%'])
    axes[0, 1].set_title('Mean Daily Return by State')
    axes[0, 1].set_xlabel('State')
    axes[0, 1].set_ylabel('Mean Daily Return (%)')
    
    axes[1, 0].bar(state_summary['state'], state_summary['volatility_daily_%'])
    axes[1, 0].set_title('Return Volatility by State')
    axes[1, 0].set_xlabel('State')
    axes[1, 0].set_ylabel('Volatility (%)')
    
    axes[1, 1].bar(state_summary['state'], state_summary['avg_duration_days'])
    axes[1, 1].set_title('Average State Duration')
    axes[1, 1].set_xlabel('State')
    axes[1, 1].set_ylabel('Average Duration (days)')
    
    plt.suptitle(f'{ticker} HMM K={k_val} State Characteristics')
    plt.tight_layout()
    plt.savefig(f"analysis/{ticker}_K{k_val}/state_characteristics.png", dpi=150)
    plt.close()
    
    return {
        "ticker": ticker,
        "k": k_val,
        "state_summary": state_summary,
        "transition_analysis": trans_analysis,
        "hmm_results": results,
        "naive_results": naive_results
    }

if __name__ == "__main__":
    # Winning cases from the extended experiments (where HMM beat naive train mean return)
    # We'll analyze the top ones by DPA improvement
    cases = [
        ("DIS", 3),   # Showed the biggest improvement
        ("ADBE", 4),  # Second biggest
        ("META", 4),
        ("NFLX", 4),
        ("XLP", 4),
        ("JPM", 4),
        ("PG", 4),
        ("NVDA", 4),
        ("SPY", 3),
    ]
    
    all_analyses = []
    for ticker, k in cases:
        try:
            analysis = analyze_case(ticker, k)
            all_analyses.append(analysis)
            print(f"\nCompleted analysis for {ticker} K={k}\n{'='*50}")
        except Exception as e:
            print(f"Failed to analyze {ticker} K={k}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save overall analysis summary
    summary_rows = []
    for analysis in all_analyses:
        summary_rows.append({
            "ticker": analysis["ticker"],
            "k": analysis["k"],
            "hmm_dpa": analysis["hmm_results"]["DPA_direction_accuracy"],
            "naive_dpa": analysis["naive_results"]["DPA_direction_accuracy"],
            "dpa_improvement": analysis["hmm_results"]["DPA_direction_accuracy"] - analysis["naive_results"]["DPA_direction_accuracy"],
            "hmm_ll_train": analysis["hmm_results"]["log_likelihood_train"],
            "hmm_ll_test": analysis["hmm_results"]["log_likelihood_test"],
            "avg_diagonal": analysis["transition_analysis"]["average_diagonal_probability"],
            "avg_entropy": analysis["transition_analysis"]["average_row_entropy"]
        })
    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Summary of winning cases ===")
    print(summary_df)
    summary_df.to_csv("analysis/winning_cases_summary.csv", index=False)
    print("Saved summary to analysis/winning_cases_summary.csv")