"""
Economic interpretation of regimes for winning cases: SPY, NVDA, DIS.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np

from data import load_asset, validate_data, build_features, chronological_split
from model import train_gaussian_hmm, decode_past_only_states
from evaluation import summarize_states

def interpret_regimes(ticker, k_val, start_date="2004-01-01"):
    print(f"\n{'='*60}")
    print(f"Economic interpretation of HMM regimes for {ticker} (K={k_val})")
    print('='*60)
    
    # Load and prepare data
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
    
    # State summary on training data (to understand regimes)
    state_summary = summarize_states(train_df, train_states)
    print("\nState characteristics (training data):")
    print(state_summary.to_string(index=False))
    
    # Determine which state is which based on mean return and volatility
    # We'll sort by mean daily return
    state_summary_sorted = state_summary.sort_values('mean_daily_return_%', ascending=False)
    print("\nStates sorted by mean daily return (high to low):")
    for idx, row in state_summary_sorted.iterrows():
        print(f"  State {row['state']}: Mean return = {row['mean_daily_return_%']:.3f}% per day, "
              f"Volatility = {row['volatility_daily_%']:.3f}% per day, "
              f"Frequency = {row['frequency_%']:.1f}% of days, "
              f"Avg duration = {row['avg_duration_days']:.1f} days")
    
    # Provide economic interpretation
    print("\nEconomic interpretation:")
    # Identify high-return, low-volatility state (bullish calm)
    # Identify low-return, high-volatility state (bearish turbulent)
    # Identify medium states
    # We'll assign based on the sorted table
    if len(state_summary_sorted) >= 3:
        high_ret = state_summary_sorted.iloc[0]
        med_ret = state_summary_sorted.iloc[1]
        low_ret = state_summary_sorted.iloc[2]
        print(f"  State {int(high_ret['state'])}: Highest mean return ({high_ret['mean_daily_return_%']:.3f}%/day). "
              f"This could represent a 'bull market' regime, especially if volatility is moderate.")
        print(f"  State {int(low_ret['state'])}: Lowest mean return ({low_ret['mean_daily_return_%']:.3f}%/day). "
              f"This could represent a 'bear market' or high-stress regime, particularly if volatility is high ({low_ret['volatility_daily_%']:.3f}%/day).")
        print(f"  State {int(med_ret['state'])}: Intermediate returns. "
              f"This may represent a 'sideways' or 'transition' regime.")
    else:
        for idx, row in state_summary_sorted.iterrows():
            print(f"  State {row['state']}: Return = {row['mean_daily_return_%']:.3f}%/day, "
                  f"Volatility = {row['volatility_daily_%']:.3f}%/day.")
    
    # Also look at the transition matrix to see persistence
    from analysis import analyze_transitions
    state_names = [f"State {i}" for i in range(k_val)]
    trans_analysis = analyze_transitions(model.transmat_, state_names)
    print(f"\nTransition persistence (average probability of staying in same state): {trans_analysis['average_diagonal_probability']:.3f}")
    if trans_analysis['average_diagonal_probability'] > 0.9:
        print("  High persistence -> regimes tend to last for extended periods.")
    elif trans_analysis['average_diagonal_probability'] > 0.8:
        print("  Moderate persistence -> regimes show some stability but can change frequently.")
    else:
        print("  Low persistence -> states are transient, may reflect noisy classification.")
    
    # Plot the price with states for the last year to visualize
    import matplotlib.pyplot as plt
    plot_df = pd.concat([train_df, test_df]).copy()
    plot_df['state'] = np.concatenate([train_states, test_states])
    plot_df['split'] = ['train']*len(train_df) + ['test']*len(test_df)
    # Take last 250 trading days (~1 year)
    plot_df_last = plot_df.tail(250)
    
    plt.figure(figsize=(14, 6))
    scatter = plt.scatter(plot_df_last.index, plot_df_last[price_col], 
                          c=plot_df_last['state'], cmap='viridis', s=15)
    plt.title(f"{ticker} Price (last ~1 year) colored by HMM state (K={k_val})")
    plt.xlabel("Date")
    plt.ylabel("Price (Adjusted Close)")
    cbar = plt.colorbar(scatter)
    cbar.set_label('State')
    plt.tight_layout()
    plot_dir = f"interpretation/{ticker}"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/price_states_lastyear.png", dpi=150)
    plt.close()
    print(f"\nSaved price-state plot to {plot_dir}/price_states_lastyear.png")
    
    return state_summary, trans_analysis

if __name__ == "__main__":
    cases = [
        ("SPY", 3),
        ("NVDA", 4),
        ("DIS", 3),
    ]
    
    for ticker, k in cases:
        try:
            interpret_regimes(ticker, k)
        except Exception as e:
            print(f"Failed to interpret {ticker} K={k}: {e}")
            import traceback
            traceback.print_exc()