"""
Model comparison script: compare Gaussian HMM variants, other HMM, and simple ML classifiers.
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def get_hmm_predictions(model, train_df, train_states, test_states):
    """Returns predicted next returns and predicted directions."""
    pred_returns = hmm_next_return_predictions(model, train_df, train_states, test_states)
    pred_dir = (pred_returns > 0).astype(int)
    return pred_returns, pred_dir

def get_ml_predictions(X_train, y_train, X_test, model_obj):
    """Trains a classifier and returns predicted directions."""
    model_obj.fit(X_train, y_train)
    pred_dir = model_obj.predict(X_test)
    # For return prediction, we don't have a direct estimate; we can use the mean of the class?
    # But we'll just return None for return predictions.
    return None, pred_dir

def run_model_comparison(ticker, start_date="2004-01-01"):
    print(f"\n{'='*60}")
    print(f"Model comparison for {ticker}")
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
    
    # Target: direction of next log return
    y_train = (train_df["next_log_return"] > 0).astype(int)
    y_test = (test_df["next_log_return"] > 0).astype(int)
    
    # Store results
    results = []
    
    # 1. Gaussian HMM with different K and covariance types
    k_values = [1, 2, 3, 4, 5]
    cov_types = ['full', 'tied', 'diag', 'spherical']
    for k in k_values:
        for cov in cov_types:
            try:
                model, runtime = train_gaussian_hmm(X_train, n_states=k, covariance_type=cov, random_state=42)
                train_states = model.predict(X_train)
                test_states = decode_past_only_states(model=model, X_train_scaled=X_train, X_test_scaled=X_test, context_window=100)
                pred_returns, pred_dir = get_hmm_predictions(model, train_df, train_states, test_states)
                # Evaluate
                eval_dict = evaluate_predictions(f"HMM K={k} {cov}", test_df, pred_returns, 
                                                runtime_sec=runtime,
                                                log_likelihood_train=model.score(X_train),
                                                log_likelihood_test=model.score(X_test))
                eval_dict['model_type'] = f'HMM K={k} {cov}'
                results.append(eval_dict)
                print(f"HMM K={k} {cov}: DPA={eval_dict['DPA_direction_accuracy']:.4f}")
            except Exception as e:
                print(f"Failed HMM K={k} {cov}: {e}")
    
    # 2. Logistic Regression
    try:
        lr = LogisticRegression(random_state=42, max_iter=1000)
        _, pred_dir_lr = get_ml_predictions(X_train, y_train, X_test, lr)
        # For DPA we need the predictions; we don't have return predictions so we can't compute MAE/RMSE
        dpa_lr = np.mean(pred_dir_lr == y_test)
        results.append({
            'model': 'Logistic Regression',
            'DPA_direction_accuracy': dpa_lr,
            'MAE_return': np.nan,
            'RMSE_return': np.nan,
            'MAPE_price_%': np.nan,
            'log_likelihood_train': np.nan,
            'log_likelihood_test': np.nan,
            'runtime_sec': np.nan,
            'model_type': 'Logistic Regression'
        })
        print(f"Logistic Regression: DPA={dpa_lr:.4f}")
    except Exception as e:
        print(f"Failed Logistic Regression: {e}")
    
    # 3. Random Forest
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        _, pred_dir_rf = get_ml_predictions(X_train, y_train, X_test, rf)
        dpa_rf = np.mean(pred_dir_rf == y_test)
        results.append({
            'model': 'Random Forest',
            'DPA_direction_accuracy': dpa_rf,
            'MAE_return': np.nan,
            'RMSE_return': np.nan,
            'MAPE_price_%': np.nan,
            'log_likelihood_train': np.nan,
            'log_likelihood_test': np.nan,
            'runtime_sec': np.nan,
            'model_type': 'Random Forest'
        })
        print(f"Random Forest: DPA={dpa_rf:.4f}")
    except Exception as e:
        print(f"Failed Random Forest: {e}")
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    # Reorder columns
    cols = ['model_type', 'model', 'DPA_direction_accuracy', 'MAE_return', 'RMSE_return', 'MAPE_price_%',
            'log_likelihood_train', 'log_likelihood_test', 'runtime_sec']
    results_df = results_df[cols]
    print(f"\nResults summary for {ticker}:")
    print(results_df[['model_type', 'DPA_direction_accuracy']].to_string(index=False))
    
    # Save
    os.makedirs('model_comparison', exist_ok=True)
    save_path = f'model_comparison/{ticker}_model_comparison.csv'
    results_df.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")
    
    return results_df

if __name__ == "__main__":
    # Run on a few assets: the winning cases and a couple more
    tickers = ["SPY", "NVDA", "DIS", "ADBE", "QQQ"]  # QQQ as a typical asset where naive won
    all_results = {}
    for ticker in tickers:
        try:
            df = run_model_comparison(ticker)
            all_results[ticker] = df
        except Exception as e:
            print(f"Failed to run comparison for {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    # Optionally, combine and save a summary
    # For now, we leave individual files.