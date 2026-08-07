"""
Test script to verify the HMM market regimes project setup.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data import load_asset, validate_data, build_features, chronological_split
from model import train_gaussian_hmm, decode_past_only_states, hmm_next_return_predictions
from evaluation import evaluate_predictions, summarize_states

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def test_data_loading():
    print("Testing data loading...")
    # Load a small amount of data for testing
    df = load_asset("SPY", "2023-01-01", "2023-12-31")
    print(f"Loaded data shape: {df.shape}")
    assert not df.empty, "Data should not be empty"
    return df

def test_feature_building():
    print("Testing feature building...")
    df = load_asset("SPY", "2023-01-01", "2023-12-31")
    df, price_col = build_features(df)
    print(f"Feature-engineered data shape: {df.shape}")
    expected_features = ["log_return", "rolling_volatility_20", "daily_range", "volume_change", 
                         "next_log_return", "next_close", "current_close"]
    for feat in expected_features:
        assert feat in df.columns, f"Missing feature: {feat}"
    print("All expected features present.")
    return df, price_col

def test_train_test_split():
    print("Testing train/test split...")
    df = load_asset("SPY", "2023-01-01", "2023-12-31")
    df, _ = build_features(df)
    train_df, test_df = chronological_split(df, test_size=0.3)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    assert len(train_df) + len(test_df) == len(df), "Split should preserve all rows"
    # Check that the split is chronological (no overlap in indices)
    assert train_df.index.max() < test_df.index.min(), "Split should be chronological"
    print("Train/test split is chronological.")
    return train_df, test_df

def test_scaling():
    print("Testing feature scaling...")
    df = load_asset("SPY", "2023-01-01", "2023-12-31")
    df, _ = build_features(df)
    train_df, test_df = chronological_split(df, test_size=0.3)
    feature_cols = ["log_return", "rolling_volatility_20", "daily_range", "volume_change"]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)
    print(f"Scaled train shape: {X_train.shape}, Scaled test shape: {X_test.shape}")
    # Check that training data has mean ~0 and std ~1
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-10), "Training data should have zero mean"
    assert np.allclose(X_train.std(axis=0), 1, atol=1e-10), "Training data should have unit std"
    print("Scaling verified.")
    return X_train, X_test

def test_hmm_training():
    print("Testing HMM training...")
    df = load_asset("SPY", "2023-01-01", "2023-12-31")
    df, _ = build_features(df)
    train_df, test_df = chronological_split(df, test_size=0.3)
    feature_cols = ["log_return", "rolling_volatility_20", "daily_range", "volume_change"]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)
    
    # Train a simple HMM with 2 states
    model, runtime = train_gaussian_hmm(X_train, n_states=2, random_state=42)
    print(f"HMM trained in {runtime:.2f} seconds")
    print(f"Model score (train): {model.score(X_train):.4f}")
    print(f"Model score (test): {model.score(X_test):.4f}")
    # Check that the model has the expected attributes
    assert hasattr(model, "transmat_"), "Model should have transition matrix"
    assert hasattr(model, "means_"), "Model should have means"
    assert hasattr(model, "covars_"), "Model should have covariances"
    print("HMM training verified.")
    return model, X_train, X_test

def test_state_decoding():
    print("Testing state decoding...")
    df = load_asset("SPY", "2023-01-01", "2023-12-31")
    df, _ = build_features(df)
    train_df, test_df = chronological_split(df, test_size=0.3)
    feature_cols = ["log_return", "rolling_volatility_20", "daily_range", "volume_change"]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)
    
    model, _ = train_gaussian_hmm(X_train, n_states=2, random_state=42)
    train_states = model.predict(X_train)
    test_states = decode_past_only_states(model, X_train, X_test, context_window=50)
    print(f"Train states shape: {train_states.shape}")
    print(f"Test states shape: {test_states.shape}")
    assert len(train_states) == len(train_df), "Train states length should match train data"
    assert len(test_states) == len(test_df), "Test states length should match test data"
    print("State decoding verified.")
    return train_states, test_states

def test_evaluation():
    print("Testing evaluation...")
    df = load_asset("SPY", "2023-01-01", "2023-12-31")
    df, _ = build_features(df)
    train_df, test_df = chronological_split(df, test_size=0.3)
    feature_cols = ["log_return", "rolling_volatility_20", "daily_range", "volume_change"]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)
    
    model, _ = train_gaussian_hmm(X_train, n_states=2, random_state=42)
    train_states = model.predict(X_train)
    test_states = decode_past_only_states(model, X_train, X_test, context_window=50)
    
    # Predict returns
    pred_returns = hmm_next_return_predictions(model, train_df, train_states, test_states)
    
    # Evaluate
    results = evaluate_predictions("Test HMM", test_df, pred_returns, runtime_sec=1.0)
    print(f"Evaluation results: {results}")
    assert "DPA_direction_accuracy" in results, "Should have DPA metric"
    assert "MAE_return" in results, "Should have MAE metric"
    assert "RMSE_return" in results, "Should have RMSE metric"
    assert "MAPE_price_%" in results, "Should have MAPE metric"
    print("Evaluation verified.")
    return results

def test_state_summarization():
    print("Testing state summarization...")
    df = load_asset("SPY", "2023-01-01", "2023-12-31")
    df, _ = build_features(df)
    train_df, _ = chronological_split(df, test_size=0.3)
    feature_cols = ["log_return", "rolling_volatility_20", "daily_range", "volume_change"]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    
    model, _ = train_gaussian_hmm(X_train, n_states=2, random_state=42)
    train_states = model.predict(X_train)
    
    state_summary = summarize_states(train_df, train_states)
    print(f"State summary:\n{state_summary}")
    assert len(state_summary) == 2, "Should have 2 states"
    assert "state" in state_summary.columns, "Should have state column"
    assert "frequency_%" in state_summary.columns, "Should have frequency column"
    assert "mean_daily_return_%" in state_summary.columns, "Should have mean return column"
    print("State summarization verified.")
    return state_summary

def main():
    print("="*60)
    print("Testing HMM Market Regimes Project Setup")
    print("="*60)
    
    try:
        test_data_loading()
        test_feature_building()
        test_train_test_split()
        test_scaling()
        test_hmm_training()
        test_state_decoding()
        test_evaluation()
        test_state_summarization()
        
        print("="*60)
        print("All tests passed! The project setup is working correctly.")
        print("="*60)
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()