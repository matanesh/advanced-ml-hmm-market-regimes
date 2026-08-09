"""
Main script to run HMM experiment for a single asset.
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Import our modules
from src.data import load_asset, validate_data, build_features, chronological_split
from src.model import train_gaussian_hmm, decode_past_only_states, hmm_next_return_predictions
from src.evaluation import (
    evaluate_predictions,
    direction_labels,
    average_duration,
    summarize_states,
)
from src.analysis import analyze_transitions, plot_transition_heatmap, plot_state_characteristics

# Configuration class
class ExperimentConfig:
    def __init__(self,
                 ticker: str = "SPY",
                 start_date: str = "2014-01-01",
                 end_date: Optional[str] = None,
                 test_size: float = 0.30,
                 feature_cols: Optional[list] = None,
                 ma_window: int = 5,
                 context_window: int = 100,
                 k_values: list = [2, 3, 4],
                 covariance_type: str = "full",
                 random_state: int = 42,
                 n_iter: int = 300,
                 tol: float = 1e-4):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.test_size = test_size
        self.feature_cols = feature_cols or ["log_return", "rolling_volatility_20", "daily_range", "volume_change"]
        self.ma_window = ma_window
        self.context_window = context_window
        self.k_values = k_values
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "test_size": self.test_size,
            "feature_cols": self.feature_cols,
            "ma_window": self.ma_window,
            "context_window": self.context_window,
            "k_values": self.k_values,
            "covariance_type": self.covariance_type,
            "random_state": self.random_state,
            "n_iter": self.n_iter,
            "tol": self.tol
        }

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


def run_experiment(config: ExperimentConfig, save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a single HMM experiment for the given configuration.

    Parameters:
    -----------
    config : ExperimentConfig
        Configuration for the experiment.
    save_dir : str, optional
        Directory to save results (plots, etc.). If None, results are not saved.

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing experiment results, including:
        - config: the configuration used
        - results_df: DataFrame with evaluation metrics for each model
        - best_model: name of the best model by DPA
        - hmm_models: dictionary of trained HMM models (for each K)
        - hmm_state_tables: dictionary of state summary tables (for each K)
        - train_df, test_df: the training and testing data
        - runtime: total experiment runtime
    """
    start_time = time.time()

    # Set random seed for reproducibility
    np.random.seed(config.random_state)

    # 1. Load data
    print(f"Loading data for {config.ticker} from {config.start_date} to {config.end_date or 'present'}...")
    raw_df = load_asset(config.ticker, config.start_date, config.end_date)
    print(f"Raw data shape: {raw_df.shape}")

    # 2. Validate data
    raw_df = validate_data(raw_df)
    print(f"Validated data shape: {raw_df.shape}")

    # 3. Build features
    print("Building features...")
    df, price_col = build_features(raw_df)
    print(f"Feature-engineered data shape: {df.shape}")
    print(f"Features used: {config.feature_cols}")
    print(f"Price column used: {price_col}")

    # 4. Split data chronologically
    print(f"Splitting data chronologically with test_size={config.test_size}...")
    train_df, test_df = chronological_split(df, config.test_size)
    print(f"Train set: {len(train_df)} observations from {train_df.index.min().date()} to {train_df.index.max().date()}")
    print(f"Test set:  {len(test_df)} observations from {test_df.index.min().date()} to {test_df.index.max().date()}")

    # 5. Scale features (fit on train, transform on train and test)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[config.feature_cols].values)
    X_test_scaled = scaler.transform(test_df[config.feature_cols].values)

    # 6. EDA plots (optional, can be skipped if save_dir is None)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        # Price over time
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df[price_col])
        plt.title(f"{config.ticker} price over time")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{config.ticker}_price_over_time.png"), dpi=150)
        plt.close()

        # Log returns
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df["log_return"])
        plt.title(f"{config.ticker} daily log returns")
        plt.xlabel("Date")
        plt.ylabel("Log return")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{config.ticker}_log_returns.png"), dpi=150)
        plt.close()

        # Rolling volatility
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df["rolling_volatility_20"])
        plt.title(f"{config.ticker} rolling volatility, 20 days")
        plt.xlabel("Date")
        plt.ylabel("Rolling volatility")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{config.ticker}_rolling_volatility.png"), dpi=150)
        plt.close()

    # 7. Baselines
    print("Running baselines...")
    results = []

    # Naive - train mean return
    start = time.time()
    pred = np.full(len(test_df), train_df["next_log_return"].mean())
    results.append(evaluate_predictions(
        "Naive - train mean return",
        test_df,
        pred,
        time.time() - start
    ))

    # Naive - persistence
    start = time.time()
    pred = test_df["log_return"].values
    results.append(evaluate_predictions(
        "Naive - persistence",
        test_df,
        pred,
        time.time() - start
    ))

    # Moving Average
    start = time.time()
    full_df = pd.concat([train_df, test_df]).copy()
    full_df["ma_pred_return"] = full_df["log_return"].rolling(config.ma_window).mean()
    pred = full_df.loc[test_df.index, "ma_pred_return"].fillna(train_df["next_log_return"].mean()).values
    results.append(evaluate_predictions(
        f"Moving Average {config.ma_window}",
        test_df,
        pred,
        time.time() - start
    ))

    # Discrete Markov Chain
    start = time.time()
    train_curr_dir = direction_labels(train_df["log_return"].values)
    train_next_ret = train_df["next_log_return"].values
    exp_next = {}
    for d in [0, 1]:
        vals = train_next_ret[train_curr_dir == d]
        exp_next[d] = vals.mean() if len(vals) else train_df["next_log_return"].mean()
    pred = np.array([exp_next[d] for d in direction_labels(test_df["log_return"].values)])
    results.append(evaluate_predictions(
        "Discrete Markov Chain",
        test_df,
        pred,
        time.time() - start
    ))

    # 8. Gaussian HMM models
    print("Training Gaussian HMM models for K =", config.k_values)
    hmm_models = {}
    hmm_state_tables = {}
    hmm_analyses = {}  # For storing transition analysis

    for k in config.k_values:
        print(f"  Training HMM with K={k}...")
        model, runtime = train_gaussian_hmm(
            X_train_scaled,
            n_states=k,
            covariance_type=config.covariance_type,
            n_iter=config.n_iter,
            tol=config.tol,
            random_state=config.random_state,
            verbose=False
        )
        train_states = model.predict(X_train_scaled)
        test_states = decode_past_only_states(
            model,
            X_train_scaled,
            X_test_scaled,
            context_window=config.context_window
        )
        pred_hmm = hmm_next_return_predictions(
            model,
            train_df,
            train_states,
            test_states
        )
        results.append(evaluate_predictions(
            f"Gaussian HMM K={k}",
            test_df,
            pred_hmm,
            runtime,
            model.score(X_train_scaled),
            model.score(X_test_scaled)
        ))
        hmm_models[k] = {
            "model": model,
            "train_states": train_states,
            "test_states_past_only": test_states,
            "pred_return": pred_hmm
        }
        hmm_state_tables[k] = summarize_states(train_df, train_states)
        # Perform transition analysis
        state_names = [f"State {i}" for i in range(k)]
        hmm_analyses[k] = analyze_transitions(model.transmat_, state_names)

    # 9. Compile results
    results_df = pd.DataFrame(results).sort_values("DPA_direction_accuracy", ascending=False).reset_index(drop=True)
    best_model = results_df.iloc[0]["model"]

    # 10. Save results if save_dir is provided
    if save_dir is not None:
        # Save configuration
        config.save(os.path.join(save_dir, "config.json"))

        # Save results DataFrame
        results_df.to_csv(os.path.join(save_dir, "results.csv"), index=False)

        # Save state tables for each K
        for k, state_table in hmm_state_tables.items():
            state_table.to_csv(os.path.join(save_dir, f"state_table_K{k}.csv"), index=False)

        # Save transition matrices and analyses for each K
        for k, model_dict in hmm_models.items():
            model = model_dict["model"]
            transmat_df = pd.DataFrame(
                model.transmat_,
                index=[f"from_state_{i}" for i in range(k)],
                columns=[f"to_state_{i}" for i in range(k)]
            )
            transmat_df.to_csv(os.path.join(save_dir, f"transition_matrix_K{k}.csv"))
            
            # Save transition analysis
            analysis = hmm_analyses[k]
            # Convert numpy arrays to lists for JSON serialization, handling complex numbers
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    if obj.dtype.kind == 'c':  # complex
                        return [{"real": x.real, "imag": x.imag} for x in obj]
                    else:
                        return obj.tolist()
                elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                    return float(obj)
                elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                    return int(obj)
                else:
                    return obj

            analysis_serializable = {
                "n_states": analysis["n_states"],
                "state_names": analysis["state_names"],
                "average_diagonal_probability": float(analysis["average_diagonal_probability"]),
                "average_row_entropy": float(analysis["average_row_entropy"]),
                "stationary_distribution": convert_to_serializable(analysis["stationary_distribution"]),
                "mean_recurrence_time": convert_to_serializable(analysis["mean_recurrence_time"]),
                "spectral_gap": float(analysis["spectral_gap"]),
                "eigenvalues": convert_to_serializable(analysis["eigenvalues"])
            }
            with open(os.path.join(save_dir, f"transition_analysis_K{k}.json"), 'w') as f:
                json.dump(analysis_serializable, f, indent=2)

        # Generate and save plots for the best K (by DPA) or for K=3 as default
        best_k = None
        for k in config.k_values:
            if f"Gaussian HMM K={k}" == best_model:
                best_k = k
                break
        if best_k is None:
            best_k = 3  # fallback to K=3

        # Use the best_k model for plotting
        model_dict = hmm_models[best_k]
        model = model_dict["model"]
        train_states = model_dict["train_states"]
        test_states = model_dict["test_states_past_only"]

        # Create a combined dataframe for plotting
        plot_df = pd.concat([train_df, test_df]).copy()
        plot_df[f"state_k{best_k}"] = np.concatenate([train_states, test_states])
        plot_df["split"] = ["train"] * len(train_df) + ["test"] * len(test_df)

        # Price colored by states
        plt.figure(figsize=(13, 5))
        sc = plt.scatter(plot_df.index, plot_df[price_col], c=plot_df[f"state_k{best_k}"], s=8)
        plt.title(f"{config.ticker} price colored by Gaussian HMM hidden states (K={best_k})")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True, alpha=0.3)
        plt.colorbar(sc, label="Hidden state")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{config.ticker}_price_states_K{best_k}.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # Log returns colored by states
        plt.figure(figsize=(13, 4))
        sc = plt.scatter(plot_df.index, plot_df["log_return"], c=plot_df[f"state_k{best_k}"], s=8)
        plt.title(f"{config.ticker} log returns colored by hidden states (K={best_k})")
        plt.xlabel("Date")
        plt.ylabel("Log return")
        plt.grid(True, alpha=0.3)
        plt.colorbar(sc, label="Hidden state")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{config.ticker}_returns_states_K{best_k}.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # Transition matrix heatmap
        state_names = [f"State {i}" for i in range(best_k)]
        fig = plot_transition_heatmap(
            model.transmat_,
            state_names=state_names,
            title=f"Transition matrix - Gaussian HMM K={best_k}",
            save_path=os.path.join(save_dir, f"{config.ticker}_transition_matrix_K{best_k}.png")
        )
        plt.close(fig)

        # State characteristics plot
        fig = plot_state_characteristics(
            hmm_state_tables[best_k],
            save_path=os.path.join(save_dir, f"{config.ticker}_state_characteristics_K{best_k}.png")
        )
        plt.close(fig)

    # 11. Print summary
    print("\n=== Experiment Results ===")
    print(results_df.to_string(index=False))
    print(f"\nBest model by DPA: {best_model}")

    # 12. Return results
    total_runtime = time.time() - start_time
    return {
        "config": config,
        "results_df": results_df,
        "best_model": best_model,
        "hmm_models": hmm_models,
        "hmm_state_tables": hmm_state_tables,
        "hmm_analyses": hmm_analyses,
        "train_df": train_df,
        "test_df": test_df,
        "total_runtime": total_runtime
    }


def run_experiment_grid(configs: list, base_save_dir: str = "experiments") -> list:
    """
    Run a grid of experiments (multiple configurations).

    Parameters:
    -----------
    configs : list of ExperimentConfig
        List of configurations to run.
    base_save_dir : str, default "experiments"
        Base directory to save experiment results. Each experiment will be saved in a subdirectory.

    Returns:
    --------
    list of Dict[str, Any]
        List of result dictionaries for each experiment.
    """
    os.makedirs(base_save_dir, exist_ok=True)
    all_results = []

    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Running experiment {i+1}/{len(configs)}: {config.ticker}")
        print(f"{'='*60}")

        # Create a unique save directory for this experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(base_save_dir, f"run_{timestamp}_{config.ticker}")
        if len(configs) > 1:
            # If we are running multiple configs, add more detail to the directory name
            save_dir = os.path.join(base_save_dir, f"run_{timestamp}_{config.ticker}_K{config.k_values}")

        # Run the experiment
        result = run_experiment(config, save_dir=save_dir)
        all_results.append(result)

        # Print a summary of the experiment
        print(f"\nExperiment {i+1} completed in {result['total_runtime']:.2f} seconds.")
        print(f"Best model: {result['best_model']}")
        print(f"Results saved to: {save_dir}")

    return all_results


if __name__ == "__main__":
    # Example usage: run a single experiment for SPY with default settings
    config = ExperimentConfig()
    result = run_experiment(config, save_dir="experiments/spy_default")
    print("\nExperiment completed!")