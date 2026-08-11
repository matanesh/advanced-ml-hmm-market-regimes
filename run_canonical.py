#!/usr/bin/env python3
"""Canonical Gaussian-HMM experiment runner.

This runner preserves the original five-asset canonical question while fixing
submission-chain issues identified during external review:

- the held-out Test set is not scored before K/seed selection is locked;
- validation observations may be used as *past context* at Test time, but never
  as training targets for the locked HMM parameters;
- market downloads can be cached for reproducible reruns;
- artifact paths are registered before the manifest is written, so the on-disk
  manifest is complete;
- posterior uncertainty is saved as a diagnostic without changing the original
  hard-state HMM forecast used in the canonical model comparison.

Every run is written to a new timestamped directory. Existing canonical
artifacts are never overwritten.
"""

from __future__ import annotations

import json
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import (
    TargetSafeSplits,
    build_features,
    load_asset,
    target_safe_train_validation_test_split,
    validate_data,
)
from src.evaluation import direction_labels, evaluate_predictions, summarize_states
from src.model import (
    decode_past_only_posteriors,
    decode_past_only_states,
    gaussian_hmm_diagnostics,
    hmm_next_return_predictions,
    posterior_confidence,
    posterior_entropy,
    train_gaussian_hmm,
)

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class CanonicalConfig:
    """Immutable configuration for a canonical run."""

    assets: Tuple[str, ...] = ("SPY", "GLD", "TLT", "BTC-USD", "DIS")
    start_date: str = "2014-01-01"
    end_date: str = "2024-12-31"
    validation_size: float = 0.15
    test_size: float = 0.15
    feature_cols: Tuple[str, ...] = (
        "log_return",
        "rolling_volatility_20",
        "daily_range",
        "volume_change",
    )
    k_values: Tuple[int, ...] = (2, 3, 4)
    covariance_type: str = "full"
    n_iter: int = 300
    tol: float = 1e-4
    seeds: Tuple[int, ...] = (42, 123, 456)
    context_window: int = 100
    ma_window: int = 5
    cache_dir: str = "data_cache"
    output_root: str = "experiments_canonical"
    protocol_version: str = "1.1"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, tuple):
                result[key] = list(value)
        return result


@dataclass
class ModelResult:
    """Serializable metrics for one baseline/HMM configuration."""

    asset: str
    fold: int
    model_name: str
    model_family: str
    k: Optional[int] = None
    seed: Optional[int] = None
    selected_by: Optional[str] = None
    dpa: float = 0.0
    mae_return: float = 0.0
    rmse_return: float = 0.0
    mape_price_pct: float = 0.0
    log_likelihood_train: float = float("nan")
    log_likelihood_val: float = float("nan")
    log_likelihood_test: float = float("nan")
    converged: Optional[bool] = None
    iterations: Optional[int] = None
    n_parameters: Optional[int] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    runtime_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, float) and np.isnan(value):
                result[key] = None
        return result


@dataclass
class FoldResult:
    asset: str
    fold: int
    train_dates: Tuple[str, str]
    val_dates: Tuple[str, str]
    test_dates: Tuple[str, str]
    hmm_results_all_starts: List[ModelResult]
    selected_hmm: ModelResult
    test_results: List[ModelResult]


class CanonicalRunner:
    """Execute the fixed canonical protocol and emit versioned artifacts."""

    def __init__(self, config: CanonicalConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"canonical_{self.timestamp}"
        self.output_dir = Path(config.output_root) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest: Dict[str, Any] = {
            "protocol_version": config.protocol_version,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "git_commit": self._get_git_commit(),
            "config": config.to_dict(),
            "assets": list(config.assets),
            "folds_completed": [],
            "artifacts": {},
            "protocol_notes": [
                "K and seed are selected using validation log-likelihood only.",
                "The Test set is not scored before model selection is locked.",
                "The locked HMM is trained on Train only; Validation may be used as past inference context at Test time.",
                "Posterior uncertainty is diagnostic and does not replace the canonical hard-state forecast.",
            ],
        }

    @staticmethod
    def _get_git_commit() -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.stdout.strip()[:12] or "unknown"
        except Exception:
            return "unknown"

    def run(self) -> List[FoldResult]:
        all_results: List[FoldResult] = []
        for asset in self.config.assets:
            print("\n" + "=" * 64)
            print(f"Processing {asset}")
            print("=" * 64)
            try:
                fold = self._process_asset(asset)
                all_results.append(fold)
                self.manifest["folds_completed"].append(
                    {"asset": asset, "fold": 0, "status": "completed"}
                )
            except Exception as exc:
                print(f"ERROR processing {asset}: {exc}")
                self.manifest["folds_completed"].append(
                    {"asset": asset, "fold": 0, "status": "failed", "error": str(exc)}
                )

        self._write_results_csv(all_results)
        self._write_model_diagnostics_json(all_results)
        # Manifest is deliberately written last so its artifact map is complete.
        self._write_manifest()

        print("\n" + "=" * 64)
        print(f"Canonical run complete: {self.output_dir}")
        print("=" * 64)
        return all_results

    def _process_asset(self, asset: str) -> FoldResult:
        raw = load_asset(
            asset,
            self.config.start_date,
            self.config.end_date,
            cache_dir=self.config.cache_dir,
        )
        raw = validate_data(raw)
        df, price_col = build_features(raw)
        splits = target_safe_train_validation_test_split(
            df,
            validation_size=self.config.validation_size,
            test_size=self.config.test_size,
        )
        print(
            f"  Train {len(splits.train)} | Val {len(splits.validation)} | Test {len(splits.test)}"
        )

        feature_cols = list(self.config.feature_cols)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(splits.train[feature_cols].to_numpy())
        X_val = scaler.transform(splits.validation[feature_cols].to_numpy())
        X_test = scaler.transform(splits.test[feature_cols].to_numpy())

        starts = self._fit_candidates(asset, splits, X_train, X_val)
        converged = [
            result
            for result in starts
            if result.converged and np.isfinite(result.log_likelihood_val)
        ]
        if not converged:
            raise RuntimeError(f"No converged HMM fits for {asset}")

        selected = max(converged, key=lambda result: result.log_likelihood_val)
        selected.selected_by = "validation_log_likelihood"
        print(
            f"  Selected K={selected.k}, seed={selected.seed}, "
            f"val LL={selected.log_likelihood_val:.2f}"
        )

        test_results, final_payload = self._evaluate_locked_model(
            asset,
            splits,
            scaler,
            X_train,
            X_val,
            X_test,
            selected,
        )

        fold = FoldResult(
            asset=asset,
            fold=0,
            train_dates=(
                splits.train.index.min().date().isoformat(),
                splits.train.index.max().date().isoformat(),
            ),
            val_dates=(
                splits.validation.index.min().date().isoformat(),
                splits.validation.index.max().date().isoformat(),
            ),
            test_dates=(
                splits.test.index.min().date().isoformat(),
                splits.test.index.max().date().isoformat(),
            ),
            hmm_results_all_starts=starts,
            selected_hmm=selected,
            test_results=test_results,
        )
        self._save_asset_artifacts(asset, fold, splits, price_col, final_payload)
        return fold

    def _fit_candidates(
        self,
        asset: str,
        splits: TargetSafeSplits,
        X_train: np.ndarray,
        X_val: np.ndarray,
    ) -> List[ModelResult]:
        """Fit all K x seed candidates without touching the Test set."""
        results: List[ModelResult] = []
        for k in self.config.k_values:
            for seed in self.config.seeds:
                start = time.time()
                try:
                    model, fit_time = train_gaussian_hmm(
                        X_train,
                        n_states=k,
                        covariance_type=self.config.covariance_type,
                        n_iter=self.config.n_iter,
                        tol=self.config.tol,
                        random_state=seed,
                    )
                    diag = gaussian_hmm_diagnostics(model, X_train)
                    val_ll = float(model.score(X_val))
                    result = ModelResult(
                        asset=asset,
                        fold=0,
                        model_name=f"Gaussian HMM K={k} seed={seed}",
                        model_family="hmm",
                        k=k,
                        seed=seed,
                        log_likelihood_train=float(diag["train_log_likelihood"]),
                        log_likelihood_val=val_ll,
                        # Test likelihood intentionally remains NaN until selection is locked.
                        converged=bool(diag["converged"]),
                        iterations=int(diag["iterations"]),
                        n_parameters=int(diag["n_parameters"]),
                        aic=float(diag["aic"]),
                        bic=float(diag["bic"]),
                        runtime_sec=float(fit_time),
                    )
                    results.append(result)
                    status = "OK" if result.converged else "NOT-CONVERGED"
                    print(f"    K={k} seed={seed}: {status}, val LL={val_ll:.2f}")
                except Exception as exc:
                    print(f"    K={k} seed={seed}: FAILED ({exc})")
                    results.append(
                        ModelResult(
                            asset=asset,
                            fold=0,
                            model_name=f"Gaussian HMM K={k} seed={seed}",
                            model_family="hmm",
                            k=k,
                            seed=seed,
                            converged=False,
                            runtime_sec=float(time.time() - start),
                        )
                    )
        return results

    def _baseline_predictions(
        self,
        history_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """Generate baselines from information available before each next-day target."""
        history_mean = float(history_df["next_log_return"].mean())
        predictions: Dict[str, np.ndarray] = {
            "Naive - train mean": np.full(len(test_df), history_mean),
            "Naive - persistence": test_df["log_return"].to_numpy(dtype=float),
        }

        full = pd.concat([history_df, test_df]).sort_index().copy()
        full["ma_pred"] = full["log_return"].rolling(self.config.ma_window).mean()
        predictions[f"Moving Average {self.config.ma_window}"] = (
            full.loc[test_df.index, "ma_pred"].fillna(history_mean).to_numpy(dtype=float)
        )

        current_direction = direction_labels(history_df["log_return"].to_numpy())
        next_returns = history_df["next_log_return"].to_numpy(dtype=float)
        expected: Dict[int, float] = {}
        for direction in (0, 1):
            values = next_returns[current_direction == direction]
            expected[direction] = float(values.mean()) if len(values) else history_mean
        test_direction = direction_labels(test_df["log_return"].to_numpy())
        predictions["Discrete Markov Chain"] = np.asarray(
            [expected[int(direction)] for direction in test_direction], dtype=float
        )
        return predictions

    def _evaluate_locked_model(
        self,
        asset: str,
        splits: TargetSafeSplits,
        scaler: StandardScaler,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        selected: ModelResult,
    ) -> Tuple[List[ModelResult], Dict[str, Any]]:
        """Evaluate baselines and the selected HMM on Test after selection is locked."""
        history_df = pd.concat([splits.train, splits.validation]).sort_index().copy()
        results: List[ModelResult] = []

        for name, prediction in self._baseline_predictions(history_df, splits.test).items():
            start = time.time()
            evaluation = evaluate_predictions(name, splits.test, prediction)
            results.append(
                ModelResult(
                    asset=asset,
                    fold=0,
                    model_name=name,
                    model_family="baseline",
                    dpa=float(evaluation["DPA_direction_accuracy"]),
                    mae_return=float(evaluation["MAE_return"]),
                    rmse_return=float(evaluation["RMSE_return"]),
                    mape_price_pct=float(evaluation["MAPE_price_%"]),
                    runtime_sec=float(time.time() - start),
                )
            )

        model, fit_time = train_gaussian_hmm(
            X_train,
            n_states=int(selected.k),
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            tol=self.config.tol,
            random_state=int(selected.seed),
        )
        diag = gaussian_hmm_diagnostics(model, X_train)
        train_states = model.predict(X_train)

        # Validation is past by the time Test begins. It is therefore allowed as
        # inference context, while the HMM parameters and state-return means remain
        # fitted on Train only.
        X_history_context = np.vstack([X_train, X_val])
        test_states = decode_past_only_states(
            model, X_history_context, X_test, self.config.context_window
        )
        test_posteriors = decode_past_only_posteriors(
            model, X_history_context, X_test, self.config.context_window
        )
        entropy = posterior_entropy(test_posteriors, normalize=True)
        confidence = posterior_confidence(test_posteriors)

        prediction = hmm_next_return_predictions(
            model, splits.train, train_states, test_states
        )
        test_ll = float(model.score(X_test))
        evaluation = evaluate_predictions(
            f"Gaussian HMM K={selected.k} (selected)",
            splits.test,
            prediction,
            fit_time,
            float(diag["train_log_likelihood"]),
            test_ll,
        )
        results.append(
            ModelResult(
                asset=asset,
                fold=0,
                model_name=evaluation["model"],
                model_family="hmm",
                k=selected.k,
                seed=selected.seed,
                selected_by=selected.selected_by,
                dpa=float(evaluation["DPA_direction_accuracy"]),
                mae_return=float(evaluation["MAE_return"]),
                rmse_return=float(evaluation["RMSE_return"]),
                mape_price_pct=float(evaluation["MAPE_price_%"]),
                log_likelihood_train=float(diag["train_log_likelihood"]),
                log_likelihood_val=selected.log_likelihood_val,
                log_likelihood_test=test_ll,
                converged=bool(diag["converged"]),
                iterations=int(diag["iterations"]),
                n_parameters=int(diag["n_parameters"]),
                aic=float(diag["aic"]),
                bic=float(diag["bic"]),
                runtime_sec=float(fit_time),
            )
        )

        payload = {
            "model": model,
            "train_states": train_states,
            "test_states": test_states,
            "test_posteriors": test_posteriors,
            "test_entropy": entropy,
            "test_confidence": confidence,
        }
        return results, payload

    def _save_asset_artifacts(
        self,
        asset: str,
        fold: FoldResult,
        splits: TargetSafeSplits,
        price_col: str,
        payload: Dict[str, Any],
    ) -> None:
        asset_dir = self.output_dir / asset
        asset_dir.mkdir(parents=True, exist_ok=True)
        model = payload["model"]
        k = int(fold.selected_hmm.k)

        train_posteriors = model.predict_proba(
            StandardScaler()
            .fit(splits.train[list(self.config.feature_cols)].to_numpy())
            .transform(splits.train[list(self.config.feature_cols)].to_numpy())
        )
        # The train state summary remains based on the locked model's hard state
        # sequence; posterior fields are diagnostic additions.
        train_entropy = posterior_entropy(train_posteriors, normalize=True)
        state_summary = summarize_states(
            splits.train,
            payload["train_states"],
            posterior_probabilities=train_posteriors,
            posterior_entropy_values=train_entropy,
        )
        state_table_path = asset_dir / f"state_table_K{k}.csv"
        state_summary.to_csv(state_table_path, index=False)

        transmat = pd.DataFrame(
            model.transmat_,
            index=[f"from_state_{i}" for i in range(k)],
            columns=[f"to_state_{i}" for i in range(k)],
        )
        trans_path = asset_dir / f"transition_matrix_K{k}.csv"
        transmat.to_csv(trans_path)

        from src.analysis import analyze_transitions, plot_state_characteristics, plot_transition_heatmap

        transition_analysis = analyze_transitions(
            model.transmat_, [f"State {i}" for i in range(k)]
        )
        serializable_analysis = {
            "n_states": transition_analysis["n_states"],
            "state_names": transition_analysis["state_names"],
            "average_diagonal_probability": float(
                transition_analysis["average_diagonal_probability"]
            ),
            "average_row_entropy": float(transition_analysis["average_row_entropy"]),
            "stationary_distribution": np.asarray(
                transition_analysis["stationary_distribution"]
            ).tolist(),
            "mean_recurrence_time": np.asarray(
                transition_analysis["mean_recurrence_time"]
            ).tolist(),
            "spectral_gap": float(transition_analysis["spectral_gap"]),
            "eigenvalues": [
                {"real": float(np.real(value)), "imag": float(np.imag(value))}
                for value in transition_analysis["eigenvalues"]
            ],
        }
        analysis_path = asset_dir / f"transition_analysis_K{k}.json"
        with analysis_path.open("w", encoding="utf-8") as handle:
            json.dump(serializable_analysis, handle, indent=2)

        posterior_daily = splits.test[
            ["log_return", "rolling_volatility_20", "daily_range", "volume_change", "current_close"]
        ].copy()
        posterior_daily["state_viterbi_past_only"] = payload["test_states"]
        posterior_daily["posterior_state_argmax"] = np.argmax(
            payload["test_posteriors"], axis=1
        )
        posterior_daily["posterior_confidence"] = payload["test_confidence"]
        posterior_daily["posterior_entropy"] = payload["test_entropy"]
        for state in range(k):
            posterior_daily[f"p_state_{state}"] = payload["test_posteriors"][:, state]
        posterior_path = asset_dir / f"posterior_test_K{k}.csv"
        posterior_daily.to_csv(posterior_path)

        try:
            import matplotlib.pyplot as plt

            # For visualization only, validation/test states are decoded using past
            # prefixes. No future test observation influences an earlier point.
            feature_cols = list(self.config.feature_cols)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(splits.train[feature_cols].to_numpy())
            X_val = scaler.transform(splits.validation[feature_cols].to_numpy())
            X_test = scaler.transform(splits.test[feature_cols].to_numpy())
            val_states = decode_past_only_states(
                model, X_train, X_val, self.config.context_window
            )
            test_states = decode_past_only_states(
                model, np.vstack([X_train, X_val]), X_test, self.config.context_window
            )
            plot_df = pd.concat([splits.train, splits.validation, splits.test]).copy()
            plot_df["state"] = np.concatenate(
                [payload["train_states"], val_states, test_states]
            )

            fig, ax = plt.subplots(figsize=(13, 5))
            scatter = ax.scatter(
                plot_df.index, plot_df[price_col], c=plot_df["state"], s=8
            )
            ax.set_title(f"{asset} price colored by HMM states (K={k})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.grid(alpha=0.3)
            fig.colorbar(scatter, ax=ax, label="Hidden state")
            fig.tight_layout()
            price_plot = asset_dir / f"{asset}_price_states_K{k}.png"
            fig.savefig(price_plot, dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(13, 4))
            scatter = ax.scatter(
                plot_df.index, plot_df["log_return"], c=plot_df["state"], s=8
            )
            ax.set_title(f"{asset} log returns colored by HMM states (K={k})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Log return")
            ax.grid(alpha=0.3)
            fig.colorbar(scatter, ax=ax, label="Hidden state")
            fig.tight_layout()
            return_plot = asset_dir / f"{asset}_returns_states_K{k}.png"
            fig.savefig(return_plot, dpi=150, bbox_inches="tight")
            plt.close(fig)

            transition_plot = asset_dir / f"{asset}_transition_matrix_K{k}.png"
            fig = plot_transition_heatmap(
                model.transmat_,
                [f"State {i}" for i in range(k)],
                title=f"Transition Matrix - {asset} HMM K={k}",
                save_path=str(transition_plot),
            )
            plt.close(fig)

            state_plot = asset_dir / f"{asset}_state_characteristics_K{k}.png"
            fig = plot_state_characteristics(
                state_summary,
                save_path=str(state_plot),
            )
            plt.close(fig)
        except Exception as exc:
            print(f"  Plot warning for {asset}: {exc}")

        self.manifest["artifacts"].setdefault("asset_files", {})[asset] = {
            "test_results": f"{asset}/test_results.csv",
            "state_table": str(state_table_path.relative_to(self.output_dir)),
            "transition_matrix": str(trans_path.relative_to(self.output_dir)),
            "transition_analysis": str(analysis_path.relative_to(self.output_dir)),
            "posterior_test": str(posterior_path.relative_to(self.output_dir)),
        }

    def _write_results_csv(self, folds: List[FoldResult]) -> None:
        rows: List[Dict[str, Any]] = []
        for fold in folds:
            asset_dir = self.output_dir / fold.asset
            asset_dir.mkdir(parents=True, exist_ok=True)
            asset_rows = []
            for result in fold.test_results:
                row = result.to_dict()
                row["asset"] = fold.asset
                row["fold"] = fold.fold
                rows.append(row)
                asset_rows.append(row)
            pd.DataFrame(asset_rows).to_csv(asset_dir / "test_results.csv", index=False)

        path = self.output_dir / "results.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        self.manifest["artifacts"]["results_csv"] = "results.csv"

    def _write_model_diagnostics_json(self, folds: List[FoldResult]) -> None:
        payload: Dict[str, Any] = {}
        for fold in folds:
            asset_payload: Dict[str, Any] = {}
            for result in fold.hmm_results_all_starts:
                key = f"K={result.k}_seed={result.seed}"
                asset_payload[key] = {
                    "k": result.k,
                    "seed": result.seed,
                    "converged": result.converged,
                    "iterations": result.iterations,
                    "train_log_likelihood": result.log_likelihood_train,
                    "validation_log_likelihood": result.log_likelihood_val,
                    # There is intentionally no pre-selection Test likelihood here.
                    "n_parameters": result.n_parameters,
                    "aic": result.aic,
                    "bic": result.bic,
                    "runtime_sec": result.runtime_sec,
                }
            asset_payload["selected"] = {
                "k": fold.selected_hmm.k,
                "seed": fold.selected_hmm.seed,
                "selected_by": fold.selected_hmm.selected_by,
            }
            payload[fold.asset] = asset_payload

        path = self.output_dir / "model_diagnostics.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.manifest["artifacts"]["model_diagnostics"] = "model_diagnostics.json"

    def _write_manifest(self) -> None:
        self.manifest["artifacts"]["manifest"] = "manifest.json"
        path = self.output_dir / "manifest.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, indent=2)


def main() -> None:
    runner = CanonicalRunner(CanonicalConfig())
    results = runner.run()
    print("\nCANONICAL RUN SUMMARY")
    print("=" * 64)
    for fold in results:
        print(f"\n{fold.asset}:")
        for result in fold.test_results:
            selected = " <- SELECTED" if result.model_family == "hmm" else ""
            print(
                f"  {result.model_name:40s} "
                f"DPA={result.dpa:.4f} MAE={result.mae_return:.6f}{selected}"
            )


if __name__ == "__main__":
    main()
