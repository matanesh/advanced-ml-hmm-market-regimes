
# ============================================================
# HMM for Stock Market Regime Detection and Next-Day Direction
# פרויקט: Gaussian HMM לניתוח נתוני שוק ההון
# ============================================================


import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
from hmmlearn.hmm import GaussianHMM

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TICKER = "SPY"          # אפשר לשנות ל-AAPL/MSFT/JPM אחרי שהכל עובד
START_DATE = "2014-01-01"
END_DATE = None
TEST_SIZE = 0.30
FEATURE_COLS = ["log_return", "rolling_volatility_20", "daily_range", "volume_change"]
MA_WINDOW = 5
CONTEXT_WINDOW = 100


def flatten_yfinance_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            try:
                df = df.xs(ticker, axis=1, level=-1)
            except Exception:
                df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(0)
    return df


def download_prices(ticker: str, start: str, end=None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    df = flatten_yfinance_columns(df, ticker).sort_index()
    if df.empty:
        raise ValueError("No data was downloaded. Check ticker/date/internet connection.")
    return df


def build_features(raw_df: pd.DataFrame):
    df = raw_df.copy()
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    df["rolling_volatility_20"] = df["log_return"].rolling(20).std()
    df["daily_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["volume_change"] = np.log(df["Volume"] / df["Volume"].shift(1))
    df["next_log_return"] = df["log_return"].shift(-1)
    df["next_close"] = df[price_col].shift(-1)
    df["current_close"] = df[price_col]
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return df, price_col


def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.30):
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def direction_labels(returns):
    return np.where(np.asarray(returns) >= 0, 1, 0)


def evaluate_predictions(name: str, test_df: pd.DataFrame, pred_return, runtime_sec=np.nan, log_likelihood_train=np.nan, log_likelihood_test=np.nan):
    pred_return = np.asarray(pred_return)
    actual_return = test_df["next_log_return"].values
    actual_close = test_df["next_close"].values
    current_close = test_df["current_close"].values
    pred_close = current_close * np.exp(pred_return)
    dpa = accuracy_score(direction_labels(actual_return), direction_labels(pred_return))
    mae_ret = mean_absolute_error(actual_return, pred_return)
    rmse_ret = np.sqrt(mean_squared_error(actual_return, pred_return))
    mape_price = np.mean(np.abs((actual_close - pred_close) / actual_close)) * 100
    return {"model": name, "DPA_direction_accuracy": dpa, "MAE_return": mae_ret, "RMSE_return": rmse_ret, "MAPE_price_%": mape_price, "log_likelihood_train": log_likelihood_train, "log_likelihood_test": log_likelihood_test, "runtime_sec": runtime_sec}


def average_duration(states, state):
    lengths, cur = [], 0
    for s in states:
        if s == state:
            cur += 1
        else:
            if cur > 0:
                lengths.append(cur)
            cur = 0
    if cur > 0:
        lengths.append(cur)
    return float(np.mean(lengths)) if lengths else 0.0


def summarize_states(df: pd.DataFrame, states):
    tmp = df.copy(); tmp["state"] = states
    rows = []
    for s in sorted(tmp["state"].unique()):
        part = tmp[tmp["state"] == s]
        rows.append({"state": int(s), "frequency_%": 100*len(part)/len(tmp), "mean_daily_return_%": 100*part["log_return"].mean(), "volatility_daily_%": 100*part["log_return"].std(), "mean_rolling_volatility_%": 100*part["rolling_volatility_20"].mean(), "avg_duration_days": average_duration(states, s)})
    return pd.DataFrame(rows).sort_values("state")


def train_gaussian_hmm(X_train_scaled, n_states: int, covariance_type="full"):
    start = time.time()
    model = GaussianHMM(n_components=n_states, covariance_type=covariance_type, n_iter=300, tol=1e-4, random_state=RANDOM_STATE, verbose=False)
    model.fit(X_train_scaled)
    return model, time.time() - start


def decode_past_only_states(model, X_train_scaled, X_test_scaled, context_window=100):
    states = []
    context = X_train_scaled[-context_window:] if len(X_train_scaled) > context_window else X_train_scaled
    for i in range(len(X_test_scaled)):
        seq = np.vstack([context, X_test_scaled[:i+1]])
        states.append(model.predict(seq)[-1])
    return np.array(states)


def hmm_next_return_predictions(model, train_df, train_states, test_states_past_only):
    n_states = model.n_components
    tmp = train_df.copy(); tmp["state"] = train_states
    global_mean = train_df["log_return"].mean()
    state_return_mean = np.zeros(n_states)
    for s in range(n_states):
        vals = tmp.loc[tmp["state"] == s, "log_return"]
        state_return_mean[s] = vals.mean() if len(vals) else global_mean
    return np.array([np.dot(model.transmat_[s], state_return_mean) for s in test_states_past_only])

# 1. Load data
raw = download_prices(TICKER, START_DATE, END_DATE)
df, price_col = build_features(raw)
print(f"Ticker: {TICKER}")
print(f"Rows after feature engineering: {len(df):,}")
print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
print(f"Price column used: {price_col}")
print(df.head())

# 2. Split and scale
train_df, test_df = temporal_train_test_split(df, TEST_SIZE)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df[FEATURE_COLS].values)
X_test_scaled = scaler.transform(test_df[FEATURE_COLS].values)
print(f"Train: {train_df.index.min().date()} to {train_df.index.max().date()} | {len(train_df):,} rows")
print(f"Test:  {test_df.index.min().date()} to {test_df.index.max().date()} | {len(test_df):,} rows")

# 3. EDA plots
plt.figure(figsize=(12,4)); plt.plot(df.index, df[price_col]); plt.title(f"{TICKER} price over time"); plt.xlabel("Date"); plt.ylabel("Price"); plt.grid(True, alpha=0.3); plt.show()
plt.figure(figsize=(12,4)); plt.plot(df.index, df["log_return"]); plt.title(f"{TICKER} daily log returns"); plt.xlabel("Date"); plt.ylabel("Log return"); plt.grid(True, alpha=0.3); plt.show()
plt.figure(figsize=(12,4)); plt.plot(df.index, df["rolling_volatility_20"]); plt.title(f"{TICKER} rolling volatility, 20 days"); plt.xlabel("Date"); plt.ylabel("Rolling volatility"); plt.grid(True, alpha=0.3); plt.show()

# 4. Baselines
results = []
start=time.time(); pred=np.full(len(test_df), train_df["next_log_return"].mean()); results.append(evaluate_predictions("Naive - train mean return", test_df, pred, time.time()-start))
start=time.time(); pred=test_df["log_return"].values; results.append(evaluate_predictions("Naive - persistence", test_df, pred, time.time()-start))
start=time.time(); full_df=pd.concat([train_df,test_df]).copy(); full_df["ma_pred_return"]=full_df["log_return"].rolling(MA_WINDOW).mean(); pred=full_df.loc[test_df.index,"ma_pred_return"].fillna(train_df["next_log_return"].mean()).values; results.append(evaluate_predictions(f"Moving Average {MA_WINDOW}", test_df, pred, time.time()-start))
start=time.time(); train_curr_dir=direction_labels(train_df["log_return"].values); train_next_ret=train_df["next_log_return"].values; exp_next={};
for d in [0,1]:
    vals=train_next_ret[train_curr_dir==d]
    exp_next[d]=vals.mean() if len(vals) else train_df["next_log_return"].mean()
pred=np.array([exp_next[d] for d in direction_labels(test_df["log_return"].values)]); results.append(evaluate_predictions("Discrete Markov Chain", test_df, pred, time.time()-start))

# 5. Gaussian HMM models
hmm_models = {}; hmm_state_tables = {}
for k in [2,3,4]:
    model, runtime = train_gaussian_hmm(X_train_scaled, k)
    train_states = model.predict(X_train_scaled)
    test_states = decode_past_only_states(model, X_train_scaled, X_test_scaled, CONTEXT_WINDOW)
    pred_hmm = hmm_next_return_predictions(model, train_df, train_states, test_states)
    results.append(evaluate_predictions(f"Gaussian HMM K={k}", test_df, pred_hmm, runtime, model.score(X_train_scaled), model.score(X_test_scaled)))
    hmm_models[k] = {"model": model, "train_states": train_states, "test_states_past_only": test_states, "pred_return": pred_hmm}
    hmm_state_tables[k] = summarize_states(train_df, train_states)

# 6. Results
results_df = pd.DataFrame(results).sort_values("DPA_direction_accuracy", ascending=False).reset_index(drop=True)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
print(results_df.to_string())
print("Best model by DPA:", results_df.iloc[0]["model"])

# 7. State interpretation for K=3
CENTRAL_K=3
central = hmm_models[CENTRAL_K]
model = central["model"]
print("State interpretation table for HMM K=3, based on training data:")
print(hmm_state_tables[CENTRAL_K].to_string())
print("Transition matrix for HMM K=3:")
print(pd.DataFrame(model.transmat_, index=[f"from_state_{i}" for i in range(CENTRAL_K)], columns=[f"to_state_{i}" for i in range(CENTRAL_K)]).to_string())

# 8. Visualizations
plot_df = pd.concat([train_df, test_df]).copy()
plot_df["state_k3"] = np.concatenate([central["train_states"], central["test_states_past_only"]])
plot_df["split"] = ["train"]*len(train_df) + ["test"]*len(test_df)

plt.figure(figsize=(13,5)); sc=plt.scatter(plot_df.index, plot_df[price_col], c=plot_df["state_k3"], s=8); plt.title(f"{TICKER} price colored by Gaussian HMM hidden states (K=3)"); plt.xlabel("Date"); plt.ylabel("Price"); plt.grid(True, alpha=0.3); plt.colorbar(sc,label="Hidden state"); plt.savefig("hmm_price_states.png", dpi=150, bbox_inches='tight'); plt.show()
plt.figure(figsize=(13,4)); sc=plt.scatter(plot_df.index, plot_df["log_return"], c=plot_df["state_k3"], s=8); plt.title(f"{TICKER} log returns colored by hidden states (K=3)"); plt.xlabel("Date"); plt.ylabel("Log return"); plt.grid(True, alpha=0.3); plt.colorbar(sc,label="Hidden state"); plt.savefig("hmm_returns_states.png", dpi=150, bbox_inches='tight'); plt.show()
plt.figure(figsize=(6,5)); plt.imshow(model.transmat_); plt.title("Transition matrix - Gaussian HMM K=3"); plt.xlabel("To state"); plt.ylabel("From state"); plt.colorbar(label="Probability")
for i in range(CENTRAL_K):
    for j in range(CENTRAL_K):
        plt.text(j, i, f"{model.transmat_[i,j]:.2f}", ha="center", va="center")
plt.savefig("hmm_transition_matrix.png", dpi=150, bbox_inches='tight')
plt.show()
state_groups=[plot_df.loc[plot_df["state_k3"]==s,"log_return"].values for s in range(CENTRAL_K)]
plt.figure(figsize=(8,5)); plt.boxplot(state_groups); plt.xticks(range(1, CENTRAL_K+1), [f"State {s}" for s in range(CENTRAL_K)]); plt.title("Distribution of daily log returns by hidden state"); plt.ylabel("Log return"); plt.grid(True, alpha=0.3); plt.savefig("hmm_state_boxplot.png", dpi=150, bbox_inches='tight'); plt.show()

print("\nTemplate for conclusions:")
print("1. Check which model achieved the best DPA and whether the improvement over baselines is meaningful.")
print("2. Check whether HMM states have different mean return and volatility.")
print("3. Check whether the transition matrix shows persistence: high diagonal values mean stable regimes.")
print("4. Do not claim profitable prediction; present the project as regime analysis and cautious forecasting comparison.")
