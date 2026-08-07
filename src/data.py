"""
Data loading and validation module for HMM market regimes project.
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def flatten_yfinance_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Flatten multi-index columns from yfinance download when multiple tickers are requested.
    """
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            try:
                df = df.xs(ticker, axis=1, level=-1)
            except Exception:
                df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(0)
    return df


def load_asset(ticker: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Download historical price data for a given ticker from Yahoo Finance.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'SPY', 'AAPL')
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, downloads up to present.

    Returns:
    --------
    pd.DataFrame
        DataFrame with OHLCV data indexed by date.
    """
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    df = flatten_yfinance_columns(df, ticker).sort_index()
    if df.empty:
        raise ValueError(f"No data was downloaded for ticker {ticker}. Check ticker/date/internet connection.")
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the downloaded data.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw data from Yahoo Finance.

    Returns:
    --------
    pd.DataFrame
        Cleaned data with infinite values replaced and NaNs dropped.
    """
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    # Drop rows with NaN values
    df = df.dropna().copy()
    return df


def build_features(df: pd.DataFrame, price_col: str = "Adj Close") -> Tuple[pd.DataFrame, str]:
    """
    Build features for HMM model from raw price data.

    Features:
    ---------
    - log_return: log of price ratio (Adj Close / previous Adj Close)
    - rolling_volatility_20: 20-day rolling standard deviation of log returns
    - daily_range: (High - Low) / Close
    - volume_change: log of volume ratio (Volume / previous Volume)
    - next_log_return: next day's log return (target for prediction)
    - next_close: next day's closing price
    - current_close: current day's closing price

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data.
    price_col : str, default "Adj Close"
        Column name to use for price (usually "Adj Close" or "Close").

    Returns:
    --------
    Tuple[pd.DataFrame, str]
        DataFrame with added features and the actual price column used.
    """
    # Determine which price column to use
    if price_col not in df.columns:
        # Fallback to Close if Adj Close is not available
        price_col = "Close" if "Close" in df.columns else df.columns[-1]
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Calculate features
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    df["rolling_volatility_20"] = df["log_return"].rolling(20).std()
    df["daily_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["volume_change"] = np.log(df["Volume"] / df["Volume"].shift(1))
    df["next_log_return"] = df["log_return"].shift(-1)
    df["next_close"] = df[price_col].shift(-1)
    df["current_close"] = df[price_col]
    
    # Replace infinite values and drop NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    
    return df, price_col


def chronological_split(df: pd.DataFrame, test_size: float = 0.30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets chronologically (no shuffling).

    Parameters:
    -----------
    df : pd.DataFrame
        Feature-engineered data.
    test_size : float, default 0.30
        Proportion of data to use for testing (from the end).

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Training and testing DataFrames.
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df