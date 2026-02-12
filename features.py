import numpy as np
import pandas as pd

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def make_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns: date, open, high, low, close, volume (adj_close optional)
    Output: df with engineered features, aligned by date.
    """
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    out["log_close"] = np.log(out["close"])
    out["log_return_1"] = out["log_close"].diff()

    # Lagged returns (causal features)
    for lag in range(1, 11):
        out[f"ret_lag_{lag}"] = out["log_return_1"].shift(lag)

    # Rolling stats of returns
    for w in [5, 10, 20]:
        out[f"ret_mean_{w}"] = out["log_return_1"].rolling(w).mean()
        out[f"ret_std_{w}"] = out["log_return_1"].rolling(w).std()

    # Price-based rolling features
    for w in [5, 10, 20]:
        out[f"close_sma_{w}"] = out["close"].rolling(w).mean()
        out[f"close_min_{w}"] = out["close"].rolling(w).min()
        out[f"close_max_{w}"] = out["close"].rolling(w).max()

    # Volatility proxy (high-low range relative to close)
    out["hl_range"] = (out["high"] - out["low"]) / (out["close"] + 1e-12)

    # Volume features (if present)
    if "volume" in out.columns:
        out["log_volume"] = np.log(out["volume"].replace(0, np.nan))
        out["vol_chg_1"] = out["log_volume"].diff()
        out["vol_sma_10"] = out["volume"].rolling(10).mean()
    else:
        out["log_volume"] = np.nan
        out["vol_chg_1"] = np.nan
        out["vol_sma_10"] = np.nan

    # Momentum indicators
    out["rsi_14"] = rsi(out["close"], period=14)
    out["ema_12"] = ema(out["close"], span=12)
    out["ema_26"] = ema(out["close"], span=26)
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal_9"] = ema(out["macd"], span=9)

    # Calendar features (often helpful for commodities)
    out["dow"] = out["date"].dt.dayofweek.astype(int)  # 0=Mon
    out["month"] = out["date"].dt.month.astype(int)

    return out

def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select feature columns explicitly to prevent leakage."""
    feature_cols = [c for c in df.columns if c.startswith((
        "ret_lag_", "ret_mean_", "ret_std_", "close_sma_", "close_min_", "close_max_"
    ))]
    feature_cols += [
        "hl_range", "rsi_14", "ema_12", "ema_26", "macd", "macd_signal_9",
        "dow", "month", "vol_chg_1", "vol_sma_10"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    return feature_cols

def make_supervised(feature_df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """
    Predict future close price at t+horizon using features at time t.
    Returns: X, y, aligned_frame (date/close at time t), feature_cols
    """
    df = feature_df.copy()
    df["target_close_fwd"] = df["close"].shift(-horizon)

    feature_cols = select_feature_columns(df)

    X = df[feature_cols].copy()
    y = df["target_close_fwd"].copy()

    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    aligned = df.loc[mask, ["date", "close"]].reset_index(drop=True)

    return X, y, aligned, feature_cols

def make_inference_X(feature_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build an inference matrix without needing a known future target.
    Uses the most recent row with all features present.
    Returns: X_all, aligned(date, close at time t)
    """
    X = feature_df[feature_cols].copy()
    mask = X.notna().all(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    aligned = feature_df.loc[mask, ["date", "close"]].reset_index(drop=True)
    return X, aligned
