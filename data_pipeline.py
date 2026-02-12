import os
import pandas as pd
import yfinance as yf

from .config import settings

def cache_path(ticker: str, interval: str, start: str, end: str) -> str:
    # Keep '=' so filenames match common finance tickers like GC=F. Replace only separators.
    safe_ticker = ticker.replace("/", "_")
    return os.path.join(settings.cache_dir, f"{safe_ticker}_{interval}_{start}_{end}.csv")

def download_ohlcv(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("No data downloaded. Check internet access, ticker, or date range.")

    df = df.reset_index()
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Expected: date/datetime + open/high/low/close/adj_close/volume (adj_close may be missing for some tickers)
    if "date" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})

    if "date" not in df.columns:
        raise ValueError(f"Downloaded data missing a date column. Columns: {list(df.columns)}")

    keep = [c for c in ["date", "open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    df = df[keep].copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Remove rows with missing close (cannot train)
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    return df

def load_or_download(ticker: str = settings.ticker) -> pd.DataFrame:
    os.makedirs(settings.cache_dir, exist_ok=True)
    path = cache_path(ticker, settings.interval, settings.start_date, settings.end_date)

    if os.path.exists(path):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    df = download_ohlcv(ticker=ticker, start=settings.start_date, end=settings.end_date, interval=settings.interval)
    df.to_csv(path, index=False)
    return df
