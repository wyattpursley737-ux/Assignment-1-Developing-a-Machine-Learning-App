import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import settings
from .data_pipeline import load_or_download
from .features import make_feature_frame, make_supervised
from .models import get_model_spec, build_pipeline

def chronological_train_test_split(X: pd.DataFrame, y: pd.Series, aligned: pd.DataFrame, test_size: float):
    n = len(X)
    split = int(np.floor(n * (1.0 - test_size)))
    X_train, X_test = X.iloc[:split].copy(), X.iloc[split:].copy()
    y_train, y_test = y.iloc[:split].copy(), y.iloc[split:].copy()
    a_train, a_test = aligned.iloc[:split].copy(), aligned.iloc[split:].copy()
    return X_train, X_test, y_train, y_test, a_train, a_test

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-12))) * 100.0)

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, current_close: np.ndarray) -> float:
    """Measures whether predicted direction (up/down vs current) matches actual direction."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    current_close = np.asarray(current_close, dtype=float)

    true_dir = np.sign(y_true - current_close)
    pred_dir = np.sign(y_pred - current_close)
    return float(np.mean(true_dir == pred_dir) * 100.0)

def ensure_dirs():
    os.makedirs(settings.artifacts_dir, exist_ok=True)
    os.makedirs(settings.outputs_dir, exist_ok=True)

def train(model_name: str = "random_forest", ticker: str = settings.ticker, horizon: int = settings.horizon_days):
    ensure_dirs()

    raw = load_or_download(ticker=ticker)
    feat = make_feature_frame(raw)
    X, y, aligned, feature_cols = make_supervised(feat, horizon=horizon)

    X_train, X_test, y_train, y_test, a_train, a_test = chronological_train_test_split(
        X, y, aligned, test_size=settings.test_size
    )

    spec = get_model_spec(model_name=model_name)
    pipe = build_pipeline(spec)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mape_val = mape(y_test.to_numpy(), y_pred)

    dir_acc = directional_accuracy(
        y_true=y_test.to_numpy(),
        y_pred=y_pred,
        current_close=a_test["close"].to_numpy(),
    )

    stamp = datetime.now().isoformat(timespec="seconds")
    artifact = {
        "app_name": settings.app_name,
        "commodity_name": settings.commodity_name,
        "ticker": ticker,
        "interval": settings.interval,
        "start_date": settings.start_date,
        "end_date": settings.end_date,
        "horizon_days": horizon,
        "model_name": spec.name,
        "n_records_raw": int(len(raw)),
        "n_records_supervised": int(len(X)),
        "n_features": int(X.shape[1]),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "metrics": {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE_percent": mape_val,
            "DirectionalAccuracy_percent": dir_acc,
        },
        "trained_at": stamp,
        "feature_columns": list(feature_cols),
    }

    model_path = os.path.join(settings.artifacts_dir, "model.joblib")
    meta_path = os.path.join(settings.artifacts_dir, "metadata.json")
    joblib.dump(pipe, model_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    sns.set_theme(style="whitegrid")

    fig1 = plt.figure(figsize=(10, 4))
    plt.plot(a_test["date"], y_test.to_numpy(), label="Actual future close", linewidth=1.5)
    plt.plot(a_test["date"], y_pred, label="Predicted future close", linewidth=1.5)
    plt.title(f"{settings.commodity_name} ({ticker}) - {horizon}-Day Ahead Close Prediction ({spec.name})")
    plt.xlabel("As-of date (features at time t)")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(settings.outputs_dir, "pred_vs_actual.png")
    plt.savefig(plot_path, dpi=160)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(6, 4))
    residuals = y_test.to_numpy() - y_pred
    sns.histplot(residuals, bins=40, kde=True)
    plt.title("Residuals (Actual - Predicted)")
    plt.xlabel("Residual")
    plt.tight_layout()
    resid_path = os.path.join(settings.outputs_dir, "residuals_hist.png")
    plt.savefig(resid_path, dpi=160)
    plt.close(fig2)

    log_path = os.path.join(settings.outputs_dir, "run_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== CommodityCast Training Run ===\n")
        f.write(f"Timestamp: {stamp}\n")
        f.write(f"Commodity: {settings.commodity_name}\n")
        f.write(f"Ticker: {ticker}\n")
        f.write(f"Date range: {settings.start_date} to {settings.end_date} ({settings.interval})\n")
        f.write(f"Horizon (days ahead): {horizon}\n")
        f.write(f"Model: {spec.name}\n")
        f.write(f"Raw records: {len(raw)}\n")
        f.write(f"Supervised records: {len(X)}\n")
        f.write(f"Features: {X.shape[1]}\n")
        f.write(f"Train size: {len(X_train)}\n")
        f.write(f"Test size: {len(X_test)}\n\n")
        f.write("Metrics on test set:\n")
        for k, v in artifact["metrics"].items():
            f.write(f"- {k}: {v}\n")

    print("=== TRAINING COMPLETE ===")
    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {meta_path}")
    print(f"Saved plots to: {settings.outputs_dir}")
    print(f"Saved run log to: {log_path}")
    print("Metrics:", artifact["metrics"])

    return artifact

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train commodity price prediction model.")
    parser.add_argument("--model", type=str, default="random_forest", help="random_forest | svr | mlp")
    parser.add_argument("--ticker", type=str, default=settings.ticker, help="Commodity ticker (e.g., GC=F).")    
    parser.add_argument("--horizon", type=int, default=settings.horizon_days, help="Days ahead to predict.")
    args = parser.parse_args()

    train(model_name=args.model, ticker=args.ticker, horizon=args.horizon)
