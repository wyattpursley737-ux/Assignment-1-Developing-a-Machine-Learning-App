import os
import json
from datetime import datetime

import joblib
import pandas as pd
from pandas.tseries.offsets import BDay

from .config import settings
from .data_pipeline import load_or_download
from .features import make_feature_frame, make_inference_X
from .models import rf_tree_interval

def load_artifacts():
    model_path = os.path.join(settings.artifacts_dir, "model.joblib")
    meta_path = os.path.join(settings.artifacts_dir, "metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Model artifacts not found. Train first: python -m src.train")

    pipe = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return pipe, meta

def predict_next(ticker: str = settings.ticker, horizon: int = settings.horizon_days):
    pipe, meta = load_artifacts()

    raw = load_or_download(ticker=ticker)
    feat = make_feature_frame(raw)

    feature_cols = meta.get("feature_columns", [])
    if not feature_cols:
        raise ValueError("metadata.json is missing feature_columns. Retrain with the updated code.")

    X_all, aligned = make_inference_X(feat, feature_cols)

    X_last = X_all.iloc[[-1]].copy()
    date_last = pd.Timestamp(aligned.iloc[-1]["date"])
    close_last = float(aligned.iloc[-1]["close"])

    pred = float(pipe.predict(X_last)[0])

    interval = None
    try:
        model_step = getattr(pipe, "named_steps", {}).get("model", None)
        if model_step is not None and hasattr(model_step, "estimators_"):
            x_np = X_last.to_numpy().astype(float).ravel()
            lo, hi = rf_tree_interval(model_step, x_np, 0.1, 0.9)
            interval = (lo, hi)
    except Exception:
        interval = None

    # Approximate the target date as horizon business days after the last observed date
    target_date = (date_last + BDay(horizon)).date()

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker,
        "horizon_days": horizon,
        "last_observed_date": str(date_last.date()),
        "predicted_target_date": str(target_date),
        "last_observed_close": close_last,
        "predicted_future_close": pred,
        "prediction_interval_10_90": interval,
        "model_name": meta.get("model_name"),
    }
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict next future close using trained artifacts.")
    parser.add_argument("--ticker", type=str, default=settings.ticker, help="Commodity ticker (e.g., GC=F).")
    parser.add_argument("--horizon", type=int, default=settings.horizon_days, help="Days ahead to predict.")
    args = parser.parse_args()

    res = predict_next(ticker=args.ticker, horizon=args.horizon)
    print(res)
