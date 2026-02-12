import os
import json

from .config import settings
from .train import train
from .predict import predict_next

def artifacts_exist() -> bool:
    return (
        os.path.exists(os.path.join(settings.artifacts_dir, "model.joblib")) and
        os.path.exists(os.path.join(settings.artifacts_dir, "metadata.json"))
    )

def main():
    print(f"{settings.app_name} - Commodity Price Prediction App")
    print(f"Default commodity: {settings.commodity_name} ({settings.ticker})")
    print(f"Default horizon: {settings.horizon_days} trading days ahead\n")

    while True:
        print("Menu:")
        print("1) Train model")
        print("2) Predict next future close")
        print("3) Show last training metadata")
        print("4) Exit")
        choice = input("Select (1-4): ").strip()

        if choice == "1":
            model = input("Choose model (random_forest | svr | mlp): ").strip() or "random_forest"
            horizon = input(f"Horizon days ahead (default {settings.horizon_days}): ").strip()
            horizon = int(horizon) if horizon else settings.horizon_days
            ticker = input(f"Ticker (default {settings.ticker}): ").strip() or settings.ticker
            train(model_name=model, ticker=ticker, horizon=horizon)
            print()

        elif choice == "2":
            if not artifacts_exist():
                print("No trained model found. Train first.\n")
                continue
            ticker = input(f"Ticker (default {settings.ticker}): ").strip() or settings.ticker
            horizon = input(f"Horizon days ahead (default {settings.horizon_days}): ").strip()
            horizon = int(horizon) if horizon else settings.horizon_days

            res = predict_next(ticker=ticker, horizon=horizon)
            print("\n=== Prediction ===")
            print(f"As of {res['last_observed_date']} (close={res['last_observed_close']:.2f})")
            print(f"Target date (approx.): {res['predicted_target_date']}")
            print(f"Predicted close in {res['horizon_days']} trading days: {res['predicted_future_close']:.2f}")
            if res["prediction_interval_10_90"] is not None:
                lo, hi = res["prediction_interval_10_90"]                
                print(f"Approx. 10-90% interval (RF trees): [{lo:.2f}, {hi:.2f}]")
            print(f"Model: {res['model_name']}\n")

        elif choice == "3":
            meta_path = os.path.join(settings.artifacts_dir, "metadata.json")
            if not os.path.exists(meta_path):
                print("No metadata found. Train first.\n")
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print("\n=== Last Training Metadata ===")
            for k in ["trained_at", "ticker", "horizon_days", "model_name", "n_records_raw", "n_records_supervised", "n_features"]:
                print(f"{k}: {meta.get(k)}")
            print("metrics:", meta.get("metrics"))
            print()

        elif choice == "4":
            break

        else:
            print("Invalid choice.\n")

if __name__ == "__main__":
    main()
