import os
import json
from datetime import datetime

from .config import settings
from .train import train
from .predict import predict_next

APA_REPORT_PATH = "report_apa.md"

def artifacts_exist() -> bool:
    return (
        os.path.exists(os.path.join(settings.artifacts_dir, "model.joblib")) and
        os.path.exists(os.path.join(settings.artifacts_dir, "metadata.json"))
    )

def load_metadata() -> dict:
    meta_path = os.path.join(settings.artifacts_dir, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_report(meta: dict, prediction: dict) -> None:
    """Writes an APA-style report using REAL dataset stats and REAL results from metadata.json."""
    now = datetime.now().strftime("%B %d, %Y")
    title = f"Machine Learning App for Commodity Price Prediction: {settings.app_name}"

    n_records = meta["n_records_raw"]
    n_supervised = meta["n_records_supervised"]
    n_features = meta["n_features"]
    horizon = meta["horizon_days"]
    ticker = meta["ticker"]
    model_name = meta["model_name"]
    metrics = meta["metrics"]
    feature_cols = meta.get("feature_columns", [])

    lines = []
    lines.append(title)
    lines.append("")
    lines.append("Title Page")
    lines.append(title)
    lines.append(settings.author_name)
    lines.append(settings.course_name)
    lines.append(f"Instructor: {settings.instructor_name}")
    lines.append(now)
    lines.append("")
    lines.append("1. Name and Purpose of the Application")
    lines.append(
        f"{settings.app_name} is a Python-based machine learning application that predicts the future closing price of a commodity using historical market data. "
        f"In this implementation, the commodity is {settings.commodity_name} (ticker {ticker}). The app automates the applied machine learning lifecycle: "
        f"data acquisition, preprocessing and feature engineering, supervised model training, evaluation using a chronological holdout test set, and an inference "
        f"workflow that produces a {horizon}-trading-day-ahead forecast."
    )
    lines.append("")
    lines.append("2. Algorithms Used")
    lines.append(
        "This application uses supervised regression models to predict a numeric future price. The primary model in the exemplar run is a Random Forest regressor, "
        "an ensemble of decision trees that can model non-linear relationships and interactions without requiring strong parametric assumptions (Breiman, 2001). "
        "The codebase also supports Support Vector Regression (SVR) and a Multi-Layer Perceptron regressor (a neural network) to reflect alternative model families."
    )
    lines.append("")
    lines.append("3. Dataset Information")
    lines.append("3.1 Dataset source/link")
    lines.append(
        "Historical daily OHLCV price data are downloaded from Yahoo Finance using the yfinance Python library. The exemplar ticker used is GC=F (gold futures)."
    )
    lines.append("3.2 Number of records")
    lines.append(f"The raw downloaded dataset contained {n_records} daily records (after removing rows with missing close prices).")    
    lines.append("3.3 Number of features")
    lines.append(f"The supervised learning table used {n_features} engineered feature columns.")
    lines.append("3.4 Description of features: Feature Name, Description, Data Type")
    lines.append("| Feature | Description | Data Type |")
    lines.append("|---|---|---|")
    lines.append("| ret_lag_1..ret_lag_10 | Lagged log returns | float |")
    lines.append("| ret_mean_5/10/20 | Rolling mean of 1-day log returns | float |")
    lines.append("| ret_std_5/10/20 | Rolling standard deviation of 1-day log returns | float |")
    lines.append("| close_sma_5/10/20 | Rolling simple moving average of close | float |")
    lines.append("| close_min_5/10/20 | Rolling minimum close | float |")
    lines.append("| close_max_5/10/20 | Rolling maximum close | float |")
    lines.append("| hl_range | (high-low)/close volatility proxy | float |")
    lines.append("| rsi_14 | Relative Strength Index (14) | float |")
    lines.append("| ema_12, ema_26 | Exponential moving averages (12, 26) | float |")
    lines.append("| macd, macd_signal_9 | MACD and signal line | float |")
    lines.append("| vol_chg_1 | Log-volume change (1 day) | float |")
    lines.append("| vol_sma_10 | Rolling volume SMA (10) | float |")
    lines.append("| dow, month | Day-of-week and month calendar features | int |")
    lines.append(f"| target_close_fwd | Future close price at t+{horizon} | float |")
    lines.append("")
    lines.append("3.5 Preprocessing steps (e.g., cleaning or normalization)")
    lines.append(
        "Preprocessing consisted of sorting records chronologically, removing rows with missing close prices, computing causal features (lagged returns and rolling-window statistics), "
        "and dropping rows made incomplete by lag/rolling windows and by the forward-shifted target construction. Chronological splitting was used to prevent information leakage from future data into training."
    )
    lines.append("")
    lines.append("4. Libraries, Toolkits, and Frameworks")
    lines.append(
        "- Pandas, NumPy: data cleaning, transformation, and numeric computation\n"
        "- yfinance: programmatic acquisition of historical commodity prices from Yahoo Finance\n"
        "- Scikit-learn: model training, pipelines, and evaluation metrics\n"
        "- Matplotlib, Seaborn: visualization of predictions and residuals\n"
        "- joblib: persistence of trained models"
    )
    lines.append("")
    lines.append("5. Application Design and Implementation")
    lines.append(
        "The app uses a modular pipeline: (1) download or load cached OHLCV data; (2) engineer technical and calendar features; (3) build a supervised dataset by shifting the close price forward by the prediction horizon; "
        "(4) split data chronologically into training and test sets; (5) train a model using a scikit-learn Pipeline; (6) evaluate on the holdout test set; and (7) save artifacts and plots for reproducible reporting and screenshots. "
        "For inference, the app re-computes features on the latest available data row and predicts the future close for the specified horizon."
    )
    lines.append("")
    lines.append("6. Instructions for Running the App")
    lines.append("1) Create and activate a virtual environment.")
    lines.append("2) Install dependencies: pip install -r requirements.txt")
    lines.append(f"3) Train a model: python -m src.train --model random_forest --ticker {settings.ticker} --horizon {settings.horizon_days}")
    lines.append("4) Run the interactive app: python -m src.app")
    lines.append("5) Generate this APA report: python -m src.report_generator")
    lines.append("")
    lines.append("7. Results")
    lines.append(
        f"The model was evaluated on a chronological holdout test set (last {int(settings.test_size * 100)}% of the supervised dataset). "
        f"The supervised dataset size after feature engineering and horizon shifting was {n_supervised} records."
    )
    lines.append("Test-set metrics from the exemplar run:")
    lines.append(f"- MAE: {metrics['MAE']:.4f}")
    lines.append(f"- RMSE: {metrics['RMSE']:.4f}")
    lines.append(f"- MAPE (%): {metrics['MAPE_percent']:.4f}")
    lines.append(f"- Directional Accuracy (%): {metrics['DirectionalAccuracy_percent']:.2f}")
    lines.append("")
    lines.append("Most recent forecast generated by the app:")
    lines.append(f"- Last observed date: {prediction['last_observed_date']} (close={prediction['last_observed_close']:.2f})")
    lines.append(f"- Target date (approx.): {prediction['predicted_target_date']}")
    lines.append(f"- Predicted future close: {prediction['predicted_future_close']:.2f}")
    if prediction.get("prediction_interval_10_90") is not None:
        lo, hi = prediction["prediction_interval_10_90"]
        lines.append(f"- Approx. prediction interval (10-90%): [{lo:.2f}, {hi:.2f}]")
    lines.append("")
    lines.append("Figures generated (include these as screenshots in your submission):")
    lines.append("")
    lines.append("![Predicted vs Actual](outputs/pred_vs_actual.png)")
    lines.append("")
    lines.append("![Residuals Histogram](outputs/residuals_hist.png)")
    lines.append("")
    lines.append("8. Discussion and Insights")
    lines.append(
        "Overall performance depends heavily on market regime changes and the limited information content of price-only features. A key strength of the Random Forest approach is its ability to model non-linear patterns without strict assumptions; however, "
        "commodity prices are influenced by exogenous macroeconomic drivers (rates, inflation expectations, geopolitical risk) that are not captured by OHLCV alone. Improvements include adding macroeconomic covariates, performing walk-forward validation, "
        "tuning hyperparameters with time-series cross-validation, and predicting returns or direction rather than raw prices to reduce non-stationarity."
    )
    lines.append("")
    lines.append("9. References")
    lines.append(
        "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324\n"
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830."
    )
    lines.append("")

    with open(APA_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main(retrain: bool = False):
    # Keep report and screenshots consistent: train only if needed, unless retrain is requested.
    if retrain or not artifacts_exist():
        train(model_name="random_forest", ticker=settings.ticker, horizon=settings.horizon_days)

    meta = load_metadata()
    pred = predict_next(ticker=meta.get("ticker", settings.ticker), horizon=meta.get("horizon_days", settings.horizon_days))
    write_report(meta, pred)
    print(f"Generated report: {APA_REPORT_PATH}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate APA report using existing artifacts.")
    parser.add_argument("--retrain", action="store_true", help="Retrain before generating report.")
    args = parser.parse_args()
    main(retrain=args.retrain)
