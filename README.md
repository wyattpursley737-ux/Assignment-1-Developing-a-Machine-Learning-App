# CommodityCast (Commodity Price Prediction)

A Python-based machine learning application that predicts the future closing price of a commodity using historical market data.

## Project layout
- `requirements.txt`
- `src/`
  - `config.py`
  - `data_pipeline.py`
  - `features.py`
  - `models.py`
  - `train.py`
  - `predict.py`
  - `app.py`
  - `report_generator.py`
- Outputs created when you run:
  - `data_cache/` (cached CSV download)
  - `artifacts/` (trained model + metadata)
  - `outputs/` (plots + run_log.txt)
  - `report_apa.md` (auto-generated with real counts + real results)

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python -m src.train --model random_forest --ticker GC=F --horizon 5
```

## Predict
```bash
python -m src.predict --ticker GC=F --horizon 5
```

## Interactive app
```bash
python -m src.app
```

## Generate APA report (uses existing artifacts unless --retrain is provided)
```bash
python -m src.report_generator
# or:
python -m src.report_generator --retrain
```

## Screenshots to include in your submission
- Console output from training (`python -m src.train ...`)
- `outputs/pred_vs_actual.png`
- `outputs/residuals_hist.png`
- `outputs/run_log.txt`
- Optional: console output from `python -m src.predict ...`
