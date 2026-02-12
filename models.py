from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

@dataclass
class ModelSpec:
    name: str
    model: Any
    needs_scaling: bool

def get_model_spec(model_name: str, random_state: int = 42) -> ModelSpec:
    name = model_name.lower().strip()

    if name in {"rf", "random_forest", "randomforest"}:
        model = RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        return ModelSpec(name="random_forest", model=model, needs_scaling=False)

    if name in {"svr", "svm"}:
        model = SVR(kernel="rbf", C=50.0, gamma="scale", epsilon=0.1)
        return ModelSpec(name="svr_rbf", model=model, needs_scaling=True)

    if name in {"mlp", "neural_net", "nn"}:
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=2000,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.15,
        )
        return ModelSpec(name="mlp_regressor", model=model, needs_scaling=True)

    raise ValueError("Unknown model_name. Use: random_forest | svr | mlp")

def build_pipeline(spec: ModelSpec) -> Pipeline:
    if spec.needs_scaling:
        return Pipeline([("scaler", StandardScaler()), ("model", spec.model)])
    return Pipeline([("model", spec.model)])

def rf_tree_interval(rf_model, X_row: np.ndarray, lower_q: float = 0.1, upper_q: float = 0.9) -> Tuple[float, float]:
    """
    Approximate prediction interval for RandomForestRegressor by using per-tree predictions.
    Expects rf_model to be the fitted RandomForestRegressor (not a Pipeline).
    """
    preds = np.array([est.predict(X_row.reshape(1, -1))[0] for est in rf_model.estimators_], dtype=float)
    lo = float(np.quantile(preds, lower_q))
    hi = float(np.quantile(preds, upper_q))
    return lo, hi
