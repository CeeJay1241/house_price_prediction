"""Model training and evaluation utilities (quick, testable)."""
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def get_default_models() -> Dict[str, Any]:
    return {
        "ridge": Ridge(random_state=42),
        "rf": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }


def evaluate_models(pipe, X, y, models: dict | None = None, cv: int = 5) -> dict:
    """Train and return cross-validated RMSE for each model. """
    models = models or get_default_models()
    results = {}
    for name, model in models.items():
        estimator = Pipeline([("pre", pipe), ("model", model)])
        # use neg_mean_squared_error and convert to RMSE
        scores = cross_val_score(estimator, X, y, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1)
        rmse = float(np.sqrt(-scores).mean())
        results[name] = {"rmse": rmse, "fold_scores": (-scores).tolist()}
    return results
