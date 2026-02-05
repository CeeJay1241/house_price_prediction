"""FastAPI inference app for the trained pipeline.

Run locally:
  uvicorn src.api:app --reload --port 8000

Endpoints:
  GET /health
  POST /predict  (accepts single record or list of records)
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "best_pipeline.pkl"

app = FastAPI(title="HousePricePredictor", version="0.1.0")


class PredictRequest(BaseModel):
    data: list[dict[str, Any]]


_model = None
_model_path = None


def load_model(path: str | Path | None = None):
    global _model, _model_path
    p = Path(path or os.environ.get("MODEL_PATH") or DEFAULT_MODEL)
    if not p.exists():
        raise FileNotFoundError(f"model not found: {p}")
    _model = joblib.load(p)
    _model_path = str(p)
    return _model


@app.on_event("startup")
def _startup():
    try:
        load_model(None)
    except FileNotFoundError:
        # app can still start for demo purposes
        _model = None


@app.get("/health")
def health():
    return {"status": "ok", "model_path": _model_path}


@app.post("/predict")
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded; train and save to models/best_pipeline.pkl")
    df = req.data
    import pandas as pd

    X = pd.DataFrame(df)
    preds = _model.predict(X)
    return {"predictions": preds.tolist(), "n": len(preds)}
