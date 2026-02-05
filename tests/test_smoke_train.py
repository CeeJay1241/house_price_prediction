import os
import json
import joblib
import pandas as pd
import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
from src.cli import train_from_path

def test_ames_smoke_train(tmp_path):
    # Use a small sample for speed
    data_path = Path("data/raw/train.csv")
    if not data_path.exists():
        import pytest
        pytest.skip("Ames train.csv not found")
    out_path = tmp_path / "smoke_pipeline.pkl"
    meta = train_from_path(str(data_path), str(out_path), sample_n=200, model_type="rf", do_search=False)
    assert out_path.exists(), "Pipeline not saved"
    # Check summary JSON
    summary_path = Path("configs/model_summary.json")
    assert summary_path.exists(), "model_summary.json not written"
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    assert "cv" in summary and "holdout" in summary
    assert "model_path" in summary
    # Check top features (expanded names)
    assert "top_features" in summary
    assert isinstance(summary["top_features"], list)
    assert len(summary["top_features"]) > 0
    # Check model loads
    pipe = joblib.load(out_path)
    assert hasattr(pipe, "predict")
    # Predict on a few rows
    df = pd.read_csv(data_path).sample(5, random_state=42)
    X = df.drop(columns=[c for c in ("Id","SalePrice") if c in df.columns])
    preds = pipe.predict(X)
    assert len(preds) == 5
