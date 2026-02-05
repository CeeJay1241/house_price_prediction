import joblib
import pandas as pd
from pathlib import Path

from src.cli import train_from_df
from src.data import load_california


def test_train_and_predict_tmp(tmp_path: Path):
    df = load_california(sample_n=200)
    out = tmp_path / "tmp_model.pkl"
    meta = train_from_df(df, out, model_type="rf", do_search=False)
    assert out.exists()
    pipe = joblib.load(out)
    # predict on a small batch
    X = df.drop(columns=["MedHouseVal"]) if "MedHouseVal" in df.columns else df.iloc[:, :-1]
    preds = pipe.predict(X.head(3))
    assert len(preds) == 3
    assert all([float(x) == x for x in preds])
