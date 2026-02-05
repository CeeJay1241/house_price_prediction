"""Small CLI for loading a serialized model and predicting a single row (CSV/JSON)."""
from __future__ import annotations
import argparse
import pandas as pd
import joblib
from pathlib import Path


def predict_from_csv(model_path: str, row_csv: str) -> pd.DataFrame:
    model = joblib.load(model_path)
    df = pd.read_csv(row_csv)
    preds = model.predict(df)
    out = df.copy()
    out["prediction"] = preds
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", help="path to model .pkl")
    p.add_argument("--row-csv", help="single-row CSV to predict")
    args = p.parse_args()
    if not args.row_csv:
        raise SystemExit("Provide --row-csv with a single-row CSV")
    out = predict_from_csv(args.model, args.row_csv)
    print(out.to_dict(orient="records"))
