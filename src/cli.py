"""CLI: train models and save artifacts.

Usage (examples):
  python -m src.cli train --data data/raw/train.csv --out models/best_pipeline.pkl
  python -m src.cli train --use-california --out models/cal_pipeline.pkl

The module exposes `train_from_df` so tests can call training programmatically.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from src.data import load_california
from src.features import df_to_Xy, get_numeric_features, build_preprocessor


ROOT = Path(__file__).resolve().parents[1]


def _infer_target(df: pd.DataFrame) -> str:
    # Ames: SalePrice; California loader uses MedHouseVal
    if "SalePrice" in df.columns:
        return "SalePrice"
    if "MedHouseVal" in df.columns:
        return "MedHouseVal"
    # fallback: assume last numeric column is target
    for c in reversed(df.columns.tolist()):
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("Could not infer target column")


def train_from_df(df: pd.DataFrame, out_path: str | Path, model_type: str = "lgbm", random_state: int = 42, do_search: bool = False) -> dict:
    """Train a pipeline from a DataFrame and persist artifacts.

    Returns a summary dict (metrics omitted for speed in unit tests).
    """
    import sklearn
    from sklearn.ensemble import RandomForestRegressor

    try:
        from lightgbm import LGBMRegressor
    except Exception:
        LGBMRegressor = None

    df = df.copy()
    target = _infer_target(df)
    if target != "MedHouseVal":
        # keep original target name (models expect arbitrary y)
        y = df.pop(target)
        X = df
    else:
        X, y = df_to_Xy(df)

    # drop id-like columns
    for cid in ["Id", "id", "PID"]:
        if cid in X.columns:
            X = X.drop(columns=[cid])

    # compute numeric features from feature matrix only (exclude target) to avoid column-mismatch
    num_feats = get_numeric_features(X)
    pre = build_preprocessor(numeric_features=num_feats)

    # choose model
    if model_type == "lgbm" and LGBMRegressor is not None:
        model = LGBMRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("pre", pre), ("model", model)])

    if do_search:
        param_dist = {"model__n_estimators": [100, 200, 400], "model__max_depth": [6, 12, None]}
        rs = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=6, cv=3, scoring="neg_mean_squared_error", random_state=random_state, n_jobs=-1)
        rs.fit(X, y)
        pipe = rs.best_estimator_
        best_params = getattr(rs, "best_params_", {})
    else:
        pipe.fit(X, y)
        best_params = {}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)

    # record minimal experiment metadata
    meta = {
        "model_path": str(out_path),
        "model_type": model_type,
        "best_params": best_params,
        "n_features_in_": int(getattr(pipe.named_steps["pre"], "transform", lambda x: x)(X).shape[1]) if hasattr(pipe.named_steps["pre"], "transform") else None,
    }
    (ROOT / "configs").mkdir(exist_ok=True)
    with open(ROOT / "configs" / "experiment.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    try:
        import yaml

        with open(ROOT / "configs" / "experiment.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f)
    except Exception:
        # YAML optional — not a blocker
        pass

    return meta


def train_from_path(data_path: str, out: str, use_california: bool = False, sample_n: Optional[int] = None, **kwargs) -> dict:
    if use_california:
        df = load_california(sample_n=sample_n)
    else:
        df = pd.read_csv(data_path)
        if sample_n:
            df = df.sample(sample_n, random_state=kwargs.get("random_state", 42))
    return train_from_df(df, out, **kwargs)


def _cli_train(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="src.cli")
    sub = p.add_subparsers(dest="cmd")
    t = sub.add_parser("train", help="train a model")
    t.add_argument("--data", default="data/raw/train.csv", help="CSV file or folder (default: data/raw/train.csv)")
    t.add_argument("--out", default="models/best_pipeline.pkl", help="where to save pipeline")
    t.add_argument("--use-california", action="store_true", help="use built-in California dataset instead of CSV")
    t.add_argument("--model", choices=["lgbm", "rf"], default="lgbm")
    t.add_argument("--sample-n", type=int, default=None)
    t.add_argument("--no-search", dest="do_search", action="store_false")
    t.set_defaults(do_search=True)
    t.add_argument("--seed", type=int, default=42)

    args = p.parse_args(argv)
    if args.cmd == "train":
        meta = train_from_path(args.data, args.out, use_california=args.use_california, sample_n=args.sample_n, model_type=args.model, random_state=args.seed, do_search=args.do_search)
        print("Wrote model ->", meta["model_path"])
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli_train())
