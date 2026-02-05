"""Feature engineering and preprocessing pipelines (scikit-learn compatible).

Provides a numeric pipeline and a convenience function to get X, y.
"""
from __future__ import annotations
from typing import Iterable
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def get_numeric_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "MedHouseVal" and pd.api.types.is_numeric_dtype(df[c])]


def build_preprocessor(numeric_features: Iterable[str], categorical_features: Iterable[str] | None = None) -> ColumnTransformer:
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="NA")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    transformers = [("num", num_pipe, list(numeric_features))]
    if categorical_features:
        transformers.append(("cat", cat_pipe, list(categorical_features)))
    return ColumnTransformer(transformers, remainder="drop")


def df_to_Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    y = df.pop("MedHouseVal")
    X = df
    return X, y


def get_expanded_feature_names(preprocessor, input_features: list[str]) -> list[str]:
    """Return expanded feature names from a fitted ColumnTransformer (numeric + OHE cat)."""
    # sklearn >=1.0: get_feature_names_out is available
    try:
        return preprocessor.get_feature_names_out(input_features).tolist()
    except Exception:
        # fallback: manually build names
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num':
                names.extend(cols)
            elif name == 'cat':
                # OHE: get feature names from encoder
                ohe = trans.named_steps.get('ohe') if hasattr(trans, 'named_steps') else None
                if ohe is not None and hasattr(ohe, 'get_feature_names_out'):
                    names.extend(ohe.get_feature_names_out(cols))
                else:
                    # fallback: just use col names
                    names.extend(cols)
        return [str(n) for n in names]
