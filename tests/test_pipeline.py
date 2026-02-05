import numpy as np
import pandas as pd

from src.data import load_california
from src.features import build_preprocessor, df_to_Xy, get_numeric_features


def test_preprocessor_preserves_rows_and_no_nans():
    df = load_california(sample_n=50)
    X, y = df_to_Xy(df)
    num_feats = get_numeric_features(df)
    pre = build_preprocessor(numeric_features=num_feats)
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == X.shape[0]
    # transformed output should not contain NaNs (imputer present)
    assert not np.isnan(Xt).any()
