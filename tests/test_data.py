from src.data import load_california


def test_load_california_basic():
    df = load_california(sample_n=100)
    assert df.shape[0] > 0
    assert "MedHouseVal" in df.columns
    # numeric features present
    assert "MedInc" in df.columns
