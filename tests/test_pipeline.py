import pandas as pd
import pytest
from src.pipeline import FEATURES, validate_dataset


def test_feature_schema_is_stable():
    assert FEATURES == ["cycle","voltage","current","temperature","capacity","resistance","charge_time","discharge_time"]


def test_validate_dataset_accepts_valid_data():
    df = pd.DataFrame({c: [1.0, 2.0] for c in FEATURES + ["rul"]})
    validate_dataset(df)


def test_validate_dataset_rejects_missing_column():
    df = pd.DataFrame({c: [1.0] for c in FEATURES})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(df)
