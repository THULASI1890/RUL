from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["cycle","voltage","current","temperature","capacity","resistance","charge_time","discharge_time"]
TARGET = "rul"


def validate_dataset(df: pd.DataFrame, features: Sequence[str] = FEATURES) -> None:
    required = list(features) + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {