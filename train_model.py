import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

FEATURES = [
    "cycle", "voltage", "current", "temperature", "capacity",
    "resistance", "charge_time", "discharge_time", "soh"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/battery_data.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    required = FEATURES + ["rul", "reusable"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURES]
    y_rul = df["rul"]
    y_reusable = df["reusable"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_rul, test_size=0.2, random_state=42
    )
    rul = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2
    )
    rul.fit(X_train, y_train)

    Xc_train, _, yc_train, _ = train_test_split(
        X, y_reusable, test_size=0.2, random_state=42, stratify=y_reusable
    )
    reusable = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        class_weight="balanced",
    )
    reusable.fit(Xc_train, yc_train)

    joblib.dump(
        {"model": rul, "features": FEATURES, "target": "rul", "model_type": "regressor"},
        "rul_model.pkl",
        compress=3,
    )
    joblib.dump(
        {
            "model": reusable,
            "features": FEATURES,
            "target": "reusable",
            "model_type": "classifier",
            "classes": reusable.classes_.tolist(),
        },
        "reusable_model.pkl",
        compress=3,
    )

    print("Models created successfully.")


if __name__ == "__main__":
    main()
