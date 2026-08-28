"""
Trains the status classifier + RUL regressor from engine_dataset.xlsx.

Usage:
    pip install -r ../requirements.txt pandas openpyxl
    python train_model.py /path/to/engine_dataset.xlsx

Only uses the 6 sensor fields the frontend dashboard already displays
(cylinder_head_temp, oil_pressure, vibration, rpm, fuel_flow,
exhaust_gas_temp) plus elapsed engine hours (cycle) and rolling
statistics over a window of recent readings -- matching exactly what
app/ml_interface.py's predict(window) receives in production. Re-run
this whenever you get a bigger/better dataset; it overwrites the
.joblib files in ../app/models/.
"""

import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

RENAME = {"cht": "cylinder_head_temp", "egt": "exhaust_gas_temp", "engine_hours": "cycle"}
RAW_FEATURES = ["cylinder_head_temp", "oil_pressure", "vibration", "rpm", "fuel_flow", "exhaust_gas_temp"]
ROLL_STD_FEATURES = ["cylinder_head_temp", "oil_pressure", "vibration"]
WINDOW = 21  # matches app/main.py: get_history(limit=20) + current point
MODELS_DIR = Path(__file__).parent.parent / "app" / "models"


def status_from_health(h: float) -> str:
    if h >= 97:
        return "nominal"
    if h >= 90:
        return "watch"
    return "critical"


def main(dataset_path: str) -> None:
    t0 = time.time()
    print(f"Loading {dataset_path} ...")
    df = pd.read_excel(dataset_path)
    df = df.rename(columns=RENAME)
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    print(f"Loaded in {time.time()-t0:.1f}s, shape={df.shape}")

    df["status"] = df["health_score"].apply(status_from_health)
    print(df["status"].value_counts())

    g = df.groupby("engine_id", sort=False)
    for col in RAW_FEATURES:
        df[f"roll_mean_{col}"] = g[col].transform(lambda s: s.rolling(WINDOW, min_periods=1).mean())
    for col in ROLL_STD_FEATURES:
        df[f"roll_std_{col}"] = g[col].transform(lambda s: s.rolling(WINDOW, min_periods=1).std().fillna(0))

    feature_cols = ["cycle"] + RAW_FEATURES + [f"roll_mean_{c}" for c in RAW_FEATURES] + [f"roll_std_{c}" for c in ROLL_STD_FEATURES]
    print(f"{len(feature_cols)} features:", feature_cols)

    X = df[feature_cols].values
    y_status = df["status"].values
    y_rul = df["rul_hours"].values
    groups = df["engine_id"].values

    # Split by engine_id, not by row, so test engines are never seen in training.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y_status, groups))
    X_train, X_test = X[train_idx], X[test_idx]

    print("\nTraining status classifier...")
    clf = RandomForestClassifier(n_estimators=80, max_depth=10, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_train, y_status[train_idx])
    print(classification_report(y_status[test_idx], clf.predict(X_test)))

    print("Training RUL regressor...")
    reg = RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=5, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_rul[train_idx])
    pred_rul = reg.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_rul[test_idx], pred_rul):.2f} hours")
    print(f"R2:  {r2_score(y_rul[test_idx], pred_rul):.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "status_classifier.joblib", compress=3)
    joblib.dump(reg, MODELS_DIR / "rul_regressor.joblib", compress=3)
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")
    print(f"\nSaved models to {MODELS_DIR}  (total time {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_model.py /path/to/engine_dataset.xlsx")
        sys.exit(1)
    main(sys.argv[1])
