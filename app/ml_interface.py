"""
Loads the trained status classifier + RUL regressor (trained on
engine_dataset.xlsx via training/train_v2.py) and implements the same
predict(window) contract the placeholder used, so main.py and the
frontend need zero changes.

Falls back to a simple heuristic if the model files are missing (e.g.
first run before training), so the API never hard-crashes.
"""

from pathlib import Path

import joblib
import numpy as np

MODELS_DIR = Path(__file__).parent / "models"
TRAIN_MAX_CYCLE = 299  # clip inputs to the domain the model was trained on

RAW_FEATURES = ["cylinder_head_temp", "oil_pressure", "vibration", "rpm", "fuel_flow", "exhaust_gas_temp"]
ROLL_STD_FEATURES = ["cylinder_head_temp", "oil_pressure", "vibration"]

_clf = None
_reg = None
_feature_cols = None
_load_error = None

try:
    _clf = joblib.load(MODELS_DIR / "status_classifier.joblib")
    _reg = joblib.load(MODELS_DIR / "rul_regressor.joblib")
    _feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
except Exception as exc:  # noqa: BLE001 - any load failure should fall back, not crash
    _load_error = str(exc)


def _build_features(window: list[dict]) -> dict:
    recent = window[-21:]  # matches the WINDOW=21 used at training time
    current = recent[-1]

    feats = {"cycle": min(current["cycle"], TRAIN_MAX_CYCLE)}
    for col in RAW_FEATURES:
        feats[col] = current[col]

    for col in RAW_FEATURES:
        values = [p[col] for p in recent]
        feats[f"roll_mean_{col}"] = float(np.mean(values))

    for col in ROLL_STD_FEATURES:
        values = [p[col] for p in recent]
        feats[f"roll_std_{col}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    return feats


def _heuristic_fallback(window: list[dict]) -> dict:
    current = window[-1]
    cht_over = max(0.0, current["cylinder_head_temp"] - 200) / 40
    oil_under = max(0.0, 40 - current["oil_pressure"]) / 30
    vib_over = max(0.0, current["vibration"] - 0.4) / 1.2
    anomaly_score = round(min(0.99, (cht_over + oil_under + vib_over) / 3), 2)
    status = "critical" if anomaly_score >= 0.72 else "watch" if anomaly_score >= 0.42 else "nominal"
    return {
        "status": status,
        "anomaly_score": anomaly_score,
        "rul_hours": round(max(0.0, (1 - anomaly_score) * 34), 1),
        "confidence": round(max(0.71, 0.96 - anomaly_score * 0.18), 2),
    }


def predict(window: list[dict]) -> dict:
    """
    window: recent raw telemetry points for one UAV, oldest first, each a
            dict with timestamp, cycle, cylinder_head_temp, oil_pressure,
            vibration, rpm, fuel_flow, exhaust_gas_temp.

    Returns: {"status": "nominal"|"watch"|"critical", "anomaly_score": float 0-1,
              "rul_hours": float, "confidence": float 0-1}
    """
    if _clf is None or _reg is None:
        return _heuristic_fallback(window)

    feats = _build_features(window)
    X = np.array([[feats[c] for c in _feature_cols]])

    status = str(_clf.predict(X)[0])
    probs = dict(zip(_clf.classes_, _clf.predict_proba(X)[0]))
    anomaly_score = round(float(1.0 - probs.get("nominal", 0.0)), 2)
    confidence = round(float(max(probs.values())), 2)

    rul_hours = round(max(0.0, float(_reg.predict(X)[0])), 1)

    return {
        "status": status,
        "anomaly_score": anomaly_score,
        "rul_hours": rul_hours,
        "confidence": confidence,
    }
