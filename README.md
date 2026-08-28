# Engine Digital Twin — FastAPI backend

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Visit http://localhost:8000/docs for interactive API docs.

## What's here

- `app/database.py` — SQLite persistence. Two tables: `uavs` (drone
  list) and `telemetry` (every packet, tagged by `uav_id`). On startup,
  each UAV's simulation resumes from its last saved packet instead of
  restarting at cycle zero after a redeploy/crash.
- `app/simulator.py` — generates raw sensor readings each tick,
  calibrated against `engine_dataset.xlsx`'s real nominal/critical value
  ranges (e.g. oil pressure ~4.3, RPM ~4843 — not arbitrary units).
  Engines run a 0–299hr cycle matching the dataset's TBO (Time Between
  Overhaul), then wrap to a fresh overhaul. No background wear —
  degradation only appears once a fault is injected, matching the real
  dataset (~89% of rows are nominal).
- `app/ml_interface.py` — **trained models, not a placeholder anymore.**
  Loads `app/models/*.joblib` (a RandomForest status classifier +
  RandomForest RUL regressor, trained on `engine_dataset.xlsx`) and
  implements `predict(window)`. Falls back to a simple heuristic if the
  model files are missing, so the API never hard-crashes.
- `app/models/` — the trained model artifacts (`status_classifier.joblib`,
  `rul_regressor.joblib`, `feature_cols.joblib`). Committed to the repo
  so Render can just load them — no training step needed at deploy time.
- `training/train_model.py` — retrain from a (bigger/better) dataset:
  `python train_model.py /path/to/engine_dataset.xlsx`. Overwrites the
  files in `app/models/`. Held-out evaluation (by engine_id, so test
  engines are never seen in training): status classifier ~99% accuracy,
  RUL regressor R²≈1.0 with MAE well under an hour on the provided
  dataset (RUL tracks accumulated engine hours against the 299hr TBO,
  adjusted down by a few hours when sensors show real degradation).
- `app/schemas.py` — Pydantic models. Field names match the frontend's
  `EngineState` / `TelemetryPoint` types in `src/App.tsx` exactly.
- `app/main.py` — routes:
  - `GET /healthz`
  - `GET /uavs`
  - `GET /uavs/{uav_id}/state`
  - `GET /uavs/{uav_id}/history?limit=60`
  - `POST /uavs/{uav_id}/fault` `{"severity": "soft"|"hard"}`
  - `POST /uavs/{uav_id}/reset`

## Before deploying

Edit `ALLOWED_ORIGINS` in `app/main.py` to match your actual GitHub
Pages URL.

## Deploying on Render

- Root Directory: `backend`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

## Adding a UAV

Insert a row into the `uavs` table (or add to `DEFAULT_UAVS` in
`database.py` before the first run) — no code changes needed elsewhere.

## Retraining with your teammate's improvements

If your teammate wants to try a different model (XGBoost, a neural net,
different features, etc.), they only need to touch `training/train_model.py`
and `app/ml_interface.py`'s `predict()` — the function signature
(`window: list[dict]` in, `{status, anomaly_score, rul_hours, confidence}`
dict out) is the contract; everything else in the backend stays the same.
