from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import database, ml_interface, simulator
from .schemas import EngineState, FaultRequest, HealthResponse, TelemetryPoint, UAV

app = FastAPI(title="Engine Digital Twin API")

# Replace with your actual GitHub Pages origin(s) before deploying.
ALLOWED_ORIGINS = [
    "https://adityadeshmane08.github.io",
    "http://localhost:5173",  # vite dev server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    database.init_db()
    # Resume each UAV's simulation from its last known packet instead of
    # restarting at cycle zero after a redeploy/crash.
    for uav in database.list_uavs():
        last_packet = database.get_last_packet(uav["id"])
        simulator.seed_from_history(uav["id"], last_packet)


def _require_uav(uav_id: str) -> None:
    if not database.uav_exists(uav_id):
        raise HTTPException(status_code=404, detail=f"Unknown UAV id: {uav_id}")


def _build_state(uav_id: str) -> dict:
    """Advance the sim one tick, run the ML model over the recent window,
    persist the resulting packet, and return it."""
    raw = simulator.tick(uav_id)

    window = database.get_history(uav_id, limit=20)
    window.append(raw)  # include current point for the model
    prediction = ml_interface.predict(window)

    packet = {**raw, **prediction, "fault_injected": simulator.fault_active(uav_id)}
    database.insert_telemetry(uav_id, packet)
    return packet


@app.get("/healthz", response_model=HealthResponse)
def healthz():
    return {"status": "ok"}


@app.get("/uavs", response_model=list[UAV])
def get_uavs():
    return database.list_uavs()


@app.get("/uavs/{uav_id}/state", response_model=EngineState)
def get_state(uav_id: str):
    _require_uav(uav_id)
    return _build_state(uav_id)


@app.get("/uavs/{uav_id}/history", response_model=list[TelemetryPoint])
def get_history(uav_id: str, limit: int = 60):
    _require_uav(uav_id)
    limit = min(240, max(10, limit))
    history = database.get_history(uav_id, limit=limit)
    if not history:
        # No packets yet — produce one so the chart isn't empty on first load.
        _build_state(uav_id)
        history = database.get_history(uav_id, limit=limit)
    return history


@app.post("/uavs/{uav_id}/fault", response_model=EngineState)
def post_fault(uav_id: str, body: FaultRequest):
    _require_uav(uav_id)
    simulator.inject_fault(uav_id, body.severity)
    return _build_state(uav_id)


@app.post("/uavs/{uav_id}/reset", response_model=EngineState)
def post_reset(uav_id: str):
    _require_uav(uav_id)
    simulator.reset(uav_id)
    database.clear_history(uav_id)
    return _build_state(uav_id)
