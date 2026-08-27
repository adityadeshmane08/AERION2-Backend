from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import database, ml_interface, simulator
from .schemas import EngineState, FaultRequest, HealthResponse, TelemetryPoint, UAV

app = FastAPI(title="Engine Digital Twin API")

ALLOWED_ORIGINS = [
    "https://aerion-2.vercel.app",
    "https://adityadeshmane08.github.io",
    "http://localhost:5173",
    "http://localhost:4173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# STARTUP
# ---------------------------------------------------------

@app.on_event("startup")
def on_startup() -> None:
    database.init_db()

    # Resume each UAV's simulation from its last known packet
    # instead of restarting at cycle zero after a redeploy/crash.
    for uav in database.list_uavs():
        last_packet = database.get_last_packet(uav["id"])
        simulator.seed_from_history(uav["id"], last_packet)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def _require_uav(uav_id: str) -> None:
    """Check whether the requested UAV exists."""
    if not database.uav_exists(uav_id):
        raise HTTPException(
            status_code=404,
            detail=f"Unknown UAV id: {uav_id}",
        )


def _build_state(uav_id: str) -> dict:
    """
    Advance the simulator by one tick, run the ML model,
    save telemetry, and return the resulting state.
    """

    # Generate current telemetry
    raw = simulator.tick(uav_id)

    # Get recent history for ML prediction
    window = database.get_history(uav_id, limit=20)

    # Include current telemetry point
    window.append(raw)

    # Run ML prediction
    prediction = ml_interface.predict(window)

    # Combine telemetry + ML prediction
    packet = {
        **raw,
        **prediction,
        "fault_injected": simulator.fault_active(uav_id),
    }

    # Save telemetry
    database.insert_telemetry(uav_id, packet)

    return packet


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/healthz", response_model=HealthResponse)
def healthz():
    return {"status": "ok"}


# ---------------------------------------------------------
# UAV LIST
# ---------------------------------------------------------

@app.get("/uavs", response_model=list[UAV])
def get_uavs():
    return database.list_uavs()


# ---------------------------------------------------------
# CURRENT UAV STATE
# ---------------------------------------------------------

@app.get("/uavs/{uav_id}/state", response_model=EngineState)
def get_state(uav_id: str):
    _require_uav(uav_id)

    return _build_state(uav_id)


# ---------------------------------------------------------
# UAV TELEMETRY HISTORY
# ---------------------------------------------------------

@app.get(
    "/uavs/{uav_id}/history",
    response_model=list[TelemetryPoint],
)
def get_history(
    uav_id: str,
    limit: int = 60,
):
    _require_uav(uav_id)

    # Keep limit between 10 and 240
    limit = min(240, max(10, limit))

    history = database.get_history(
        uav_id,
        limit=limit,
    )

    # Generate first packet if history is empty
    if not history:
        _build_state(uav_id)

        history = database.get_history(
            uav_id,
            limit=limit,
        )

    return history


# ---------------------------------------------------------
# INJECT FAULT
# ---------------------------------------------------------

@app.post(
    "/uavs/{uav_id}/fault",
    response_model=EngineState,
)
def post_fault(
    uav_id: str,
    body: FaultRequest,
):
    _require_uav(uav_id)

    simulator.inject_fault(
        uav_id,
        body.severity,
    )

    return _build_state(uav_id)


# ---------------------------------------------------------
# RESET UAV
# ---------------------------------------------------------

@app.post(
    "/uavs/{uav_id}/reset",
    response_model=EngineState,
)
def post_reset(uav_id: str):
    _require_uav(uav_id)

    simulator.reset(uav_id)

    database.clear_history(uav_id)

    return _build_state(uav_id)
