"""
Pydantic models. Field names match src/App.tsx's EngineState and
TelemetryPoint types exactly — do not rename these without updating
the frontend types too.
"""

from typing import Literal

from pydantic import BaseModel

Status = Literal["nominal", "watch", "critical"]
Severity = Literal["soft", "hard"]


class UAV(BaseModel):
    id: str
    name: str
    model: str


class UAVCreate(BaseModel):
    name: str
    model: str


class TelemetryPoint(BaseModel):
    timestamp: str
    cycle: int
    cylinder_head_temp: float
    oil_pressure: float
    vibration: float
    rpm: float
    fuel_flow: float
    exhaust_gas_temp: float


class EngineState(TelemetryPoint):
    status: Status
    anomaly_score: float
    rul_hours: float
    fault_injected: bool
    confidence: float


class FaultRequest(BaseModel):
    severity: Severity


class HealthResponse(BaseModel):
    status: str
