"""
Generates raw sensor readings per UAV. This is a placeholder physics
model (ported from the frontend's old in-browser demo generator) — swap
the body of `_raw_telemetry` for a real physics-based engine model
whenever that's ready. It only produces RAW sensor fields; status /
anomaly_score / rul_hours / confidence come from ml_interface.predict,
not from here.
"""

import math
import time
from datetime import datetime, timezone

TICK_SECONDS = 0.9  # matches the frontend's polling cadence
TBO_HOURS = 299  # rated Time-Between-Overhaul the ML model was trained against


class _UavSimState:
    def __init__(self, start_cycle: int = 0):
        self.cycle = start_cycle
        self.fault: dict | None = None  # {"severity": "soft"|"hard", "start_cycle": int}
        self.last_advanced_at = time.time()

    def resume_from(self, last_packet: dict) -> None:
        """Resume from the last known packet instead of restarting at cycle 0
        after a backend restart."""
        self.cycle = last_packet["cycle"]
        self.fault = {"severity": "hard", "start_cycle": self.cycle} if last_packet["fault_injected"] else None


_states: dict[str, _UavSimState] = {}


def _state_for(uav_id: str) -> _UavSimState:
    if uav_id not in _states:
        _states[uav_id] = _UavSimState()
    return _states[uav_id]


def seed_from_history(uav_id: str, last_packet: dict | None) -> None:
    if last_packet is not None:
        _state_for(uav_id).resume_from(last_packet)


def inject_fault(uav_id: str, severity: str) -> None:
    state = _state_for(uav_id)
    state.fault = {"severity": severity, "start_cycle": state.cycle}


def reset(uav_id: str) -> None:
    _states[uav_id] = _UavSimState()


    # Baseline (nominal-condition mean) and full-fault target, calibrated
    # from engine_dataset.xlsx so simulated telemetry lands in the same
    # value ranges the ML model was trained on. Real engine life is ~89%
    # nominal in the training data, so there is intentionally no gradual
    # "background wear" here -- degradation only appears once a fault is
    # injected, same as the source dataset.


def _raw_telemetry(uav_id: str, cycle: int) -> dict:
    state = _state_for(uav_id)
    fault_progress = 0.0
    severity_mult = 1.0
    if state.fault:
        fault_progress = min(max((cycle - state.fault["start_cycle"]) / 24, 0), 1)
        severity_mult = 1.6 if state.fault["severity"] == "hard" else 1.0
    d = fault_progress * severity_mult
    osc = math.sin(cycle / 7)

    cht = 159.6 + d * 39.6 + osc * 3.0
    oil = 4.31 - d * 2.13 + math.cos(cycle / 9) * 0.08
    vib = 0.13 + d * 0.47 + abs(osc) * 0.02
    rpm = 4843 - d * 150 + math.sin(cycle / 11) * 60
    fuel = 14.4 + d * 1.5 + math.cos(cycle / 13) * 0.3
    egt = 659 + d * 61 + osc * 8

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cycle": cycle,
        "cylinder_head_temp": round(min(max(cht, 140), 260), 1),
        "oil_pressure": round(min(max(oil, 0), 6), 2),
        "vibration": round(min(max(vib, 0), 4), 2),
        "rpm": round(min(max(rpm, 3800), 5600), 1),
        "fuel_flow": round(min(max(fuel, 10), 19), 2),
        "exhaust_gas_temp": round(min(max(egt, 550), 850), 1),
    }


def tick(uav_id: str) -> dict:
    """Advance this UAV by one cycle (if enough wall-clock time passed since
    the last tick) and return the raw telemetry for the current cycle."""
    state = _state_for(uav_id)
    now = time.time()
    if now - state.last_advanced_at >= TICK_SECONDS:
        state.cycle += 1
        state.last_advanced_at = now
        if state.cycle > TBO_HOURS:
            # Scheduled overhaul: engine returns to service at zero hours.
            state.cycle = 0
            state.fault = None
    return _raw_telemetry(uav_id, state.cycle)


def fault_active(uav_id: str) -> bool:
    return _state_for(uav_id).fault is not None
