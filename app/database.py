"""
SQLite persistence for the digital twin backend.

Two tables:
- uavs: the list of drones/UAVs the dashboard can select between
- telemetry: every packet ever produced, tagged with uav_id, so history
  survives a backend restart (not just page reloads).
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "digital_twin.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS uavs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    model TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uav_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    cylinder_head_temp REAL NOT NULL,
    oil_pressure REAL NOT NULL,
    vibration REAL NOT NULL,
    rpm REAL NOT NULL,
    fuel_flow REAL NOT NULL,
    exhaust_gas_temp REAL NOT NULL,
    status TEXT NOT NULL,
    anomaly_score REAL NOT NULL,
    rul_hours REAL NOT NULL,
    fault_injected INTEGER NOT NULL,
    confidence REAL NOT NULL,
    FOREIGN KEY (uav_id) REFERENCES uavs (id)
);

CREATE INDEX IF NOT EXISTS idx_telemetry_uav_cycle
    ON telemetry (uav_id, cycle);
"""

DEFAULT_UAVS = [
    ("uav-01", "Falcon-9E", "MALE Class-II"),
    ("uav-02", "Sentinel-X", "MALE Class-III"),
]


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        existing = conn.execute("SELECT COUNT(*) AS c FROM uavs").fetchone()["c"]
        if existing == 0:
            conn.executemany(
                "INSERT INTO uavs (id, name, model) VALUES (?, ?, ?)",
                DEFAULT_UAVS,
            )


def list_uavs() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT id, name, model FROM uavs ORDER BY id").fetchall()
        return [dict(row) for row in rows]


def uav_exists(uav_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT 1 FROM uavs WHERE id = ?", (uav_id,)).fetchone()
        return row is not None


def insert_telemetry(uav_id: str, packet: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO telemetry (
                uav_id, timestamp, cycle, cylinder_head_temp, oil_pressure,
                vibration, rpm, fuel_flow, exhaust_gas_temp, status,
                anomaly_score, rul_hours, fault_injected, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uav_id,
                packet["timestamp"],
                packet["cycle"],
                packet["cylinder_head_temp"],
                packet["oil_pressure"],
                packet["vibration"],
                packet["rpm"],
                packet["fuel_flow"],
                packet["exhaust_gas_temp"],
                packet["status"],
                packet["anomaly_score"],
                packet["rul_hours"],
                int(packet["fault_injected"]),
                packet["confidence"],
            ),
        )


def get_last_packet(uav_id: str) -> dict | None:
    """Most recent packet for a UAV — used on startup to resume instead of
    restarting the simulation from cycle zero after a backend restart."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM telemetry WHERE uav_id = ? ORDER BY cycle DESC LIMIT 1",
            (uav_id,),
        ).fetchone()
        return _row_to_packet(row) if row else None


def get_history(uav_id: str, limit: int = 60) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM telemetry WHERE uav_id = ? ORDER BY cycle DESC LIMIT ?",
            (uav_id, limit),
        ).fetchall()
        return [_row_to_packet(row) for row in reversed(rows)]


def clear_history(uav_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM telemetry WHERE uav_id = ?", (uav_id,))


def _row_to_packet(row: sqlite3.Row) -> dict:
    packet = dict(row)
    packet["fault_injected"] = bool(packet["fault_injected"])
    return packet
