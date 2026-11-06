"""SQLite storage for the harness: blocked prompts, task history, daily spend and the
forecaster's key-value state. One connection from Hermes' `plugin_db()`; every access
goes through one lock because the gateway calls hooks from several threads.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Optional

try:
    from ..contracts import Estimate, PastTask, TaskFeatures, from_dict, to_dict
except ImportError:  # top-level import in tests
    from contracts import Estimate, PastTask, TaskFeatures, from_dict, to_dict

PENDING_TTL = 30 * 60  # a quote not answered in 30 minutes is dropped
HISTORY_LIMIT = 2000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pending (
    session_id TEXT PRIMARY KEY, message TEXT NOT NULL, features TEXT NOT NULL,
    estimate TEXT NOT NULL, blocked_at REAL NOT NULL, turn_id TEXT
);
CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, finished_at REAL, row TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS daily (day TEXT PRIMARY KEY, usd REAL NOT NULL);
"""


class Pending:
    def __init__(self, session_id: str, message: str, features: TaskFeatures, estimate: Estimate,
                 blocked_at: float, turn_id: Optional[str]):
        self.session_id, self.message, self.features = session_id, message, features
        self.estimate, self.blocked_at, self.turn_id = estimate, blocked_at, turn_id


class Store:
    def __init__(self, conn: sqlite3.Connection):
        self._db = conn
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(_SCHEMA)

    # --- ForecastState (contracts.ForecastState) ----------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key: str, value: Any) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, json.dumps(value)))

    # --- History (contracts.History) ----------------------------------------------------------

    def load(self) -> list[PastTask]:
        with self._lock:
            rows = self._db.execute("SELECT row FROM history ORDER BY id").fetchall()
        return [from_dict(PastTask, json.loads(r[0])) for r in rows]

    def append(self, task: PastTask) -> None:
        """Store a finished task. The raw message never goes into history."""
        task.features.message = ""
        with self._lock, self._db:
            self._db.execute("INSERT INTO history (finished_at, row) VALUES (?, ?)",
                             (task.finished_at, json.dumps(to_dict(task))))
            self._db.execute("DELETE FROM history WHERE id <= (SELECT MAX(id) FROM history) - ?", (HISTORY_LIMIT,))

    # --- Blocked prompts ----------------------------------------------------------------------

    def put_pending(self, session_id: str, message: str, features: TaskFeatures, estimate: Estimate,
                    turn_id: Optional[str] = None, now: Optional[float] = None) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO pending VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, message, json.dumps(to_dict(features)), json.dumps(to_dict(estimate)),
                 now or time.time(), turn_id))

    def get_pending(self, session_id: str, now: Optional[float] = None) -> Optional[Pending]:
        now = now or time.time()
        with self._lock, self._db:
            self._db.execute("DELETE FROM pending WHERE blocked_at < ?", (now - PENDING_TTL,))
            row = self._db.execute("SELECT * FROM pending WHERE session_id = ?", (session_id,)).fetchone()
        if not row:
            return None
        sid, message, features, estimate, blocked_at, turn_id = row
        return Pending(sid, message, from_dict(TaskFeatures, json.loads(features)),
                       from_dict(Estimate, json.loads(estimate)), blocked_at, turn_id)

    def drop_pending(self, session_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM pending WHERE session_id = ?", (session_id,))

    # --- Daily spend --------------------------------------------------------------------------

    @staticmethod
    def today(now: Optional[float] = None) -> str:
        return time.strftime("%Y-%m-%d", time.localtime(now or time.time()))

    def add_spend(self, usd: float, now: Optional[float] = None) -> float:
        day = self.today(now)
        with self._lock, self._db:
            self._db.execute("INSERT INTO daily VALUES (?, ?) ON CONFLICT(day) DO UPDATE SET usd = usd + ?",
                             (day, usd, usd))
            return self._db.execute("SELECT usd FROM daily WHERE day = ?", (day,)).fetchone()[0]

    def spent_today(self, now: Optional[float] = None) -> float:
        with self._lock:
            row = self._db.execute("SELECT usd FROM daily WHERE day = ?", (self.today(now),)).fetchone()
        return row[0] if row else 0.0
