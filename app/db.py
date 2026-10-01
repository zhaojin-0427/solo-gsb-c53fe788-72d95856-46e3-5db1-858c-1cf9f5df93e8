"""SQLite persistence for mosaic tasks and idempotent retries.

Schema stores one row per task:

* the request body (canonical JSON),
* a stable SHA-256 payload hash,
* the client ``request_id`` (unique when given),
* an *input summary* (per-tile shape, cloud fraction, grid mode …),
* the full result payload (deterministic mosaic, sources, evidence, stats).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS mosaics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT UNIQUE,
    payload_hash    TEXT NOT NULL,
    request_json    TEXT NOT NULL,
    input_summary   TEXT NOT NULL,
    result_json     TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mosaics_hash ON mosaics(payload_hash);
"""


class Database:
    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._local = threading.local()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._connect()
            self._local.conn = c
        return c

    # ------------------------------------------------------------------ #

    @staticmethod
    def canonical_payload(request_dict: dict[str, Any]) -> str:
        return json.dumps(
            request_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    @staticmethod
    def payload_hash(canonical: str) -> str:
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def create(
        self,
        *,
        request_id: Optional[str],
        request_dict: dict[str, Any],
        input_summary: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> tuple[int, bool, dict[str, Any]]:
        """Insert a task.

        Returns ``(id, reused, stored_result)``. ``reused`` is True when an
        existing row was returned (idempotent retry).
        """
        canonical = self.canonical_payload(request_dict)
        phash = self.payload_hash(canonical)
        now = result.get("created_at") or datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

        with self.conn:
            if request_id is not None:
                row = self.conn.execute(
                    "SELECT id, result_json, request_json FROM mosaics WHERE request_id = ?",
                    (request_id,),
                ).fetchone()
                if row is not None:
                    stored_req = json.loads(row["request_json"])
                    if self.canonical_payload(stored_req) != canonical:
                        raise ValueError(
                            "request_id was already used with a different request body"
                        )
                    return row["id"], True, json.loads(row["result_json"])

            cur = self.conn.execute(
                """
                INSERT INTO mosaics
                    (request_id, payload_hash, request_json, input_summary,
                     result_json, created_at)
                VALUES (?, ?, ?, ?, '', ?)
                """,
                (request_id, phash, canonical, json.dumps(input_summary, ensure_ascii=False), now),
            )
            mosaic_id = int(cur.lastrowid)
            stored = dict(result)
            stored["id"] = mosaic_id
            stored.setdefault("created_at", now)
            self.conn.execute(
                "UPDATE mosaics SET result_json = ? WHERE id = ?",
                (json.dumps(stored, ensure_ascii=False, allow_nan=False), mosaic_id),
            )
            return mosaic_id, False, stored

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Fetch by numeric id or client request_id."""
        row = None
        if key.isdigit():
            row = self.conn.execute(
                "SELECT result_json FROM mosaics WHERE id = ?", (int(key),)
            ).fetchone()
        if row is None:
            row = self.conn.execute(
                "SELECT result_json FROM mosaics WHERE request_id = ?", (key,)
            ).fetchone()
        return json.loads(row["result_json"]) if row else None

    def exists_hash(self, phash: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT result_json FROM mosaics WHERE payload_hash = ? ORDER BY id LIMIT 1",
            (phash,),
        ).fetchone()
        return json.loads(row["result_json"]) if row else None
