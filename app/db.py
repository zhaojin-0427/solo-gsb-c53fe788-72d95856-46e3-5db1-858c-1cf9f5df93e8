"""SQLite 持久化：任务、输入摘要与完整请求/结果，支持幂等重试。

幂等语义：
* 服务对规范化请求体计算 SHA-256（``request_hash``）；
* 相同请求体重试 -> 直接返回首次结果（HTTP 200，不重新计算）；
* 相同 ``request_id`` 但请求体不同 -> 409 冲突；
* 数据库在写入时对 (hash, request_id) 做唯一约束，并发下以
  ``OperationalError``/``IntegrityError`` 重试读取兜底。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from .config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mosaics (
    id              TEXT PRIMARY KEY,
    request_hash    TEXT NOT NULL UNIQUE,
    request_id      TEXT UNIQUE,
    created_at      TEXT NOT NULL,
    input_summary   TEXT NOT NULL,
    request_json    TEXT NOT NULL,
    result_json     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mosaics_request_id ON mosaics(request_id);
"""

_init_lock = threading.Lock()
_initialized = False


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(settings.db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(settings.db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db() -> None:
    global _initialized
    with _init_lock:
        if _initialized:
            return
        with _connect() as conn:
            conn.executescript(_SCHEMA)
        _initialized = True


def reset_init_for_tests() -> None:  # pragma: no cover - 测试辅助
    global _initialized
    _initialized = False


def canonical_hash(payload: dict[str, Any]) -> str:
    import hashlib

    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def input_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """从请求体提取紧凑摘要（不包含可能很大的矩阵数据）。"""
    tiles = []
    for i, t in enumerate(payload.get("tiles", [])):
        if t.get("bands") is not None:
            bands = t["bands"]
            n_bands = len(bands)
            height = len(bands[0]) if n_bands else 0
            width = len(bands[0][0]) if height else 0
            bands_desc = {"encoding": "array", "bands": n_bands, "height": height, "width": width}
        else:
            bands_desc = {"encoding": "png"}
        if t.get("cloud_mask") is not None:
            cm = t["cloud_mask"]
            bands_desc["cloud"] = {"encoding": "array", "height": len(cm), "width": len(cm[0]) if cm else 0}
        else:
            bands_desc["cloud"] = {"encoding": "png"}
        tiles.append(
            {
                "order": i,
                "id": t.get("id"),
                "acquired_at": t.get("acquired_at"),
                "pixel_size": t.get("pixel_size"),
                "origin": t.get("origin"),
                "nodata": t.get("nodata"),
                "shape": bands_desc,
            }
        )
    return {
        "tile_count": len(tiles),
        "tiles": tiles,
        "target": payload.get("target"),
        "max_time_delta_seconds": payload.get("max_time_delta_seconds"),
        "conflict_value_tolerance": payload.get("conflict_value_tolerance"),
        "request_id": payload.get("request_id"),
    }


def find_by_hash(request_hash: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT result_json FROM mosaics WHERE request_hash = ?", (request_hash,)
        ).fetchone()
    return json.loads(row["result_json"]) if row else None


def find_by_request_id(request_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT result_json FROM mosaics WHERE request_id = ?", (request_id,)
        ).fetchone()
    return json.loads(row["result_json"]) if row else None


def get_mosaic(mosaic_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT result_json FROM mosaics WHERE id = ?", (mosaic_id,)
        ).fetchone()
    return json.loads(row["result_json"]) if row else None


def save_mosaic(
    mosaic_id: str,
    request_hash: str,
    request_id: str | None,
    payload: dict[str, Any],
    result: dict[str, Any],
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO mosaics
                (id, request_hash, request_id, created_at, input_summary,
                 request_json, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mosaic_id,
                request_hash,
                request_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(input_summary(payload), ensure_ascii=False),
                json.dumps(payload, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
            ),
        )
        conn.commit()
