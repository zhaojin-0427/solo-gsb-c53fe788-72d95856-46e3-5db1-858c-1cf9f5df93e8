"""pytest 公共夹具：隔离的临时 SQLite 库 + TestClient。

关键：必须在任何 ``app.*`` 模块被导入之前设置 ``MOSAIC_DB_PATH``，
``app.config.settings`` 只在导入时读取一次环境变量。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# 确保以仓库根目录可导入 app 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_TEST_DB_DIR = tempfile.mkdtemp(prefix="mosaic_api_tests_")
os.environ["MOSAIC_DB_PATH"] = str(Path(_TEST_DB_DIR) / "test_mosaics.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture()
def client():
    db.init_db()
    with TestClient(app) as c:
        yield c
