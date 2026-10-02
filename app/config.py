"""应用配置。所有设置均可用环境变量覆盖，便于容器部署。"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    db_path: str = os.environ.get("MOSAIC_DB_PATH", "/data/mosaics.db")
    # 单任务规模保护：输出网格像素总数与单边长度上限
    max_pixels: int = _int("MOSAIC_MAX_PIXELS", 4_000_000)
    max_dim: int = _int("MOSAIC_MAX_DIM", 8192)
    max_tiles: int = _int("MOSAIC_MAX_TILES", 16)
    min_tiles: int = _int("MOSAIC_MIN_TILES", 2)
    # 仿射对齐（亚像元偏移、分辨率差异）容差
    align_tol: float = _float("MOSAIC_ALIGN_TOL", 1e-6)
    # 冲突证据与空洞组件的最多返回条数（服务内部仍计算全部计数）
    max_conflict_evidence: int = _int("MOSAIC_MAX_EVIDENCE", 200)
    max_hole_components: int = _int("MOSAIC_MAX_HOLES", 50)
    # 预览图最长边
    preview_max_edge: int = _int("MOSAIC_PREVIEW_EDGE", 512)


settings = Settings()
