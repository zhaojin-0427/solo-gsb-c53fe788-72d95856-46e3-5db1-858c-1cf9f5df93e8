"""请求/响应数据模型（Pydantic v2）。

输入的波段矩阵与云掩膜支持两种等价编码：

* ``values``：嵌套 JSON 数组（直观，便于复核）；
* ``png``：base64 编码的 PNG（字节友好，Pillow 编解码）。

对于 ``bands``，PNG 每一个通道按 R,G,B,A 顺序依次对应第 0、1、2、3 波段，
``dtype`` 描述原始像元数据类型（``uint8``/``uint16``/``float32``）。
``cloud_mask`` 使用单波段 8 位 PNG，非零像元为云。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

DtypeName = Literal["uint8", "uint16", "float32"]


class XY(BaseModel):
    x: float
    y: float


class TargetGrid(BaseModel):
    """输出网格定义。

    ``origin`` 为输出网格左上角的投影坐标（像元左上角，而非像元中心）；
    ``pixel_size`` 为像元分辨率，x 方向为正、y 方向为负（北朝上）。
    """

    origin: XY
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    pixel_size: tuple[float, float]
    nodata: float = -9999.0


class TileIn(BaseModel):
    """单个时相切片。``bands`` 与 ``cloud_mask`` 二选一编码方式即可。"""

    id: str = Field(min_length=1, max_length=128)
    acquired_at: str = Field(description="ISO 8601 采集时间；naive 时间按 UTC 处理")
    pixel_size: tuple[float, float] = Field(description="(sx, sy)，要求与目标分辨率一致")
    origin: XY = Field(description="切片左上角投影坐标（像元左上角）")
    nodata: float = -9999.0
    bands: list[list[list[Any]]] | None = Field(
        default=None,
        description="形状 [band, height, width] 的嵌套数组",
    )
    bands_png: str | None = Field(default=None, description="base64 PNG（通道即波段）")
    bands_dtype: DtypeName = "uint16"
    cloud_mask: list[list[int]] | None = Field(default=None, description="形状 [height, width]，1=云")
    cloud_mask_png: str | None = Field(default=None, description="base64 单波段 PNG，非零=云")

    @model_validator(mode="after")
    def _require_one_encoding(self) -> "TileIn":
        if (self.bands is None) == (self.bands_png is None):
            raise ValueError("bands 与 bands_png 必须且只能提供一种")
        if (self.cloud_mask is None) == (self.cloud_mask_png is None):
            raise ValueError("cloud_mask 与 cloud_mask_png 必须且只能提供一种")
        if not self.id.strip():
            raise ValueError("tile id 不能为空字符串")
        return self


class MosaicRequest(BaseModel):
    tiles: list[TileIn] = Field(min_length=2, max_length=16)
    target: TargetGrid
    max_time_delta_seconds: float = Field(ge=0)
    conflict_value_tolerance: float = Field(
        default=0.0,
        ge=0,
        description="次优像元与入选像元差异超过该阈值（任一波段）即记为冲突",
    )
    request_id: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def _unique_tile_ids(self) -> "MosaicRequest":
        ids = [t.id for t in self.tiles]
        if len(set(ids)) != len(ids):
            raise ValueError("tiles 中存在重复的 id")
        return self


# ------------------------------------------------------------------ 响应模型


class BBox(BaseModel):
    min_row: int
    min_col: int
    max_row: int
    max_col: int


class HoleOut(BaseModel):
    size: int
    bbox: BBox


class CoverageOut(BaseModel):
    covered_fraction: float
    covered_pixels: int
    hole_pixels: int
    total_pixels: int
    holes: list[HoleOut]


class ConflictEvidenceOut(BaseModel):
    row: int
    col: int
    winner: str
    runner_up: str
    winner_values: list[float]
    runner_up_values: list[float]
    abs_difference: list[float]
    acquired_at: str
    runner_up_acquired_at: str


class OverlapOut(BaseModel):
    pixels_with_multiple_candidates: int
    conflicting_pixels: int
    evidence: list[ConflictEvidenceOut]


class TileStatOut(BaseModel):
    tile_id: str
    acquired_at: str
    used_pixels: int
    excluded: bool
    exclude_reason: str | None = None
    cloud_fraction: float
    origin_col: int
    origin_row: int


class MosaicOut(BaseModel):
    id: str
    created_at: str
    status: Literal["done"] = "done"
    target: dict[str, Any]
    max_time_delta_seconds: float
    reference_time: str
    bands: list[list[list[float]]]
    nodata: float
    source_map: list[list[int]]
    source_ids: list[str]
    coverage: CoverageOut
    overlap: OverlapOut
    tiles: list[TileStatOut]
    preview_png: str = Field(description="data:image/png;base64,... 彩色预览")
    source_map_png: str = Field(description="data:image/png;base64,... 来源索引着色图")
    request_hash: str
    request_id: str | None = None
