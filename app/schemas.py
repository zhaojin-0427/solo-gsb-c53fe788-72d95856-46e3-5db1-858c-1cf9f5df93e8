"""Pydantic request/response schemas for the cloud-gap mosaic API.

Coordinate convention
---------------------
The grid is a north-up affine raster grid::

    X = origin_x + col * pixel_size_x
    Y = origin_y - row * pixel_size_y

i.e. rows increase southward. ``pixel_size`` is always given as a positive
number; the sign of the affine is fixed by the service.

All tiles and the target share one coordinate reference frame (the service is
CRS-agnostic — numbers in, numbers out).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Hard caps keep single-request memory/cpu bounded.
MAX_BANDS = 8
MAX_TILE_SIDE = 2048
MAX_GRID_SIDE = 2048


def parse_acquired_at(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; naive timestamps are treated as UTC."""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"invalid ISO-8601 timestamp {value!r} "
            "(expected e.g. 2026-09-01T10:15:00Z)"
        ) from exc
    if dt.tzinfo is None:
        from datetime import timezone

        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class TargetGrid(BaseModel):
    """Output window expressed in the shared affine coordinate frame."""

    origin_x: float = Field(..., description="X coordinate of the NW pixel corner")
    origin_y: float = Field(..., description="Y coordinate of the NW pixel corner")
    pixel_size: float = Field(..., gt=0, description="Ground units per output pixel (>0)")
    width: int = Field(..., ge=1, le=MAX_GRID_SIDE, description="Output columns")
    height: int = Field(..., ge=1, le=MAX_GRID_SIDE, description="Output rows")


class RasterTile(BaseModel):
    """One co-registered acquisition slice."""

    tile_id: str = Field(..., min_length=1, max_length=128, description="Unique source label")
    acquired_at: str = Field(
        ..., description="ISO-8601 acquisition time, e.g. 2026-09-01T10:15:00Z"
    )
    pixel_size: float = Field(..., gt=0)
    origin_x: float
    origin_y: float
    bands: list[list[list[float]]] = Field(
        ...,
        description="Band-major matrix [band][row][col]; all tiles use the same band count",
    )
    nodata: Optional[float] = Field(
        None, description="No-data sentinel value; pixels equal to it are invalid"
    )
    cloud_mask: Optional[list[list[int]]] = Field(
        None,
        description="Optional HxW matrix of 0/1 (1 = cloud). Defaults to all clear.",
    )

    @field_validator("acquired_at")
    @classmethod
    def _validate_time(cls, v: str) -> str:
        parse_acquired_at(v)
        return v

    @property
    def band_count(self) -> int:
        return len(self.bands)

    @property
    def height(self) -> int:
        return len(self.bands[0]) if self.bands else 0

    @property
    def width(self) -> int:
        return len(self.bands[0][0]) if self.bands and self.bands[0] else 0


class MosaicRequest(BaseModel):
    """Body of ``POST /mosaics``."""

    target: TargetGrid
    tiles: list[RasterTile] = Field(..., min_length=2, max_length=16)
    max_time_delta_seconds: Optional[float] = Field(
        None,
        ge=0,
        description="Only tiles within this many seconds of the reference time are eligible",
    )
    reference_time: Optional[str] = Field(
        None,
        description="ISO-8601 temporal target. Defaults to the median tile acquisition time.",
    )
    conflict_tolerance: float = Field(
        0.0,
        ge=0,
        description="Absolute band-value difference at which overlapping pixels are a conflict",
    )
    max_evidence: int = Field(
        500, ge=1, le=5000, description="Per-result cap on conflict/hole evidence cells"
    )
    request_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=128,
        description="Client idempotency key; resubmitting the same request_id returns the stored result",
        pattern=r"^[A-Za-z0-9._\-]+$",
    )

    @field_validator("reference_time")
    @classmethod
    def _validate_ref_time(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            parse_acquired_at(v)
        return v


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #


class InputSummary(BaseModel):
    tile_id: str
    acquired_at: str
    shape: list[int]
    pixel_size: float
    origin: list[float]
    nodata: Optional[float]
    cloud_fraction: float
    covered_target_pixels: int
    mode: Literal["native", "aggregate_finer", "replicate_coarser"]


class HoleSample(BaseModel):
    row: int
    col: int
    reason: Literal[
        "no_coverage",
        "cloud",
        "nodata",
        "outside_time_window",
    ]


class ConflictEvidence(BaseModel):
    row: int
    col: int
    band: int
    sources: list[str]
    values: list[float]
    difference: float


class MosaicResult(BaseModel):
    id: int
    request_id: Optional[str]
    status: Literal["completed"]
    target: TargetGrid
    reference_time: str
    max_time_delta_seconds: Optional[float]
    mosaic: list[list[list[Optional[float]]]]
    """Band-major output matrix [band][row][col]; holes are ``null``."""

    sources: list[list[Optional[str]]]
    """Per-pixel winning tile id; holes are ``null``."""

    selected_index: list[list[int]]
    """Per-pixel 1-based index into ``input_summary``; 0 marks a hole."""

    hole_mask: list[list[int]]
    """1 where no eligible valid pixel exists."""

    hole_reasons: list[list[str]]
    """One of no_coverage / cloud / nodata / outside_time_window for holes."""

    coverage_fraction: float
    band_coverage_fraction: list[float]
    conflict_count: int
    conflict_fraction: float
    conflict_evidence: list[ConflictEvidence]
    holes: list[HoleSample]
    warnings: list[str]
    input_summary: list[InputSummary]
    created_at: str
    reused: Optional[bool] = Field(
        None,
        description="True when the response was served from a stored idempotent retry",
    )
