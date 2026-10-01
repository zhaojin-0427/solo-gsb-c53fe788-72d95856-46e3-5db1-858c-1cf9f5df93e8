"""Deterministic multi-temporal cloud-gap mosaicking engine.

Selection priority (fixed, documented in the README):

1. Cloud fraction of the *source tile* (lower wins) — the least-cloudy
   acquisition is preferred everywhere it contributes;
2. temporal proximity ``|acquired_at - reference_time|`` (smaller wins);
3. per-pixel distance to the tile-footprint edge (farther from seams wins);
4. lexicographically smallest ``tile_id`` (final deterministic tie-break).

Pixels whose only candidates fall outside ``max_time_delta_seconds`` become
holes with reason ``outside_time_window``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from typing import Optional

import numpy as np

from .distance import distance_to_mask
from .schemas import (
    MAX_TILE_SIDE,
    MosaicRequest,
    TargetGrid,
    parse_acquired_at,
)

ALIGN_TOL = 1e-6


class MosaicError(ValueError):
    """Validation failure that should be reported as HTTP 422."""


@dataclass
class TileData:
    tile_id: str
    acquired_at: datetime
    nodata: Optional[float]
    values: np.ndarray            # float32 (B, H, W), NaN at no-data
    cloud: np.ndarray             # bool (H, W)
    cloud_fraction: float
    pixel_size: float
    origin_x: float
    origin_y: float
    height: int
    width: int
    # Output-space arrays, filled later:
    cover: Optional[np.ndarray] = None      # bool (Hout, Wout): footprint
    ocloud: Optional[np.ndarray] = None     # bool (Hout, Wout): cloud fraction == 1
    ofrac: Optional[np.ndarray] = None      # float32 (Hout, Wout): subpixel cloud fraction
    ovalues: Optional[np.ndarray] = None    # float32 (B, Hout, Wout), NaN outside
    edge_dist: Optional[np.ndarray] = None  # float32 (Hout, Wout), ground units
    present: Optional[np.ndarray] = None    # bool: cover & not cloudy & some valid band
    mode: str = "native"
    covered_target_pixels: int = 0


def _epoched(dt: datetime) -> float:
    return dt.timestamp()


def _validate_and_load(req: MosaicRequest) -> tuple[TargetGrid, list[TileData]]:
    target = req.target
    ids: set[str] = set()
    band_count: Optional[int] = None
    loaded: list[TileData] = []

    for tile in req.tiles:
        if tile.tile_id in ids:
            raise MosaicError(f"duplicate tile_id {tile.tile_id!r}")
        ids.add(tile.tile_id)

        b = tile.band_count
        if b < 1:
            raise MosaicError(f"tile {tile.tile_id!r}: at least one band is required")
        if b > 8:
            raise MosaicError(f"tile {tile.tile_id!r}: at most 8 bands are supported")
        if band_count is None:
            band_count = b
        elif b != band_count:
            raise MosaicError(
                f"tile {tile.tile_id!r}: band count {b} differs from {band_count}"
            )

        h, w = tile.height, tile.width
        if h < 1 or w < 1:
            raise MosaicError(f"tile {tile.tile_id!r}: empty band matrix")
        if h > MAX_TILE_SIDE or w > MAX_TILE_SIDE:
            raise MosaicError(
                f"tile {tile.tile_id!r}: dimensions {w}x{h} exceed "
                f"{MAX_TILE_SIDE}px side limit"
            )
        for bi, band in enumerate(tile.bands):
            if len(band) != h or any(len(row) != w for row in band):
                raise MosaicError(
                    f"tile {tile.tile_id!r} band {bi}: matrix must be rectangular {w}x{h}"
                )

        values = np.asarray(tile.bands, dtype=np.float32)
        if tile.nodata is not None and not math.isnan(tile.nodata):
            values = np.where(values == np.float32(tile.nodata), np.nan, values)
        # Explicit NaN input also counts as no-data.
        valid = ~np.isnan(values)

        if tile.cloud_mask is not None:
            cm = tile.cloud_mask
            if len(cm) != h or any(len(row) != w for row in cm):
                raise MosaicError(
                    f"tile {tile.tile_id!r}: cloud_mask shape must be {w}x{h}"
                )
            flat = {v for row in cm for v in row}
            if not flat <= {0, 1}:
                raise MosaicError(f"tile {tile.tile_id!r}: cloud_mask must contain only 0/1")
            cloud = np.asarray(cm, dtype=bool)
        else:
            cloud = np.zeros((h, w), dtype=bool)

        observed = valid.any(axis=0) | cloud
        frac = float(cloud.sum() / observed.size) if observed.size else 0.0

        loaded.append(
            TileData(
                tile_id=tile.tile_id,
                acquired_at=parse_acquired_at(tile.acquired_at),
                nodata=tile.nodata,
                values=values,
                cloud=cloud,
                cloud_fraction=frac,
                pixel_size=float(tile.pixel_size),
                origin_x=float(tile.origin_x),
                origin_y=float(tile.origin_y),
                height=h,
                width=w,
            )
        )

    return target, loaded


def _check_grid_affine(tile: TileData, target: TargetGrid) -> tuple[int, int, int, int]:
    """Validate co-registration; return (step, r0, c0, mode category).

    step is the integer ratio of pixel sizes (>= 1): output pixels per tile
    pixel for coarser/native tiles, or tile pixels per output pixel (>= 2)
    for finer tiles — distinguished by the returned mode.
    """
    pst = target.pixel_size
    ratio = tile.pixel_size / pst
    nearest = int(round(ratio))

    def aligned_origin() -> None:
        # Tile NW corner must fall on an output-grid pixel corner.
        fy = (tile.origin_y - target.origin_y) / pst
        fx = (tile.origin_x - target.origin_x) / pst
        if abs(fy - round(fy)) > ALIGN_TOL or abs(fx - round(fx)) > ALIGN_TOL:
            raise MosaicError(
                f"tile {tile.tile_id!r}: origin ({tile.origin_x}, {tile.origin_y}) "
                f"does not land on the target output grid"
            )

    if ratio >= 1.0:
        if nearest < 1:
            nearest = 1
        if abs(ratio - nearest) > ALIGN_TOL * max(1.0, nearest):
            raise MosaicError(
                f"tile {tile.tile_id!r}: pixel_size ratio {ratio:.6g} is not an "
                "integer multiple of target pixel_size"
            )
        aligned_origin()
        r0 = int(round((target.origin_y - tile.origin_y) / pst))
        c0 = int(round((tile.origin_x - target.origin_x) / pst))
        mode = "native" if nearest == 1 else "replicate_coarser"
        return nearest, r0, c0, mode

    # Finer tile: target / tile ratio must be an integer >= 2.
    frac = pst / tile.pixel_size
    fint = int(round(frac))
    if fint < 2 or abs(frac - fint) > ALIGN_TOL * max(1.0, fint):
        raise MosaicError(
            f"tile {tile.tile_id!r}: finer pixel_size ratio {frac:.6g} is not an "
            "integer (>=2) divisor of target pixel_size"
        )
    aligned_origin()
    r0 = int(round((target.origin_y - tile.origin_y) / pst))
    c0 = int(round((tile.origin_x - target.origin_x) / pst))
    return fint, r0, c0, "aggregate_finer"


def _place_coarse(
    tile: TileData, target: TargetGrid, step: int, r0: int, c0: int, mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample a native- or coarser-resolution tile into output space."""
    H, W = target.height, target.width
    cover = np.zeros((H, W), dtype=bool)
    ofrac = np.zeros((H, W), dtype=np.float32)
    ovalues = np.full((tile.values.shape[0], H, W), np.nan, dtype=np.float32)

    th, tw = tile.height, tile.width
    # Tile pixel (i,j) occupies output rows [r0+i*step, r0+(i+1)*step).
    r_start = max(r0, 0)
    r_end = min(r0 + th * step, H)
    c_start = max(c0, 0)
    c_end = min(c0 + tw * step, W)
    if r_end <= r_start or c_end <= c_start:
        return cover, ofrac, ovalues, np.zeros_like(ofrac, dtype=bool)

    rows = np.arange(r_start, r_end)
    cols = np.arange(c_start, c_end)
    ti = np.clip((rows - r0) // step, 0, th - 1)
    tj = np.clip((cols - c0) // step, 0, tw - 1)

    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    ii = np.clip((rr - r0) // step, 0, th - 1)
    jj = np.clip((cc - c0) // step, 0, tw - 1)

    cover[r_start:r_end, c_start:c_end] = True
    sub_cloud = tile.cloud[ii, jj]
    ofrac[r_start:r_end, c_start:c_end] = sub_cloud.astype(np.float32)
    for b in range(tile.values.shape[0]):
        ovalues[b, r_start:r_end, c_start:c_end] = tile.values[b, ii, jj]

    ocloud = ofrac >= 1.0
    return cover, ofrac, ovalues, ocloud


def _place_fine(
    tile: TileData, target: TargetGrid, factor: int, r0: int, c0: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Aggregate a finer-resolution tile: mean of clear valid subpixels."""
    warnings: list[str] = []
    H, W = target.height, target.width
    B = tile.values.shape[0]
    cover = np.zeros((H, W), dtype=bool)
    ofrac = np.zeros((H, W), dtype=np.float32)
    ovalues = np.full((B, H, W), np.nan, dtype=np.float32)

    th, tw = tile.height, tile.width
    # Output pixel (r,c) uses fine rows [r0*factor + r*factor, ...+factor).
    r_start = max(r0, 0)
    c_start = max(c0, 0)
    r_end = min((th // factor) + r0, H)
    c_end = min((tw // factor) + c0, W)
    if r_end <= r_start or c_end <= c_start:
        return cover, ofrac, ovalues, np.zeros((H, W), dtype=bool), warnings

    if th % factor or tw % factor:
        warnings.append(
            f"tile {tile.tile_id!r}: {th - th % factor}/{th} fine rows and "
            f"{tw - tw % factor}/{tw} fine cols align to the target grid; "
            "partial edge pixels ignored"
        )

    rows = np.arange(r_start, r_end)
    cols = np.arange(c_start, c_end)
    fine_r0 = (rows - r0) * factor
    fine_c0 = (cols - c0) * factor
    rr, cc = np.meshgrid(fine_r0, fine_c0, indexing="ij")

    # Subpixel index grids (factor x factor per output cell), vectorised.
    dr = np.arange(factor).reshape(factor, 1)
    dc = np.arange(factor).reshape(1, factor)
    fi = rr[:, None, :, None] + dr[None, :, None, :]   # (rows, f, cols, f)
    fj = cc[:, None, :, None] + dc[None, :, None, :]

    sub_cloud = tile.cloud[fi, fj]
    ofrac[r_start:r_end, c_start:c_end] = sub_cloud.mean(axis=(1, 3), dtype=np.float32)
    cover[r_start:r_end, c_start:c_end] = True

    clear = ~sub_cloud
    for b in range(B):
        sub = tile.values[b][fi, fj]
        usable = clear & ~np.isnan(sub)
        count = usable.sum(axis=(1, 3), dtype=np.float32)
        sums = np.where(usable, sub, 0.0).sum(axis=(1, 3), dtype=np.float32)
        agg = np.divide(
            sums, count, out=np.full_like(sums, np.nan), where=count > 0
        )
        ovalues[b, r_start:r_end, c_start:c_end] = agg

    ocloud = ofrac >= 1.0
    return cover, ofrac, ovalues, ocloud, warnings


def build_mosaic(req: MosaicRequest) -> dict:
    """Run the whole pipeline and return a serializable result dict."""
    target, tiles = _validate_and_load(req)
    H, W = target.height, target.width
    warnings: list[str] = []

    times = [t.acquired_at for t in tiles]
    if req.reference_time is not None:
        ref_time = parse_acquired_at(req.reference_time)
    else:
        ref_time = datetime.fromtimestamp(
            median(_epoched(t) for t in times), tz=timezone.utc
        )
        warnings.append(
            "reference_time not supplied; using median acquisition time "
            f"{ref_time.isoformat()}"
        )

    B = tiles[0].values.shape[0]
    max_dt = req.max_time_delta_seconds

    for tile in tiles:
        step, r0, c0, mode = _check_grid_affine(tile, target)
        if mode == "aggregate_finer":
            cover, ofrac, ovalues, ocloud, w = _place_fine(
                tile, target, step, r0, c0
            )
            warnings.extend(w)
        else:
            cover, ofrac, ovalues, ocloud = _place_coarse(
                tile, target, step, r0, c0, mode
            )
        tile.mode = mode
        tile.cover, tile.ofrac, tile.ovalues, tile.ocloud = cover, ofrac, ovalues, ocloud
        tile.covered_target_pixels = int(cover.sum())

        # Edge distance measured over the geometric footprint, ground units.
        tile.edge_dist = distance_to_mask(cover, target.pixel_size)

        any_valid_band = ~np.isnan(ovalues).all(axis=0)
        tile.present = cover & ~ocloud & any_valid_band

        if not cover.any():
            warnings.append(
                f"tile {tile.tile_id!r} does not intersect the target window"
            )

    n = len(tiles)
    time_delta = np.asarray(
        [abs(_epoched(t.acquired_at) - _epoched(ref_time)) for t in tiles],
        dtype=np.float64,
    )
    if max_dt is not None:
        time_ok = np.asarray([d <= max_dt + 1e-9 for d in time_delta], dtype=bool)
        excluded = [t.tile_id for t, ok in zip(tiles, time_ok) if not ok]
        if excluded:
            warnings.append(
                "tiles outside max_time_delta window, excluded from selection: "
                + ", ".join(excluded)
            )
    else:
        time_ok = np.ones(n, dtype=bool)

    # Fixed priority 1 (tile cloud fraction) and 4 (tile_id tie-break) are
    # resolved implicitly: tiles are processed in (cloud_fraction, tile_id)
    # order, so a later tile may only replace the incumbent when the cloud
    # fraction is equal AND its time proximity / edge distance is better;
    # remaining ties keep the earlier (lexicographically smaller) id.
    order = sorted(
        range(n), key=lambda i: (tiles[i].cloud_fraction, tiles[i].tile_id)
    )

    win = np.full((H, W), -1, dtype=np.int64)
    have = np.zeros((H, W), dtype=bool)
    win_time = np.zeros((H, W), dtype=np.float64)
    win_edge = np.zeros((H, W), dtype=np.float32)

    win_cloud = np.zeros((H, W), dtype=np.float32)

    for i in order:
        t = tiles[i]
        cand = t.present & time_ok[i]
        if not cand.any():
            continue
        same_cloud = win_cloud == np.float32(t.cloud_fraction)
        time_better = time_delta[i] < win_time - 1e-9
        time_same = abs(time_delta[i] - win_time) <= 1e-9
        edge_better = t.edge_dist > win_edge + 1e-7
        replace = same_cloud & (time_better | (time_same & edge_better))
        take = cand & (~have | replace)
        win[take] = i
        win_cloud[take] = t.cloud_fraction
        win_time[take] = time_delta[i]
        win_edge[take] = t.edge_dist[take]
        have |= cand

    # Band values from winners.
    mosaic = np.full((B, H, W), np.nan, dtype=np.float32)
    for b in range(B):
        for i in range(n):
            sel = win == i
            if sel.any():
                mosaic[b][sel] = tiles[i].ovalues[b][sel]

    covered = win >= 0
    hole = ~covered

    # ------------------------------------------------------------------ holes
    cover_any = np.zeros((H, W), dtype=bool)
    cloud_any = np.zeros((H, W), dtype=bool)
    nodata_any = np.zeros((H, W), dtype=bool)
    present_any = np.zeros((H, W), dtype=bool)
    time_present = np.zeros((H, W), dtype=bool)
    for i, t in enumerate(tiles):
        cover_any |= t.cover
        cloud_any |= t.ocloud
        nodata_any |= t.cover & ~t.ocloud & np.isnan(t.ovalues).all(axis=0)
        present_any |= t.present
        if time_ok[i]:
            time_present |= t.present

    reason_grid = np.empty((H, W), dtype=object)
    reason_grid[:] = ""
    reason_grid[hole & ~cover_any] = "no_coverage"
    reason_grid[hole & cover_any & ~cloud_any & nodata_any] = "nodata"
    reason_grid[hole & cloud_any] = "cloud"
    reason_grid[hole & present_any & ~time_present] = "outside_time_window"
    # Fallback (mixed nodata/cloud with no single dominant evidence): nodata.
    reason_grid[hole & (reason_grid == "")] = "nodata"

    # --------------------------------------------------------------- conflicts
    tol = float(req.conflict_tolerance)
    conflict_count = 0
    band_conflict = np.zeros((B, H, W), dtype=bool)
    if n >= 2:
        # Range of eligible, clear, valid values per pixel per band.
        for b in range(B):
            vmin = np.full((H, W), np.inf, dtype=np.float32)
            vmax = np.full((H, W), -np.inf, dtype=np.float32)
            cnt = np.zeros((H, W), dtype=np.int16)
            for i, t in enumerate(tiles):
                if not time_ok[i]:
                    continue
                v = t.ovalues[b]
                ok = t.cover & ~t.ocloud & ~np.isnan(v)
                vmin = np.minimum(vmin, np.where(ok, v, np.inf))
                vmax = np.maximum(vmax, np.where(ok, v, -np.inf))
                cnt += ok
            with np.errstate(invalid="ignore"):
                band_conflict[b] = (cnt >= 2) & ((vmax - vmin) > tol)
        conflict_pixels = band_conflict.any(axis=0)
        conflict_count = int(conflict_pixels.sum())
    else:
        conflict_pixels = np.zeros((H, W), dtype=bool)

    # ------------------------------------------------------------ serialize
    def clean_float(x: float) -> Optional[float]:
        if x is None:
            return None
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return None
        return xf

    def matrix_from(arr3: np.ndarray) -> list:
        return [
            [[clean_float(v) for v in row] for row in band] for band in arr3
        ]

    sources: list[list[Optional[str]]] = [
        [tiles[win[r, c]].tile_id if win[r, c] >= 0 else None for c in range(W)]
        for r in range(H)
    ]
    index1: list[list[int]] = [
        [int(win[r, c]) + 1 if win[r, c] >= 0 else 0 for c in range(W)]
        for r in range(H)
    ]
    hole_mask_list = [
        [int(hole[r, c]) for c in range(W)] for r in range(H)
    ]
    reasons_list = [
        [str(reason_grid[r, c]) if hole[r, c] else "" for c in range(W)]
        for r in range(H)
    ]

    # Evidence samples (deterministic row-major order, capped).
    cap = req.max_evidence
    hole_ys, hole_xs = np.where(hole)
    holes_out = []
    for r, c in zip(hole_ys.tolist(), hole_xs.tolist()):
        if len(holes_out) >= cap:
            break
        holes_out.append({"row": r, "col": c, "reason": str(reason_grid[r, c])})

    conflict_evidence = []
    if conflict_count:
        cys, cxs = np.where(conflict_pixels)
        for r, c in zip(cys.tolist(), cxs.tolist()):
            bands_hit = [b for b in range(B) if band_conflict[b, r, c]]
            for b in bands_hit:
                if len(conflict_evidence) >= cap:
                    break
                entries = []
                for i, t in enumerate(tiles):
                    if not time_ok[i]:
                        continue
                    v = t.ovalues[b, r, c]
                    if t.cover[r, c] and not t.ocloud[r, c] and not np.isnan(v):
                        entries.append((t.tile_id, float(v)))
                if len(entries) < 2:
                    continue
                vals = [v for _, v in entries]
                conflict_evidence.append(
                    {
                        "row": r,
                        "col": c,
                        "band": b,
                        "sources": [tid for tid, _ in entries],
                        "values": [clean_float(v) for v in vals],
                        "difference": clean_float(max(vals) - min(vals)),
                    }
                )
            if len(conflict_evidence) >= cap:
                break

    total_px = H * W
    coverage = float(covered.sum() / total_px)
    band_cov = [
        float((~np.isnan(mosaic[b])).sum() / total_px) for b in range(B)
    ]

    input_summary = [
        {
            "tile_id": t.tile_id,
            "acquired_at": t.acquired_at.isoformat().replace("+00:00", "Z"),
            "shape": [int(t.values.shape[0]), t.height, t.width],
            "pixel_size": t.pixel_size,
            "origin": [t.origin_x, t.origin_y],
            "nodata": t.nodata,
            "cloud_fraction": round(float(t.cloud_fraction), 6),
            "covered_target_pixels": t.covered_target_pixels,
            "mode": t.mode,
        }
        for t in tiles
    ]

    result = {
        "target": target,
        "tiles": tiles,
        "reference_time": ref_time,
        "warnings": warnings,
        "mosaic": matrix_from(mosaic),
        "sources": sources,
        "selected_index": index1,
        "hole_mask": hole_mask_list,
        "hole_reasons": reasons_list,
        "coverage_fraction": coverage,
        "band_coverage_fraction": band_cov,
        "conflict_count": conflict_count,
        "conflict_fraction": float(conflict_count / total_px),
        "conflict_evidence": conflict_evidence,
        "holes": holes_out,
        "band_count": B,
        "win": win,
        "input_summary": input_summary,
    }
    return result  # type: ignore[return-value]
