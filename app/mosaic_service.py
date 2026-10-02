"""多时相云隙拼接核心算法与任务编排。

选源固定优先级（逐像元，仅在时相合格范围内比较）：

1. 切片总云量 ``cloud_fraction`` 升序（云少优先）；
2. 与参考时间（所有切片中的最新采集时间）的时间差升序；
3. 到切片边缘的距离降序（优先使用切片内部像元，规避边缘）；
4. 切片在请求中的序号 ``tile_order`` 升序（最终确定性裁决）。

仿射对齐：切片必须与目标网格同分辨率（允许 1e-6 相对容差），
且左上角必须落在整数像元偏移上，否则返回 422。
"""
from __future__ import annotations

import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .config import settings
from .models import (
    BBox,
    ConflictEvidenceOut,
    CoverageOut,
    HoleOut,
    MosaicOut,
    MosaicRequest,
    OverlapOut,
    TileStatOut,
)
from .png_codec import decode_bands, decode_cloud, make_preview_png, make_source_map_png

_NEIGH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_KEY_COMPONENTS = 4
# 字典序键的“空槽”哨兵，排在任何真实候选之后
_SENTINEL = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float64)


class MosaicError(ValueError):
    """可返回给客户端的 422 业务错误。"""


def _parse_time(raw: str, tile_id: str) -> datetime:
    text = raw.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise MosaicError(f"tile {tile_id} 的 acquired_at 不是合法 ISO 8601 时间: {raw!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_close_resolution(a: float, b: float) -> bool:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= settings.align_tol * scale


def _integer_offset(origin: float, target_origin: float, pixel_size: float, axis: str) -> int:
    off = (origin - target_origin) / pixel_size
    rounded = round(off)
    if abs(off - rounded) > settings.align_tol:
        raise MosaicError(
            f"{axis} 方向原点 {origin} 相对目标原点 {target_origin} 的偏移 {off} "
            f"不是整数像元（容差 {settings.align_tol}），无法落入统一网格"
        )
    return int(rounded)


def _lex_compare(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对形状 [4, ...] 的键做逐元素字典序比较。

    返回 (a<b, a==b) 的布尔数组，与数组形状一致。
    """
    less = a[0] < b[0]
    equal = a[0] == b[0]
    for k in range(1, _KEY_COMPONENTS):
        less = less | (equal & (a[k] < b[k]))
        equal = equal & (a[k] == b[k])
    return less, equal


def _hole_components(filled: np.ndarray) -> list[HoleOut]:
    """对未填充像元做 4 连通标记，按面积降序、先行后列顺序返回。"""
    h, w = filled.shape
    visited = np.zeros_like(filled, dtype=bool)
    found: list[tuple[int, int, int, int, int, int, int]] = []
    for r0 in range(h):
        for c0 in range(w):
            if filled[r0, c0] or visited[r0, c0]:
                continue
            queue: deque[tuple[int, int]] = deque([(r0, c0)])
            visited[r0, c0] = True
            size = 0
            rmin = rmax = r0
            cmin = cmax = c0
            while queue:
                r, c = queue.popleft()
                size += 1
                rmin, rmax = min(rmin, r), max(rmax, r)
                cmin, cmax = min(cmin, c), max(cmax, c)
                for dr, dc in _NEIGH:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not visited[nr, nc]
                        and not filled[nr, nc]
                    ):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            found.append((size, r0, c0, rmin, cmin, rmax, cmax))
    found.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [
        HoleOut(
            size=size,
            bbox=BBox(min_row=rmin, min_col=cmin, max_row=rmax, max_col=cmax),
        )
        for size, _, _, rmin, cmin, rmax, cmax in found[: settings.max_hole_components]
    ]


def build_mosaic(req: MosaicRequest, request_hash: str) -> MosaicOut:
    target = req.target
    sx, sy = target.pixel_size
    if sx <= 0:
        raise MosaicError("pixel_size.x 必须为正")
    if sy >= 0:
        raise MosaicError("pixel_size.y 必须为负（北朝上栅格约定）")
    H, W = target.height, target.width
    if H > settings.max_dim or W > settings.max_dim:
        raise MosaicError(f"输出网格单边不能超过 {settings.max_dim} 像元")
    if H * W > settings.max_pixels:
        raise MosaicError(f"输出网格总像元数不能超过 {settings.max_pixels}（当前 {H * W}）")

    # ---- 解析时间、解码数据 -------------------------------------------------
    times = [_parse_time(t.acquired_at, t.id) for t in req.tiles]
    reference_time = max(times)
    max_delta = float(req.max_time_delta_seconds)

    decoded: list[np.ndarray] = []
    clouds: list[np.ndarray] = []
    band_count: int | None = None
    for tile in req.tiles:
        tsx, tsy = tile.pixel_size
        if not _is_close_resolution(tsx, sx) or not _is_close_resolution(tsy, sy):
            raise MosaicError(
                f"tile {tile.id} 分辨率 {tile.pixel_size} 与目标 {target.pixel_size} 不一致"
            )
        bands = decode_bands(tile)
        if band_count is None:
            band_count = bands.shape[0]
        elif bands.shape[0] != band_count:
            raise MosaicError(
                f"tile {tile.id} 波段数 {bands.shape[0]} 与首个切片 {band_count} 不一致"
            )
        _, th, tw = bands.shape
        cloud = decode_cloud(tile, th, tw)
        decoded.append(bands.astype(np.float64, copy=False))
        clouds.append(cloud)
    assert band_count is not None

    # ---- 逐切片静态属性 -----------------------------------------------------
    tile_infos: list[dict[str, Any]] = []
    for order, tile in enumerate(req.tiles):
        bands = decoded[order]
        _, th, tw = bands.shape
        dt_seconds = (reference_time - times[order]).total_seconds()
        excluded = dt_seconds > max_delta
        reason = (
            f"距参考时间 {dt_seconds:g}s 超过 max_time_delta_seconds={max_delta:g}s"
            if excluded
            else None
        )
        cloud_fraction = float(clouds[order].mean()) if th * tw else 0.0
        # 切片局部坐标中到最近边缘的距离（轴对齐，确定性的边缘度量）
        rows = np.arange(th, dtype=np.float64).reshape(th, 1)
        cols = np.arange(tw, dtype=np.float64).reshape(1, tw)
        edge_dist = np.minimum(
            np.minimum(np.broadcast_to(rows, (th, tw)), np.broadcast_to(cols, (th, tw))),
            np.minimum(
                np.broadcast_to(th - 1 - rows, (th, tw)),
                np.broadcast_to(tw - 1 - cols, (th, tw)),
            ),
        )
        tile_infos.append(
            {
                "order": order,
                "id": tile.id,
                "time": times[order],
                "dt": float(dt_seconds),
                "cloud_fraction": cloud_fraction,
                "excluded": excluded,
                "reason": reason,
                "bands": bands,
                "cloud": clouds[order],
                "edge_dist": edge_dist,
                "nodata": float(tile.nodata),
                "origin_col": None,
                "origin_row": None,
            }
        )

    # ---- 逐像元选源状态 -----------------------------------------------------
    mosaic = np.full((band_count, H, W), float(target.nodata), dtype=np.float64)
    filled = np.zeros((H, W), dtype=bool)
    source_idx = np.full((H, W), -1, dtype=np.int32)
    best_key = np.broadcast_to(_SENTINEL[:, None, None], (_KEY_COMPONENTS, H, W)).copy()
    best_values = np.full((band_count, H, W), np.nan, dtype=np.float64)
    second_idx = np.full((H, W), -1, dtype=np.int32)
    second_key = np.broadcast_to(_SENTINEL[:, None, None], (_KEY_COMPONENTS, H, W)).copy()
    second_values = np.full((band_count, H, W), np.nan, dtype=np.float64)
    overlap_multi = np.zeros((H, W), dtype=bool)

    for info in tile_infos:
        if info["excluded"]:
            continue
        tile = req.tiles[info["order"]]
        bands = info["bands"]
        _, th, tw = bands.shape

        col0 = _integer_offset(tile.origin.x, target.origin.x, sx, "x")
        row0 = _integer_offset(tile.origin.y, target.origin.y, sy, "y")
        info["origin_col"] = col0
        info["origin_row"] = row0

        r0, c0 = max(0, row0), max(0, col0)
        r1, c1 = min(H, row0 + th), min(W, col0 + tw)
        if r1 <= r0 or c1 <= c0:
            info["excluded"] = True
            info["reason"] = "切片范围与目标网格不相交"
            continue
        tr0, tc0 = r0 - row0, c0 - col0
        tr1 = tr0 + (r1 - r0)
        tc1 = tc0 + (c1 - c0)

        win_bands = bands[:, tr0:tr1, tc0:tc1]
        win_cloud = info["cloud"][tr0:tr1, tc0:tc1]
        win_edge = info["edge_dist"][tr0:tr1, tc0:tc1]

        valid = ~win_cloud
        for k in range(band_count):
            valid &= win_bands[k] != info["nodata"]
        if not valid.any():
            continue

        # 候选键：云量↑、时间差↑、边缘距离↓（取负）、切片序号↑
        cand_key = np.empty((_KEY_COMPONENTS, r1 - r0, c1 - c0), dtype=np.float64)
        cand_key[0] = info["cloud_fraction"]
        cand_key[1] = info["dt"]
        cand_key[2] = -win_edge
        cand_key[3] = float(info["order"])

        w_filled = filled[r0:r1, c0:c1]
        w_source = source_idx[r0:r1, c0:c1]
        w_best_key = best_key[:, r0:r1, c0:c1]
        w_second_key = second_key[:, r0:r1, c0:c1]

        cand_better, cand_equal = _lex_compare(cand_key, w_best_key)
        different = w_source != info["order"]
        first_place = valid & (~w_filled | (cand_better & different) | (cand_equal & different))
        # 等键也由“更小序号”决定：后续切片序号更大，cand_equal 理论上仅在
        # 同切片出现（不可能），这里保守处理为不抢占。
        first_place = valid & (~w_filled | (cand_better & different))
        loser = valid & w_filled & ~cand_better & different
        # 出现来自第二个切片的候选：无论胜负都构成重叠
        overlap_multi[r0:r1, c0:c1] |= valid & w_filled & different

        # 旧最优在被挤下前保留快照（用于次优比较）
        old_best_key = w_best_key.copy()
        old_best_idx = w_source.copy()
        old_best_values = best_values[:, r0:r1, c0:c1].copy()

        # ---- 写入新最优 ----
        for k in range(_KEY_COMPONENTS):
            w_best_key[k] = np.where(first_place, cand_key[k], w_best_key[k])
        for k in range(band_count):
            best_values[k, r0:r1, c0:c1] = np.where(
                first_place, win_bands[k], best_values[k, r0:r1, c0:c1]
            )
            mosaic[k, r0:r1, c0:c1] = np.where(
                first_place, win_bands[k], mosaic[k, r0:r1, c0:c1]
            )
        source_idx[r0:r1, c0:c1] = np.where(first_place, info["order"], w_source)
        filled[r0:r1, c0:c1] = w_filled | valid

        # ---- 次优维护（来自不同于当前最优的切片）----
        w_second_idx = second_idx[r0:r1, c0:c1]
        # a) 当前切片落败 -> 尝试以 cand_key 占据次优
        cand_beats_second, _ = _lex_compare(cand_key, w_second_key)
        take_second_loser = loser & cand_beats_second
        for k in range(_KEY_COMPONENTS):
            w_second_key[k] = np.where(take_second_loser, cand_key[k], w_second_key[k])
        for k in range(band_count):
            second_values[k, r0:r1, c0:c1] = np.where(
                take_second_loser, win_bands[k], second_values[k, r0:r1, c0:c1]
            )
        w_second_idx = np.where(take_second_loser, info["order"], w_second_idx)

        # b) 旧最优退位 -> 与现存次优比较后取较优者
        demoted = first_place & w_filled & (old_best_idx >= 0)
        old_beats_second, _ = _lex_compare(old_best_key, w_second_key)
        take_second_old = demoted & old_beats_second
        for k in range(_KEY_COMPONENTS):
            w_second_key[k] = np.where(take_second_old, old_best_key[k], w_second_key[k])
        for k in range(band_count):
            second_values[k, r0:r1, c0:c1] = np.where(
                take_second_old, old_best_values[k], second_values[k, r0:r1, c0:c1]
            )
        w_second_idx = np.where(take_second_old, old_best_idx, w_second_idx)
        second_idx[r0:r1, c0:c1] = w_second_idx

    # ---- 汇总 ---------------------------------------------------------------
    total = H * W
    covered = int(filled.sum())
    coverage = CoverageOut(
        covered_fraction=(covered / total) if total else 0.0,
        covered_pixels=covered,
        hole_pixels=total - covered,
        total_pixels=total,
        holes=_hole_components(filled),
    )

    tol = float(req.conflict_value_tolerance)
    with np.errstate(invalid="ignore"):
        diff = np.abs(best_values - second_values)
        disagree = np.any(diff > tol, axis=0) & (second_idx >= 0)
    conflicting = int(disagree.sum())
    multi_count = int(overlap_multi.sum())

    evidence: list[ConflictEvidenceOut] = []
    source_ids = [t.id for t in req.tiles]
    coords = np.argwhere(disagree)  # 行优先顺序，确定性
    for r, c in coords[: settings.max_conflict_evidence]:
        wi, si = int(source_idx[r, c]), int(second_idx[r, c])
        evidence.append(
            ConflictEvidenceOut(
                row=int(r),
                col=int(c),
                winner=source_ids[wi],
                runner_up=source_ids[si],
                winner_values=[float(v) for v in best_values[:, r, c]],
                runner_up_values=[float(v) for v in second_values[:, r, c]],
                abs_difference=[float(v) for v in diff[:, r, c]],
                acquired_at=_iso(tile_infos[wi]["time"]),
                runner_up_acquired_at=_iso(tile_infos[si]["time"]),
            )
        )
    overlap = OverlapOut(
        pixels_with_multiple_candidates=multi_count,
        conflicting_pixels=conflicting,
        evidence=evidence,
    )

    used_counts = [0 for _ in req.tiles]
    flat_idx, counts = np.unique(source_idx[filled], return_counts=True)
    for fi, cnt in zip((int(x) for x in flat_idx), (int(x) for x in counts), strict=True):
        used_counts[fi] = cnt
    tile_stats = [
        TileStatOut(
            tile_id=info["id"],
            acquired_at=_iso(info["time"]),
            used_pixels=used_counts[info["order"]],
            excluded=bool(info["excluded"]),
            exclude_reason=info["reason"],
            cloud_fraction=info["cloud_fraction"],
            origin_col=info["origin_col"] if info["origin_col"] is not None else 0,
            origin_row=info["origin_row"] if info["origin_row"] is not None else 0,
        )
        for info in tile_infos
    ]

    preview = make_preview_png(mosaic, filled, float(target.nodata), settings.preview_max_edge)
    source_png = make_source_map_png(source_idx, settings.preview_max_edge)

    return MosaicOut(
        id=uuid.uuid4().hex,
        created_at=_iso(datetime.now(timezone.utc)),
        target={
            "origin": target.origin.model_dump(),
            "width": W,
            "height": H,
            "pixel_size": list(target.pixel_size),
            "nodata": float(target.nodata),
        },
        max_time_delta_seconds=max_delta,
        reference_time=_iso(reference_time),
        bands=mosaic.tolist(),
        nodata=float(target.nodata),
        source_map=source_idx.tolist(),
        source_ids=source_ids,
        coverage=coverage,
        overlap=overlap,
        tiles=tile_stats,
        preview_png=preview,
        source_map_png=source_png,
        request_hash=request_hash,
        request_id=req.request_id,
    )
