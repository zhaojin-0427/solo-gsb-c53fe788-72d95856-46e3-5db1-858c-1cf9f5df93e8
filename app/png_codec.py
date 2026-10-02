"""Pillow / NumPy 编解码工具。

* 输入切片：JSON 嵌套数组 <-> base64 PNG 两种等价编码；
* 输出产物：真彩色预览 PNG（无数据透明）、逐像元来源索引着色 PNG。

多波段 PNG 的通道约定：L=1、LA=2、RGB=3、RGBA=4，按 R,G,B,A 顺序映射波段。
``float32`` 或超过 4 个波段时请使用 JSON 数组编码（PNG 无法无损表达这些情况）。
"""
from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image

from .models import TileIn

_MODE_CHANNELS = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}


def _b64_to_image(data: str) -> Image.Image:
    raw = base64.b64decode(data, validate=True)
    return Image.open(io.BytesIO(raw))


def _image_to_array(img: Image.Image, dtype: str) -> np.ndarray:
    """返回形状 [band, height, width] 的数组。"""
    if dtype == "float32":
        raise ValueError("PNG 编码不支持 float32，请改用 bands JSON 数组")
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        arr = np.moveaxis(arr, 2, 0)
    else:
        raise ValueError(f"无法解析的 PNG 维度: {arr.shape}")
    # Pillow 的 I;16 可能以大端字节序暴露，统一转回目标 dtype
    return arr.astype(_DTYPES[dtype], copy=False)


_DTYPES = {"uint8": np.uint8, "uint16": np.uint16, "float32": np.float32}


def decode_bands(tile: TileIn) -> np.ndarray:
    """解码波段矩阵为 [band, height, width] 数组。"""
    if tile.bands is not None:
        try:
            arr = np.asarray(tile.bands, dtype=_DTYPES[tile.bands_dtype])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"tile {tile.id} 的 bands 不是规则的三维数组: {exc}") from exc
        if arr.ndim != 3:
            raise ValueError(
                f"tile {tile.id} 的 bands 形状应为 [band, height, width]，实际维度 {arr.ndim}"
            )
        return arr
    assert tile.bands_png is not None
    try:
        img = _b64_to_image(tile.bands_png)
        img.load()
    except Exception as exc:  # Pillow raises various errors
        raise ValueError(f"tile {tile.id} 的 bands_png 无法解码: {exc}") from exc
    if img.mode not in _MODE_CHANNELS:
        raise ValueError(
            f"tile {tile.id} 的 bands_png 模式 {img.mode} 不受支持，"
            "请使用 L/LA/RGB/RGBA（分别对应 1/2/3/4 波段）"
        )
    arr = _image_to_array(img, tile.bands_dtype)
    return arr


def decode_cloud(tile: TileIn, height: int, width: int) -> np.ndarray:
    """解码云掩膜为形状 [height, width] 的 bool 数组（True=云）。"""
    if tile.cloud_mask is not None:
        try:
            arr = np.asarray(tile.cloud_mask, dtype=np.uint8)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"tile {tile.id} 的 cloud_mask 不是规则二维数组: {exc}") from exc
    else:
        assert tile.cloud_mask_png is not None
        try:
            img = _b64_to_image(tile.cloud_mask_png)
            img.load()
        except Exception as exc:
            raise ValueError(f"tile {tile.id} 的 cloud_mask_png 无法解码: {exc}") from exc
        if img.mode != "L":
            img = img.convert("L")
        arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape != (height, width):
        raise ValueError(
            f"tile {tile.id} 的 cloud_mask 形状 {arr.shape} 与波段矩阵 ({height}, {width}) 不一致"
        )
    return arr > 0


# ------------------------------------------------------------------ 输出 PNG


def _png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def make_preview_png(
    mosaic: np.ndarray,
    filled: np.ndarray,
    nodata: float,
    max_edge: int,
) -> str:
    """生成彩色预览 data URL。

    3+ 波段取前 3 波段做 2% 线性拉伸；1/2 波段取首波段灰度复制为 RGB；
    未填充/无数据像元透明。
    """
    b, h, w = mosaic.shape
    if b >= 3:
        rgb = np.stack([mosaic[0], mosaic[1], mosaic[2]], axis=-1).astype(np.float64)
    else:
        gray = mosaic[0].astype(np.float64)
        rgb = np.stack([gray, gray, gray], axis=-1)

    mask = filled
    out = np.zeros((h, w, 4), dtype=np.uint8)
    if mask.any():
        for k in range(3):
            ch = rgb[:, :, k][mask]
            lo, hi = np.percentile(ch, 2.0), np.percentile(ch, 98.0)
            if hi > lo:
                scaled = np.clip((rgb[:, :, k] - lo) / (hi - lo), 0.0, 1.0)
            else:
                scaled = np.zeros_like(rgb[:, :, k])
            out[:, :, k] = (scaled * 255.0 + 0.5).astype(np.uint8)
        out[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)

    img = Image.fromarray(out, mode="RGBA")
    if max(h, w) > max_edge:
        scale = max_edge / max(h, w)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.NEAREST)
    return _png_data_url(img)


# 最多 16 个时相的确定色调（RGB），-1 空洞为透明
_SOURCE_COLORS = np.array(
    [
        (228, 26, 28),
        (55, 126, 184),
        (77, 175, 74),
        (152, 78, 163),
        (255, 127, 0),
        (255, 255, 51),
        (166, 86, 40),
        (247, 129, 191),
        (153, 153, 153),
        (0, 190, 190),
        (102, 51, 153),
        (200, 200, 0),
        (0, 102, 51),
        (255, 178, 102),
        (128, 0, 64),
        (0, 0, 128),
    ],
    dtype=np.uint8,
)


def make_source_map_png(source_map: np.ndarray, max_edge: int) -> str:
    """逐像元来源索引着色（-1/未填充为透明）。"""
    h, w = source_map.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    valid = source_map >= 0
    if valid.any():
        idx = np.clip(source_map, 0, len(_SOURCE_COLORS) - 1)
        out[:, :, :3] = _SOURCE_COLORS[idx]
        out[:, :, 3] = np.where(valid, 255, 0).astype(np.uint8)
    img = Image.fromarray(out, mode="RGBA")
    if max(h, w) > max_edge:
        scale = max_edge / max(h, w)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.NEAREST)
    return _png_data_url(img)
