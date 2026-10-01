"""Pillow-based preview rendering for mosaic results."""

from __future__ import annotations

import io
from typing import Any, Optional

import numpy as np
from PIL import Image


def render_preview(result: dict[str, Any], max_width: int = 512) -> bytes:
    """Render band 0 as a grayscale PNG; uncovered holes are tinted red.

    Scaling is a deterministic min/max linear stretch. The image is
    nearest-neighbour downscaled when wider than ``max_width``.
    """
    mosaic = np.asarray(result["mosaic"][0], dtype=np.float32)
    hole = np.asarray(result["hole_mask"], dtype=bool)
    h, w = mosaic.shape

    valid = ~np.isnan(mosaic)
    if valid.any():
        vmin = float(np.nanmin(mosaic))
        vmax = float(np.nanmax(mosaic))
    else:
        vmin, vmax = 0.0, 1.0
    span = vmax - vmin if vmax > vmin else 1.0

    gray = np.zeros((h, w), dtype=np.uint8)
    gray[valid] = np.clip(
        np.round((mosaic[valid] - vmin) / span * 255.0), 0, 255
    ).astype(np.uint8)

    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    rgb[hole] = np.array([220, 40, 40], dtype=np.uint8)

    img = Image.fromarray(rgb, mode="RGB")
    if w > max_width:
        scale = max_width / w
        img = img.resize((max_width, max(1, round(h * scale))), Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()
