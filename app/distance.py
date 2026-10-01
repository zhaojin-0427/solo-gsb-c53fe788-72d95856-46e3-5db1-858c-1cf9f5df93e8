"""Exact Euclidean distance transform (Felzenszwalb & Huttenlocher, 2004).

Given a 2-D boolean feature mask, returns for every feature pixel the
Euclidean distance (in ground units) to the nearest non-feature pixel. The
input is padded with a one-pixel non-feature ring, so edges of the feature
footprint count as boundaries too.

The transform is separable: an exact 1-D squared-distance transform is
applied along rows then columns. Each 1-D pass runs the classic
envelope-of-parabolas algorithm with its per-line envelope pointers advanced
in parallel with vectorised NumPy operations, so the Python-level loops are
only ``O(n)`` long while all arithmetic stays in C.
"""

from __future__ import annotations

import numpy as np


def _edt_1d_batch(f: np.ndarray) -> np.ndarray:
    """Exact 1-D squared EDT on the last axis for a batch of lines.

    Parameters
    ----------
    f:
        Array of shape ``(M, n)`` holding line costs (0 at feature sites,
        a finite positive value otherwise). Every line must contain at least
        one finite site.

    Returns
    -------
    ndarray
        ``(M, n)`` float32 array of squared distances to the cheapest site.

    Notes
    -----
    Classic envelope-of-parabolas (Felzenszwalb & Huttenlocher). Per-line
    envelope pointers are advanced in parallel, so the Python-level loops are
    ``O(n)`` long while the comparisons/arithmetic are vectorised across all
    lines. Intersections use the strict ``s < z[k]`` pop rule so integer tie
    positions are awarded to the later (farther-index) parabola.
    """
    f = f.astype(np.float64, copy=False)
    m, n = f.shape
    idx = np.arange(m)
    INF = np.inf

    v = np.zeros((m, n), dtype=np.int64)          # envelope sites per line
    z = np.empty((m, n + 1), dtype=np.float64)    # envelope boundaries
    k = np.full(m, -1, dtype=np.int64)            # last active site pointer
    z[:, 0] = -INF

    def intersect(lines: np.ndarray, q: int) -> np.ndarray:
        a = v[lines, k[lines]].astype(np.float64)
        bq = np.float64(q)
        fa = f[lines, v[lines, k[lines]]]
        return ((a * a + fa) - (bq * bq + f[lines, q])) / (2.0 * a - 2.0 * bq)

    for q in range(n):
        finite_lines = np.where(np.isfinite(f[:, q]))[0]
        # Pop sites while the new parabola overtakes strictly before z[k].
        pop = np.zeros(m, dtype=bool)
        pop[finite_lines] = k[finite_lines] >= 0
        while pop.any():
            lines = np.where(pop)[0]
            s = intersect(lines, q)
            bad = s < z[lines, k[lines]]
            pop_lines = lines[bad]
            k[pop_lines] -= 1
            pop = np.zeros(m, dtype=bool)
            still = pop_lines[k[pop_lines] >= 0]
            pop[still] = True
        s = np.full(m, -INF, dtype=np.float64)
        active = np.zeros(m, dtype=bool)
        active[finite_lines] = k[finite_lines] >= 0
        if active.any():
            lines = np.where(active)[0]
            s[lines] = intersect(lines, q)
        k[finite_lines] += 1
        v[finite_lines, k[finite_lines]] = q
        z[finite_lines, k[finite_lines]] = s[finite_lines]
        z[finite_lines, k[finite_lines] + 1] = INF

    # Evaluate: advance each line's pointer while scanning q.
    ptr = np.zeros(m, dtype=np.int64)
    out = np.full((m, n), np.inf, dtype=np.float32)
    live = k >= 0
    for q in range(n):
        # z[ptr+1] <= q  =>  move on (ties at integer points move to the
        # next parabola, matching the envelope construction above).
        while True:
            nxt = ptr + 1
            go = live & (nxt <= k) & (z[idx, nxt] <= q)
            if not go.any():
                break
            ptr[go] += 1
        sites = v[idx, ptr]
        dq = q - sites
        row = (dq.astype(np.float64) ** 2 + f[idx, sites]).astype(np.float32)
        out[live, q] = row[live]
    return out


def distance_to_mask(mask: np.ndarray, pixel_size: float = 1.0) -> np.ndarray:
    """Distance of every True pixel to the nearest False-region boundary.

    Parameters
    ----------
    mask:
        2-D boolean array; ``True`` = inside the feature (valid tile area).
    pixel_size:
        Multiplier converting pixel distance to ground units.

    Returns
    -------
    ndarray
        Float32 array of the same shape: ground distance to the nearest
        exterior pixel or footprint edge for feature pixels; 0 elsewhere.
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    out = np.zeros((h, w), dtype=np.float32)
    if not mask.any():
        return out

    # Non-feature ring around the grid => every 1-D line sees a finite site,
    # and footprint borders behave as true boundaries.
    padded = np.zeros((h + 2, w + 2), dtype=np.float32)
    padded[1:-1, 1:-1] = np.where(mask, np.inf, 0.0)

    d = _edt_1d_batch(padded)                  # along columns (x)
    d = _edt_1d_batch(d.T).T                   # along rows (y)
    d = np.sqrt(np.maximum(d, 0.0)).astype(np.float32)
    interior = d[1:-1, 1:-1]
    return np.where(mask, interior, 0.0).astype(np.float32) * float(pixel_size)
