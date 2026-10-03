"""Shared coordinate and point-budget helpers for 2D/3D flight paths.

Every rendered flight path uses an ENU world frame:

* X = East
* Y = North
* Z = Up

Keeping the conversion here prevents drag/drop, automatic panels, and layout
restore from silently choosing different axis orders.  The downsampling helper
returns source indices so timestamps and other per-sample metadata can be
sliced with exactly the same selection as the path coordinates.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence

import numpy as np


PATH_COORDINATE_FRAME = "ENU"
PATH_AXIS_NAMES = ("East", "North", "Up")
EARTH_RADIUS_M = 6_378_137.0
DEFAULT_2D_PATH_MAX_POINTS = 8_000
DEFAULT_3D_PATH_MAX_POINTS = 5_000


def _as_equal_1d_arrays(values: Sequence[object], *, names: Sequence[str]) -> list[np.ndarray]:
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    if any(array.ndim != 1 for array in arrays):
        raise ValueError(f"{', '.join(names)} must be one-dimensional")
    lengths = {len(array) for array in arrays}
    if len(lengths) > 1:
        raise ValueError(f"{', '.join(names)} must have equal lengths")
    return arrays


def local_ned_to_enu(north, east, down, *, origin_index: int = 0):
    """Convert a PX4 local NED path to origin-relative ENU coordinates."""

    north_arr, east_arr, down_arr = _as_equal_1d_arrays(
        (north, east, down), names=("north", "east", "down")
    )
    if len(north_arr) == 0:
        return east_arr.copy(), north_arr.copy(), down_arr.copy()
    if not 0 <= int(origin_index) < len(north_arr):
        raise IndexError("origin_index is outside the path")
    origin_index = int(origin_index)
    east_enu = east_arr - float(east_arr[origin_index])
    north_enu = north_arr - float(north_arr[origin_index])
    up_enu = -(down_arr - float(down_arr[origin_index]))
    return east_enu, north_enu, up_enu


def global_lla_to_enu(latitude_deg, longitude_deg, altitude_m, *, origin_index: int = 0):
    """Approximate a WGS84 latitude/longitude/altitude path as local ENU.

    This intentionally retains the application's existing equirectangular
    projection for short flight paths while making its output-axis contract
    explicit.  Longitude deltas are wrapped across the antimeridian.
    """

    lat, lon, alt = _as_equal_1d_arrays(
        (latitude_deg, longitude_deg, altitude_m),
        names=("latitude_deg", "longitude_deg", "altitude_m"),
    )
    if len(lat) == 0:
        return lon.copy(), lat.copy(), alt.copy()
    if not 0 <= int(origin_index) < len(lat):
        raise IndexError("origin_index is outside the path")
    origin_index = int(origin_index)
    lat0 = float(lat[origin_index])
    lon0 = float(lon[origin_index])
    alt0 = float(alt[origin_index])
    deg_to_rad = np.pi / 180.0
    delta_lon = (lon - lon0 + 180.0) % 360.0 - 180.0
    east = delta_lon * deg_to_rad * EARTH_RADIUS_M * np.cos(lat0 * deg_to_rad)
    north = (lat - lat0) * deg_to_rad * EARTH_RADIUS_M
    up = alt - alt0
    return east, north, up


def select_path_indices(*coordinates, max_points: int) -> np.ndarray:
    """Select a bounded, ordered subset of path samples.

    The first and last samples and each coordinate's global finite minimum and
    maximum are mandatory.  A small share of the remaining budget is assigned
    to the sharpest turns; the rest repeatedly bisects the largest uncovered
    index interval to retain coverage across the full flight.  The result is
    sorted and never exceeds ``max_points``.

    A limit too small to contain all mandatory samples raises ``ValueError``
    instead of silently dropping an endpoint or extremum.
    """

    if not coordinates:
        raise ValueError("at least one coordinate array is required")
    try:
        max_points = int(max_points)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_points must be an integer") from exc
    if max_points < 2:
        raise ValueError("max_points must be at least 2")

    names = tuple(f"coordinate_{idx}" for idx in range(len(coordinates)))
    arrays = _as_equal_1d_arrays(coordinates, names=names)
    count = len(arrays[0])
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    if count == 0:
        return np.empty(0, dtype=np.int64)

    mandatory = {0, count - 1}
    for array in arrays:
        finite = np.flatnonzero(np.isfinite(array))
        if finite.size == 0:
            continue
        finite_values = array[finite]
        mandatory.add(int(finite[int(np.argmin(finite_values))]))
        mandatory.add(int(finite[int(np.argmax(finite_values))]))

    if len(mandatory) > max_points:
        raise ValueError(
            f"max_points={max_points} cannot preserve {len(mandatory)} "
            "endpoint/extremum samples"
        )

    selected = set(mandatory)
    remaining = max_points - len(selected)

    # Preserve the most significant direction changes before filling temporal
    # coverage.  Angle is scale-independent because ENU dimensions use meters.
    if remaining > 0 and count >= 3:
        points = np.column_stack(arrays)
        incoming = points[1:-1] - points[:-2]
        outgoing = points[2:] - points[1:-1]
        incoming_norm = np.linalg.norm(incoming, axis=1)
        outgoing_norm = np.linalg.norm(outgoing, axis=1)
        denom = incoming_norm * outgoing_norm
        valid = np.isfinite(denom) & (denom > np.finfo(np.float64).eps)
        turn_score = np.full(count - 2, -np.inf, dtype=np.float64)
        if np.any(valid):
            cosine = np.sum(incoming[valid] * outgoing[valid], axis=1) / denom[valid]
            turn_score[valid] = 1.0 - np.clip(cosine, -1.0, 1.0)

        candidate_local = np.flatnonzero(turn_score > 1e-9)
        if candidate_local.size:
            candidate_indices = candidate_local + 1
            candidate_scores = turn_score[candidate_local]
            turn_budget = min(remaining, max(1, max_points // 10))
            # Partition before sorting so million-sample paths do not pay for
            # a full O(n log n) ordering just to retain a few hundred turns.
            candidate_limit = min(
                candidate_local.size,
                turn_budget + len(selected),
            )
            if candidate_limit < candidate_local.size:
                partition = np.argpartition(-candidate_scores, candidate_limit - 1)[:candidate_limit]
                candidate_indices = candidate_indices[partition]
                candidate_scores = candidate_scores[partition]
            # Deterministic within the retained candidates: descending score,
            # then ascending source index.
            order = np.lexsort((candidate_indices, -candidate_scores))
            added = 0
            for candidate in candidate_indices[order]:
                candidate = int(candidate)
                if candidate in selected:
                    continue
                selected.add(candidate)
                added += 1
                if added >= turn_budget:
                    break

    # Fill the point budget by repeatedly bisecting the largest uncovered
    # source-index interval.  This avoids clustering all spare points around
    # extrema/turns and retains deterministic full-flight coverage.
    interval_heap: list[tuple[int, int, int]] = []
    ordered = sorted(selected)
    for left, right in zip(ordered, ordered[1:]):
        if right - left > 1:
            heapq.heappush(interval_heap, (-(right - left), left, right))

    while len(selected) < max_points and interval_heap:
        _negative_gap, left, right = heapq.heappop(interval_heap)
        middle = (left + right) // 2
        if middle <= left or middle >= right or middle in selected:
            continue
        selected.add(middle)
        if middle - left > 1:
            heapq.heappush(interval_heap, (-(middle - left), left, middle))
        if right - middle > 1:
            heapq.heappush(interval_heap, (-(right - middle), middle, right))

    return np.asarray(sorted(selected), dtype=np.int64)
