from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


DEFAULT_TARGET_POINTS_PER_VIEW = 1000
DEFAULT_LEVEL_POINT_FLOOR = 2000
DEFAULT_DOWNSAMPLE_FACTOR = 4
MAX_LEVEL_COUNT = 12


@dataclass(frozen=True)
class LODLevel:
    """One level of the peak-preserving LOD pyramid.

    level == 0 is a reference to the raw (ts, values) arrays. level >= 1 stores
    min/max pairs interleaved in time order: for each bucket the two samples
    carrying the bucket's extrema are emitted back-to-back so a single polyline
    renders an envelope (upper + lower) without additional draw calls.
    """

    level: int
    timestamps: np.ndarray
    values: np.ndarray

    @property
    def point_count(self) -> int:
        return int(self.timestamps.shape[0])


@dataclass
class LODPyramid:
    levels: List[LODLevel]
    source_row_count: int
    source_fingerprint: int
    target_points_per_view: int = DEFAULT_TARGET_POINTS_PER_VIEW

    @classmethod
    def build(
        cls,
        timestamps: np.ndarray,
        values: np.ndarray,
        *,
        target_points_per_view: int = DEFAULT_TARGET_POINTS_PER_VIEW,
        level_point_floor: int = DEFAULT_LEVEL_POINT_FLOOR,
        downsample_factor: int = DEFAULT_DOWNSAMPLE_FACTOR,
    ) -> "LODPyramid":
        if timestamps.shape != values.shape:
            raise ValueError(
                f"timestamps and values must share shape; got {timestamps.shape} vs {values.shape}"
            )
        if timestamps.ndim != 1:
            raise ValueError(f"LODPyramid requires 1-D arrays; got ndim={timestamps.ndim}")
        if downsample_factor < 2:
            raise ValueError(f"downsample_factor must be >= 2; got {downsample_factor}")

        ts64 = np.ascontiguousarray(timestamps, dtype=np.float64)
        vals64 = np.ascontiguousarray(values, dtype=np.float64)

        levels: List[LODLevel] = [LODLevel(level=0, timestamps=ts64, values=vals64)]

        current_ts = ts64
        current_vals = vals64
        while len(levels) < MAX_LEVEL_COUNT:
            if current_ts.shape[0] < downsample_factor * 2:
                break
            next_ts, next_vals = _peak_downsample(current_ts, current_vals, downsample_factor)
            if next_ts.shape[0] >= current_ts.shape[0]:
                break
            levels.append(LODLevel(level=len(levels), timestamps=next_ts, values=next_vals))
            if next_ts.shape[0] <= level_point_floor:
                break
            current_ts = next_ts
            current_vals = next_vals

        return cls(
            levels=levels,
            source_row_count=int(timestamps.shape[0]),
            source_fingerprint=_compute_fingerprint(timestamps, values),
            target_points_per_view=int(target_points_per_view),
        )

    def pick_level(
        self,
        t_start: float,
        t_end: float,
        target_points: Optional[int] = None,
    ) -> int:
        if not self.levels:
            return 0
        if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
            return self.levels[-1].level
        raw_budget = int(target_points) if target_points is not None else self.target_points_per_view
        raw_budget = max(1, raw_budget)
        lod_budget = 2 * raw_budget
        for level in self.levels:
            ts = level.timestamps
            if ts.shape[0] == 0:
                continue
            i_lo = int(np.searchsorted(ts, t_start, side="left"))
            i_hi = int(np.searchsorted(ts, t_end, side="right"))
            visible = i_hi - i_lo
            budget = raw_budget if level.level == 0 else lod_budget
            if visible <= budget:
                return level.level
        return self.levels[-1].level

    def slice(
        self,
        level_idx: int,
        t_start: float,
        t_end: float,
        *,
        context_points: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.levels:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty
        level_idx = max(0, min(level_idx, len(self.levels) - 1))
        level = self.levels[level_idx]
        ts = level.timestamps
        if ts.shape[0] == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty
        i_lo = int(np.searchsorted(ts, t_start, side="left")) - context_points
        i_hi = int(np.searchsorted(ts, t_end, side="right")) + context_points
        i_lo = max(0, i_lo)
        i_hi = min(ts.shape[0], i_hi)
        if i_hi <= i_lo:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty
        return ts[i_lo:i_hi], level.values[i_lo:i_hi]

    def matches(self, timestamps: np.ndarray, values: np.ndarray) -> bool:
        return (
            self.source_row_count == int(timestamps.shape[0])
            and self.source_fingerprint == _compute_fingerprint(timestamps, values)
        )

    @property
    def level_count(self) -> int:
        return len(self.levels)

    @property
    def total_points(self) -> int:
        return sum(level.point_count for level in self.levels)


def _peak_downsample(
    timestamps: np.ndarray,
    values: np.ndarray,
    factor: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = values.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    num_buckets = (n + factor - 1) // factor
    full_count = n // factor

    out_ts = np.empty(2 * num_buckets, dtype=np.float64)
    out_vals = np.empty(2 * num_buckets, dtype=np.float64)

    if full_count > 0:
        body_len = full_count * factor
        reshaped_vals = values[:body_len].reshape(full_count, factor)
        nan_mask = np.isnan(reshaped_vals)
        if nan_mask.any():
            # ``ndarray.argmin/argmax`` can select a NaN, which discards a
            # real extremum from that bucket.  That makes spikes disappear in
            # a coarse/full-range view and reappear only after zooming.  Reuse
            # one work buffer for both reductions to keep the hot path compact.
            work_vals = reshaped_vals.copy()
            all_nan = nan_mask.all(axis=1)
            work_vals[nan_mask] = np.inf
            argmin_local = work_vals.argmin(axis=1)
            work_vals[nan_mask] = -np.inf
            argmax_local = work_vals.argmax(axis=1)

            # Preserve the time extent of an entirely missing bucket.  Two
            # NaNs are intentional: the fixed two-points-per-bucket shape is
            # part of the pyramid's level-size contract.
            if all_nan.any():
                argmin_local[all_nan] = 0
                argmax_local[all_nan] = factor - 1
        else:
            argmin_local = reshaped_vals.argmin(axis=1)
            argmax_local = reshaped_vals.argmax(axis=1)
        bucket_starts = np.arange(0, body_len, factor, dtype=np.int64)
        min_idx = bucket_starts + argmin_local
        max_idx = bucket_starts + argmax_local
        first_idx = np.minimum(min_idx, max_idx)
        second_idx = np.maximum(min_idx, max_idx)
        out_ts[: 2 * full_count : 2] = timestamps[first_idx]
        out_ts[1 : 2 * full_count : 2] = timestamps[second_idx]
        out_vals[: 2 * full_count : 2] = values[first_idx]
        out_vals[1 : 2 * full_count : 2] = values[second_idx]

    if full_count < num_buckets:
        tail_offset = full_count * factor
        tail_vals = values[tail_offset:]
        tail_ts = timestamps[tail_offset:]
        valid_local = np.flatnonzero(~np.isnan(tail_vals))
        if valid_local.size:
            valid_vals = tail_vals[valid_local]
            tmin_local = int(valid_local[int(valid_vals.argmin())])
            tmax_local = int(valid_local[int(valid_vals.argmax())])
        else:
            tmin_local = 0
            tmax_local = tail_vals.shape[0] - 1
        first_local = min(tmin_local, tmax_local)
        second_local = max(tmin_local, tmax_local)
        out_ts[2 * full_count] = tail_ts[first_local]
        out_ts[2 * full_count + 1] = tail_ts[second_local]
        out_vals[2 * full_count] = tail_vals[first_local]
        out_vals[2 * full_count + 1] = tail_vals[second_local]

    return out_ts, out_vals


def _compute_fingerprint(timestamps: np.ndarray, values: np.ndarray) -> int:
    n = int(timestamps.shape[0])
    if n == 0:
        return 0
    return hash(
        (
            n,
            float(timestamps[0]),
            float(timestamps[-1]),
            float(values[0]),
            float(values[-1]),
            float(values[n // 2]),
        )
    )
