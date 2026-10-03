from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.lod_pyramid import (  # noqa: E402
    DEFAULT_DOWNSAMPLE_FACTOR,
    MAX_LEVEL_COUNT,
    LODLevel,
    LODPyramid,
    _compute_fingerprint,
    _peak_downsample,
)


def _sine_signal(n: int, freq: float = 10.0, duration: float = 10.0):
    ts = np.linspace(0.0, duration, n, dtype=np.float64)
    vals = np.sin(2.0 * np.pi * freq * ts).astype(np.float64)
    return ts, vals


class TestPeakDownsample:
    def test_empty_input(self):
        ts = np.array([], dtype=np.float64)
        vals = np.array([], dtype=np.float64)
        out_ts, out_vals = _peak_downsample(ts, vals, 4)
        assert out_ts.shape == (0,)
        assert out_vals.shape == (0,)

    def test_output_length_full_buckets(self):
        n = 100
        ts = np.arange(n, dtype=np.float64)
        vals = np.arange(n, dtype=np.float64)
        out_ts, out_vals = _peak_downsample(ts, vals, 4)
        assert out_ts.shape == (50,)
        assert out_vals.shape == (50,)

    def test_output_length_with_tail(self):
        n = 102
        ts = np.arange(n, dtype=np.float64)
        vals = np.arange(n, dtype=np.float64)
        out_ts, out_vals = _peak_downsample(ts, vals, 4)
        assert out_ts.shape == (52,)
        assert out_vals.shape == (52,)

    def test_min_max_pairs_time_ordered(self):
        ts = np.arange(16, dtype=np.float64)
        vals = np.array(
            [1, 10, 2, 3, 4, 15, 5, 6, 7, 20, 8, 9, 11, 25, 12, 13],
            dtype=np.float64,
        )
        out_ts, out_vals = _peak_downsample(ts, vals, 4)
        assert out_vals.tolist() == [1, 10, 4, 15, 7, 20, 11, 25]
        assert out_ts.tolist() == [0, 1, 4, 5, 8, 9, 12, 13]

    def test_timestamps_non_decreasing(self):
        ts, vals = _sine_signal(1000)
        out_ts, _ = _peak_downsample(ts, vals, 4)
        assert np.all(np.diff(out_ts) >= 0)

    def test_preserves_global_min_max(self):
        ts, vals = _sine_signal(5000)
        out_ts, out_vals = _peak_downsample(ts, vals, 4)
        assert np.isclose(out_vals.min(), vals.min(), atol=1e-12)
        assert np.isclose(out_vals.max(), vals.max(), atol=1e-12)

    def test_tail_bucket_extrema(self):
        ts = np.arange(18, dtype=np.float64)
        vals = np.zeros(18, dtype=np.float64)
        vals[16] = 99.0
        vals[17] = -50.0
        out_ts, out_vals = _peak_downsample(ts, vals, 4)
        # Tail spans indices 16..17. Min=-50@17, Max=99@16 → first=16 (max), second=17 (min)
        assert out_ts[-2:].tolist() == [16.0, 17.0]
        assert out_vals[-2:].tolist() == [99.0, -50.0]

    def test_nan_bucket_preserves_finite_extrema(self):
        ts = np.arange(8, dtype=np.float64)
        vals = np.array(
            [np.nan, -18.397, np.nan, 101.066, 5.0, np.nan, 9.0, 7.0],
            dtype=np.float64,
        )
        out_ts, out_vals = _peak_downsample(ts, vals, 4)
        assert out_ts[:2].tolist() == [1.0, 3.0]
        assert out_vals[:2].tolist() == [-18.397, 101.066]
        assert out_ts[2:].tolist() == [4.0, 6.0]
        assert out_vals[2:].tolist() == [5.0, 9.0]

    def test_all_nan_bucket_preserves_time_extent(self):
        ts = np.arange(4, dtype=np.float64)
        vals = np.full(4, np.nan, dtype=np.float64)
        out_ts, out_vals = _peak_downsample(ts, vals, 4)
        assert out_ts.tolist() == [0.0, 3.0]
        assert np.isnan(out_vals).all()

    def test_nan_tail_preserves_finite_extrema(self):
        ts = np.arange(6, dtype=np.float64)
        vals = np.array([1.0, 2.0, 3.0, 4.0, np.nan, -20.0], dtype=np.float64)
        out_ts, out_vals = _peak_downsample(ts, vals, 4)
        assert out_ts[-2:].tolist() == [5.0, 5.0]
        assert out_vals[-2:].tolist() == [-20.0, -20.0]


class TestLODPyramidBuild:
    def test_small_array_single_level(self):
        ts = np.arange(5, dtype=np.float64)
        vals = np.arange(5, dtype=np.float64)
        pyramid = LODPyramid.build(ts, vals)
        assert pyramid.level_count == 1
        assert pyramid.levels[0].level == 0

    def test_medium_array_builds_multiple_levels(self):
        n = 10_000
        ts = np.arange(n, dtype=np.float64)
        vals = np.sin(ts * 0.01)
        pyramid = LODPyramid.build(ts, vals, level_point_floor=2000)
        assert pyramid.level_count >= 2
        counts = [lvl.point_count for lvl in pyramid.levels]
        for i in range(1, len(counts)):
            assert counts[i] < counts[i - 1], f"level count must strictly decrease; got {counts}"
        assert pyramid.levels[-1].point_count <= 2 * 2000

    def test_level_0_is_raw_reference(self):
        ts = np.arange(100, dtype=np.float64)
        vals = np.arange(100, dtype=np.float64)
        pyramid = LODPyramid.build(ts, vals)
        np.testing.assert_array_equal(pyramid.levels[0].timestamps, ts)
        np.testing.assert_array_equal(pyramid.levels[0].values, vals)

    def test_min_max_preserved_every_level(self):
        n = 50_000
        ts = np.arange(n, dtype=np.float64) * 0.001
        rng = np.random.default_rng(42)
        vals = rng.standard_normal(n).astype(np.float64)
        pyramid = LODPyramid.build(ts, vals)
        for lvl in pyramid.levels:
            assert np.isclose(lvl.values.min(), vals.min(), atol=1e-12), f"min broken at level {lvl.level}"
            assert np.isclose(lvl.values.max(), vals.max(), atol=1e-12), f"max broken at level {lvl.level}"

    def test_finite_min_max_preserved_every_level_with_nans(self):
        n = 50_000
        ts = np.arange(n, dtype=np.float64) * 0.001
        vals = np.linspace(3.0, 103.0, n, dtype=np.float64)
        vals[::7] = np.nan
        vals[25_001] = -18.397
        pyramid = LODPyramid.build(ts, vals)
        expected_min = np.nanmin(vals)
        expected_max = np.nanmax(vals)
        for lvl in pyramid.levels:
            assert np.isclose(np.nanmin(lvl.values), expected_min, atol=1e-12), (
                f"finite min broken at level {lvl.level}"
            )
            assert np.isclose(np.nanmax(lvl.values), expected_max, atol=1e-12), (
                f"finite max broken at level {lvl.level}"
            )

    def test_monotonic_timestamps_every_level(self):
        ts, vals = _sine_signal(20_000)
        pyramid = LODPyramid.build(ts, vals)
        for lvl in pyramid.levels:
            assert np.all(np.diff(lvl.timestamps) >= 0), f"non-monotonic ts at level {lvl.level}"

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            LODPyramid.build(
                np.arange(5, dtype=np.float64),
                np.arange(6, dtype=np.float64),
            )

    def test_non_1d_input_raises(self):
        ts = np.zeros((10, 2), dtype=np.float64)
        vals = np.zeros((10, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="1-D"):
            LODPyramid.build(ts, vals)

    def test_downsample_factor_floor(self):
        ts = np.arange(100, dtype=np.float64)
        vals = np.arange(100, dtype=np.float64)
        with pytest.raises(ValueError, match="downsample_factor"):
            LODPyramid.build(ts, vals, downsample_factor=1)

    def test_level_count_capped(self):
        n = 100_000
        ts = np.arange(n, dtype=np.float64)
        vals = np.sin(ts * 0.0001)
        pyramid = LODPyramid.build(ts, vals, level_point_floor=1)
        assert pyramid.level_count <= MAX_LEVEL_COUNT

    def test_integer_input_coerced_to_float64(self):
        ts = np.arange(1000, dtype=np.int64)
        vals = np.arange(1000, dtype=np.int32)
        pyramid = LODPyramid.build(ts, vals)
        assert pyramid.levels[0].timestamps.dtype == np.float64
        assert pyramid.levels[0].values.dtype == np.float64


class TestPickLevel:
    def test_full_range_picks_coarse(self):
        n = 50_000
        ts = np.arange(n, dtype=np.float64) * 0.001
        vals = np.sin(ts)
        pyramid = LODPyramid.build(ts, vals, target_points_per_view=1000)
        lvl = pyramid.pick_level(float(ts[0]), float(ts[-1]))
        assert lvl >= 1, "full range over 50K must not use raw level 0"

    def test_tight_range_picks_level_0(self):
        n = 50_000
        ts = np.arange(n, dtype=np.float64) * 0.001
        vals = np.sin(ts)
        pyramid = LODPyramid.build(ts, vals, target_points_per_view=1000)
        lvl = pyramid.pick_level(float(ts[100]), float(ts[200]))
        assert lvl == 0

    def test_picked_level_visible_within_budget(self):
        n = 100_000
        ts = np.arange(n, dtype=np.float64) * 0.001
        vals = np.sin(ts)
        target = 1500
        pyramid = LODPyramid.build(ts, vals, target_points_per_view=target)
        lvl_idx = pyramid.pick_level(float(ts[0]), float(ts[-1]))
        level = pyramid.levels[lvl_idx]
        i_lo = int(np.searchsorted(level.timestamps, float(ts[0]), side="left"))
        i_hi = int(np.searchsorted(level.timestamps, float(ts[-1]), side="right"))
        visible = i_hi - i_lo
        budget = target if lvl_idx == 0 else 2 * target
        if lvl_idx != pyramid.levels[-1].level:
            assert visible <= budget

    def test_invalid_range_returns_coarsest(self):
        n = 1000
        ts = np.arange(n, dtype=np.float64)
        vals = np.sin(ts * 0.01)
        pyramid = LODPyramid.build(ts, vals)
        lvl = pyramid.pick_level(5.0, 5.0)
        assert lvl == pyramid.levels[-1].level


class TestSlice:
    def test_slice_covers_requested_range(self):
        n = 10_000
        ts = np.arange(n, dtype=np.float64) * 0.01
        vals = np.sin(ts)
        pyramid = LODPyramid.build(ts, vals)
        out_ts, out_vals = pyramid.slice(0, 10.0, 20.0)
        assert out_ts.shape == out_vals.shape
        assert out_ts.shape[0] > 0
        assert out_ts[0] <= 10.0
        assert out_ts[-1] >= 20.0

    def test_slice_shape_consistency_all_levels(self):
        n = 20_000
        ts = np.arange(n, dtype=np.float64) * 0.001
        vals = np.sin(ts)
        pyramid = LODPyramid.build(ts, vals)
        for lvl_idx in range(pyramid.level_count):
            out_ts, out_vals = pyramid.slice(lvl_idx, 2.0, 5.0)
            assert out_ts.shape == out_vals.shape

    def test_slice_level_out_of_bounds_clamped(self):
        ts = np.arange(100, dtype=np.float64)
        vals = np.arange(100, dtype=np.float64)
        pyramid = LODPyramid.build(ts, vals)
        out_ts, _ = pyramid.slice(999, 0.0, 100.0)
        assert out_ts.shape[0] > 0

    def test_slice_out_of_data_range_empty(self):
        ts = np.arange(100, dtype=np.float64)
        vals = np.arange(100, dtype=np.float64)
        pyramid = LODPyramid.build(ts, vals)
        out_ts, out_vals = pyramid.slice(0, 1000.0, 2000.0)
        assert out_ts.shape[0] <= 1
        assert out_vals.shape[0] == out_ts.shape[0]


class TestFingerprintAndMatches:
    def test_identical_arrays_same_fingerprint(self):
        ts = np.arange(100, dtype=np.float64)
        vals = np.sin(ts)
        f1 = _compute_fingerprint(ts, vals)
        f2 = _compute_fingerprint(ts.copy(), vals.copy())
        assert f1 == f2

    def test_length_difference_changes_fingerprint(self):
        ts1 = np.arange(100, dtype=np.float64)
        ts2 = np.arange(101, dtype=np.float64)
        vals1 = np.sin(ts1)
        vals2 = np.sin(ts2)
        assert _compute_fingerprint(ts1, vals1) != _compute_fingerprint(ts2, vals2)

    def test_value_difference_changes_fingerprint(self):
        ts = np.arange(100, dtype=np.float64)
        vals1 = np.sin(ts * 0.01)
        vals2 = vals1.copy()
        vals2[50] += 10.0
        assert _compute_fingerprint(ts, vals1) != _compute_fingerprint(ts, vals2)

    def test_matches_returns_true_for_source(self):
        ts = np.arange(1000, dtype=np.float64)
        vals = np.sin(ts * 0.01)
        pyramid = LODPyramid.build(ts, vals)
        assert pyramid.matches(ts, vals)

    def test_matches_returns_false_after_mutation(self):
        ts = np.arange(1000, dtype=np.float64)
        vals = np.sin(ts * 0.01)
        pyramid = LODPyramid.build(ts, vals)
        mutated = vals.copy()
        mutated[500] = 999.0
        assert not pyramid.matches(ts, mutated)

    def test_empty_fingerprint_zero(self):
        empty = np.array([], dtype=np.float64)
        assert _compute_fingerprint(empty, empty) == 0


class TestEdgeCases:
    def test_empty_pyramid(self):
        empty = np.array([], dtype=np.float64)
        pyramid = LODPyramid.build(empty, empty)
        assert pyramid.level_count == 1
        assert pyramid.levels[0].point_count == 0
        out_ts, out_vals = pyramid.slice(0, 0.0, 1.0)
        assert out_ts.shape == (0,)
        assert out_vals.shape == (0,)

    def test_single_element(self):
        ts = np.array([1.0])
        vals = np.array([5.0])
        pyramid = LODPyramid.build(ts, vals)
        assert pyramid.level_count == 1

    def test_all_equal_values(self):
        n = 1000
        ts = np.arange(n, dtype=np.float64)
        vals = np.full(n, 42.0)
        pyramid = LODPyramid.build(ts, vals)
        for lvl in pyramid.levels:
            assert np.all(lvl.values == 42.0)

    def test_default_downsample_factor_constant(self):
        assert DEFAULT_DOWNSAMPLE_FACTOR >= 2
