from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.path_geometry import (  # noqa: E402
    PATH_AXIS_NAMES,
    PATH_COORDINATE_FRAME,
    global_lla_to_enu,
    local_ned_to_enu,
    select_path_indices,
)


def test_local_ned_conversion_obeys_shared_enu_contract():
    north = np.array([100.0, 101.0, 104.0])
    east = np.array([200.0, 202.0, 203.0])
    down = np.array([-5.0, -8.0, -6.0])

    x, y, z = local_ned_to_enu(north, east, down)

    assert PATH_COORDINATE_FRAME == "ENU"
    assert PATH_AXIS_NAMES == ("East", "North", "Up")
    np.testing.assert_allclose(x, [0.0, 2.0, 3.0])
    np.testing.assert_allclose(y, [0.0, 1.0, 4.0])
    np.testing.assert_allclose(z, [0.0, 3.0, 1.0])


def test_global_lla_conversion_obeys_same_enu_axis_order():
    latitude = np.array([37.0, 37.0001, 37.0001])
    longitude = np.array([127.0, 127.0, 127.0001])
    altitude = np.array([50.0, 53.0, 55.0])

    x, y, z = global_lla_to_enu(latitude, longitude, altitude)

    assert x[1] == pytest.approx(0.0, abs=1e-9)
    assert y[1] > 11.0  # latitude change is North/Y
    assert x[2] > 8.0   # longitude change is East/X at 37 degrees
    assert y[2] == pytest.approx(y[1])
    np.testing.assert_allclose(z, [0.0, 3.0, 5.0])


@pytest.mark.parametrize("count, max_points", [(8_001, 8_000), (9_999, 5_000)])
def test_path_selection_never_exceeds_point_budget(count, max_points):
    x = np.linspace(0.0, 100.0, count)
    y = np.sin(x)

    indices = select_path_indices(x, y, max_points=max_points)

    assert len(indices) == max_points
    assert indices[0] == 0
    assert indices[-1] == count - 1
    assert np.all(np.diff(indices) > 0)


def test_path_selection_preserves_endpoints_and_all_coordinate_extrema():
    count = 20_000
    x = np.linspace(-2.0, 2.0, count)
    y = np.cos(np.linspace(0.0, 20.0, count))
    z = np.sin(np.linspace(0.0, 30.0, count))
    x[123] = -999.0
    x[456] = 999.0
    y[789] = -888.0
    y[1_234] = 888.0
    z[4_321] = -777.0
    z[9_876] = 777.0

    indices = select_path_indices(x, y, z, max_points=128)
    selected = set(indices.tolist())

    expected = {
        0,
        count - 1,
        int(np.argmin(x)),
        int(np.argmax(x)),
        int(np.argmin(y)),
        int(np.argmax(y)),
        int(np.argmin(z)),
        int(np.argmax(z)),
    }
    assert expected <= selected
    assert len(indices) <= 128


def test_path_selection_prioritizes_a_sharp_non_extreme_turn():
    count = 1_001
    x = np.arange(count, dtype=np.float64)
    y = 100.0 * np.sin(2.0 * np.pi * x / (count - 1))
    # A small one-sample kink has a much sharper angle than the smooth global
    # extrema, while not itself being a coordinate minimum or maximum.
    y[500] += 1.0

    indices = select_path_indices(x, y, max_points=32)

    assert 500 in set(indices.tolist())


def test_path_selection_returns_all_indices_when_under_budget():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([2.0, 1.0, 0.0])
    np.testing.assert_array_equal(
        select_path_indices(x, y, max_points=10),
        np.arange(3),
    )


def test_path_selection_rejects_budget_that_cannot_retain_extrema():
    x = np.array([0.0, -5.0, 1.0, 7.0, 2.0])
    y = np.array([0.0, 2.0, -9.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="cannot preserve"):
        select_path_indices(x, y, max_points=3)


def test_path_selection_requires_aligned_arrays():
    with pytest.raises(ValueError, match="equal lengths"):
        select_path_indices(np.arange(3), np.arange(4), max_points=3)
