"""Headless regression coverage for the code-native 3D vehicle markers."""

from __future__ import annotations

import gc
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PX4_INTERACTIVE_FAST_RENDER_PREVIEW", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    import pyqtgraph as pg

    from core.log_model import LogDataset, TopicInstance
    from gui.main_window import AdvancedPlot, MainWindow, Workspace
except Exception as exc:  # pragma: no cover - only without optional GUI deps
    pytest.skip(f"Qt graph stack unavailable: {exc}", allow_module_level=True)


# These tests exercise numpy geometry and capture setData calls.  The one
# render/restore check deliberately forces the production projected-3D path.
pg.setConfigOptions(useOpenGL=False)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture()
def marker_plot(qapp):
    plot = AdvancedPlot(main_window=None, workspace=None)
    try:
        yield plot
    finally:
        plot.close()
        plot.deleteLater()
        qapp.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()


class _LineItemCapture:
    def __init__(self):
        self.kwargs = None

    def setData(self, **kwargs):
        self.kwargs = kwargs


def _local_position_dataset(source_path="vehicle-marker.ulg"):
    topic = TopicInstance(
        "vehicle_local_position",
        0,
        dataframe=pl.DataFrame(
            {
                "timestamp_sec": [0.0, 1.0, 2.0, 3.0],
                "x": [0.0, 1.0, 2.0, 4.0],
                "y": [0.0, 2.0, 4.0, 7.0],
                "z": [0.0, -0.5, -1.0, -1.5],
            }
        ),
    )
    topic.refresh_signals()
    dataset = LogDataset(
        source_format="px4_ulog",
        source_path=source_path,
        capabilities={"px4", "timeseries"},
    )
    dataset.add_topic(topic)
    return dataset


def test_marker_type_normalisation_and_detection_keep_helicopter_distinct():
    aliases = {
        "Auto": "auto",
        "fixed-wing": "fixed_wing",
        "airplane": "fixed_wing",
        "MULTI COPTER": "quadcopter",
        "quadrotor": "quadcopter",
        "rotary-wing": "helicopter",
        "ground rover": "rover",
    }
    for raw, expected in aliases.items():
        assert AdvancedPlot._normalise_true_3d_vehicle_marker_type(raw) == expected

    assert AdvancedPlot._normalise_true_3d_vehicle_marker_type("auto", allow_auto=False) is None
    assert AdvancedPlot._normalise_true_3d_vehicle_marker_type("submarine") is None

    detected = AdvancedPlot._vehicle_marker_type_from_aircraft_label
    # Helicopter must win even if a source label also contains a broad MC term.
    assert detected("Traditional Helicopter / Multicopter") == "helicopter"
    assert detected("PX4 Helicopter") == "helicopter"
    assert detected("Multicopter") == "quadcopter"
    assert detected("Quadrotor UAV") == "quadcopter"
    assert detected("ArduCopter") == "quadcopter"
    assert detected("Copter") == "quadcopter"
    assert detected("Ground Rover") == "rover"
    assert detected("Fixed Wing") == "fixed_wing"
    assert detected("ArduPlane") == "fixed_wing"
    assert detected("VTOL") == "unknown"
    assert detected("ROS 2") == "unknown"
    assert detected("Unknown") == "unknown"


def test_each_vehicle_wireframe_is_detailed_finite_scaled_and_distinct():
    marker_types = ("fixed_wing", "quadcopter", "helicopter", "rover")
    geometries = {}

    for marker_type in marker_types:
        unit = AdvancedPlot._build_true_3d_aircraft_body_segments(1.0, marker_type)
        scaled = AdvancedPlot._build_true_3d_aircraft_body_segments(2.5, marker_type)
        geometries[marker_type] = unit

        assert unit.dtype == np.float32
        assert unit.ndim == 2 and unit.shape[1] == 3
        assert unit.shape[0] >= 200
        assert unit.shape[0] % 2 == 0
        assert np.all(np.isfinite(unit))
        assert np.unique(np.round(unit, decimals=5), axis=0).shape[0] >= 60
        assert np.all(np.ptp(unit, axis=0) > 0.5)

        edges = unit.reshape(-1, 2, 3)
        edge_lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
        assert np.all(edge_lengths > 1e-6)
        np.testing.assert_allclose(scaled, unit * 2.5, rtol=1e-6, atol=1e-6)

    for left, right in combinations(marker_types, 2):
        left_geometry = geometries[left]
        right_geometry = geometries[right]
        assert left_geometry.shape != right_geometry.shape or not np.array_equal(
            left_geometry,
            right_geometry,
        )


def test_auto_selection_tracks_detected_type_but_explicit_type_wins(marker_plot):
    assert marker_plot._true_3d_detected_vehicle_type == "unknown"
    assert marker_plot._effective_true_3d_vehicle_marker_type() == "fixed_wing"
    assert marker_plot._true_3d_vehicle_marker_display_name() == "Fixed Wing (Auto Fallback)"
    assert marker_plot._true_3d_vehicle_auto_menu_label() == "Auto (Fallback: Fixed Wing)"

    assert marker_plot.configure_true_3d_vehicle_marker(
        marker_type="auto",
        detected_aircraft_type="Helicopter",
        refresh=False,
    )
    assert marker_plot._true_3d_vehicle_marker_type == "auto"
    assert marker_plot._true_3d_detected_vehicle_type == "helicopter"
    assert marker_plot._effective_true_3d_vehicle_marker_type() == "helicopter"
    assert marker_plot._true_3d_vehicle_marker_display_name() == "Helicopter (Auto)"

    assert marker_plot.configure_true_3d_vehicle_marker(
        marker_type="quadcopter",
        detected_aircraft_type="Ground Rover",
        refresh=False,
    )
    assert marker_plot._true_3d_detected_vehicle_type == "rover"
    assert marker_plot._effective_true_3d_vehicle_marker_type() == "quadcopter"
    assert marker_plot._true_3d_vehicle_marker_display_name() == "Quadcopter"
    assert marker_plot._true_3d_vehicle_auto_menu_label() == "Auto (Detected: Rover)"

    assert marker_plot.configure_true_3d_vehicle_marker(
        marker_type="rover",
        detected_aircraft_type="Helicopter",
        refresh=False,
    )
    assert marker_plot._effective_true_3d_vehicle_marker_type() == "rover"
    assert marker_plot._true_3d_vehicle_auto_menu_label() == "Auto (Detected: Helicopter)"

    assert marker_plot.configure_true_3d_vehicle_marker(
        marker_type="unsupported",
        scale_factor=float("nan"),
        refresh=False,
    )
    assert marker_plot._true_3d_vehicle_marker_type == "auto"
    assert marker_plot._effective_true_3d_vehicle_marker_type() == "helicopter"
    assert marker_plot._true_3d_aircraft_scale_factor == pytest.approx(1.0)


def test_vehicle_marker_type_and_scale_round_trip_through_layout(qapp):
    file_name = "vehicle-marker-layout.ulg"
    dataset = _local_position_dataset(file_name)
    window = MainWindow()
    workspace = Workspace(window)
    workspace.create_grid(1, 2)
    source_plot = workspace.get_plot(0, 0)
    restored_plot = workspace.get_plot(0, 1)

    try:
        window.loaded_datasets[file_name] = dataset
        window.loaded_aircraft_types[file_name] = "Helicopter"
        source_plot._true_3d_available = False
        restored_plot._true_3d_available = False

        assert window._render_3d_flight_path_in_plot(source_plot, file_name, dataset)
        assert source_plot._effective_true_3d_vehicle_marker_type() == "helicopter"
        assert source_plot.configure_true_3d_vehicle_marker(
            marker_type="rover",
            scale_factor=2.34,
            refresh=False,
        )

        node = window._serialize_plot_layout(source_plot)
        node = json.loads(json.dumps(node))
        assert node["special"]["vehicle_marker_type"] == "rover"
        assert node["special"]["vehicle_marker_scale"] == pytest.approx(2.3)

        missing_items = []
        window._restore_plot_from_layout(restored_plot, node, missing_items)

        assert missing_items == []
        assert restored_plot._true_3d_vehicle_marker_type == "rover"
        assert restored_plot._true_3d_detected_vehicle_type == "helicopter"
        assert restored_plot._effective_true_3d_vehicle_marker_type() == "rover"
        assert restored_plot._true_3d_aircraft_scale_factor == pytest.approx(2.3)
        assert restored_plot.layout_special_spec["vehicle_marker_type"] == "rover"
        assert restored_plot.layout_special_spec["vehicle_marker_scale"] == pytest.approx(2.3)

        # Layouts created before vehicle markers had no marker fields.  Loading
        # one into a plot that currently has explicit settings must restore the
        # legacy defaults rather than leak the previous plot state.
        legacy_node = json.loads(json.dumps(node))
        legacy_node["special"].pop("vehicle_marker_type")
        legacy_node["special"].pop("vehicle_marker_scale")
        window._restore_plot_from_layout(restored_plot, legacy_node, missing_items)
        assert missing_items == []
        assert restored_plot._true_3d_vehicle_marker_type == "auto"
        assert restored_plot._effective_true_3d_vehicle_marker_type() == "helicopter"
        assert restored_plot._true_3d_aircraft_scale_factor == pytest.approx(1.0)
        assert restored_plot.layout_special_spec["vehicle_marker_type"] == "auto"
        assert restored_plot.layout_special_spec["vehicle_marker_scale"] == pytest.approx(1.0)

        # Corrupt persisted values are sanitized to the same safe defaults.
        malformed_node = json.loads(json.dumps(node))
        malformed_node["special"]["vehicle_marker_type"] = "submarine"
        malformed_node["special"]["vehicle_marker_scale"] = "not-a-number"
        window._restore_plot_from_layout(restored_plot, malformed_node, missing_items)
        assert missing_items == []
        assert restored_plot._true_3d_vehicle_marker_type == "auto"
        assert restored_plot._effective_true_3d_vehicle_marker_type() == "helicopter"
        assert restored_plot._true_3d_aircraft_scale_factor == pytest.approx(1.0)
    finally:
        workspace.close()
        window.close()
        workspace.deleteLater()
        window.deleteLater()
        qapp.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()
        gc.collect()


def test_no_attitude_uses_nonzero_path_tangent_without_live_opengl(marker_plot):
    # The repeated samples emulate a hover.  The fallback must search past the
    # duplicate points and align body-forward with the first stable ENU tangent.
    tangent = np.array([4.0, 2.0, 2.0], dtype=np.float64)
    marker_plot._true_3d_points = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], tangent],
        dtype=np.float64,
    )
    marker_plot._3d_value_extras = None

    roll, pitch, yaw = marker_plot._true_3d_fallback_attitude_at_idx(1)
    assert roll == pytest.approx(0.0)
    assert pitch == pytest.approx(math.atan2(2.0, math.hypot(4.0, 2.0)))
    assert yaw == pytest.approx(math.atan2(4.0, 2.0))

    capture = _LineItemCapture()
    marker_plot._true_3d_enabled = True
    marker_plot._true_3d_aircraft_item = capture
    marker_plot._true_3d_aircraft_body_segments = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    marker_plot._true_3d_aircraft_ned_axes_item = None

    assert marker_plot._update_true_3d_aircraft_marker(1)
    assert capture.kwargs is not None
    rendered = np.asarray(capture.kwargs["pos"], dtype=np.float64)
    np.testing.assert_allclose(rendered[0], marker_plot._true_3d_points[1], atol=1e-7)
    np.testing.assert_allclose(
        rendered[1] - rendered[0],
        tangent / np.linalg.norm(tangent),
        rtol=1e-6,
        atol=1e-6,
    )
    assert capture.kwargs["mode"] == "lines"


def test_large_downsampled_path_uses_full_resolution_tangent(marker_plot):
    sample_count = 6001
    times = np.arange(sample_count, dtype=np.float64)
    angles = np.linspace(0.0, math.pi, sample_count)
    full_points = np.column_stack(
        (100.0 * np.cos(angles), 100.0 * np.sin(angles), angles * 4.0)
    )
    marker_plot._true_3d_points = full_points[::20]
    marker_plot._true_3d_pick_points = full_points
    marker_plot._true_3d_pick_time_arr = times
    marker_plot._3d_value_extras = None
    marker_plot._true_3d_enabled = True
    marker_plot._true_3d_aircraft_body_segments = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    marker_plot._true_3d_aircraft_ned_axes_item = None
    capture = _LineItemCapture()
    marker_plot._true_3d_aircraft_item = capture

    target_idx = 4500
    assert marker_plot._set_true_3d_cursor_at_time(float(target_idx), show_label=False)

    rendered = np.asarray(capture.kwargs["pos"], dtype=np.float64)
    rendered_forward = rendered[1] - rendered[0]
    expected = full_points[target_idx + 1] - full_points[target_idx - 1]
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(rendered_forward, expected, rtol=1e-5, atol=1e-5)
