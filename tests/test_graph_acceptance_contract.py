"""Headless acceptance checks for the graph contracts shared by real ULG tests.

These tests intentionally use a small in-memory dataset.  The field acceptance
runner uses the two real sortie files; keeping those private files out of the
pytest dependency makes the normal suite reproducible on another machine.
"""

from __future__ import annotations

import os
import sys
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
    from PySide6.QtWidgets import QApplication
    import pyqtgraph as pg
    from core.log_model import LogDataset, TopicInstance
    from gui.main_window import MainWindow, Workspace
except Exception as exc:  # pragma: no cover - exercised only without GUI deps
    pytest.skip(f"Qt graph stack unavailable: {exc}", allow_module_level=True)

# The production app uses an OpenGL viewport.  A headless platform has no GL
# context, so use the equivalent QPainter path here; true GL remains a required
# real-Windows acceptance item in docs/ULG_GRAPH_ACCEPTANCE_PLAN.md.
pg.setConfigOptions(useOpenGL=False)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture()
def graph_context(qapp):
    t = np.linspace(0.0, 60.0, 6001, dtype=np.float64)
    frame = pl.DataFrame(
        {
            "timestamp_sec": t,
            "roll": np.sin(t * 0.7),
            "pitch": np.cos(t * 0.43),
            "nav_state": (np.floor(t / 5.0) % 4).astype(np.int16),
        }
    )
    topic = TopicInstance("acceptance_signal", 0, dataframe=frame)
    topic.refresh_signals()
    dataset = LogDataset(
        source_format="px4_ulog",
        source_path="synthetic.ulg",
        capabilities={"px4", "timeseries"},
    )
    dataset.add_topic(topic)

    window = MainWindow()
    qapp.processEvents()
    window.loaded_datasets["synthetic.ulg"] = dataset
    workspace = Workspace(window)
    workspace.resize(1200, 700)
    workspace.show()
    qapp.processEvents()
    try:
        yield window, workspace
    finally:
        workspace.close()
        window.close()
        workspace.deleteLater()
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("curve_count", [1, 2, 3])
def test_one_panel_accepts_one_two_or_three_curves(graph_context, curve_count):
    window, workspace = graph_context
    plot = workspace.first_plot
    signal_names = ("roll", "pitch", "nav_state")

    for signal_name in signal_names[:curve_count]:
        assert plot.render_signal(
            "synthetic.ulg",
            "acceptance_signal_0",
            signal_name,
        )

    assert plot.is_time_plot
    assert len(plot.plotted_signals) == curve_count
    assert workspace.global_min_x == pytest.approx(0.0)
    assert workspace.global_max_x == pytest.approx(60.0)


@pytest.mark.parametrize("panel_count", [1, 2, 3])
def test_one_two_or_three_time_panels_share_authoritative_range(
    graph_context, qapp, panel_count
):
    _window, workspace = graph_context
    workspace.create_grid(1, panel_count)
    qapp.processEvents()

    plots = [workspace.get_plot(0, idx) for idx in range(panel_count)]
    signal_names = ("roll", "pitch", "nav_state")
    for idx, plot in enumerate(plots):
        assert plot.render_signal(
            "synthetic.ulg",
            "acceptance_signal_0",
            signal_names[idx],
        )

    workspace.set_time_range(12.5, 27.25, clamp_to_global=True)
    qapp.processEvents()
    for plot in plots:
        x_range = plot.plot.getViewBox().viewRange()[0]
        assert x_range[0] == pytest.approx(12.5, abs=1e-6)
        assert x_range[1] == pytest.approx(27.25, abs=1e-6)

    workspace.reset_zoom()
    qapp.processEvents()
    for plot in plots:
        x_range = plot.plot.getViewBox().viewRange()[0]
        assert x_range[0] == pytest.approx(0.0, abs=1e-6)
        assert x_range[1] == pytest.approx(60.0, abs=1e-6)


def test_2d_space_zoom_is_independent_but_time_marker_tracks_cursor(
    graph_context, qapp
):
    _window, workspace = graph_context
    plot = workspace.first_plot
    t = np.linspace(0.0, 20.0, 401, dtype=np.float64)
    east = 25.0 * np.sin(t * 0.2)
    north = t * 3.0

    assert plot.render_2d_flight_path(east, north, "2D acceptance", timestamps=t)
    assert plot._flight_path_2d_enabled
    assert not plot.is_time_plot

    workspace.time_cursor.setValue(9.0)
    workspace._pending_cursor_t = 9.0
    workspace._update_cursor_overlays(force=True)
    qapp.processEvents()
    idx = int(np.argmin(np.abs(t - 9.0)))
    marker_x, marker_y = plot._flight_path_2d_marker_item.getData()
    assert float(marker_x[0]) == pytest.approx(float(east[idx]))
    assert float(marker_y[0]) == pytest.approx(float(north[idx]))

    before = tuple(tuple(axis) for axis in plot.plot.getViewBox().viewRange())
    workspace.set_time_range(3.0, 7.0, clamp_to_global=False)
    qapp.processEvents()
    after = tuple(tuple(axis) for axis in plot.plot.getViewBox().viewRange())
    np.testing.assert_allclose(np.asarray(after), np.asarray(before), atol=1e-9)


def test_projected_3d_fallback_tracks_time_without_becoming_time_plot(
    graph_context, qapp
):
    _window, workspace = graph_context
    plot = workspace.first_plot
    plot._true_3d_available = False
    t = np.linspace(0.0, 12.0, 241, dtype=np.float64)
    east = np.cos(t * 0.4) * 20.0
    north = np.sin(t * 0.4) * 20.0
    up = t * 1.5

    assert plot.render_3d_path(east, north, up, "3D acceptance", timestamps=t)
    assert plot._projected_3d_enabled
    assert not plot._true_3d_enabled
    assert not plot.is_time_plot

    workspace.time_cursor.setValue(6.0)
    workspace._pending_cursor_t = 6.0
    workspace._update_cursor_overlays(force=True)
    qapp.processEvents()
    expected_idx = int(np.argmin(np.abs(t - 6.0)))
    assert plot._projected_3d_state["cursor_idx"] == expected_idx
