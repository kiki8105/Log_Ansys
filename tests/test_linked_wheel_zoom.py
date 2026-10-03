"""Acceptance tests for live linked wheel zoom in one workspace."""

from __future__ import annotations

import os
import sys
import time
import gc
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
    from gui.main_window import MainWindow, Workspace
except Exception as exc:  # pragma: no cover - only without optional GUI deps
    pytest.skip(f"Qt graph stack unavailable: {exc}", allow_module_level=True)


pg.setConfigOptions(useOpenGL=False)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def _build_workspace(qapp, panel_count: int, sample_count: int = 20_001):
    t = np.linspace(0.0, 120.0, sample_count, dtype=np.float64)
    frame = pl.DataFrame(
        {
            "timestamp_sec": t,
            "roll": np.sin(t * 0.71),
            "pitch": np.cos(t * 0.43),
            "yaw": np.sin(t * 0.17) * np.cos(t * 0.03),
        }
    )
    topic = TopicInstance("wheel_signal", 0, dataframe=frame)
    topic.refresh_signals()
    dataset = LogDataset(
        source_format="px4_ulog",
        source_path="wheel-synthetic.ulg",
        capabilities={"px4", "timeseries"},
    )
    dataset.add_topic(topic)

    window = MainWindow()
    window.loaded_datasets["wheel-synthetic.ulg"] = dataset
    workspace = Workspace(window)
    workspace.resize(1200, 720)
    workspace.create_grid(1, panel_count)
    workspace.show()
    qapp.processEvents()

    names = ("roll", "pitch", "yaw")
    plots = [workspace.get_plot(0, idx) for idx in range(panel_count)]
    for idx, plot in enumerate(plots):
        assert plot.render_signal(
            "wheel-synthetic.ulg",
            "wheel_signal_0",
            names[idx],
        )
    workspace.set_time_range(0.0, 120.0, clamp_to_global=True)
    qapp.processEvents()
    return window, workspace, plots


def _close_workspace(qapp, window, workspace):
    workspace.close()
    window.close()
    workspace.deleteLater()
    window.deleteLater()
    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qapp.processEvents()
    gc.collect()


def _ranges_match(plots, start_t, end_t, atol=1e-6):
    for plot in plots:
        current = plot.plot.getViewBox().viewRange()[0]
        if abs(float(current[0]) - float(start_t)) > atol:
            return False
        if abs(float(current[1]) - float(end_t)) > atol:
            return False
    return True


@pytest.mark.parametrize("panel_count", [1, 2, 3])
def test_wheel_live_range_moves_one_two_or_three_time_panels_together(
    qapp, panel_count
):
    window, workspace, plots = _build_workspace(qapp, panel_count)
    try:
        workspace._wheel_active = True

        # A cold/first frame is applied inline, so all current ViewBox ranges
        # already agree when the input callback returns.
        workspace._wheel_begin_or_update(20.0, 80.0, source_plot=plots[0])
        assert _ranges_match(plots, 20.0, 80.0)

        # A second event also updates every logical ViewBox immediately while
        # Qt coalesces the expensive viewport paints. It must not wait for the
        # 300 ms idle commit used by the old source-only implementation.
        started_at = time.perf_counter()
        workspace._wheel_begin_or_update(26.0, 64.0, source_plot=plots[0])
        callback_ms = (time.perf_counter() - started_at) * 1000.0
        assert _ranges_match(plots, 26.0, 64.0)
        assert callback_ms <= 50.0
        qapp.processEvents()
        assert _ranges_match(plots, 26.0, 64.0)

        # Idle remains the exact authoritative commit (1e-6 seconds).
        workspace._wheel_live_commit_range = (26.0, 64.0)
        workspace._wheel_live_commit_source = plots[0]
        workspace._run_wheel_commit()
        qapp.processEvents()
        assert _ranges_match(plots, 26.0, 64.0, atol=1e-6)
    finally:
        workspace._wheel_active = False
        _close_workspace(qapp, window, workspace)


def test_live_wheel_sync_excludes_2d_and_3d_spatial_views(qapp):
    window, workspace, plots = _build_workspace(qapp, 3)
    try:
        source_plot, path_2d_plot, path_3d_plot = plots
        t = np.linspace(0.0, 30.0, 601, dtype=np.float64)
        east = 30.0 * np.sin(t * 0.2)
        north = t * 2.0
        up = 0.5 * t
        assert path_2d_plot.render_2d_flight_path(
            east, north, "2D path", timestamps=t
        )
        path_3d_plot._true_3d_available = False
        assert path_3d_plot.render_3d_path(
            east, north, up, "3D path", timestamps=t
        )
        assert source_plot.is_time_plot
        assert not path_2d_plot.is_time_plot
        assert not path_3d_plot.is_time_plot

        path_2d_plot.plot.getViewBox().disableAutoRange()
        path_3d_plot.plot.getViewBox().disableAutoRange()
        qapp.processEvents()
        before_2d = np.asarray(path_2d_plot.plot.getViewBox().viewRange())
        before_3d = np.asarray(path_3d_plot.plot.getViewBox().viewRange())
        workspace._wheel_active = True
        workspace._wheel_begin_or_update(10.0, 40.0, source_plot=source_plot)
        qapp.processEvents()

        np.testing.assert_allclose(
            np.asarray(path_2d_plot.plot.getViewBox().viewRange()),
            before_2d,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            np.asarray(path_3d_plot.plot.getViewBox().viewRange()),
            before_3d,
            atol=1e-9,
        )
    finally:
        workspace._wheel_active = False
        _close_workspace(qapp, window, workspace)


def test_large_series_follower_range_apply_p95_stays_under_50_ms(qapp):
    window, workspace, plots = _build_workspace(
        qapp, panel_count=3, sample_count=100_001
    )
    try:
        workspace._wheel_active = True
        durations_ms = []
        for idx in range(30):
            start_t = 5.0 + (idx % 10)
            end_t = 95.0 - (idx % 7)
            workspace._wheel_live_follower_pending_range = (start_t, end_t)
            workspace._wheel_live_follower_pending_source = plots[0]
            workspace._wheel_live_follower_pending_requested_at = time.perf_counter()
            started_at = time.perf_counter()
            workspace._apply_pending_wheel_live_follower_sync()
            durations_ms.append((time.perf_counter() - started_at) * 1000.0)

        p95_ms = float(np.percentile(np.asarray(durations_ms), 95))
        assert p95_ms <= 50.0, f"live follower range apply p95={p95_ms:.3f} ms"
        assert workspace._wheel_live_follower_sync_last_plot_count <= 2
    finally:
        workspace._wheel_active = False
        _close_workspace(qapp, window, workspace)
