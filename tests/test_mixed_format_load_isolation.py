"""Failure-isolation contracts for mixed-format loading in one window.

These tests deliberately keep an already-rendered PX4 curve alive while a
foreign reader fails.  Reader parsing is stubbed because this module verifies
the GUI transaction boundary, not the format parsers themselves.
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
    from readers.base import LoadResult
    import gui.main_window as main_window_module
    from gui.main_window import LogLoadWorker, MainWindow, Workspace
except Exception as exc:  # pragma: no cover - only without optional GUI deps
    pytest.skip(f"Qt graph stack unavailable: {exc}", allow_module_level=True)


pg.setConfigOptions(useOpenGL=False)


def _dataset(file_name: str, source_format: str, capabilities: set[str], duration: float = 4.0):
    timestamps = np.linspace(0.0, duration, 41, dtype=np.float64)
    frame = pl.DataFrame(
        {
            "timestamp_sec": timestamps,
            "value": np.sin(timestamps),
        }
    )
    topic = TopicInstance("common", 0, dataframe=frame)
    topic.refresh_signals()
    dataset = LogDataset(
        source_format=source_format,
        source_path=file_name,
        capabilities=capabilities,
    )
    dataset.add_topic(topic)
    return dataset


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture()
def px4_session(qapp):
    window = MainWindow()
    window.resize(1000, 650)
    window.show()
    qapp.processEvents()

    px4 = _dataset("flight.ulg", "px4_ulog", {"px4", "timeseries"})
    assert window._apply_loaded_log_result(
        {
            "file_path": "flight.ulg",
            "filename": "flight.ulg",
            "dataset": px4,
            "aircraft_type": "Fixed Wing",
            "metadata": {"evaluation": {"overall_status": "synthetic"}},
            "format_id": "px4_ulog",
        }
    )
    workspace = window.tab_widget.currentWidget()
    assert isinstance(workspace, Workspace)
    assert workspace.first_plot.render_signal("flight.ulg", "common_0", "value")
    qapp.processEvents()

    try:
        yield window, workspace, px4
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def _session_snapshot(window: MainWindow, workspace: Workspace):
    plot = workspace.first_plot
    return {
        "datasets": dict(window.loaded_datasets),
        "aircraft": dict(window.loaded_aircraft_types),
        "metadata": dict(window.loaded_log_metadata),
        "active": window.active_analysis_log,
        "tree_rows": window.tree_model.invisibleRootItem().rowCount(),
        "plotted": tuple(plot.plotted_signals),
        "cache_keys": tuple(plot.signal_cache),
        "global_range": (workspace.global_min_x, workspace.global_max_x),
    }


@pytest.mark.parametrize(
    ("file_name", "error_text"),
    (
        ("broken.tlog", "malformed MAVLink telemetry"),
        ("broken.bag", "malformed ROS1 bag"),
        ("broken.db3", "malformed ROS2 bag"),
    ),
)
def test_failed_foreign_load_reports_error_without_mutating_px4_session(
    monkeypatch,
    tmp_path,
    px4_session,
    file_name,
    error_text,
):
    window, workspace, _px4 = px4_session
    source = tmp_path / file_name
    source.write_bytes(b"not a valid log")
    before = _session_snapshot(window, workspace)

    class FailingEngine:
        def load_result(self, _file_path):
            raise ValueError(error_text)

    monkeypatch.setattr(main_window_module, "LogIOEngine", FailingEngine)
    emitted_payloads = []
    summaries = []
    worker = LogLoadWorker([str(source)], existing_names=set(window.loaded_datasets))
    worker.fileLoaded.connect(emitted_payloads.append)
    worker.finished.connect(summaries.append)
    worker.run()

    assert emitted_payloads == []
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["loaded_count"] == 0
    assert summary["failed"] == [{"file_path": str(source), "error": error_text}]
    assert _session_snapshot(window, workspace) == before

    warnings = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    window._on_log_load_finished(summary)
    assert len(warnings) == 1
    assert warnings[0][0] == "Log Load"
    assert str(source) in warnings[0][1]
    assert error_text in warnings[0][1]
    assert _session_snapshot(window, workspace) == before


def test_partial_mixed_batch_commits_valid_ros_and_reports_failed_ardupilot(
    monkeypatch,
    tmp_path,
    px4_session,
    qapp,
):
    window, workspace, px4 = px4_session
    ros_path = tmp_path / "valid_ros.bag"
    bad_path = tmp_path / "broken.tlog"
    ros_path.write_bytes(b"synthetic ROS payload")
    bad_path.write_bytes(b"broken MAVLink payload")
    ros = _dataset(str(ros_path), "ros1_bag", {"ros", "ros1", "timeseries"}, duration=9.0)

    class MixedEngine:
        def load_result(self, file_path):
            if Path(file_path) == ros_path:
                return LoadResult(
                    dataset=ros,
                    format_id="ros1_bag",
                    metadata={"message_count": 41},
                    capabilities={"ros", "ros1", "timeseries"},
                    source_path=str(ros_path),
                )
            raise ValueError("malformed MAVLink telemetry")

    monkeypatch.setattr(main_window_module, "LogIOEngine", MixedEngine)
    summaries = []
    worker = LogLoadWorker(
        [str(ros_path), str(bad_path)],
        existing_names=set(window.loaded_datasets),
    )
    worker.fileLoaded.connect(window._apply_loaded_log_result)
    worker.finished.connect(summaries.append)
    worker.run()
    qapp.processEvents()

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["loaded_count"] == 1
    assert summary["failed"] == [
        {"file_path": str(bad_path), "error": "malformed MAVLink telemetry"}
    ]

    ros_name = ros_path.name
    assert window.loaded_datasets == {"flight.ulg": px4, ros_name: ros}
    assert set(window.loaded_aircraft_types) == {"flight.ulg", ros_name}
    assert set(window.loaded_log_metadata) == {"flight.ulg", ros_name}
    assert window.tree_model.invisibleRootItem().rowCount() == 2
    assert bad_path.name not in window.loaded_datasets
    assert list(workspace.first_plot.plotted_signals) == [
        "flight.ulg|common_0|value|timestamp_sec|False"
    ]

    # A successfully committed ROS source can immediately join the existing
    # PX4 panel; the failed sibling never becomes visible to the workspace.
    assert workspace.first_plot.render_signal(ros_name, "common_0", "value")
    qapp.processEvents()
    assert len(workspace.first_plot.plotted_signals) == 2
    assert workspace.global_max_x == pytest.approx(9.0)

    warnings = []
    monkeypatch.setattr(window, "show_log_info_dialog", lambda **_kwargs: None)
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    window._on_log_load_finished(summary)
    assert len(warnings) == 1
    assert warnings[0][0] == "Log Load"
    assert str(bad_path) in warnings[0][1]
    assert "malformed MAVLink telemetry" in warnings[0][1]


def test_ui_apply_rolls_back_all_state_when_tree_commit_fails(
    monkeypatch,
    px4_session,
):
    window, workspace, _px4 = px4_session
    before = _session_snapshot(window, workspace)
    ros = _dataset("candidate.bag", "ros1_bag", {"ros", "ros1", "timeseries"})

    def fail_tree_commit(*_args, **_kwargs):
        raise RuntimeError("synthetic tree insertion failure")

    monkeypatch.setattr(window, "_add_to_tree", fail_tree_commit)
    with pytest.raises(RuntimeError, match="synthetic tree insertion failure"):
        window._apply_loaded_log_result(
            {
                "file_path": "candidate.bag",
                "filename": "candidate.bag",
                "dataset": ros,
                "aircraft_type": "ROS 1",
                "metadata": {"evaluation": {"overall_status": "unavailable"}},
                "format_id": "ros1_bag",
            }
        )

    assert _session_snapshot(window, workspace) == before
