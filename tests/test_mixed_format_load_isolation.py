"""Failure-isolation contracts for mixed-format loading in one window.

These tests deliberately keep an already-rendered PX4 curve alive while a
foreign reader fails.  Reader parsing is stubbed because this module verifies
the GUI transaction boundary, not the format parsers themselves.
"""

from __future__ import annotations

import os
import sys
import json
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
        "source_ids": dict(window.loaded_source_ids),
        "source_paths": dict(window.loaded_source_paths),
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
    assert summary["emitted_count"] == 0
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
    worker.fileLoaded.connect(window._on_log_file_loaded)
    worker.finished.connect(summaries.append)
    worker.run()
    qapp.processEvents()

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["emitted_count"] == 1
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


def test_same_basename_from_different_directories_loads_as_distinct_sources(
    monkeypatch,
    tmp_path,
    qapp,
):
    first_path = tmp_path / "sortie-a" / "flight.ulg"
    second_path = tmp_path / "sortie-b" / "flight.ulg"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    first_path.write_bytes(b"first synthetic ULog")
    second_path.write_bytes(b"second synthetic ULog")

    datasets = {
        str(first_path): _dataset(str(first_path), "px4_ulog", {"px4", "timeseries"}),
        str(second_path): _dataset(str(second_path), "px4_ulog", {"px4", "timeseries"}),
    }

    class SameBasenameEngine:
        def load_result(self, file_path):
            return LoadResult(
                dataset=datasets[str(file_path)],
                format_id="px4_ulog",
                capabilities={"px4", "timeseries"},
                source_path=str(file_path),
            )

    monkeypatch.setattr(main_window_module, "LogIOEngine", SameBasenameEngine)
    monkeypatch.setattr(main_window_module, "_preprocess_loaded_dataset", lambda _dataset: None)
    monkeypatch.setattr(main_window_module, "_detect_aircraft_type_for_dataset", lambda _dataset: "Fixed Wing")
    monkeypatch.setattr(
        main_window_module,
        "_extract_log_metadata_for_dataset",
        lambda *_args, **_kwargs: {"evaluation": {"overall_status": "synthetic"}},
    )

    payloads = []
    summaries = []
    worker = LogLoadWorker([str(first_path), str(second_path)])
    worker.fileLoaded.connect(payloads.append)
    worker.finished.connect(summaries.append)
    worker.run()

    assert len(payloads) == 2
    assert {payload["filename"] for payload in payloads} == {"flight.ulg"}
    assert len({payload["source_id"] for payload in payloads}) == 2
    assert summaries == [
        {
            "emitted_count": 2,
            "skipped_missing": [],
            "skipped_duplicate": [],
            "failed": [],
        }
    ]

    window = MainWindow()
    window.resize(1000, 650)
    window.show()
    qapp.processEvents()
    try:
        assert window._apply_loaded_log_result(payloads[0])
        assert window._apply_loaded_log_result(payloads[1])
        qapp.processEvents()

        source_keys = list(window.loaded_datasets)
        assert len(source_keys) == 2
        assert source_keys[0] == "flight.ulg"
        assert source_keys[1] != source_keys[0]
        assert set(window.loaded_source_ids.values()) == {
            payloads[0]["source_id"],
            payloads[1]["source_id"],
        }
        tree_root = window.tree_model.invisibleRootItem()
        assert tree_root.rowCount() == 2
        assert len({tree_root.child(row).text() for row in range(2)}) == 2

        plot = window.tab_widget.currentWidget().first_plot
        assert plot.render_signal(source_keys[0], "common_0", "value")
        assert plot.render_signal(source_keys[1], "common_0", "value")
        legend_labels = [item[1].text for item in plot.plot.plotItem.legend.items]
        assert len(legend_labels) == 2
        assert len(set(legend_labels)) == 2

        plot.layout_special_spec = {
            "kind": "synthetic-test",
            "file_name": source_keys[0],
            # Simulate a transitional layout restored into this plot.  A new
            # export must scrub both absolute-path fields.
            "source_id": payloads[0]["source_id"],
            "source_path": str(first_path),
        }
        layout = window._serialize_plot_layout(plot)
        serialized_layout = json.dumps(layout, ensure_ascii=False)
        assert str(first_path) not in serialized_layout
        assert str(second_path) not in serialized_layout
        assert "sortie-a" not in serialized_layout
        assert "sortie-b" not in serialized_layout
        assert "source_path" not in serialized_layout
        assert "source_id" not in serialized_layout
        assert {signal["file_name"] for signal in layout["signals"]} == {"flight.ulg"}
        assert {signal["source_token"] for signal in layout["signals"]} == {
            main_window_module._source_layout_token(source_id)
            for source_id in window.loaded_source_ids.values()
        }
        assert all("source_id" not in signal for signal in layout["signals"])
        assert all("source_path" not in signal for signal in layout["signals"])
        assert layout["special"]["source_token"] == main_window_module._source_layout_token(
            payloads[0]["source_id"]
        )
        assert layout["special"]["file_name"] == "flight.ulg"
        # A pre-identity layout cannot safely choose between two equal
        # basenames; it must remain unresolved instead of selecting whichever
        # source happened to load first.
        assert window._resolve_layout_source_key("flight.ulg") is None

        # Source identity, not whichever log happened to claim the legacy
        # basename key, drives restoration after a different load order.
        reverse_window = MainWindow()
        reverse_window.show()
        qapp.processEvents()
        try:
            assert reverse_window._apply_loaded_log_result(payloads[1])
            assert reverse_window._apply_loaded_log_result(payloads[0])
            for signal in layout["signals"]:
                restored_key = reverse_window._resolve_layout_source_key(
                    signal["file_name"],
                    source_token=signal["source_token"],
                )
                assert main_window_module._source_layout_token(
                    reverse_window.loaded_source_ids[restored_key]
                ) == signal["source_token"]
        finally:
            reverse_window.close()
            reverse_window.deleteLater()
            qapp.processEvents()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_identity_layout_never_falls_back_to_different_moved_same_basename(qapp):
    original_path = str(PROJECT_ROOT / "original-sortie" / "flight.ulg")
    replacement_path = str(PROJECT_ROOT / "moved-sortie" / "flight.ulg")
    original = _dataset(original_path, "px4_ulog", {"px4", "timeseries"}, duration=4.0)
    replacement = _dataset(replacement_path, "px4_ulog", {"px4", "timeseries"}, duration=9.0)

    source_window = MainWindow()
    replacement_window = MainWindow()
    source_window.show()
    replacement_window.show()
    qapp.processEvents()
    try:
        assert source_window._apply_loaded_log_result(
            {
                "file_path": original_path,
                "filename": "flight.ulg",
                "source_id": main_window_module._canonical_source_identity(original_path),
                "dataset": original,
                "aircraft_type": "Fixed Wing",
                "metadata": {},
            }
        )
        source_plot = source_window.tab_widget.currentWidget().first_plot
        assert source_plot.render_signal("flight.ulg", "common_0", "value")
        signal_spec = source_window._serialize_plot_layout(source_plot)["signals"][0]

        assert replacement_window._apply_loaded_log_result(
            {
                "file_path": replacement_path,
                "filename": "flight.ulg",
                "source_id": main_window_module._canonical_source_identity(replacement_path),
                "dataset": replacement,
                "aircraft_type": "Fixed Wing",
                "metadata": {},
            }
        )
        assert replacement_window._resolve_layout_source_key(
            signal_spec["file_name"],
            source_token=signal_spec["source_token"],
        ) is None
        assert replacement_window._resolve_layout_source_key(
            signal_spec["file_name"],
            source_id=main_window_module._canonical_source_identity(original_path),
        ) is None

        target_plot = replacement_window.tab_widget.currentWidget().first_plot
        missing_items = []
        replacement_window._restore_plot_signals_into(target_plot, [signal_spec], missing_items)
        assert target_plot.plotted_signals == {}
        assert any("재연결 필요" in item for item in missing_items)

        # Layouts created before identity metadata existed retain their old
        # basename-only behavior for backward compatibility.
        legacy_spec = dict(signal_spec)
        legacy_spec.pop("source_token")
        assert replacement_window._resolve_layout_source_key(
            legacy_spec["file_name"]
        ) == "flight.ulg"
    finally:
        source_window.close()
        replacement_window.close()
        source_window.deleteLater()
        replacement_window.deleteLater()
        qapp.processEvents()


def test_worker_skips_only_the_same_source_identity(monkeypatch, tmp_path):
    source = tmp_path / "flight.ulg"
    source.write_bytes(b"synthetic ULog")
    dataset = _dataset(str(source), "px4_ulog", {"px4", "timeseries"})

    class CountingEngine:
        calls = 0

        def load_result(self, file_path):
            self.calls += 1
            return LoadResult(
                dataset=dataset,
                format_id="px4_ulog",
                capabilities={"px4", "timeseries"},
                source_path=str(file_path),
            )

    monkeypatch.setattr(main_window_module, "LogIOEngine", CountingEngine)
    source_id = main_window_module._canonical_source_identity(str(source))
    summaries = []
    worker = LogLoadWorker([str(source)], existing_source_ids={source_id})
    worker.finished.connect(summaries.append)
    worker.run()

    assert CountingEngine.calls == 0
    assert summaries[0]["emitted_count"] == 0
    assert summaries[0]["skipped_duplicate"] == [source.name]


def test_last_log_removal_restores_multiformat_idle_text(px4_session):
    window, _workspace, _px4 = px4_session

    assert window.delete_loaded_log("flight.ulg", ask_confirm=False)

    assert window.file_drop_widget.text() == window.file_drop_widget.idle_text()
    assert "ULG" in window.file_drop_widget.text()
    assert "ROS" in window.file_drop_widget.text()
    assert "ArduPilot" in window.file_drop_widget.text()
    assert "Date(ulg)" not in window.file_drop_widget.text()


def test_delete_everything_restores_multiformat_idle_text(monkeypatch, px4_session):
    window, _workspace, _px4 = px4_session
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: main_window_module.QMessageBox.Yes,
    )

    window.delete_everything()

    assert window.file_drop_widget.text() == window.file_drop_widget.idle_text()
    assert "ULG" in window.file_drop_widget.text()
    assert "ROS" in window.file_drop_widget.text()
    assert "ArduPilot" in window.file_drop_widget.text()
    assert "Date(ulg)" not in window.file_drop_widget.text()
