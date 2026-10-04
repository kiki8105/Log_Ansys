"""Cooperative cancellation and shutdown contracts for background log loading.

The format readers do not currently expose an interrupt callback.  These tests
therefore make the intended boundary explicit: an in-flight reader is allowed
to return, but its result is never committed after cancellation and all later
files are skipped.  The GUI stays alive until its QThread has really stopped.
"""

from __future__ import annotations

import os
import sys
import threading
import time
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
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    import pyqtgraph as pg
    from core.log_model import LogDataset, TopicInstance
    from readers.base import LoadResult
    import gui.main_window as main_window_module
    from gui.main_window import LogLoadWorker, MainWindow
except Exception as exc:  # pragma: no cover - only without optional GUI deps
    pytest.skip(f"Qt graph stack unavailable: {exc}", allow_module_level=True)


pg.setConfigOptions(useOpenGL=False)


def _dataset(source_path: str) -> LogDataset:
    timestamps = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    frame = pl.DataFrame({"timestamp_sec": timestamps, "value": timestamps})
    topic = TopicInstance("common", 0, dataframe=frame)
    topic.refresh_signals()
    dataset = LogDataset(
        source_format="csv",
        source_path=source_path,
        capabilities={"timeseries"},
    )
    dataset.add_topic(topic)
    return dataset


def _result(source_path: str) -> LoadResult:
    return LoadResult(
        dataset=_dataset(source_path),
        format_id="csv",
        capabilities={"timeseries"},
        source_path=source_path,
    )


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def _process_until(qapp, predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    qapp.processEvents()
    return bool(predicate())


def test_cancel_before_run_never_opens_or_commits_file(monkeypatch, tmp_path):
    source = tmp_path / "large.csv"
    source.write_text("timestamp,value\n0,1\n", encoding="utf-8")

    class CountingEngine:
        calls = 0

        def load_result(self, file_path):
            self.calls += 1
            return _result(file_path)

    monkeypatch.setattr(main_window_module, "LogIOEngine", CountingEngine)
    payloads = []
    summaries = []
    worker = LogLoadWorker([str(source)])
    worker.fileLoaded.connect(payloads.append)
    worker.finished.connect(summaries.append)

    worker.request_cancel()
    worker.run()

    assert CountingEngine.calls == 0
    assert payloads == []
    assert summaries == [
        {
            "emitted_count": 0,
            "skipped_missing": [],
            "skipped_duplicate": [],
            "failed": [],
            "cancelled": True,
            "cancelled_file_path": str(source),
        }
    ]


def test_cancel_during_reader_waits_for_return_then_discards_result(
    monkeypatch,
    tmp_path,
    qapp,
):
    source = tmp_path / "large.csv"
    source.write_text("timestamp,value\n0,1\n", encoding="utf-8")
    entered_reader = threading.Event()
    release_reader = threading.Event()

    class BlockingEngine:
        def load_result(self, file_path):
            entered_reader.set()
            assert release_reader.wait(5.0), "test did not release synthetic reader"
            return _result(file_path)

    monkeypatch.setattr(main_window_module, "LogIOEngine", BlockingEngine)
    payloads = []
    summaries = []
    worker = LogLoadWorker([str(source)])
    worker.fileLoaded.connect(payloads.append)
    worker.finished.connect(summaries.append)
    runner = threading.Thread(target=worker.run, daemon=True)
    runner.start()
    assert entered_reader.wait(2.0)

    worker.request_cancel()
    # Cancellation does not pretend that a non-cooperative parser can be
    # interrupted.  It remains alive until that parser returns.
    assert runner.is_alive()
    release_reader.set()
    runner.join(3.0)
    assert not runner.is_alive()
    qapp.processEvents()

    assert payloads == []
    assert summaries[0]["emitted_count"] == 0
    assert summaries[0]["failed"] == []
    assert summaries[0]["cancelled"] is True
    assert summaries[0]["cancelled_file_path"] == str(source)


def test_cancel_after_committed_file_skips_remaining_batch(monkeypatch, tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("timestamp,value\n0,1\n", encoding="utf-8")
    second.write_text("timestamp,value\n0,2\n", encoding="utf-8")

    class RecordingEngine:
        calls = []

        def load_result(self, file_path):
            self.calls.append(file_path)
            return _result(file_path)

    monkeypatch.setattr(main_window_module, "LogIOEngine", RecordingEngine)
    payloads = []
    summaries = []
    worker = LogLoadWorker([str(first), str(second)])

    def commit_first_then_cancel(payload):
        payloads.append(payload)
        worker.request_cancel()

    worker.fileLoaded.connect(commit_first_then_cancel)
    worker.finished.connect(summaries.append)
    worker.run()

    assert RecordingEngine.calls == [str(first)]
    assert [payload["file_path"] for payload in payloads] == [str(first)]
    assert summaries[0]["emitted_count"] == 1
    assert summaries[0]["cancelled"] is True
    assert summaries[0]["cancelled_file_path"] == str(second)


def test_user_cancel_keeps_window_consistent_until_worker_stops(
    monkeypatch,
    tmp_path,
    qapp,
):
    source = tmp_path / "large.csv"
    source.write_text("timestamp,value\n0,1\n", encoding="utf-8")
    entered_reader = threading.Event()
    release_reader = threading.Event()

    class BlockingEngine:
        def load_result(self, file_path):
            entered_reader.set()
            assert release_reader.wait(5.0), "test did not release synthetic reader"
            return _result(file_path)

    monkeypatch.setattr(main_window_module, "LogIOEngine", BlockingEngine)
    window = MainWindow()
    window.show()
    qapp.processEvents()
    try:
        window.load_log_files([str(source)])
        assert entered_reader.wait(2.0)

        window.upload_progress_cancel_button.click()
        qapp.processEvents()
        assert window.isVisible()
        assert window._log_load_cancel_requested is True
        assert not window.upload_progress_cancel_button.isEnabled()
        assert window.loaded_datasets == {}

        release_reader.set()
        assert _process_until(qapp, lambda: window._log_load_thread is None)
        assert window.isVisible()
        assert window.loaded_datasets == {}
        assert window.tree_model.invisibleRootItem().rowCount() == 0
    finally:
        release_reader.set()
        _process_until(qapp, lambda: window._log_load_thread is None)
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_close_during_reader_defers_shutdown_and_prevents_late_commit(
    monkeypatch,
    tmp_path,
    qapp,
):
    source = tmp_path / "large.csv"
    source.write_text("timestamp,value\n0,1\n", encoding="utf-8")
    entered_reader = threading.Event()
    release_reader = threading.Event()

    class BlockingEngine:
        def load_result(self, file_path):
            entered_reader.set()
            assert release_reader.wait(5.0), "test did not release synthetic reader"
            return _result(file_path)

    monkeypatch.setattr(main_window_module, "LogIOEngine", BlockingEngine)
    window = MainWindow()
    window.show()
    qapp.processEvents()
    try:
        window.load_log_files([str(source)])
        assert entered_reader.wait(2.0)

        # The first close is ignored while the reader unwinds.  This is what
        # keeps the child QThread alive and prevents a shutdown crash.
        assert window.close() is False
        qapp.processEvents()
        assert window.isVisible()
        assert window._closing_requested is True
        assert window._log_load_cancel_requested is True
        assert window.loaded_datasets == {}

        release_reader.set()
        assert _process_until(
            qapp,
            lambda: window._log_load_thread is None and not window.isVisible(),
        )
        assert window.loaded_datasets == {}
        assert window.tree_model.invisibleRootItem().rowCount() == 0
    finally:
        release_reader.set()
        _process_until(qapp, lambda: window._log_load_thread is None)
        if window.isVisible():
            window.close()
        window.deleteLater()
        qapp.processEvents()


def test_delete_everything_during_reader_blocks_late_commit(
    monkeypatch,
    tmp_path,
    qapp,
):
    source = tmp_path / "large.csv"
    source.write_text("timestamp,value\n0,1\n", encoding="utf-8")
    entered_reader = threading.Event()
    release_reader = threading.Event()

    class BlockingEngine:
        def load_result(self, file_path):
            entered_reader.set()
            assert release_reader.wait(5.0), "test did not release synthetic reader"
            return _result(file_path)

    monkeypatch.setattr(main_window_module, "LogIOEngine", BlockingEngine)
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: main_window_module.QMessageBox.Yes,
    )
    window = MainWindow()
    window.show()
    qapp.processEvents()
    try:
        window.load_log_files([str(source)])
        assert entered_reader.wait(2.0)

        window.delete_everything()
        assert window._log_load_cancel_requested is True
        assert window.loaded_datasets == {}
        assert window.tab_widget.count() == 0

        release_reader.set()
        assert _process_until(qapp, lambda: window._log_load_thread is None)
        assert window.loaded_datasets == {}
        assert window.loaded_log_metadata == {}
        assert window.tree_model.invisibleRootItem().rowCount() == 0
        assert window._last_log_load_summary["loaded_count"] == 0
    finally:
        release_reader.set()
        _process_until(qapp, lambda: window._log_load_thread is None)
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_cancel_after_real_queued_file_signal_reports_zero_gui_commits(
    monkeypatch,
    tmp_path,
    qapp,
):
    source = tmp_path / "queued.csv"
    source.write_text("timestamp,value\n0,1\n", encoding="utf-8")
    entered_reader = threading.Event()
    release_reader = threading.Event()
    payload_emitted = threading.Event()

    class BlockingEngine:
        def load_result(self, file_path):
            entered_reader.set()
            assert release_reader.wait(5.0), "test did not release synthetic reader"
            return _result(file_path)

    monkeypatch.setattr(main_window_module, "LogIOEngine", BlockingEngine)
    window = MainWindow()
    window.show()
    qapp.processEvents()
    try:
        window.load_log_files([str(source)])
        assert entered_reader.wait(2.0)
        window._log_load_worker.fileLoaded.connect(
            lambda _payload: payload_emitted.set(),
            Qt.DirectConnection,
        )

        release_reader.set()
        # Do not process GUI events: the worker's fileLoaded signal is now
        # queued for MainWindow but has not reached the commit slot.
        assert payload_emitted.wait(2.0)
        assert window.loaded_datasets == {}
        assert window.cancel_log_load()

        assert _process_until(qapp, lambda: window._log_load_thread is None)
        assert window.loaded_datasets == {}
        assert window._last_log_load_summary["emitted_count"] == 1
        assert window._last_log_load_summary["loaded_count"] == 0
        assert window._last_log_load_summary["rejected_count"] == 1
    finally:
        release_reader.set()
        _process_until(qapp, lambda: window._log_load_thread is None)
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_ui_apply_failure_is_reported_as_zero_loaded(monkeypatch, qapp):
    window = MainWindow()
    window.show()
    qapp.processEvents()
    warnings = []
    monkeypatch.setattr(
        window,
        "_apply_loaded_log_result",
        lambda _payload: (_ for _ in ()).throw(RuntimeError("synthetic UI commit failure")),
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    try:
        window._log_load_committed_count = 0
        window._log_load_rejected_count = 0
        window._log_load_apply_failures = []
        window._on_log_file_loaded({"file_path": "candidate.csv", "filename": "candidate.csv"})
        window._on_log_load_finished(
            {
                "emitted_count": 1,
                "skipped_missing": [],
                "skipped_duplicate": [],
                "failed": [],
            }
        )

        assert window._last_log_load_summary["emitted_count"] == 1
        assert window._last_log_load_summary["loaded_count"] == 0
        assert window._last_log_load_summary["rejected_count"] == 1
        assert "synthetic UI commit failure" in window._last_log_load_summary["failed"][0]["error"]
        assert warnings and "synthetic UI commit failure" in warnings[0][1]
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
