"""Headless contract tests for mixed PX4/ArduPilot/ROS overlays.

These checks prove a UI capability only: each source owns an elapsed-time
axis whose zero is that log's origin.  They do not claim that equally named
signals from different ecosystems have the same physical meaning or units.
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
    import gui.main_window as main_window_module
    from gui.main_window import MainWindow, Workspace
except Exception as exc:  # pragma: no cover - only without optional GUI deps
    pytest.skip(f"Qt graph stack unavailable: {exc}", allow_module_level=True)


pg.setConfigOptions(useOpenGL=False)


SOURCE_SPECS = (
    ("px4.ulg", "px4_ulog", {"px4", "timeseries"}, 121, 12.0),
    ("mavlink.tlog", "mavlink_tlog", {"ardupilot", "mavlink", "timeseries"}, 51, 20.0),
    ("dataflash.bin", "ardupilot_dataflash", {"ardupilot", "timeseries"}, 81, 16.0),
    ("ros1.bag", "ros1_bag", {"ros", "ros1", "timeseries"}, 33, 8.0),
    ("ros2.db3", "ros2_bag", {"ros", "ros2", "timeseries"}, 301, 30.0),
)


def _dataset(file_name, source_format, capabilities, sample_count, duration):
    t = np.linspace(0.0, duration, sample_count, dtype=np.float64)
    frame = pl.DataFrame(
        {
            "timestamp_sec": t,
            # Deliberately use the same topic/field in every format: the
            # legend must still make each origin unambiguous.
            "value": np.sin(t * 0.2) + float(sample_count) / 1000.0,
            "value_sp": np.cos(t * 0.2),
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
def mixed_context(qapp):
    window = MainWindow()
    window.resize(1200, 700)
    window.show()
    qapp.processEvents()
    workspace = window.tab_widget.currentWidget()
    assert isinstance(workspace, Workspace)

    datasets = {}
    for file_name, source_format, capabilities, sample_count, duration in SOURCE_SPECS:
        dataset = _dataset(
            file_name,
            source_format,
            capabilities,
            sample_count,
            duration,
        )
        datasets[file_name] = dataset
        metadata = (
            {"evaluation": {"overall_status": "synthetic_px4"}}
            if source_format == "px4_ulog"
            else main_window_module._extract_log_metadata_for_dataset(file_name, dataset)
        )
        assert window._apply_loaded_log_result(
            {
                "file_path": file_name,
                "filename": file_name,
                "dataset": dataset,
                "aircraft_type": source_format,
                "metadata": metadata,
                "format_id": source_format,
            }
        )

    qapp.processEvents()
    try:
        yield window, workspace, datasets
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def _legend_texts(plot):
    legend = plot.plot.plotItem.legend
    return [str(getattr(label, "text", "")) for _sample, label in (legend.items or [])]


def test_all_formats_overlay_in_one_panel_with_distinct_rates_and_origins(
    mixed_context, qapp
):
    window, workspace, _datasets = mixed_context
    plot = workspace.first_plot

    for file_name, _fmt, _caps, _count, _duration in SOURCE_SPECS:
        assert plot.render_signal(file_name, "common_0", "value")

    qapp.processEvents()
    assert len(plot.plotted_signals) == len(SOURCE_SPECS)
    assert {len(cache["x"]) for cache in plot.signal_cache.values()} == {
        spec[3] for spec in SOURCE_SPECS
    }
    colors = [cache["color"].upper() for cache in plot.signal_cache.values()]
    assert len(set(colors)) == len(SOURCE_SPECS)
    assert colors == ["#FF0000", "#FF8800", "#FFD700", "#00FF00", "#0066FF"]
    assert workspace.global_min_x == pytest.approx(0.0)
    assert workspace.global_max_x == pytest.approx(30.0)

    labels = _legend_texts(plot)
    assert len(labels) == len(SOURCE_SPECS)
    assert len(set(labels)) == len(labels)
    for file_name, source_format, *_rest in SOURCE_SPECS:
        matches = [label for label in labels if file_name in label]
        assert len(matches) == 1
        assert "common_0.value" in matches[0]
        assert source_format == plot.signal_cache[
            f"{file_name}|common_0|value|timestamp_sec|False"
        ]["source_format"]

    workspace.set_time_range(3.0, 7.0, clamp_to_global=True)
    qapp.processEvents()
    x_range = plot.plot.getViewBox().viewRange()[0]
    assert x_range[0] == pytest.approx(3.0, abs=1e-6)
    assert x_range[1] == pytest.approx(7.0, abs=1e-6)

    # The foreign logs are present in the same MainWindow but remain outside
    # the PX4 evaluation contract.
    for file_name, source_format, *_rest in SOURCE_SPECS:
        if source_format == "px4_ulog":
            continue
        evaluation = window.loaded_log_metadata[file_name]["evaluation"]
        assert evaluation["overall_status"] == "unavailable"
        assert evaluation["items"] == []


def test_remove_and_readd_foreign_log_keeps_overlay_stable_and_rebuilds_bounds(
    mixed_context, qapp
):
    window, workspace, datasets = mixed_context
    plot = workspace.first_plot
    for file_name, *_rest in SOURCE_SPECS:
        assert plot.render_signal(file_name, "common_0", "value")

    assert window.delete_loaded_log("ros2.db3", ask_confirm=False)
    qapp.processEvents()
    assert len(plot.plotted_signals) == len(SOURCE_SPECS) - 1
    assert all(not uri.startswith("ros2.db3|") for uri in plot.plotted_signals)
    assert workspace.global_max_x == pytest.approx(20.0)

    ros2 = datasets["ros2.db3"]
    assert window._apply_loaded_log_result(
        {
            "file_path": "ros2.db3",
            "filename": "ros2.db3",
            "dataset": ros2,
            "aircraft_type": "ros2_bag",
            "metadata": main_window_module._extract_log_metadata_for_dataset("ros2.db3", ros2),
            "format_id": "ros2_bag",
        }
    )
    assert plot.render_signal("ros2.db3", "common_0", "value")
    qapp.processEvents()

    assert len(plot.plotted_signals) == len(SOURCE_SPECS)
    assert workspace.global_max_x == pytest.approx(30.0)
    assert len(set(_legend_texts(plot))) == len(SOURCE_SPECS)
    assert len({cache["color"].upper() for cache in plot.signal_cache.values()}) == len(SOURCE_SPECS)


def test_setpoint_red_policy_is_not_changed_by_palette_rotation(mixed_context):
    _window, workspace, _datasets = mixed_context
    plot = workspace.first_plot
    assert plot.render_signal("px4.ulg", "common_0", "value_sp")
    cache = plot.signal_cache["px4.ulg|common_0|value_sp|timestamp_sec|False"]
    assert cache["color"].upper() == "#FF4D4D"


def test_filter_legend_keeps_source_origin(mixed_context):
    _window, workspace, _datasets = mixed_context
    plot = workspace.first_plot
    uri = "ros1.bag|common_0|value|timestamp_sec|False"
    assert plot.render_signal("ros1.bag", "common_0", "value")
    plot._apply_data_filter(uri, "absolute", {}, "absolute value")
    assert "ros1.bag" in plot.signal_cache[uri]["legend_name"]
    assert "absolute value" in plot.signal_cache[uri]["legend_name"]
    assert any("ros1.bag" in label for label in _legend_texts(plot))


@pytest.mark.parametrize(
    "source_format",
    ["mavlink_tlog", "ardupilot_dataflash", "ros1_bag", "ros2_bag"],
)
def test_px4_metadata_fallback_is_strictly_isolated(
    monkeypatch, source_format
):
    dataset = _dataset(
        f"foreign-{source_format}.log",
        source_format,
        {"timeseries", "px4"},  # even a bad capability marker must not bypass format ID
        11,
        1.0,
    )

    class ForbiddenEvaluator:
        def __init__(self):
            raise AssertionError("PX4 evaluator must not be constructed for foreign logs")

    monkeypatch.setattr(main_window_module, "AutoEvaluationEngine", ForbiddenEvaluator)
    metadata = main_window_module._extract_log_metadata_for_dataset(
        dataset.source_path,
        dataset,
    )
    assert metadata["evaluation"]["overall_status"] == "unavailable"
    assert metadata["evaluation"]["items"] == []
