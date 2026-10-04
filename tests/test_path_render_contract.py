"""Regression tests for the ENU contract at the two 3D entry points."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from core.log_model import LogDataset, TopicInstance  # noqa: E402
    from gui.main_window import AdvancedPlot, MainWindow  # noqa: E402
except Exception as exc:  # pragma: no cover - only when optional GUI deps are absent
    pytest.skip(f"Qt graph stack unavailable: {exc}", allow_module_level=True)


class _ColorStub:
    @staticmethod
    def name():
        return "#123456"


class _CapturePlot:
    def __init__(self, main_window):
        self.main_window = main_window
        self._true_3d_enabled = False
        self._3d_path_style = {"color": _ColorStub(), "style": "solid", "width": 2.0}
        self._true_3d_vehicle_marker_type = "auto"
        self._true_3d_aircraft_scale_factor = 1.0
        self.layout_special_spec = None
        self.rendered = None
        self.vehicle_marker_config = None

    _parse_signal_uri = staticmethod(AdvancedPlot._parse_signal_uri)
    _signal_axis_role = staticmethod(AdvancedPlot._signal_axis_role)
    _sync_3d_style_to_layout_spec = AdvancedPlot._sync_3d_style_to_layout_spec

    def render_3d_path(self, x, y, z, title, timestamps=None, value_extras=None):
        self.rendered = (
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(z, dtype=np.float64),
            np.asarray(timestamps, dtype=np.float64),
        )
        return True

    def configure_true_3d_vehicle_marker(self, **kwargs):
        self.vehicle_marker_config = dict(kwargs)
        return True


class _MainWindowStub:
    def __init__(self, file_name, dataset):
        self.loaded_datasets = {file_name: dataset}
        self.loaded_aircraft_types = {file_name: "Fixed-Wing"}

    @staticmethod
    def _find_topic_by_prefixes(dataset, prefixes):
        values = (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
        for prefix in values:
            for topic_name in dataset.topics:
                if topic_name.startswith(prefix):
                    return topic_name
        return None

    @staticmethod
    def _find_signal_in_topic(dataset, topic_name, candidates):
        if not topic_name or topic_name not in dataset.topics:
            return None
        columns = set(dataset.topics[topic_name].dataframe.columns)
        return next((candidate for candidate in candidates if candidate in columns), None)

    @staticmethod
    def _build_3d_path_value_extras(dataset, timestamps, x, y, z):
        return None


def _dataset(topic_base, data):
    topic = TopicInstance(topic_base, 0, dataframe=pl.DataFrame(data))
    topic.refresh_signals()
    dataset = LogDataset(source_format="px4_ulog", source_path="contract.ulg")
    dataset.add_topic(topic)
    return dataset, topic.unique_name


def _render_both_entry_points(dataset, topic_name, signals):
    file_name = "contract.ulg"
    main_window = _MainWindowStub(file_name, dataset)

    drop_plot = _CapturePlot(main_window)
    uris = [f"{file_name}|{topic_name}|{signal}|timestamp_sec" for signal in signals]
    assert AdvancedPlot._try_render_3d_from_signal_uris(drop_plot, uris)

    auto_plot = _CapturePlot(main_window)
    assert MainWindow._render_3d_flight_path_in_plot(
        main_window,
        auto_plot,
        file_name,
        dataset,
    )
    assert drop_plot.vehicle_marker_config == {
        "detected_aircraft_type": "Fixed-Wing",
        "refresh": False,
    }
    assert auto_plot.vehicle_marker_config == drop_plot.vehicle_marker_config
    assert drop_plot.layout_special_spec["vehicle_marker_type"] == "auto"
    assert drop_plot.layout_special_spec["vehicle_marker_scale"] == 1.0
    return drop_plot.rendered, auto_plot.rendered


def test_direct_drop_and_auto_panel_match_for_local_ned_data():
    dataset, topic_name = _dataset(
        "vehicle_local_position",
        {
            "timestamp_sec": [0.0, 1.0, 2.0],
            "x": [100.0, 101.0, 104.0],       # North
            "y": [200.0, 202.0, 203.0],       # East
            "z": [-5.0, -8.0, -6.0],          # Down
        },
    )

    direct, automatic = _render_both_entry_points(dataset, topic_name, ("x", "y", "z"))

    for direct_axis, automatic_axis in zip(direct, automatic):
        np.testing.assert_allclose(direct_axis, automatic_axis)
    np.testing.assert_allclose(direct[0], [0.0, 2.0, 3.0])  # X = East
    np.testing.assert_allclose(direct[1], [0.0, 1.0, 4.0])  # Y = North
    np.testing.assert_allclose(direct[2], [0.0, 3.0, 1.0])  # Z = Up


def test_direct_drop_and_auto_panel_match_for_global_lla_data():
    dataset, topic_name = _dataset(
        "vehicle_global_position",
        {
            "timestamp_sec": [0.0, 1.0, 2.0],
            "lon": [127.0, 127.0, 127.0001],
            "lat": [37.0, 37.0001, 37.0001],
            "alt": [50.0, 53.0, 55.0],
        },
    )

    direct, automatic = _render_both_entry_points(dataset, topic_name, ("lon", "lat", "alt"))

    for direct_axis, automatic_axis in zip(direct, automatic):
        np.testing.assert_allclose(direct_axis, automatic_axis)
    assert direct[0][2] > 8.0    # longitude change is East/X
    assert direct[1][1] > 11.0   # latitude change is North/Y
    np.testing.assert_allclose(direct[2], [0.0, 3.0, 5.0])
