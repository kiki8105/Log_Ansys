"""Validate one-window PX4/ArduPilot/ROS overlay capability on real logs.

This is a UI interoperability test, not a claim that the chosen signals have
the same semantics or units.  Every reader supplies elapsed seconds from its
own log origin; explicit cross-log clock/event alignment is outside this gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


# Pin the promoted multi-format policy before reader modules are imported so
# an inherited ULG-only diagnostic environment cannot narrow this acceptance run.
os.environ["PX4_LOG_FORMAT_POLICY"] = "multiformat_stable"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PX4_INTERACTIVE_FAST_RENDER_PREVIEW", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtWidgets import QApplication
import numpy as np
import pyqtgraph as pg

from engines.io_engine import LogIOEngine
from gui.main_window import (
    MainWindow,
    Workspace,
    _metadata_from_load_result,
    _preprocess_loaded_dataset,
)
from readers.release_policy import FORMAT_POLICY_ENV, MULTIFORMAT_STABLE
from storage.parquet_cache import ParquetCacheManager


PREFERRED_SIGNALS = {
    "px4_ulog": (
        ("vehicle_attitude_0", "q[0]"),
        ("battery_status_0", "voltage_v"),
    ),
    "mavlink_tlog": (
        ("ATTITUDE_257", "roll"),
        ("VFR_HUD_257", "airspeed"),
    ),
    "ardupilot_dataflash": (
        ("ATT_0", "Roll"),
        ("GPS_0", "Spd"),
    ),
    "ros1_bag": (
        ("imu_data_0", "orientation.z"),
        ("gravity_vector_0", "vector.x"),
    ),
    "ros2_bag": (
        ("imu_0", "orientation.x"),
        ("battery_0", "voltage"),
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest().upper()


def _finite_xy(topic, signal_name):
    frame = topic.dataframe
    if frame is None or "timestamp_sec" not in frame.columns or signal_name not in frame.columns:
        return None
    try:
        x = np.asarray(frame["timestamp_sec"].to_numpy(), dtype=np.float64)
        y = np.asarray(frame[signal_name].to_numpy(), dtype=np.float64)
    except Exception:
        return None
    if x.size != y.size or x.size < 2:
        return None
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(finite)) < 2:
        return None
    return x[finite], y[finite]


def _select_signal(dataset):
    source_format = str(dataset.source_format)
    for topic_name, signal_name in PREFERRED_SIGNALS.get(source_format, ()):
        topic = dataset.topics.get(topic_name)
        if topic is not None and signal_name in topic.signals:
            xy = _finite_xy(topic, signal_name)
            if xy is not None:
                return topic_name, signal_name, xy

    candidates = []
    for topic_name, topic in dataset.topics.items():
        for signal_name in topic.signals:
            xy = _finite_xy(topic, signal_name)
            if xy is None:
                continue
            x, y = xy
            variation = float(np.nanmax(y) - np.nanmin(y)) if y.size else 0.0
            candidates.append((variation > 0.0, int(y.size), topic_name, signal_name, xy))
    if not candidates:
        raise RuntimeError(f"{source_format}: renderable numeric time series not found")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _varies, _count, topic_name, signal_name, xy = candidates[0]
    return topic_name, signal_name, xy


def _legend_texts(plot):
    legend = plot.plot.plotItem.legend
    return [str(getattr(label, "text", "")) for _sample, label in (legend.items or [])]


def _plot_bounds(plot):
    minima = []
    maxima = []
    for cache in plot.signal_cache.values():
        x = np.asarray(cache.get("x"), dtype=np.float64)
        finite = x[np.isfinite(x) & (x >= -5.0)]
        if finite.size:
            minima.append(float(np.min(finite)))
            maxima.append(float(np.max(finite)))
    return (min(minima), max(maxima)) if minima else (None, None)


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ulg", type=Path, required=True, help="PX4 ULog input")
    parser.add_argument(
        "--ardupilot",
        "--tlog",
        dest="tlog",
        type=Path,
        required=True,
        help="ArduPilot DataFlash or MAVLink telemetry log input",
    )
    parser.add_argument("--ros1", type=Path, required=True, help="ROS1 bag input")
    parser.add_argument("--ros2", type=Path, required=True, help="ROS2 bag database input")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runtime" / "test-artifacts" / "mixed-format-workspace",
    )
    return parser


def run(args) -> dict:
    started = time.perf_counter()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    paths = [args.ulg.resolve(), args.tlog.resolve(), args.ros1.resolve(), args.ros2.resolve()]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing acceptance inputs: " + ", ".join(missing))

    pg.setConfigOptions(useOpenGL=False)
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1500, 820)
    window.show()
    app.processEvents()
    workspace = window.tab_widget.currentWidget()
    if not isinstance(workspace, Workspace):
        raise RuntimeError("Initial workspace was not created")
    plot = workspace.first_plot
    plot.plot.setTitle("Mixed-format elapsed-time overlay (UI capability only)")

    engine = LogIOEngine(cache_mgr=ParquetCacheManager(cache_dir=str(cache_dir)))
    records = []
    loaded = {}
    metadata_by_name = {}
    selection_by_name = {}
    try:
        for path in paths:
            load_started = time.perf_counter()
            result = engine.load_result(str(path))
            load_ms = (time.perf_counter() - load_started) * 1000.0
            if result.format_id == "px4_ulog":
                _preprocess_loaded_dataset(result.dataset)
                metadata = {
                    "evaluation": {
                        "overall_score": None,
                        "overall_status": "not_run_in_mixed_ui_acceptance",
                        "overall_status_label": "PX4 전용 평가 별도 승인 대상",
                        "items": [],
                    }
                }
                aircraft_type = "PX4"
            else:
                metadata = _metadata_from_load_result(result)
                aircraft_type = result.format_id

            file_name = path.name
            payload = {
                "file_path": str(path),
                "filename": file_name,
                "dataset": result.dataset,
                "aircraft_type": aircraft_type,
                "metadata": metadata,
                "format_id": result.format_id,
            }
            if not window._apply_loaded_log_result(payload):
                raise RuntimeError(f"MainWindow rejected loaded source: {file_name}")
            topic_name, signal_name, (x, _y) = _select_signal(result.dataset)
            if not plot.render_signal(file_name, topic_name, signal_name):
                raise RuntimeError(f"Could not render {file_name}|{topic_name}|{signal_name}")
            loaded[file_name] = (result, payload)
            metadata_by_name[file_name] = metadata
            selection_by_name[file_name] = (topic_name, signal_name)
            records.append(
                {
                    "file_name": file_name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "format_id": result.format_id,
                    "reader_id": engine.last_load_info.get("reader_id"),
                    "load_type": engine.last_load_info.get("load_type"),
                    "load_ms": round(load_ms, 3),
                    "topic_count": len(result.dataset.topics),
                    "selected_topic": topic_name,
                    "selected_signal": signal_name,
                    "selected_samples": int(x.size),
                    "selected_start_sec": round(float(np.min(x)), 6),
                    "selected_end_sec": round(float(np.max(x)), 6),
                    "evaluation_status": metadata.get("evaluation", {}).get("overall_status"),
                }
            )

        app.processEvents()
        labels_before = _legend_texts(plot)
        expected_min, expected_max = _plot_bounds(plot)
        if expected_min is None or expected_max is None:
            raise RuntimeError("No mixed plot bounds")
        # Capture a window where every selected source actually has samples;
        # a percentage of the longest log would hide both short ROS traces.
        common_start = max(record["selected_start_sec"] for record in records)
        common_end = min(record["selected_end_sec"] for record in records)
        if common_end <= common_start:
            raise RuntimeError("Selected real signals do not share an elapsed-time overlap")
        common_span = common_end - common_start
        zoom_start = common_start + common_span * 0.02
        zoom_end = common_end - common_span * 0.02
        workspace.set_time_range(zoom_start, zoom_end, clamp_to_global=True)
        app.processEvents()
        view_range = plot.plot.getViewBox().viewRange()[0]

        overlay_png = output_dir / "mixed_overlay.png"
        pixmap = plot._capture_plot_pixmap(hide_wheel_overlay=True)
        if pixmap is None or pixmap.isNull() or not pixmap.save(str(overlay_png), "PNG"):
            raise RuntimeError("Failed to capture mixed overlay PNG")

        window_png = output_dir / "mixed_workspace.png"
        window_pixmap = window.grab()
        if window_pixmap.isNull() or not window_pixmap.save(str(window_png), "PNG"):
            raise RuntimeError("Failed to capture mixed workspace PNG")

        # Exercise the destructive-in-memory path without touching originals:
        # remove the longest ArduPilot source, verify the remaining panel and
        # shrunken global range, then add the same parsed dataset back.
        remove_name = args.tlog.name
        count_before_remove = len(plot.plotted_signals)
        if not window.delete_loaded_log(remove_name, ask_confirm=False):
            raise RuntimeError("ArduPilot in-memory remove failed")
        app.processEvents()
        count_after_remove = len(plot.plotted_signals)
        range_after_remove = [workspace.global_min_x, workspace.global_max_x]

        removed_result, removed_payload = loaded[remove_name]
        if not window._apply_loaded_log_result(dict(removed_payload)):
            raise RuntimeError("ArduPilot in-memory re-add failed")
        removed_topic, removed_signal = selection_by_name[remove_name]
        if not plot.render_signal(remove_name, removed_topic, removed_signal):
            raise RuntimeError("ArduPilot re-render failed")
        app.processEvents()
        count_after_readd = len(plot.plotted_signals)
        labels_after = _legend_texts(plot)
        colors_after = [
            str(cache.get("color", "")).upper()
            for cache in plot.signal_cache.values()
        ]
        expected_readd_min, expected_readd_max = _plot_bounds(plot)

        foreign_evaluation_isolated = all(
            metadata_by_name[record["file_name"]].get("evaluation", {}).get("overall_status")
            == "unavailable"
            for record in records
            if record["format_id"] != "px4_ulog"
        )
        checks = {
            "multiformat_policy_active": os.environ.get(FORMAT_POLICY_ENV) == MULTIFORMAT_STABLE,
            "all_four_sources_loaded": len(window.loaded_datasets) == 4,
            "one_panel_four_curves": count_after_readd == 4,
            "different_sample_rates": len({record["selected_samples"] for record in records}) == 4,
            "legend_file_origin_unique": (
                len(labels_before) == 4
                and len(set(labels_before)) == 4
                and all(record["file_name"] in " ".join(labels_before) for record in records)
                and len(set(labels_after)) == 4
            ),
            "overlay_colors_unique": len(colors_after) == 4 and len(set(colors_after)) == 4,
            "global_range_matches_curve_union": (
                abs(float(workspace.global_min_x) - float(expected_readd_min)) <= 1e-6
                and abs(float(workspace.global_max_x) - float(expected_readd_max)) <= 1e-6
            ),
            "zoom_applied": (
                abs(float(view_range[0]) - float(zoom_start)) <= 1e-6
                and abs(float(view_range[1]) - float(zoom_end)) <= 1e-6
            ),
            "remove_kept_other_curves": count_after_remove == count_before_remove - 1,
            "readd_restored_curve": count_after_readd == count_before_remove,
            "remove_recomputed_global_range": (
                range_after_remove[1] is not None
                and float(range_after_remove[1]) < float(expected_max)
            ),
            "px4_evaluation_isolated_from_foreign_logs": foreign_evaluation_isolated,
            "png_artifacts_written": overlay_png.is_file() and window_png.is_file(),
        }
        return {
            "status": "pass" if all(checks.values()) else "fail",
            "scope": "mixed-format UI overlay capability only",
            "semantic_warning": (
                "Signals are intentionally heterogeneous and are not asserted to share meaning or units. "
                "Each x-axis is elapsed seconds from that source's own origin; no event/clock alignment is claimed."
            ),
            "policy": os.environ.get(FORMAT_POLICY_ENV),
            "headless": True,
            "inputs": records,
            "legend_labels": labels_after,
            "curve_colors": colors_after,
            "global_range_sec": [workspace.global_min_x, workspace.global_max_x],
            "removed_and_readded": remove_name,
            "range_after_remove_sec": range_after_remove,
            "zoom_probe_sec": {
                "requested": [zoom_start, zoom_end],
                "observed": [float(view_range[0]), float(view_range[1])],
            },
            "checks": checks,
            "artifacts": {
                "overlay_png": str(overlay_png),
                "workspace_png": str(window_png),
            },
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    try:
        report = run(args)
    except Exception as exc:
        report = {
            "status": "error",
            "scope": "mixed-format UI overlay capability only",
            "error": f"{type(exc).__name__}: {exc}",
        }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"REPORT={report_path}")
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
