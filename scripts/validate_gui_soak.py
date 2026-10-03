"""Exercise a real ULog workspace through a repeatable native-Qt GUI soak.

Private flight logs are supplied only at runtime and are never repository
fixtures.  The JSON report and screenshots default to the git-ignored
``runtime/`` tree.  This runner intentionally drives the same Workspace and
AdvancedPlot methods used by the UI instead of replacing them with test
doubles.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the native Qt ULG GUI soak")
    parser.add_argument("ulg", type=Path, help="ULog source to exercise")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runtime/test-artifacts/gui-soak"),
        help="Ignored directory for the private report and screenshots",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=100,
        help="Zoom/pan/reset/split/close cycles (default: 100)",
    )
    parser.add_argument(
        "--viewport-sample-every",
        type=int,
        default=10,
        help="Capture/inspect every Nth cycle plus the first and last",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _rss_bytes() -> int:
    """Return current process RSS/working-set without an optional dependency."""

    if sys.platform == "win32":
        from ctypes import wintypes

        size_t = ctypes.c_size_t

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", size_t),
                ("WorkingSetSize", size_t),
                ("QuotaPeakPagedPoolUsage", size_t),
                ("QuotaPagedPoolUsage", size_t),
                ("QuotaPeakNonPagedPoolUsage", size_t),
                ("QuotaNonPagedPoolUsage", size_t),
                ("PagefileUsage", size_t),
                ("PeakPagefileUsage", size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        handle = get_current_process()
        ok = get_process_memory_info(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            raise ctypes.WinError()
        return int(counters.WorkingSetSize)

    proc_status = Path("/proc/self/status")
    if proc_status.is_file():
        for line in proc_status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    try:
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss if sys.platform == "darwin" else rss * 1024
    except Exception as exc:  # pragma: no cover - platform fallback
        raise RuntimeError("RSS measurement is unavailable") from exc


def _pick_signals(dataset, count: int = 3) -> list[tuple[str, str]]:
    preferred = (
        ("vehicle_attitude_0", "q[0]"),
        ("vehicle_angular_velocity_0", "xyz[1]"),
        ("battery_status_0", "voltage_v"),
        ("vehicle_local_position_0", "z"),
        ("vehicle_status_0", "nav_state"),
    )
    picked: list[tuple[str, str]] = []
    for topic_name, signal_name in preferred:
        topic = dataset.topics.get(topic_name)
        if topic is not None and signal_name in topic.signals:
            picked.append((topic_name, signal_name))
        if len(picked) == count:
            return picked
    for topic_name, topic in dataset.topics.items():
        for signal_name in topic.signals:
            candidate = (topic_name, signal_name)
            if candidate not in picked:
                picked.append(candidate)
            if len(picked) == count:
                return picked
    return picked


def _settle(app, *widgets, turns: int = 4) -> None:
    from PySide6 import QtCore

    for widget in widgets:
        if widget is None:
            continue
        try:
            widget.updateGeometry()
            widget.update()
        except (RuntimeError, AttributeError):
            pass
    for _ in range(max(1, int(turns))):
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)


def _range(plot) -> tuple[float, float]:
    values = plot.plot.getViewBox().viewRange()[0]
    return float(values[0]), float(values[1])


def _range_error(plots, expected: tuple[float, float]) -> float:
    errors = []
    for plot in plots:
        start_t, end_t = _range(plot)
        errors.extend((abs(start_t - expected[0]), abs(end_t - expected[1])))
    return max(errors, default=float("inf"))


def _finite_valid_range(plot) -> bool:
    try:
        start_t, end_t = _range(plot)
        return math.isfinite(start_t) and math.isfinite(end_t) and end_t > start_t
    except (RuntimeError, TypeError, ValueError):
        return False


def _capture_desktop_crop(widget):
    """Capture native child pixels using the composited desktop surface."""

    from PySide6.QtCore import QPoint, QRect
    from PySide6.QtGui import QGuiApplication

    screen = widget.screen() or QGuiApplication.primaryScreen()
    if screen is None:
        return None
    desktop = screen.grabWindow(0)
    if desktop is None or desktop.isNull():
        return None
    screen_rect = screen.geometry()
    origin = widget.mapToGlobal(QPoint(0, 0))
    scale_x = desktop.width() / max(1, screen_rect.width())
    scale_y = desktop.height() / max(1, screen_rect.height())
    crop = QRect(
        int(round((origin.x() - screen_rect.x()) * scale_x)),
        int(round((origin.y() - screen_rect.y()) * scale_y)),
        max(1, int(round(widget.width() * scale_x))),
        max(1, int(round(widget.height() * scale_y))),
    ).intersected(desktop.rect())
    if crop.isEmpty():
        return None
    return desktop.copy(crop)


def _pixmap_has_content(pixmap) -> bool:
    """Quickly reject null, all-black, or uniform capture candidates."""

    import numpy as np
    from PySide6.QtGui import QImage

    if pixmap is None or pixmap.isNull() or pixmap.width() < 2 or pixmap.height() < 2:
        return False
    image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
    width, height = int(image.width()), int(image.height())
    bytes_per_line = int(image.bytesPerLine())
    rgba = np.frombuffer(
        image.constBits(), dtype=np.uint8, count=image.sizeInBytes()
    )
    rgb = rgba.reshape(height, bytes_per_line)[:, : width * 4]
    rgb = rgb.reshape(height, width, 4)[::12, ::12, :3]
    return bool(
        rgb.size
        and int(np.max(rgb)) > 5
        and int(np.max(rgb)) - int(np.min(rgb)) > 5
    )


def _capture_plot_surface(plot, host_widget):
    """Capture a pyqtgraph surface including its native OpenGL viewport."""

    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QGuiApplication, QPixmap

    viewport = plot.plot.viewport()
    screen = host_widget.screen() or QGuiApplication.primaryScreen()
    if screen is not None and host_widget.isVisible():
        origin = viewport.mapTo(host_widget, QPoint(0, 0))
        pixmap = screen.grabWindow(
            int(host_widget.winId()),
            int(origin.x()),
            int(origin.y()),
            int(viewport.width()),
            int(viewport.height()),
        )
        if _pixmap_has_content(pixmap):
            return pixmap, "screen_window_crop"
        pixmap = _capture_desktop_crop(viewport)
        if _pixmap_has_content(pixmap):
            return pixmap, "desktop_screen_crop"
    grab_framebuffer = getattr(viewport, "grabFramebuffer", None)
    if callable(grab_framebuffer):
        image = grab_framebuffer()
        if image is not None and not image.isNull():
            pixmap = QPixmap.fromImage(image)
            if _pixmap_has_content(pixmap):
                return pixmap, "opengl_framebuffer"
    return plot.plot.grab(), "widget_grab"


def _capture_workspace_surface(workspace):
    from PySide6.QtGui import QGuiApplication

    screen = workspace.screen() or QGuiApplication.primaryScreen()
    if screen is not None and workspace.isVisible():
        pixmap = screen.grabWindow(int(workspace.winId()))
        if _pixmap_has_content(pixmap):
            return pixmap, "screen_window"
    pixmap = _capture_desktop_crop(workspace)
    if _pixmap_has_content(pixmap):
        return pixmap, "desktop_screen_crop"
    return workspace.grab(), "widget_grab"


def _activate_native_surface(app, workspace) -> dict[str, Any]:
    """Bring the soak window forward before objective compositor capture."""

    from PySide6.QtTest import QTest

    workspace.showNormal()
    workspace.raise_()
    workspace.activateWindow()
    handle = workspace.windowHandle()
    if handle is not None:
        try:
            handle.requestActivate()
        except Exception:
            pass
    _settle(app, workspace, turns=3)
    QTest.qWait(15)
    _settle(app, workspace, turns=2)
    handle = workspace.windowHandle()
    return {
        "visible": bool(workspace.isVisible()),
        "active": bool(workspace.isActiveWindow()),
        "exposed": bool(handle is not None and handle.isExposed()),
        "minimized": bool(workspace.isMinimized()),
        "size": [int(workspace.width()), int(workspace.height())],
    }


def _viewport_metrics(plot, host_widget) -> dict[str, Any]:
    """Inspect a rendered viewport for null, uniform, or all-black output."""

    import numpy as np
    from PySide6.QtGui import QImage

    viewport = plot.plot.viewport()
    pixmap, capture_mode = _capture_plot_surface(plot, host_widget)
    result: dict[str, Any] = {
        "logical_size": [int(viewport.width()), int(viewport.height())],
        "pixmap_null": bool(pixmap.isNull()),
        "capture_mode": capture_mode,
        "valid_range": _finite_valid_range(plot),
        "all_black_indicator": True,
        "uniform_indicator": True,
    }
    if pixmap.isNull():
        return result
    try:
        image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        width, height = int(image.width()), int(image.height())
        bytes_per_line = int(image.bytesPerLine())
        rgba = np.frombuffer(
            image.constBits(), dtype=np.uint8, count=image.sizeInBytes()
        )
        rgba = rgba.reshape(height, bytes_per_line)[:, : width * 4]
        rgb = rgba.reshape(height, width, 4)[:, :, :3]
        # Keep analysis bounded on high-DPI displays while sampling the whole
        # surface rather than a special region chosen for this particular log.
        stride = max(1, int(math.sqrt(max(1, width * height) / 12_000)))
        sampled = rgb[::stride, ::stride].reshape(-1, 3)
        black_fraction = float(np.mean(np.all(sampled <= 4, axis=1)))
        luminance = sampled.astype(np.float32).mean(axis=1)
        luminance_span = float(np.max(luminance) - np.min(luminance))
        channel_span = int(np.max(sampled) - np.min(sampled))
        channel_std = float(np.std(sampled.astype(np.float32)))
        result.update(
            {
                "capture_size": [width, height],
                "sample_stride": stride,
                "sample_count": int(len(sampled)),
                "black_fraction": black_fraction,
                "luminance_span": luminance_span,
                "channel_span": channel_span,
                "channel_std": channel_std,
                "all_black_indicator": bool(black_fraction >= 0.995),
                "uniform_indicator": bool(
                    channel_span < 3 or (luminance_span < 3.0 and channel_std < 1.0)
                ),
            }
        )
    except Exception as exc:
        result["inspection_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _aggregate_timings(cycles: list[dict[str, Any]], action: str) -> dict[str, float]:
    import numpy as np

    values = np.asarray(
        [float(item["timings_ms"][action]) for item in cycles], dtype=np.float64
    )
    return {
        "min_ms": round(float(np.min(values)), 3),
        "median_ms": round(float(np.median(values)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "max_ms": round(float(np.max(values)), 3),
    }


def main() -> int:
    args = _parse_args()
    if args.cycles < 1:
        raise ValueError("--cycles must be at least 1")
    if args.viewport_sample_every < 1:
        raise ValueError("--viewport-sample-every must be at least 1")

    source = args.ulg.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing ULog source: {source}")

    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PX4_LOG_FORMAT_POLICY", "ulg_stable")
    os.environ.setdefault("PX4_INTERACTIVE_FAST_RENDER_PREVIEW", "0")

    # PySide must create the shared-context application before pyqtgraph and
    # the product UI instantiate any QOpenGLWidget-backed plot surfaces.
    from PySide6 import QtCore
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    try:
        QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts, True)
    except Exception:
        pass
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    qpa_name = str(app.platformName()).lower()
    if qpa_name in {"offscreen", "minimal", "vnc"}:
        raise RuntimeError(
            f"Native GUI soak requires a real window-system QPA; got {qpa_name!r}"
        )

    from engines.io_engine import LogIOEngine
    from gui.main_window import MainWindow, Workspace, _preprocess_loaded_dataset
    from storage.parquet_cache import ParquetCacheManager

    qt_messages: list[dict[str, Any]] = []
    qt_message_counts: dict[str, int] = {}

    def qt_message_handler(mode, _context, message):
        label = getattr(mode, "name", str(mode))
        qt_message_counts[label] = qt_message_counts.get(label, 0) + 1
        if len(qt_messages) < 200:
            qt_messages.append({"type": label, "message": str(message)})

    previous_qt_handler = QtCore.qInstallMessageHandler(qt_message_handler)
    unhandled: list[str] = []
    previous_excepthook = sys.excepthook

    def exception_hook(exc_type, exc_value, exc_tb):
        unhandled.append(
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb))[-8000:]
        )

    sys.excepthook = exception_hook

    workspace = None
    window = None
    started_at = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "completed": False,
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
            "size_bytes": source.stat().st_size,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "qt": QtCore.qVersion(),
            "qpa": qpa_name,
            "pid": os.getpid(),
        },
        "requested_cycles": int(args.cycles),
        "cycles": [],
        "errors": [],
    }

    try:
        with tempfile.TemporaryDirectory(prefix="gui-soak-cache-", dir=output_dir) as cache_dir:
            load_started = time.perf_counter()
            engine = LogIOEngine(cache_mgr=ParquetCacheManager(cache_dir=cache_dir))
            load_result = engine.load_result(str(source))
            report["load"] = {
                "format_id": load_result.format_id,
                "load_type": engine.last_load_info.get("load_type"),
                "elapsed_ms": round(_elapsed_ms(load_started), 3),
            }
            if load_result.format_id != "px4_ulog":
                raise RuntimeError(
                    f"Expected px4_ulog input, received {load_result.format_id!r}"
                )
            dataset = load_result.dataset
            _preprocess_loaded_dataset(dataset)
            selected = _pick_signals(dataset, 3)
            if len(selected) < 3:
                raise RuntimeError("At least three numeric time signals are required")

            window = MainWindow()
            window.loaded_datasets[source.name] = dataset
            workspace = Workspace(window)
            workspace.resize(1500, 850)
            workspace.create_grid(1, 3)
            workspace.show()
            workspace.raise_()
            workspace.activateWindow()
            _settle(app, workspace, turns=8)

            plots = [workspace.get_plot(0, index) for index in range(3)]
            rendered = [
                plot.render_signal(source.name, *selected[index])
                for index, plot in enumerate(plots)
            ]
            _settle(app, workspace, *(plot.plot for plot in plots), turns=8)
            if not all(rendered):
                raise RuntimeError("Initial three-panel render failed")

            lo = float(workspace.global_min_x)
            hi = float(workspace.global_max_x)
            if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
                raise RuntimeError(f"Invalid workspace range: {(lo, hi)!r}")
            full_range = (lo, hi)
            full_span = hi - lo

            gc.collect()
            _settle(app, workspace, turns=4)
            rss_start = _rss_bytes()
            rss_peak = rss_start
            rss_samples = [rss_start]
            initial_surface_state = _activate_native_surface(app, workspace)
            initial_viewports = [_viewport_metrics(plot, workspace) for plot in plots]
            report["initial"] = {
                "signals": [f"{topic}.{signal}" for topic, signal in selected],
                "global_range": [lo, hi],
                "plot_count": len(workspace._iter_workspace_plots()),
                "rss_bytes": rss_start,
                "surface_state": initial_surface_state,
                "viewports": initial_viewports,
            }

            for cycle_number in range(1, args.cycles + 1):
                cycle_started = time.perf_counter()
                timings: dict[str, float] = {}
                paint_drains: dict[str, float] = {}
                failures: list[str] = []
                range_errors: dict[str, float] = {}

                # Zoom: exercise the wheel-linked range path, varying center
                # and width while remaining inside the log's global extent.
                zoom_width_fraction = 0.18 + 0.02 * (cycle_number % 7)
                zoom_width = full_span * zoom_width_fraction
                center_fraction = 0.25 + 0.5 * ((cycle_number % 19) / 18.0)
                zoom_center = lo + full_span * center_fraction
                zoom_start = max(lo, min(hi - zoom_width, zoom_center - zoom_width / 2.0))
                zoom_target = (zoom_start, zoom_start + zoom_width)
                action_started = time.perf_counter()
                workspace._wheel_active = True
                workspace._wheel_begin_or_update(
                    *zoom_target,
                    source_plot=plots[cycle_number % len(plots)],
                    started_at=action_started,
                    local_debug={},
                )
                timings["zoom"] = _elapsed_ms(action_started)
                paint_started = time.perf_counter()
                idle_timer = getattr(workspace, "_wheel_idle_timer", None)
                if idle_timer is not None and idle_timer.isActive():
                    idle_timer.stop()
                # Mirror the real wheel burst lifecycle.  Merely clearing
                # _wheel_active leaves a stale commit timer that can overwrite
                # the following pan operation hundreds of milliseconds later.
                workspace._handle_wheel_idle_transition()
                _settle(app, workspace, turns=2)
                paint_drains["zoom"] = _elapsed_ms(paint_started)
                range_errors["zoom"] = _range_error(plots, zoom_target)
                zoom_observed = [list(_range(plot)) for plot in plots]
                if range_errors["zoom"] > 1e-6:
                    failures.append("zoom_link_range")

                # Pan: preserve the zoom width but move its center.
                pan_room = full_span - zoom_width
                pan_fraction = ((cycle_number * 7) % 97) / 96.0
                pan_start = lo + pan_room * pan_fraction
                pan_target = (pan_start, pan_start + zoom_width)
                action_started = time.perf_counter()
                workspace.set_time_range(*pan_target, clamp_to_global=True)
                timings["pan"] = _elapsed_ms(action_started)
                paint_started = time.perf_counter()
                _settle(app, workspace, turns=2)
                paint_drains["pan"] = _elapsed_ms(paint_started)
                range_errors["pan"] = _range_error(plots, pan_target)
                pan_observed = [list(_range(plot)) for plot in plots]
                if range_errors["pan"] > 1e-6:
                    failures.append("pan_link_range")

                action_started = time.perf_counter()
                workspace.reset_zoom()
                timings["reset"] = _elapsed_ms(action_started)
                paint_started = time.perf_counter()
                _settle(app, workspace, turns=2)
                paint_drains["reset"] = _elapsed_ms(paint_started)
                range_errors["reset"] = _range_error(plots, full_range)
                reset_observed = [list(_range(plot)) for plot in plots]
                if range_errors["reset"] > 1e-6:
                    failures.append("reset_link_range")

                # Alternate horizontal/vertical split topology, render into
                # the new leaf, then close it and verify the original three
                # linked plots survive.
                split_target = plots[cycle_number % len(plots)]
                orientation = Qt.Horizontal if cycle_number % 2 else Qt.Vertical
                action_started = time.perf_counter()
                new_plot = split_target.split_layout(orientation)
                if new_plot is not None:
                    new_plot.render_signal(
                        source.name,
                        *selected[(cycle_number + 1) % len(selected)],
                    )
                timings["split"] = _elapsed_ms(action_started)
                paint_started = time.perf_counter()
                _settle(app, workspace, turns=3)
                paint_drains["split"] = _elapsed_ms(paint_started)
                split_count = len(workspace._iter_workspace_plots())
                if new_plot is None or split_count != 4:
                    failures.append(f"split_plot_count:{split_count}")

                action_started = time.perf_counter()
                if new_plot is not None:
                    new_plot.close_plot()
                timings["close"] = _elapsed_ms(action_started)
                paint_started = time.perf_counter()
                _settle(app, workspace, turns=4)
                paint_drains["close"] = _elapsed_ms(paint_started)
                close_count = len(workspace._iter_workspace_plots())
                if close_count != 3:
                    failures.append(f"close_plot_count:{close_count}")
                if any(not _finite_valid_range(plot) for plot in plots):
                    failures.append("invalid_view_range")

                viewport_checks: list[dict[str, Any]] = []
                surface_state: dict[str, Any] | None = None
                should_sample = (
                    cycle_number == 1
                    or cycle_number == args.cycles
                    or cycle_number % args.viewport_sample_every == 0
                )
                if should_sample:
                    surface_state = _activate_native_surface(app, workspace)
                    _settle(
                        app,
                        workspace,
                        *(plot.plot.viewport() for plot in plots),
                        turns=3,
                    )
                    viewport_checks = [
                        _viewport_metrics(plot, workspace) for plot in plots
                    ]
                    for index, metrics in enumerate(viewport_checks):
                        if metrics.get("pixmap_null"):
                            failures.append(f"viewport_{index}_null")
                        if not metrics.get("valid_range"):
                            failures.append(f"viewport_{index}_range")
                        if metrics.get("all_black_indicator"):
                            failures.append(f"viewport_{index}_black")
                        if metrics.get("uniform_indicator"):
                            failures.append(f"viewport_{index}_uniform")
                        if metrics.get("inspection_error"):
                            failures.append(f"viewport_{index}_inspection")

                rss_now = _rss_bytes()
                rss_samples.append(rss_now)
                rss_peak = max(rss_peak, rss_now)
                report["cycles"].append(
                    {
                        "cycle": cycle_number,
                        "elapsed_ms": round(_elapsed_ms(cycle_started), 3),
                        "timings_ms": {
                            key: round(value, 3) for key, value in timings.items()
                        },
                        "paint_drain_ms": {
                            key: round(value, 3) for key, value in paint_drains.items()
                        },
                        "range_errors_s": range_errors,
                        "ranges": {
                            "zoom": {
                                "target": list(zoom_target),
                                "observed": zoom_observed,
                            },
                            "pan": {
                                "target": list(pan_target),
                                "observed": pan_observed,
                            },
                            "reset": {
                                "target": list(full_range),
                                "observed": reset_observed,
                            },
                        },
                        "plot_count_after_close": close_count,
                        "rss_bytes": rss_now,
                        "failures": failures,
                        "surface_state": surface_state,
                        "viewport_checks": viewport_checks,
                    }
                )

            # Measure retained working set both before and after deferred Qt
            # deletion and Python collection.  The settled value is used for
            # acceptance, while the raw value helps diagnose allocator caches.
            rss_end_raw = _rss_bytes()
            _settle(app, workspace, turns=8)
            gc.collect()
            _settle(app, workspace, turns=8)
            rss_end = _rss_bytes()
            rss_peak = max(rss_peak, rss_end_raw, rss_end)

            final_range_error = _range_error(plots, full_range)
            final_surface_state = _activate_native_surface(app, workspace)
            final_viewports = [_viewport_metrics(plot, workspace) for plot in plots]
            final_screenshot = output_dir / "final_workspace.png"
            final_workspace_pixmap, final_workspace_capture_mode = (
                _capture_workspace_surface(workspace)
            )
            final_workspace_pixmap.save(str(final_screenshot))
            for index, plot in enumerate(plots, start=1):
                final_plot_pixmap, _capture_mode = _capture_plot_surface(plot, workspace)
                final_plot_pixmap.save(str(output_dir / f"final_viewport_{index}.png"))

            cycle_failures = [
                {"cycle": item["cycle"], "failures": item["failures"]}
                for item in report["cycles"]
                if item["failures"]
            ]
            delta = rss_end - rss_start
            delta_percent = (delta / rss_start * 100.0) if rss_start else None
            allowed_growth = max(100 * 1024 * 1024, int(rss_start * 0.10))
            memory_pass = delta <= allowed_growth
            viewport_failures = [
                metrics
                for metrics in final_viewports
                if metrics.get("pixmap_null")
                or metrics.get("all_black_indicator")
                or metrics.get("uniform_indicator")
                or not metrics.get("valid_range")
                or metrics.get("inspection_error")
            ]
            critical_qt_messages = [
                item
                for item in qt_messages
                if "critical" in item["type"].lower()
                or "fatal" in item["type"].lower()
                or any(
                    token in item["message"].lower()
                    for token in ("failed to create context", "makecurrent", "shader error")
                )
            ]
            report.update(
                {
                    "completed": True,
                    "completed_cycles": len(report["cycles"]),
                    "duration_ms": round(_elapsed_ms(started_at), 3),
                    "timing_summary": {
                        action: _aggregate_timings(report["cycles"], action)
                        for action in ("zoom", "pan", "reset", "split", "close")
                    },
                    "paint_drain_summary": {
                        action: _aggregate_timings(
                            [
                                {"timings_ms": item["paint_drain_ms"]}
                                for item in report["cycles"]
                            ],
                            action,
                        )
                        for action in ("zoom", "pan", "reset", "split", "close")
                    },
                    "range_summary": {
                        "final_expected": list(full_range),
                        "final_observed": [list(_range(plot)) for plot in plots],
                        "final_max_error_s": final_range_error,
                        "maximum_cycle_error_s": max(
                            (
                                max(item["range_errors_s"].values())
                                for item in report["cycles"]
                            ),
                            default=float("inf"),
                        ),
                    },
                    "memory": {
                        "rss_start_bytes": rss_start,
                        "rss_peak_bytes": rss_peak,
                        "rss_end_raw_bytes": rss_end_raw,
                        "rss_end_settled_bytes": rss_end,
                        "rss_delta_bytes": delta,
                        "rss_delta_percent": round(delta_percent, 3)
                        if delta_percent is not None
                        else None,
                        "allowed_growth_bytes": allowed_growth,
                        "acceptance": "pass" if memory_pass else "fail",
                        "sample_count": len(rss_samples),
                    },
                    "final": {
                        "plot_count": len(workspace._iter_workspace_plots()),
                        "surface_state": final_surface_state,
                        "viewports": final_viewports,
                        "screenshot": str(final_screenshot),
                        "screenshot_capture_mode": final_workspace_capture_mode,
                    },
                    "qt_messages": {
                        "counts": qt_message_counts,
                        "captured": qt_messages,
                        "critical": critical_qt_messages,
                        "capture_limit": 200,
                    },
                    "unhandled_exceptions": unhandled,
                    "cycle_failures": cycle_failures,
                }
            )
            report["acceptance"] = {
                "status": "pass"
                if (
                    len(report["cycles"]) == args.cycles
                    and not cycle_failures
                    and not viewport_failures
                    and not critical_qt_messages
                    and not unhandled
                    and memory_pass
                    and final_range_error <= 1e-6
                    and len(workspace._iter_workspace_plots()) == 3
                )
                else "fail",
                "criteria": {
                    "all_cycles_completed": len(report["cycles"]) == args.cycles,
                    "no_cycle_failures": not cycle_failures,
                    "no_final_viewport_failures": not viewport_failures,
                    "no_critical_qt_messages": not critical_qt_messages,
                    "no_unhandled_exceptions": not unhandled,
                    "rss_growth_within_budget": memory_pass,
                    "final_linked_range_exact": final_range_error <= 1e-6,
                    "final_plot_count_is_three": len(workspace._iter_workspace_plots()) == 3,
                },
                "memory_budget_note": (
                    "Settled RSS growth must be <= the looser of 100 MiB or "
                    "10% of initialized GUI RSS."
                ),
            }
    except Exception as exc:
        report["errors"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        report["duration_ms"] = round(_elapsed_ms(started_at), 3)
        report["qt_messages"] = {
            "counts": qt_message_counts,
            "captured": qt_messages,
            "capture_limit": 200,
        }
        report["unhandled_exceptions"] = unhandled
        report["acceptance"] = {"status": "fail", "criteria": {"completed": False}}
    finally:
        if workspace is not None:
            try:
                workspace.close()
                workspace.deleteLater()
            except RuntimeError:
                pass
        if window is not None:
            try:
                window.close()
                window.deleteLater()
            except RuntimeError:
                pass
        _settle(app, turns=8)
        sys.excepthook = previous_excepthook
        QtCore.qInstallMessageHandler(previous_qt_handler)

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "report": str(report_path),
        "status": report.get("acceptance", {}).get("status", "fail"),
        "completed_cycles": report.get("completed_cycles", len(report["cycles"])),
        "rss_delta_bytes": report.get("memory", {}).get("rss_delta_bytes"),
        "cycle_failure_count": len(report.get("cycle_failures", ())),
        "qt_critical_count": len(report.get("qt_messages", {}).get("critical", ())),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
