"""Run repeatable graph acceptance checks against one or more real ULog files.

This runner is intentionally separate from pytest: private flight logs are not
repository fixtures, and render timings are evidence rather than deterministic
CI assertions.  It writes a JSON report plus time-series, 2D, and 3D visual
evidence per log.
True OpenGL and physical wheel feel still require the Windows GPU checklist in
``docs/ULG_GRAPH_ACCEPTANCE_PLAN.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ULG graph acceptance contracts")
    parser.add_argument("ulg", nargs="+", type=Path, help="ULog source(s) to validate")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runtime/test-artifacts/ulg-graph-acceptance"),
    )
    parser.add_argument(
        "--onscreen",
        action="store_true",
        help="Show the Qt window. Headless QPainter rendering is the default.",
    )
    parser.add_argument(
        "--cache-cycles",
        type=int,
        default=1,
        help="Number of isolated cold+warm cache cycles per ULog (default: 1).",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _range(plot) -> tuple[float, float]:
    values = plot.plot.getViewBox().viewRange()[0]
    return float(values[0]), float(values[1])


def _range_error(plots, expected: tuple[float, float]) -> float:
    return max(
        max(abs(_range(plot)[0] - expected[0]), abs(_range(plot)[1] - expected[1]))
        for plot in plots
    )


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
        if len(picked) >= count:
            return picked
    for topic_name, topic in dataset.topics.items():
        for signal_name in topic.signals:
            candidate = (topic_name, signal_name)
            if candidate not in picked:
                picked.append(candidate)
            if len(picked) >= count:
                return picked
    return picked


def _check(condition: bool, *, detail: Any = None) -> dict[str, Any]:
    return {"status": "pass" if condition else "fail", "detail": detail}


def _settle_qt_render(app, *widgets, turns: int = 8) -> None:
    """Drain layout/paint work after replacing a workspace grid.

    ``Workspace.create_grid`` schedules both old-widget deletion and splitter
    rebalancing.  A single ``processEvents`` call can therefore finish the
    layout timer without delivering the subsequent GraphicsView paint.  That
    produced a valid 2D path state and ViewBox range but a blank screenshot.
    Several bounded event-loop turns keep this deterministic without sleeping.
    """

    from PySide6 import QtCore

    for widget in widgets:
        if widget is None:
            continue
        try:
            widget.updateGeometry()
            widget.update()
        except Exception:
            pass
    for _ in range(max(1, int(turns))):
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        for widget in widgets:
            if widget is None:
                continue
            try:
                widget.update()
            except Exception:
                pass


def _ensure_widget_exposed(app, widget, *, attempts: int = 20) -> dict[str, Any]:
    """Wait briefly until a sequential native test window is composited."""

    from PySide6.QtTest import QTest

    widget.show()
    widget.raise_()
    widget.activateWindow()
    handle = widget.windowHandle()
    if handle is not None:
        try:
            handle.requestActivate()
        except Exception:
            pass
    for attempt in range(1, max(1, int(attempts)) + 1):
        _settle_qt_render(app, widget, turns=2)
        handle = widget.windowHandle()
        exposed = bool(handle is not None and handle.isExposed())
        usable_size = bool(widget.width() > 1 and widget.height() > 1)
        if widget.isVisible() and exposed and usable_size:
            return {
                "ready": True,
                "attempts": attempt,
                "size": [int(widget.width()), int(widget.height())],
            }
        QTest.qWait(25)
    return {
        "ready": False,
        "attempts": max(1, int(attempts)),
        "visible": bool(widget.isVisible()),
        "exposed": bool(widget.windowHandle() and widget.windowHandle().isExposed()),
        "size": [int(widget.width()), int(widget.height())],
    }


def _path_geometry_visibility(plot) -> dict[str, Any]:
    """Return objective data/view/scene checks for the primary 2D path."""

    import numpy as np

    state = getattr(plot, "_flight_path_2d_state", None)
    item = getattr(plot, "_flight_path_2d_curve_item", None)
    result: dict[str, Any] = {
        "finite_point_count": 0,
        "item_visible": False,
        "data_intersects_view": False,
        "scene_bounds_nonempty": False,
    }
    if not isinstance(state, dict) or item is None:
        return result
    try:
        x = np.asarray(state.get("x", ()), dtype=np.float64)
        y = np.asarray(state.get("y", ()), dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        result["finite_point_count"] = int(len(x))
        if len(x) >= 2:
            (view_x0, view_x1), (view_y0, view_y1) = plot.plot.getViewBox().viewRange()
            data_x0, data_x1 = float(np.min(x)), float(np.max(x))
            data_y0, data_y1 = float(np.min(y)), float(np.max(y))
            result.update(
                {
                    "data_bounds": [[data_x0, data_x1], [data_y0, data_y1]],
                    "view_range": [
                        [float(view_x0), float(view_x1)],
                        [float(view_y0), float(view_y1)],
                    ],
                    "data_intersects_view": bool(
                        data_x1 >= view_x0
                        and data_x0 <= view_x1
                        and data_y1 >= view_y0
                        and data_y0 <= view_y1
                    ),
                }
            )
        result["item_visible"] = bool(item.isVisible())
        # PlotDataItem is only a container and may report an empty scene rect;
        # its child PlotCurveItem owns the actual painted geometry.
        painted_item = getattr(item, "curve", None) or item
        scene_rect = painted_item.sceneBoundingRect()
        result["scene_bounds_nonempty"] = bool(
            scene_rect.isValid() and not scene_rect.isEmpty()
        )
        result["scene_bounds"] = [
            float(scene_rect.x()),
            float(scene_rect.y()),
            float(scene_rect.width()),
            float(scene_rect.height()),
        ]
    except Exception as exc:
        result["inspection_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _pixmap_has_content(pixmap, *, minimum_width: int = 2, minimum_height: int = 2) -> bool:
    """Reject null, tiny, uniformly black, or uniformly blank captures."""

    import numpy as np
    from PySide6.QtGui import QImage

    if (
        pixmap is None
        or pixmap.isNull()
        or pixmap.width() < minimum_width
        or pixmap.height() < minimum_height
    ):
        return False
    image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
    width, height = int(image.width()), int(image.height())
    bytes_per_line = int(image.bytesPerLine())
    rgba = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.sizeInBytes())
    rgba = rgba.reshape(height, bytes_per_line)[:, : width * 4].reshape(height, width, 4)
    rgb = rgba[::8, ::8, :3]
    return bool(rgb.size and int(rgb.max()) > 5 and int(rgb.max()) - int(rgb.min()) > 5)


def _capture_desktop_crop(widget):
    """Capture a visible widget via desktop coordinates, independent of HWND reuse."""

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


def _capture_plot_surface(plot_widget, *, host_widget=None):
    """Capture a PlotWidget, including its QOpenGLWidget viewport.

    QWidget.grab() omits native OpenGL child surfaces on Windows.  pyqtgraph
    uses such a viewport when ``useOpenGL=True``, so native acceptance first
    crops the composited window pixels from QScreen.  A framebuffer read and
    normal QWidget grab remain fallbacks for other platforms and headless runs.
    """

    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QGuiApplication, QPixmap

    viewport = plot_widget.viewport()
    if host_widget is not None and host_widget.isVisible():
        screen = host_widget.screen() or QGuiApplication.primaryScreen()
        if screen is not None:
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
        desktop_crop = _capture_desktop_crop(viewport)
        if _pixmap_has_content(desktop_crop):
            return desktop_crop, "desktop_screen_crop"
    grab_framebuffer = getattr(viewport, "grabFramebuffer", None)
    if callable(grab_framebuffer):
        image = grab_framebuffer()
        if image is not None and not image.isNull():
            pixmap = QPixmap.fromImage(image)
            if _pixmap_has_content(pixmap):
                return pixmap, "opengl_framebuffer"
    return plot_widget.grab(), "widget_grab"


def _capture_widget_surface(widget, *, prefer_screen: bool):
    """Capture a top-level widget with native child surfaces when available."""

    from PySide6.QtGui import QGuiApplication

    if prefer_screen and widget.isVisible():
        screen = widget.screen() or QGuiApplication.primaryScreen()
        if screen is not None:
            pixmap = screen.grabWindow(int(widget.winId()))
            if _pixmap_has_content(pixmap):
                return pixmap, "screen_window"
        desktop_crop = _capture_desktop_crop(widget)
        if _pixmap_has_content(desktop_crop):
            return desktop_crop, "desktop_screen_crop"
    return widget.grab(), "widget_grab"


def _path_pixel_visibility(
    pixmap,
    color,
    *,
    tolerance: int = 24,
    scene_bounds=None,
    logical_size=None,
) -> dict[str, Any]:
    """Count pixels close to the path pen color in a captured PlotWidget.

    The tolerance admits anti-aliasing while the minimum count below rejects
    an axes-only or entirely blank capture.  The plot-only pixmap prevents the
    blue application chrome from creating false positives.
    """

    import numpy as np
    from PySide6.QtGui import QColor, QImage

    target = QColor(color)
    result: dict[str, Any] = {
        "target_rgb": [target.red(), target.green(), target.blue()],
        "tolerance": int(tolerance),
        "near_color_pixel_count": 0,
        "minimum_required": 100,
        "visible": False,
    }
    if pixmap is None or pixmap.isNull() or not target.isValid():
        return result
    try:
        image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        width, height = int(image.width()), int(image.height())
        bytes_per_line = int(image.bytesPerLine())
        rgba = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.sizeInBytes())
        rgba = rgba.reshape(height, bytes_per_line)[:, : width * 4].reshape(height, width, 4)
        rgb = rgba[:, :, :3].astype(np.int16, copy=False)
        inspection_region = [0, 0, width, height]
        if scene_bounds is not None and logical_size is not None:
            logical_width, logical_height = map(float, logical_size)
            bounds = list(map(float, scene_bounds))
            if logical_width > 0.0 and logical_height > 0.0 and len(bounds) == 4:
                scale_x = width / logical_width
                scale_y = height / logical_height
                padding = 6.0
                x0 = max(0, int((bounds[0] - padding) * scale_x))
                y0 = max(0, int((bounds[1] - padding) * scale_y))
                x1 = min(width, int((bounds[0] + bounds[2] + padding) * scale_x) + 1)
                y1 = min(height, int((bounds[1] + bounds[3] + padding) * scale_y) + 1)
                if x1 > x0 and y1 > y0:
                    inspection_region = [x0, y0, x1, y1]
                    rgb = rgb[y0:y1, x0:x1]
        target_rgb = np.asarray(result["target_rgb"], dtype=np.int16)
        near = np.max(np.abs(rgb - target_rgb), axis=2) <= int(tolerance)
        count = int(np.count_nonzero(near))
        result.update(
            {
                "capture_size": [width, height],
                "inspection_region_px": inspection_region,
                "near_color_pixel_count": count,
                "visible": bool(count >= result["minimum_required"]),
            }
        )
    except Exception as exc:
        result["inspection_error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    args = _parse_args()
    if not args.onscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ.setdefault("PX4_LOG_FORMAT_POLICY", "ulg_stable")
    os.environ.setdefault("PX4_INTERACTIVE_FAST_RENDER_PREVIEW", "0")

    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from PySide6 import QtCore
    from PySide6.QtWidgets import QApplication
    import numpy as np
    import pyqtgraph as pg

    from engines.io_engine import LogIOEngine
    from gui.main_window import (
        MainWindow,
        Workspace,
        _detect_aircraft_type_for_dataset,
        _extract_log_metadata_for_dataset,
        _preprocess_loaded_dataset,
    )
    from storage.parquet_cache import ParquetCacheManager

    # Offscreen platforms do not provide a valid OpenGL context.  QPainter
    # still exercises the plot state/data contracts without noisy GL errors.
    if not args.onscreen:
        pg.setConfigOptions(useOpenGL=False)

    sources = [path.expanduser().resolve() for path in args.ulg]
    if args.cache_cycles < 1:
        raise ValueError("--cache-cycles must be at least 1")
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ULog source(s): " + ", ".join(missing))

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Match src/main.py: pyqtgraph's 3D items can span multiple widgets,
        # so the native acceptance runner must create shareable GL contexts
        # before QApplication exists.
        QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts, True)
    except Exception:
        pass
    app = QApplication.instance() or QApplication([])
    # The runner opens and closes one top-level Workspace per log.  Keep Qt's
    # application dispatcher alive after the first window closes; otherwise a
    # subsequent workspace can have a valid QWidget state but no composited
    # native surface (1x1/all-black QScreen captures on Windows).
    app.setQuitOnLastWindowClosed(False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "qt": QtCore.qVersion(),
            "qpa": os.environ.get("QT_QPA_PLATFORM", "native"),
            "quit_on_last_window_closed": app.quitOnLastWindowClosed(),
            "true_gl_result": "manual_required" if not args.onscreen else "not_automatically_asserted",
        },
        "logs": [],
        "multi_log": {},
    }
    loaded: list[dict[str, Any]] = []
    # Keep native pyqtgraph/QOpenGLWidget owners alive across sequential logs.
    # Destroying the first top-level workspace before constructing the second
    # can invalidate the next GraphicsView GL surface on Windows even with
    # AA_ShareOpenGLContexts (valid data/ViewBox but corrupt scene transform
    # and all-black capture).  Hidden retained widgets cost no paint time and
    # are released together after the run.
    retained_gui_instances: list[tuple[Any, Any]] = []

    for source in sources:
        cache_cycles: list[dict[str, Any]] = []
        load_result = None
        for cycle_index in range(args.cache_cycles):
            # Each cycle starts from a brand-new cache, then immediately reads
            # it back.  TemporaryDirectory prevents the private flight data's
            # generated Parquet cache from becoming a persistent test artifact.
            with tempfile.TemporaryDirectory(
                prefix=f"cache-cycle-{cycle_index + 1}-",
                dir=str(output_dir),
            ) as cache_dir:
                cache_manager = ParquetCacheManager(cache_dir=cache_dir)
                cold_engine = LogIOEngine(cache_mgr=cache_manager)
                started = time.perf_counter()
                cold_result = cold_engine.load_result(str(source))
                cold_ms = _elapsed_ms(started)

                warm_engine = LogIOEngine(cache_mgr=cache_manager)
                started = time.perf_counter()
                warm_result = warm_engine.load_result(str(source))
                warm_ms = _elapsed_ms(started)

                cold_signals = {
                    name: tuple(topic.signals)
                    for name, topic in cold_result.dataset.topics.items()
                }
                warm_signals = {
                    name: tuple(topic.signals)
                    for name, topic in warm_result.dataset.topics.items()
                }
                contract_equal = (
                    cold_result.format_id == warm_result.format_id
                    and cold_signals == warm_signals
                )
                cache_cycles.append(
                    {
                        "cycle": cycle_index + 1,
                        "cold_ms": cold_ms,
                        "warm_ms": warm_ms,
                        "cold_load_type": cold_engine.last_load_info.get("load_type"),
                        "warm_load_type": warm_engine.last_load_info.get("load_type"),
                        "signal_contract_equal": contract_equal,
                    }
                )
                if load_result is None:
                    load_result = cold_result

        assert load_result is not None
        dataset = load_result.dataset
        _preprocess_loaded_dataset(dataset)
        aircraft_type = _detect_aircraft_type_for_dataset(dataset)
        metadata = _extract_log_metadata_for_dataset(
            str(source), dataset=dataset, aircraft_type=aircraft_type
        )
        selected = _pick_signals(dataset, 3)
        if len(selected) < 3:
            raise RuntimeError(f"{source.name}: fewer than three numeric signals")

        window = MainWindow()
        app.processEvents()
        window.loaded_datasets[source.name] = dataset
        window.loaded_aircraft_types[source.name] = aircraft_type
        window.loaded_log_metadata[source.name] = metadata
        window.active_analysis_log = source.name
        workspace = Workspace(window)
        workspace.resize(1400, 780)
        if args.onscreen:
            window_exposure = _ensure_widget_exposed(app, workspace)
        else:
            workspace.show()
            _settle_qt_render(app, workspace, turns=3)
            window_exposure = {
                "ready": bool(workspace.isVisible()),
                "attempts": 1,
                "size": [int(workspace.width()), int(workspace.height())],
                "mode": "offscreen",
            }

        cases: dict[str, Any] = {}
        cases["cold_warm_cache"] = {
            **_check(
                all(
                    cycle["cold_load_type"] == "Full Parsing"
                    and cycle["warm_load_type"] == "Cache Load"
                    and cycle["signal_contract_equal"]
                    for cycle in cache_cycles
                ),
                detail={"cycles": len(cache_cycles)},
            ),
            "cycles": cache_cycles,
        }

        for curve_count in (1, 2, 3):
            workspace.create_grid(1, 1)
            plot = workspace.get_plot(0, 0)
            started = time.perf_counter()
            rendered = [
                plot.render_signal(source.name, topic, signal)
                for topic, signal in selected[:curve_count]
            ]
            app.processEvents()
            cases[f"curves_{curve_count}"] = {
                **_check(all(rendered) and len(plot.plotted_signals) == curve_count),
                "render_ms": _elapsed_ms(started),
                "signals": [f"{topic}.{signal}" for topic, signal in selected[:curve_count]],
            }

        screenshot_path = output_dir / f"{source.stem}_3panels.png"
        for panel_count in (1, 2, 3):
            workspace.create_grid(1, panel_count)
            plots = [workspace.get_plot(0, idx) for idx in range(panel_count)]
            started = time.perf_counter()
            rendered = [
                plot.render_signal(source.name, *selected[idx])
                for idx, plot in enumerate(plots)
            ]
            app.processEvents()
            lo = float(workspace.global_min_x)
            hi = float(workspace.global_max_x)
            target = (lo + 0.25 * (hi - lo), lo + 0.55 * (hi - lo))
            workspace.set_time_range(*target, clamp_to_global=True)
            app.processEvents()
            error = _range_error(plots, target)
            cases[f"panels_{panel_count}"] = {
                **_check(all(rendered) and error <= 1e-6, detail={"range_error_s": error}),
                "render_and_range_ms": _elapsed_ms(started),
            }
            if panel_count == 3:
                if args.onscreen:
                    workspace.raise_()
                    workspace.activateWindow()
                _settle_qt_render(app, workspace, turns=3)
                time_series_pixmap, time_series_capture_mode = _capture_widget_surface(
                    workspace,
                    prefer_screen=args.onscreen,
                )
                time_series_pixmap.save(str(screenshot_path))
                # Flush unrelated render/grab work before measuring the wheel
                # input path.  The acceptance budget concerns the callback and
                # logical follower ranges; a whole processEvents() drain can
                # include unrelated queued paints and is recorded separately.
                for _ in range(3):
                    app.processEvents()

                callback_samples_ms: list[float] = []
                follower_samples_ms: list[float] = []
                immediate_errors: list[float] = []
                wheel_target = target
                workspace._wheel_active = True
                for notch in range(30):
                    left_fraction = 0.28 + (notch % 10) * 0.006
                    right_fraction = 0.62 - (notch % 7) * 0.005
                    wheel_target = (
                        lo + left_fraction * (hi - lo),
                        lo + right_fraction * (hi - lo),
                    )
                    started = time.perf_counter()
                    workspace._wheel_begin_or_update(
                        *wheel_target,
                        source_plot=plots[0],
                        started_at=started,
                        local_debug={},
                    )
                    callback_samples_ms.append(_elapsed_ms(started))
                    immediate_errors.append(_range_error(plots, wheel_target))
                    follower_samples_ms.append(
                        float(
                            getattr(
                                workspace,
                                "_wheel_live_follower_sync_last_latency_ms",
                                0.0,
                            )
                            or 0.0
                        )
                    )

                callback_p95_ms = float(np.percentile(callback_samples_ms, 95))
                follower_p95_ms = float(np.percentile(follower_samples_ms, 95))
                immediate_error = max(immediate_errors, default=float("inf"))
                paint_started = time.perf_counter()
                app.processEvents()
                queued_paint_drain_ms = _elapsed_ms(paint_started)
                live_error = _range_error(plots, wheel_target)
                workspace._wheel_active = False
                cases["wheel_live_3_panels"] = {
                    **_check(
                        immediate_error <= 1e-6
                        and live_error <= 1e-6
                        and callback_p95_ms <= 50.0
                        and follower_p95_ms <= 50.0,
                        detail={
                            "immediate_range_error_s": immediate_error,
                            "final_range_error_s": live_error,
                        },
                    ),
                    "notches": len(callback_samples_ms),
                    "callback_p95_ms": round(callback_p95_ms, 3),
                    "follower_sync_p95_ms": round(follower_p95_ms, 3),
                    "queued_paint_event_drain_ms": queued_paint_drain_ms,
                    "paint_note": (
                        "Headless event-queue drain is diagnostic only; true frame "
                        "latency/stutter requires the Windows GPU manual gate."
                    ),
                }

        workspace.create_grid(1, 1)
        plot = workspace.get_plot(0, 0)
        _settle_qt_render(app, workspace, plot, plot.plot, plot.plot.viewport(), turns=3)
        started = time.perf_counter()
        ok_2d = window._render_2d_flight_path_in_plot(plot, source.name, dataset)
        _settle_qt_render(app, workspace, plot, plot.plot, plot.plot.viewport())
        render_2d_ms = _elapsed_ms(started)
        state_2d = plot._flight_path_2d_state if ok_2d else None
        screenshot_2d_path = output_dir / f"{source.stem}_2d.png"
        screenshot_2d_workspace_path = output_dir / f"{source.stem}_2d_workspace.png"
        if args.onscreen:
            workspace.raise_()
            workspace.activateWindow()
            _settle_qt_render(app, workspace, plot.plot.viewport(), turns=3)
        plot_pixmap, plot_capture_mode = _capture_plot_surface(
            plot.plot,
            host_widget=workspace if args.onscreen else None,
        )
        plot_pixmap.save(str(screenshot_2d_path))
        workspace_pixmap, workspace_capture_mode = _capture_widget_surface(
            workspace,
            prefer_screen=args.onscreen,
        )
        workspace_pixmap.save(str(screenshot_2d_workspace_path))
        geometry_visibility = _path_geometry_visibility(plot)
        curve_item = getattr(plot, "_flight_path_2d_curve_item", None)
        curve_pen = curve_item.opts.get("pen") if curve_item is not None else None
        path_color = curve_pen.color() if curve_pen is not None else "#2E86DE"
        pixel_visibility = _path_pixel_visibility(
            plot_pixmap,
            path_color,
            scene_bounds=geometry_visibility.get("scene_bounds"),
            logical_size=(plot.plot.viewport().width(), plot.plot.viewport().height()),
        )
        cases["flight_path_2d"] = {
            **_check(
                bool(
                    ok_2d
                    and state_2d
                    and geometry_visibility.get("finite_point_count", 0) >= 2
                    and geometry_visibility.get("item_visible")
                    and geometry_visibility.get("data_intersects_view")
                    and geometry_visibility.get("scene_bounds_nonempty")
                    and pixel_visibility.get("visible")
                ),
                detail={
                    "geometry_visible": bool(
                        geometry_visibility.get("item_visible")
                        and geometry_visibility.get("data_intersects_view")
                        and geometry_visibility.get("scene_bounds_nonempty")
                    ),
                    "path_pixels_visible": bool(pixel_visibility.get("visible")),
                },
            ),
            "render_ms": render_2d_ms,
            "point_count": int(len(state_2d.get("time", ()))) if state_2d else 0,
            "geometry_visibility": geometry_visibility,
            "pixel_visibility": {
                **pixel_visibility,
                "capture_mode": plot_capture_mode,
            },
            "screenshot": str(screenshot_2d_path),
            "workspace_screenshot": str(screenshot_2d_workspace_path),
            "workspace_capture_mode": workspace_capture_mode,
        }

        workspace.create_grid(1, 1)
        plot = workspace.get_plot(0, 0)
        if not args.onscreen:
            plot._true_3d_available = False
        started = time.perf_counter()
        ok_3d = window._render_3d_flight_path_in_plot(plot, source.name, dataset)
        app.processEvents()
        render_3d_ms = _elapsed_ms(started)
        screenshot_3d_path = output_dir / f"{source.stem}_3d.png"
        if args.onscreen:
            workspace.raise_()
            workspace.activateWindow()
            _settle_qt_render(app, workspace, turns=3)
        screenshot_3d_pixmap, screenshot_3d_capture_mode = _capture_widget_surface(
            workspace,
            prefer_screen=args.onscreen,
        )
        screenshot_3d_pixmap.save(str(screenshot_3d_path))
        path_state = plot._true_3d_points if plot._true_3d_enabled else plot._projected_3d_state
        if isinstance(path_state, dict):
            point_count_3d = int(len(path_state.get("x", ())))
        elif path_state is not None:
            point_count_3d = int(len(path_state))
        else:
            point_count_3d = 0
        cases["flight_path_3d"] = {
            **_check(bool(ok_3d and point_count_3d >= 2)),
            "render_ms": render_3d_ms,
            "mode": "true_3d" if plot._true_3d_enabled else "projected_3d",
            "point_count": point_count_3d,
            "screenshot": str(screenshot_3d_path),
            "screenshot_capture_mode": screenshot_3d_capture_mode,
        }

        eval_result = metadata.get("evaluation", {}) or {}
        item_status_counts: dict[str, int] = {}
        for item in eval_result.get("items", []) or []:
            status = str(item.get("status", "unknown"))
            item_status_counts[status] = item_status_counts.get(status, 0) + 1

        report["logs"].append(
            {
                "source": str(source),
                "sha256": _sha256(source),
                "size_bytes": source.stat().st_size,
                "load_ms": cache_cycles[0]["cold_ms"],
                "load_type": cache_cycles[0]["cold_load_type"],
                "cache_cycles": cache_cycles,
                "topic_count": len(dataset.topics),
                "aircraft_type": aircraft_type,
                "window_exposure": window_exposure,
                "evaluation": {
                    "overall_score": eval_result.get("overall_score"),
                    "overall_status": eval_result.get("overall_status"),
                    "item_status_counts": item_status_counts,
                },
                "cases": cases,
                "screenshot": str(screenshot_path),
                "screenshot_capture_mode": time_series_capture_mode,
            }
        )
        loaded.append({"name": source.name, "dataset": dataset, "selected": selected})

        if args.onscreen:
            workspace.hide()
            retained_gui_instances.append((workspace, window))
            _settle_qt_render(app, turns=3)
        else:
            workspace.close()
            window.close()
            workspace.deleteLater()
            window.deleteLater()
            _settle_qt_render(app, turns=3)

    if len(loaded) >= 2:
        first, second = loaded[0], loaded[1]
        window = MainWindow()
        app.processEvents()
        window.loaded_datasets[first["name"]] = first["dataset"]
        window.loaded_datasets[second["name"]] = second["dataset"]
        workspace = Workspace(window)
        workspace.resize(1400, 780)
        if args.onscreen:
            _ensure_widget_exposed(app, workspace)
        else:
            workspace.show()
            _settle_qt_render(app, workspace, turns=3)
        plot = workspace.first_plot
        started = time.perf_counter()
        ok_first = plot.render_signal(first["name"], *first["selected"][0])
        ok_second = plot.render_signal(second["name"], *second["selected"][0])
        _settle_qt_render(app, workspace, plot, plot.plot, plot.plot.viewport())
        overlay_path = output_dir / "two_ulg_overlay.png"
        if args.onscreen:
            workspace.raise_()
            workspace.activateWindow()
            _settle_qt_render(app, workspace, turns=3)
        overlay_pixmap, overlay_capture_mode = _capture_widget_surface(
            workspace,
            prefer_screen=args.onscreen,
        )
        overlay_pixmap.save(str(overlay_path))
        report["multi_log"] = {
            **_check(bool(ok_first and ok_second and len(plot.plotted_signals) == 2)),
            "render_ms": _elapsed_ms(started),
            "screenshot": str(overlay_path),
            "screenshot_capture_mode": overlay_capture_mode,
        }
        workspace.close()
        window.close()
        workspace.deleteLater()
        window.deleteLater()
        app.processEvents()

    for retained_workspace, retained_window in retained_gui_instances:
        retained_workspace.close()
        retained_window.close()
        retained_workspace.deleteLater()
        retained_window.deleteLater()
    _settle_qt_render(app, turns=3)

    all_cases = [case for item in report["logs"] for case in item["cases"].values()]
    if report["multi_log"]:
        all_cases.append(report["multi_log"])
    report["automatic_summary"] = {
        "passed": sum(case.get("status") == "pass" for case in all_cases),
        "failed": sum(case.get("status") == "fail" for case in all_cases),
        "manual_remaining": [
            "true OpenGL 3D rotation/pan/zoom/context stability",
            "physical wheel p95/p99 paint latency and subjective stutter",
            "100-cycle GUI soak and RSS growth",
            "100%/150% DPI packaged EXE visual check",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), **report["automatic_summary"]}, ensure_ascii=False))
    return 1 if report["automatic_summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
