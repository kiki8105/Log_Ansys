# src/main.py

import argparse
import datetime
import io
import json
import os
import re
import sys
import traceback
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────
# (2026-05-28) 전역 예외 핸들러 + 크래시 로그.
# windowed frozen exe 는 stdout/stderr 가 닫혀있어서 unhandled exception 이
# 사용자 화면에 안 보이고 그냥 창이 꺼짐 → "고객 PC 에서 갑자기 죽음" 의 70%
# 가 디버깅 불가. 이 핸들러가 traceback 을 ~/Log_ansys_crash_logs/ 에 저장.
# 가능하면 QMessageBox 로 사용자에게 파일 경로 안내.
# ──────────────────────────────────────────────────────────────────────────

def _crash_log_dir() -> Path:
    # 사용자 홈 (Documents 보다 home 이 권한 문제 적음).
    home = Path(os.path.expanduser("~"))
    target = home / "Log_ansys_crash_logs"
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception:
        target = home
    return target


def _write_crash_log(exc_type, exc_value, exc_tb) -> Path:
    log_dir = _crash_log_dir()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    path = log_dir / f"crash_{stamp}_pid{pid}.log"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("=== Log ansys crash log ===\n")
            f.write(f"Time: {datetime.datetime.now().isoformat()}\n")
            f.write(f"PID:  {pid}\n")
            f.write(f"Python: {sys.version}\n")
            f.write(f"Frozen: {bool(getattr(sys, 'frozen', False))}\n")
            f.write(f"Executable: {sys.executable}\n")
            f.write(f"Argv: {sys.argv}\n")
            f.write(f"Platform: {sys.platform}\n")
            f.write(f"\n--- Traceback ---\n")
            traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
    except Exception:
        # 크래시 로그 저장조차 실패하면 더 할 수 있는 게 없음.
        pass
    return path


def _install_global_excepthook() -> None:
    def _hook(exc_type, exc_value, exc_tb):
        # KeyboardInterrupt 는 사용자가 Ctrl+C 로 종료한 케이스 — 로그 생략.
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        log_path = _write_crash_log(exc_type, exc_value, exc_tb)
        try:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
        except Exception:
            pass
        # QMessageBox 로 사용자에게 안내 (QApplication 살아있을 때만).
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance()
            if app is not None:
                QMessageBox.critical(
                    None,
                    "Log ansys — 오류 발생",
                    "예기치 못한 오류로 중단됩니다.\n\n"
                    f"오류 로그가 저장되었습니다:\n{log_path}\n\n"
                    "이 파일을 개발자에게 전달하시면 원인 파악에 도움이 됩니다.",
                )
        except Exception:
            pass

    sys.excepthook = _hook


def _ensure_stdio_streams() -> None:
    """Guarantee stdout/stderr are writable in frozen no-console mode."""
    if getattr(sys, "stdout", None) is None or not hasattr(sys.stdout, "write"):
        sys.stdout = io.TextIOWrapper(open(os.devnull, "wb"), encoding="utf-8", errors="replace", write_through=True)
    if getattr(sys, "stderr", None) is None or not hasattr(sys.stderr, "write"):
        sys.stderr = io.TextIOWrapper(open(os.devnull, "wb"), encoding="utf-8", errors="replace", write_through=True)


# Ensure project src is importable when launched directly.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

_ensure_stdio_streams()

def _close_pyi_splash():
    """Close PyInstaller bootloader splash when running inside a frozen
    EXE built with `Splash(...)`. In dev mode `pyi_splash` is not
    injected and the import fails silently — there is no splash to
    close. (2026-05-20)"""
    try:
        import pyi_splash  # type: ignore  # injected by PyInstaller bootloader
    except Exception:
        return
    try:
        if pyi_splash.is_alive():
            pyi_splash.close()
    except Exception:
        try:
            pyi_splash.close()
        except Exception:
            pass


SPLASH_IMAGE_CANDIDATES = (
    Path("assets") / "splash" / "custom_splash.png",
)


def _resource_roots():
    roots = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        try:
            roots.append(Path(meipass).resolve())
        except Exception:
            pass
    try:
        roots.append(Path(__file__).resolve().parent.parent)
    except Exception:
        pass
    try:
        roots.append(Path.cwd())
    except Exception:
        pass

    unique = []
    seen = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except Exception:
            key = str(root)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _find_startup_splash_path():
    for root in _resource_roots():
        for rel_path in SPLASH_IMAGE_CANDIDATES:
            candidate = root / rel_path
            if candidate.is_file():
                return candidate
    return None


def _create_qt_startup_splash(app):
    # Keep Qt imports out of module scope.  The frozen ``--smoke-test`` path
    # must be able to catch and report a broken Qt runtime instead of letting
    # PyInstaller show a modal "Unhandled exception" dialog that blocks the
    # automated build until its timeout expires.
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QSplashScreen

    image_path = _find_startup_splash_path()
    if image_path is None:
        return None

    pixmap = QPixmap(str(image_path))
    if pixmap.isNull():
        return None

    screen = app.primaryScreen()
    if screen is not None:
        avail = screen.availableGeometry().size()
        max_w = max(480, int(avail.width() * 0.6))
        max_h = max(270, int(avail.height() * 0.6))
        if pixmap.width() > max_w or pixmap.height() > max_h:
            pixmap = pixmap.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    splash = QSplashScreen(pixmap)
    splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    splash.setEnabled(False)
    splash.show()
    app.processEvents()
    return splash


def _run_smoke_test() -> int:
    # (2026-05-28) build.py 가 빌드 직후 `exe --smoke-test` 로 실행 — 모든 핵심 import
    # 가 frozen 환경에서 통과하는지만 검증. 실패 시 비0 종료 → 빌드 검증 실패로 잡힘.
    print("[SMOKE] Starting smoke test ...")
    try:
        from PySide6.QtWidgets import QApplication  # noqa: F401
        from PySide6.QtGui import QPixmap  # noqa: F401
        import numpy  # noqa: F401
        import polars  # noqa: F401
        import scipy  # noqa: F401
        import pyqtgraph  # noqa: F401
        import pyulog  # noqa: F401
        import rosbags  # noqa: F401
        import pymavlink  # noqa: F401
        import yaml  # noqa: F401
        from readers.registry import default_reader_registry
        registry = default_reader_registry()
        reader_ids = set(registry.reader_ids)
        policy = getattr(registry, "release_policy", None)
        required_readers = set(getattr(policy, "enabled_reader_ids", ()) or {"px4_ulog"})
        missing_readers = sorted(required_readers - reader_ids)
        if missing_readers:
            raise RuntimeError(f"reader registration missing: {', '.join(missing_readers)}")
        from gui.main_window import MainWindow  # noqa: F401
        from engines.math_engine import MathEngine  # noqa: F401
        from engines.overview_metrics import build_overview  # noqa: F401
        from engines.kml_export import export_kmz  # noqa: F401
        print("[SMOKE] OK - all critical imports passed.")
        return 0
    except Exception as e:
        print(f"[SMOKE] FAIL - {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


def _acceptance_numeric_summary(dataset, min_samples: int) -> dict:
    """Return a dependency-light graphability summary for frozen load tests."""
    import numpy as np

    numeric_signal_count = 0
    renderable_signal_count = 0
    selected = None
    for topic_name, topic in sorted(dataset.topics.items()):
        frame = getattr(topic, "dataframe", None)
        if frame is None or "timestamp_sec" not in frame.columns:
            continue
        try:
            timestamps = np.asarray(frame["timestamp_sec"].to_numpy(), dtype=np.float64)
        except (TypeError, ValueError):
            continue
        for signal_name in sorted(topic.signals):
            if signal_name not in frame.columns:
                continue
            try:
                values = np.asarray(frame[signal_name].to_numpy(), dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if timestamps.size != values.size:
                continue
            finite_count = int(np.count_nonzero(np.isfinite(timestamps) & np.isfinite(values)))
            if finite_count == 0:
                continue
            numeric_signal_count += 1
            if finite_count >= min_samples:
                renderable_signal_count += 1
                candidate = {
                    "topic": topic_name,
                    "signal": signal_name,
                    "finite_samples": finite_count,
                }
                if selected is None or finite_count > selected["finite_samples"]:
                    selected = candidate
    return {
        "topic_count": len(dataset.topics),
        "numeric_signal_count": numeric_signal_count,
        "renderable_signal_count": renderable_signal_count,
        "selected": selected,
    }


def _acceptance_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _acceptance_output_collision(
    report_path: Path,
    temporary_path: Path,
    cache_dir: Path,
    sources: list[Path],
) -> str | None:
    """Reject acceptance outputs that could modify an input source."""

    outputs = (
        ("report", report_path, False),
        ("temporary report", temporary_path, False),
        ("cache directory", cache_dir, True),
    )
    for label, output, is_tree in outputs:
        for source in sources:
            if output == source:
                return f"{label} collides with input source: {source}"
            if source.is_dir() and _acceptance_is_relative_to(output, source):
                return f"{label} would be created inside input source directory: {source}"
            if is_tree and _acceptance_is_relative_to(source, output):
                return f"input source would be inside {label}: {source}"

    if report_path == cache_dir or temporary_path == cache_dir:
        return "report path collides with cache directory"
    if _acceptance_is_relative_to(report_path, cache_dir) or _acceptance_is_relative_to(
        temporary_path, cache_dir
    ):
        return "report path would be created inside cache directory"
    return None


def _run_acceptance_load_test(argv=None) -> int:
    """Load real fixtures inside the packaged process and write JSON evidence.

    The normal Windows build has no console, so the report file is the stable
    automation boundary.  No QApplication is created on this path.
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--acceptance-load-report", type=Path, required=True)
    parser.add_argument("--acceptance-cache-dir", type=Path, required=True)
    parser.add_argument("--acceptance-run-id", required=True)
    parser.add_argument("--acceptance-load", type=Path, action="append", required=True)
    parser.add_argument("--acceptance-min-samples", type=int, default=2)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    report_path = args.acceptance_load_report.expanduser().resolve()
    temporary_path = report_path.with_suffix(report_path.suffix + ".tmp")
    cache_dir = args.acceptance_cache_dir.expanduser().resolve()
    source_paths = [path.expanduser().resolve() for path in args.acceptance_load]
    collision = _acceptance_output_collision(
        report_path,
        temporary_path,
        cache_dir,
        source_paths,
    )
    if collision is not None:
        print(f"[ACCEPTANCE] FAIL - {collision}", file=sys.stderr)
        return 1
    if report_path.exists() or temporary_path.exists():
        print(
            "[ACCEPTANCE] FAIL - refusing to reuse an existing report or temporary path",
            file=sys.stderr,
        )
        return 1
    if cache_dir.exists():
        print(
            "[ACCEPTANCE] FAIL - acceptance cache directory must be unique and absent",
            file=sys.stderr,
        )
        return 1

    report = {
        "schema_version": 1,
        "run_id": args.acceptance_run_id,
        "status": "fail",
        "frozen": bool(getattr(sys, "frozen", False)),
        "executable": sys.executable,
        "inputs": [],
    }
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.acceptance_run_id):
        report["error"] = "--acceptance-run-id has an invalid format"
    elif args.acceptance_min_samples < 2:
        report["error"] = "--acceptance-min-samples must be at least 2"
    else:
        try:
            os.environ["PX4_LOG_FORMAT_POLICY"] = "multiformat_stable"
            from engines.io_engine import LogIOEngine
            from storage.parquet_cache import ParquetCacheManager

            engine = LogIOEngine(
                cache_mgr=ParquetCacheManager(
                    cache_dir=str(cache_dir)
                )
            )
            for path in source_paths:
                item = {"path": str(path), "status": "fail"}
                try:
                    reader, probe = engine.detect(str(path))
                    result = engine.load_result(str(path))
                    numeric = _acceptance_numeric_summary(
                        result.dataset,
                        args.acceptance_min_samples,
                    )
                    passed = numeric["renderable_signal_count"] > 0
                    item.update(
                        {
                            "status": "pass" if passed else "fail",
                            "reader_id": reader.id,
                            "probe_format_id": probe.format_id,
                            "probe_confidence": probe.confidence,
                            "format_id": result.format_id,
                            "load_type": engine.last_load_info.get("load_type"),
                            **numeric,
                        }
                    )
                    if not passed:
                        item["error"] = (
                            "no signal has the minimum finite timestamp/value sample count"
                        )
                except Exception as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                report["inputs"].append(item)
            report["status"] = (
                "pass"
                if report["inputs"]
                and all(item["status"] == "pass" for item in report["inputs"])
                else "fail"
            )
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(report_path)
    return 0 if report.get("status") == "pass" else 1


def main():
    # (2026-05-28) 가장 먼저 글로벌 예외 핸들러 설치 — QApplication 생성 전 에러도 잡힘.
    _install_global_excepthook()

    # 빌드 검증용 smoke test 분기. UI 없이 import 만 확인.
    if "--smoke-test" in sys.argv:
        sys.exit(_run_smoke_test())

    # Release acceptance path used by scripts/validate_format_acceptance.py.
    # It must run before QApplication so packaged CI can verify actual parser
    # dependencies without opening or automating the desktop window.
    if "--acceptance-load-report" in sys.argv:
        sys.exit(_run_acceptance_load_test())

    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication

    try:
        QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    except Exception:
        pass

    app = QApplication(sys.argv)
    qt_splash = _create_qt_startup_splash(app)

    from gui.main_window import APP_DISPLAY_NAME, MainWindow

    app.setApplicationName(APP_DISPLAY_NAME)
    app.setApplicationDisplayName(APP_DISPLAY_NAME)

    app.setStyleSheet("""
        QSplitter::handle { background-color: #333333; }
        QSplitter::handle:horizontal { width: 5px; }
        QSplitter::handle:vertical { height: 5px; }
        QSplitter::handle:hover { background-color: #4CAF50; }
        QTabBar::tab { padding: 8px 15px; font-weight: bold; background-color: #2b2b2b; color: #aaaaaa; }
        QTabBar::tab:selected { background-color: #1A2D57; color: white; }
    """)

    window = MainWindow()
    window.show()
    window_shown = True

    # Close the PyInstaller bootloader splash only AFTER the initial
    # workspace (Tab1) has finished construction. MainWindow emits
    # `initial_workspace_ready` from `_create_initial_workspace_once`
    # after `add_workspace()` has run. Without this gating the user
    # briefly sees a half-built window (no Tab1 / no plot / no control
    # buttons) between the splash dismissal and the workspace setup,
    # because `add_workspace()` itself executes inside the first event
    # loop iteration and can take longer than a fixed timer guess.
    # A safety timer at 5 s closes the splash regardless, so a misfired
    # signal never leaves the splash hanging. (2026-05-20)
    def _show_window_if_needed():
        nonlocal window_shown
        if not window_shown:
            window.show()
            window_shown = True

    def _on_workspace_ready():
        _show_window_if_needed()
        # Tiny extra delay so the freshly-built workspace gets one paint
        # cycle on screen before the splash is removed.
        def _close_splashes():
            try:
                if qt_splash is not None:
                    qt_splash.finish(window)
            except Exception:
                pass
            _close_pyi_splash()

        QTimer.singleShot(80, _close_splashes)

    def _on_splash_timeout():
        _show_window_if_needed()
        try:
            if qt_splash is not None:
                qt_splash.finish(window)
        except Exception:
            pass
        _close_pyi_splash()

    window.initial_workspace_ready.connect(_on_workspace_ready)
    QTimer.singleShot(5000, _on_splash_timeout)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
