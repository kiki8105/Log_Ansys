from __future__ import annotations

import os
from pathlib import Path

import build


def test_sanitized_pyinstaller_environment_removes_host_native_runtime_paths():
    separator = os.pathsep
    safe_a = str(Path("C:/safe/python"))
    injected = str(
        Path("C:/Users/example/.cache/codex-runtimes/runtime/dependencies/native/poppler/bin")
    )
    safe_b = str(Path("D:/tools"))

    environment, removed = build.sanitized_pyinstaller_environment(
        {"PATH": separator.join((safe_a, injected, safe_b)), "KEEP": "yes"}
    )

    assert environment["PATH"] == separator.join((safe_a, safe_b))
    assert environment["KEEP"] == "yes"
    assert removed == (injected,)


def test_generated_spec_places_pyside_msvc_runtime_at_executable_root():
    spec = build.spec_text(console=False, onefile=False)

    assert "qt_msvc_binaries" in spec
    assert "binaries=qt_msvc_binaries" in spec
    for name in build.QT_MSVC_RUNTIME_FILES:
        assert name in spec


def test_product_name_and_default_bundle_paths_are_log_ansys():
    spec = build.spec_text(console=False, onefile=False)

    assert build.APP_NAME == "Log ansys"
    assert build.SPEC_NAME == "Log_ansys.spec"
    assert "name='Log ansys'" in spec
    assert build.SPLASH_REL_PATH == Path("assets/splash/custom_splash.png")


def test_default_build_allows_missing_custom_icon(monkeypatch, tmp_path):
    monkeypatch.setattr(build, "project_root", lambda: tmp_path)

    assert build.find_default_icon() is None
    assert "icon_path = None" in build.spec_text(console=False, onefile=False)
