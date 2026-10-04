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


def test_generated_spec_bundles_both_pymavlink_protocol_dialects_and_fallback_data():
    spec = build.spec_text(console=False, onefile=False)

    for module_name in build.PYMAVLINK_DIALECT_MODULES:
        assert module_name in spec
    assert "datas += pymavlink_bundle_data" in spec
    assert "Path(pymavlink.__file__).resolve().parent" in spec


def test_pymavlink_bundle_data_follows_only_active_all_xml_include_graph(tmp_path):
    definitions = tmp_path / "message_definitions" / "v1.0"
    definitions.mkdir(parents=True)
    generator = tmp_path / "generator"
    generator.mkdir()
    (generator / "mavschema.xsd").write_text("<schema/>", encoding="utf-8")
    (definitions / "all.xml").write_text(
        "<mavlink><include>ardupilotmega.xml</include>"
        "<!-- <include>unused.xml</include> --></mavlink>",
        encoding="utf-8",
    )
    (definitions / "ardupilotmega.xml").write_text(
        "<mavlink><include>common.xml</include></mavlink>",
        encoding="utf-8",
    )
    (definitions / "common.xml").write_text("<mavlink/>", encoding="utf-8")
    (definitions / "unused.xml").write_text("<mavlink/>", encoding="utf-8")

    resources = build.pymavlink_bundle_data(tmp_path)

    definitions_in_bundle = {
        Path(source).name
        for source, destination in resources
        if destination == build.PYMAVLINK_DEFINITION_DEST
    }
    assert definitions_in_bundle == {"all.xml", "ardupilotmega.xml", "common.xml"}
    assert (
        str(generator / "mavschema.xsd"),
        build.PYMAVLINK_SCHEMA_DEST,
    ) in resources


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
