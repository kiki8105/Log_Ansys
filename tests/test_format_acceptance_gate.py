from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_gate_module():
    path = PROJECT_ROOT / "scripts" / "validate_format_acceptance.py"
    spec = importlib.util.spec_from_file_location("validate_format_acceptance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate_module()


def _load_generator_module():
    path = PROJECT_ROOT / "scripts" / "generate_synthetic_acceptance_fixtures.py"
    spec = importlib.util.spec_from_file_location(
        "generate_synthetic_acceptance_fixtures", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_keeps_all_missing_slots_machine_visible(tmp_path: Path) -> None:
    report = gate.run_gate(
        {},
        output_dir=tmp_path / "output",
        check_warm_cache=False,
        run_gui=False,
    )

    assert set(report["formats"]) == set(gate.FORMAT_BY_SLOT)
    assert all(
        item["status"] == "missing_fixture" for item in report["formats"].values()
    )
    assert report["status"] == "incomplete"
    assert report["exit_code"] == 2


def test_synthetic_csv_and_json_pass_technical_checks_but_not_release(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "flight.csv"
    csv_path.write_text(
        "timestamp_sec,value\n0.0,1.0\n0.1,2.0\n0.2,3.0\n",
        encoding="utf-8",
    )
    json_path = tmp_path / "flight.json"
    json_path.write_text(
        json.dumps(
            [
                {"timestamp_sec": 0.0, "value": 3.0},
                {"timestamp_sec": 0.1, "value": 4.0},
            ]
        ),
        encoding="utf-8",
    )
    fixtures = {
        "csv": gate.Fixture("csv", csv_path, "synthetic"),
        "json": gate.Fixture("json", json_path, "synthetic"),
    }

    report = gate.run_gate(
        fixtures,
        output_dir=tmp_path / "output",
        check_warm_cache=True,
        run_gui=False,
    )

    for slot in ("csv", "json"):
        item = report["formats"][slot]
        assert item["checks"]["probe"]["status"] == "pass"
        assert item["checks"]["load"]["status"] == "pass"
        assert item["checks"]["numeric_timeseries"]["status"] == "pass"
        assert item["checks"]["warm_cache"]["status"] == "pass"
        assert item["checks"]["corrupt_isolation"]["status"] == "pass"
        assert item["status"] == "manual_required"
    assert report["status"] == "incomplete"


def test_generated_non_ulg_matrix_is_technical_only_and_fully_loadable(
    tmp_path: Path,
) -> None:
    generator = _load_generator_module()
    manifest = generator.generate(tmp_path / "fixtures")
    fixtures = gate.load_fixtures(manifest)

    report = gate.run_gate(
        fixtures,
        output_dir=tmp_path / "output",
        check_warm_cache=True,
        run_gui=False,
    )

    assert report["formats"]["px4_ulog"]["status"] == "missing_fixture"
    for slot in set(gate.FORMAT_BY_SLOT) - {"px4_ulog"}:
        item = report["formats"][slot]
        assert item["status"] == "manual_required"
        for check_name in (
            "probe",
            "load",
            "numeric_timeseries",
            "warm_cache",
            "corrupt_isolation",
        ):
            assert item["checks"][check_name]["status"] == "pass"
    assert report["cross_format_session"]["status"] == "pass"
    assert report["status"] == "incomplete"


def test_manifest_rejects_one_source_claimed_as_two_variants(tmp_path: Path) -> None:
    source = tmp_path / "one.json"
    source.write_text("[]", encoding="utf-8")
    manifest = tmp_path / "fixtures.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fixtures": {
                    "csv": {"path": "one.json", "provenance": "synthetic"},
                    "json": {"path": "one.json", "provenance": "synthetic"},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot satisfy two acceptance slots"):
        gate.load_fixtures(manifest)


def test_frozen_entrypoint_contract_loads_numeric_log_without_qapplication(
    tmp_path: Path,
) -> None:
    from main import _run_acceptance_load_test

    source = tmp_path / "flight.csv"
    source.write_text(
        "timestamp_sec,value\n0.0,1.0\n0.1,2.0\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "frozen-report.json"
    exit_code = _run_acceptance_load_test(
        [
            "--acceptance-load-report",
            str(report_path),
            "--acceptance-cache-dir",
            str(tmp_path / "cache"),
            "--acceptance-run-id",
            "source-contract-test",
            "--acceptance-load",
            str(source),
            "--acceptance-min-samples",
            "2",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == "pass"
    assert report["run_id"] == "source-contract-test"
    assert report["frozen"] is False
    assert report["inputs"][0]["format_id"] == "tabular"
    assert report["inputs"][0]["renderable_signal_count"] >= 1


def test_frozen_report_maps_each_input_back_to_its_own_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "flight.csv"
    json_path = tmp_path / "flight.json"
    csv_path.write_text("timestamp_sec,value\n0,1\n1,2\n", encoding="utf-8")
    json_path.write_text("[]", encoding="utf-8")
    executable = tmp_path / "Log ansys.exe"
    executable.write_bytes(b"placeholder")
    loaded = {
        "csv": {"fixture": gate.Fixture("csv", csv_path, "real")},
        "json": {"fixture": gate.Fixture("json", json_path, "real")},
    }

    child_paths = {}

    def fake_run(command, **_kwargs):
        report_path = Path(command[command.index("--acceptance-load-report") + 1])
        cache_path = Path(command[command.index("--acceptance-cache-dir") + 1])
        run_id = command[command.index("--acceptance-run-id") + 1]
        child_paths.update(
            report=report_path,
            cache=cache_path,
            temporary_directory=report_path.parent,
        )
        cache_path.mkdir(parents=True)
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": gate.SCHEMA_VERSION,
                    "run_id": run_id,
                    "frozen": True,
                    "status": "fail",
                    "inputs": [
                        {
                            "path": str(json_path.resolve()),
                            "status": "fail",
                            "error": "No module named message_definitions",
                        },
                        {
                            "path": str(csv_path.resolve()),
                            "status": "pass",
                            "reader_id": "tabular",
                            "probe_format_id": "tabular",
                            "format_id": "tabular",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    result = gate._run_frozen_check(
        executable,
        loaded,
        output_dir=tmp_path / "output",
        min_samples=2,
        timeout_sec=10,
    )

    assert result["status"] == "fail"
    assert result["per_slot"]["csv"]["status"] == "pass"
    assert result["per_slot"]["json"]["status"] == "fail"
    assert "message_definitions" in result["per_slot"]["json"]["detail"]
    assert result["unmatched_inputs"] == []
    assert Path(result["report"]).is_file()
    assert not child_paths["report"].exists()
    assert not child_paths["cache"].exists()
    assert not child_paths["temporary_directory"].exists()


def test_frozen_identity_mismatch_fails_even_when_child_reports_pass(
    tmp_path: Path,
) -> None:
    source = tmp_path / "flight.csv"
    source.write_text("timestamp_sec,value\n0,1\n1,2\n", encoding="utf-8")
    loaded = {"csv": {"fixture": gate.Fixture("csv", source, "real")}}

    per_slot, unmatched = gate._map_frozen_inputs(
        loaded,
        [
            {
                "path": str(source.resolve()),
                "status": "pass",
                "reader_id": "px4_ulog",
                "probe_format_id": "px4_ulog",
                "format_id": "px4_ulog",
                "renderable_signal_count": 12,
            }
        ],
    )

    assert unmatched == []
    assert per_slot["csv"]["status"] == "fail"
    assert set(per_slot["csv"]["mismatches"]) == {
        "reader_id",
        "probe_format_id",
        "format_id",
    }


def test_frozen_success_uses_unique_run_and_cleans_child_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "flight.csv"
    source.write_text("timestamp_sec,value\n0,1\n1,2\n", encoding="utf-8")
    executable = tmp_path / "Log ansys.exe"
    executable.write_bytes(b"placeholder")
    loaded = {"csv": {"fixture": gate.Fixture("csv", source, "real")}}
    child_paths = {}

    def fake_run(command, **_kwargs):
        report_path = Path(command[command.index("--acceptance-load-report") + 1])
        cache_path = Path(command[command.index("--acceptance-cache-dir") + 1])
        run_id = command[command.index("--acceptance-run-id") + 1]
        child_paths.update(
            report=report_path,
            cache=cache_path,
            temporary_directory=report_path.parent,
            run_id=run_id,
        )
        cache_path.mkdir(parents=True)
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": gate.SCHEMA_VERSION,
                    "run_id": run_id,
                    "frozen": True,
                    "status": "pass",
                    "inputs": [
                        {
                            "path": str(source.resolve()),
                            "status": "pass",
                            "reader_id": "tabular",
                            "probe_format_id": "tabular",
                            "format_id": "tabular",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    result = gate._run_frozen_check(
        executable,
        loaded,
        output_dir=tmp_path / "output",
        min_samples=2,
        timeout_sec=10,
    )

    assert result["status"] == "pass"
    assert result["run_id"] == child_paths["run_id"]
    assert child_paths["run_id"] in Path(result["report"]).name
    assert Path(result["report"]).is_file()
    assert not child_paths["report"].exists()
    assert not child_paths["cache"].exists()
    assert not child_paths["temporary_directory"].exists()


@pytest.mark.parametrize(
    ("field", "wrong_value", "expected_fragment"),
    (
        ("schema_version", 999, "schema_version"),
        ("run_id", "stale-run", "run_id"),
        ("frozen", False, "frozen must be exactly true"),
    ),
)
def test_frozen_parent_rejects_untrusted_report_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    wrong_value,
    expected_fragment: str,
) -> None:
    source = tmp_path / "flight.csv"
    source.write_text("timestamp_sec,value\n0,1\n1,2\n", encoding="utf-8")
    executable = tmp_path / "Log ansys.exe"
    executable.write_bytes(b"placeholder")
    loaded = {"csv": {"fixture": gate.Fixture("csv", source, "real")}}

    def fake_run(command, **_kwargs):
        report_path = Path(command[command.index("--acceptance-load-report") + 1])
        run_id = command[command.index("--acceptance-run-id") + 1]
        payload = {
            "schema_version": gate.SCHEMA_VERSION,
            "run_id": run_id,
            "frozen": True,
            "status": "pass",
            "inputs": [],
        }
        payload[field] = wrong_value
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    result = gate._run_frozen_check(
        executable,
        loaded,
        output_dir=tmp_path / "output",
        min_samples=2,
        timeout_sec=10,
    )

    assert result["status"] == "fail"
    assert expected_fragment in result["detail"]
    assert "per_slot" not in result
    assert Path(result["report"]).is_file()


def test_frozen_parent_never_reuses_stale_canonical_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "flight.csv"
    source.write_text("timestamp_sec,value\n0,1\n1,2\n", encoding="utf-8")
    executable = tmp_path / "Log ansys.exe"
    executable.write_bytes(b"placeholder")
    output = tmp_path / "output"
    output.mkdir()
    stale = output / "frozen_load_report.json"
    stale.write_text('{"status":"pass"}', encoding="utf-8")

    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )
    result = gate._run_frozen_check(
        executable,
        {"csv": {"fixture": gate.Fixture("csv", source, "real")}},
        output_dir=output,
        min_samples=2,
        timeout_sec=10,
    )

    assert result["status"] == "fail"
    assert "did not write the unique requested report" in result["detail"]
    assert stale.read_text(encoding="utf-8") == '{"status":"pass"}'


def test_frozen_parent_rejects_final_report_input_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    source = output / "frozen_load_report-collision.json"
    original = b"critical input source"
    source.write_bytes(original)
    executable = tmp_path / "Log ansys.exe"
    executable.write_bytes(b"placeholder")
    monkeypatch.setattr(
        gate.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="collision"),
    )

    result = gate._run_frozen_check(
        executable,
        {"json": {"fixture": gate.Fixture("json", source, "real")}},
        output_dir=output,
        min_samples=2,
        timeout_sec=10,
    )

    assert result["status"] == "fail"
    assert "collides with input source" in result["detail"]
    assert source.read_bytes() == original


def test_acceptance_child_rejects_unknown_arguments_without_writes(
    tmp_path: Path,
) -> None:
    from main import _run_acceptance_load_test

    source = tmp_path / "flight.csv"
    original = b"timestamp_sec,value\n0,1\n1,2\n"
    source.write_bytes(original)
    report_path = tmp_path / "report.json"
    cache_path = tmp_path / "cache"

    with pytest.raises(SystemExit) as exc_info:
        _run_acceptance_load_test(
            [
                "--acceptance-load-report",
                str(report_path),
                "--acceptance-cache-dir",
                str(cache_path),
                "--acceptance-run-id",
                "unknown-argument-test",
                "--acceptance-load",
                str(source),
                "--not-a-real-option",
            ]
        )

    assert exc_info.value.code == 2
    assert source.read_bytes() == original
    assert not report_path.exists()
    assert not cache_path.exists()


def test_parent_gate_rejects_fixture_inside_output_tree_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    source = output_dir / "report.json"
    original = b'[{"timestamp_sec": 0.0, "value": 1.0}]'
    source.write_bytes(original)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_format_acceptance.py",
            "--fixture",
            f"json={source}",
            "--provenance",
            "json=real",
            "--output-dir",
            str(output_dir),
            "--no-gui",
        ],
    )

    exit_code = gate.main()

    assert exit_code == 1
    assert source.read_bytes() == original
    captured = capsys.readouterr().out
    assert "input source would be inside disposable acceptance output directory" in captured
    assert "REPORT_NOT_WRITTEN=" in captured


def test_parent_gate_rejects_manifest_inside_output_tree_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = tmp_path / "flight.json"
    fixture.write_text(
        '[{"timestamp_sec": 0.0, "value": 1.0}]',
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    manifest = output_dir / "report.json"
    manifest_payload = {
        "schema_version": gate.SCHEMA_VERSION,
        "fixtures": {
            "json": {
                "path": str(fixture),
                "provenance": "real",
            }
        },
    }
    original = json.dumps(manifest_payload).encode("utf-8")
    manifest.write_bytes(original)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_format_acceptance.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--no-gui",
        ],
    )

    exit_code = gate.main()

    assert exit_code == 1
    assert manifest.read_bytes() == original
    captured = capsys.readouterr().out
    assert "input source would be inside disposable acceptance output directory" in captured
    assert "REPORT_NOT_WRITTEN=" in captured


@pytest.mark.parametrize("collision_kind", ("report", "temporary_report", "cache"))
def test_acceptance_child_rejects_output_input_collisions_without_modifying_source(
    tmp_path: Path,
    collision_kind: str,
) -> None:
    from main import _run_acceptance_load_test

    case_dir = tmp_path / collision_kind
    case_dir.mkdir()
    report_path = case_dir / "report.json"
    cache_path = case_dir / "cache"
    if collision_kind == "report":
        source = report_path
    elif collision_kind == "temporary_report":
        source = report_path.with_suffix(report_path.suffix + ".tmp")
    else:
        source = cache_path
    original = b"do not overwrite this source"
    source.write_bytes(original)

    exit_code = _run_acceptance_load_test(
        [
            "--acceptance-load-report",
            str(report_path),
            "--acceptance-cache-dir",
            str(cache_path),
            "--acceptance-run-id",
            f"collision-{collision_kind}",
            "--acceptance-load",
            str(source),
        ]
    )

    assert exit_code == 1
    assert source.read_bytes() == original
    if report_path != source:
        assert not report_path.exists()


def test_partial_frozen_failure_only_fails_the_matching_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "flight.csv"
    csv_path.write_text(
        "timestamp_sec,value\n0.0,1.0\n0.1,2.0\n",
        encoding="utf-8",
    )
    json_path = tmp_path / "flight.json"
    json_path.write_text(
        '[{"timestamp_sec": 0.0, "value": 1.0}, '
        '{"timestamp_sec": 0.1, "value": 2.0}]',
        encoding="utf-8",
    )

    def fake_frozen_check(*_args, **_kwargs):
        return {
            "status": "fail",
            "detail": "one or more fixtures failed in the frozen process",
            "per_slot": {
                "csv": gate._check("pass", "fixture loaded inside the frozen process"),
                "json": gate._check(
                    "fail", "No module named message_definitions"
                ),
            },
            "inputs": [],
        }

    monkeypatch.setattr(gate, "_run_frozen_check", fake_frozen_check)
    report = gate.run_gate(
        {
            "csv": gate.Fixture("csv", csv_path, "real"),
            "json": gate.Fixture("json", json_path, "real"),
        },
        output_dir=tmp_path / "output",
        check_warm_cache=True,
        run_gui=True,
        frozen_executable=tmp_path / "Log ansys.exe",
        require_frozen=True,
    )

    assert report["packaged_load"]["status"] == "fail"
    assert report["formats"]["csv"]["checks"]["frozen_load"]["status"] == "pass"
    assert report["formats"]["csv"]["status"] == "pass"
    assert report["formats"]["json"]["checks"]["frozen_load"]["status"] == "fail"
    assert report["formats"]["json"]["status"] == "fail"
    assert report["status"] == "fail"
