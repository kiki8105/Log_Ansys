"""Machine-readable acceptance gate for nine core log variants.

The gate deliberately separates three claims:

* a reader can probe and load a fixture;
* the loaded dataset contains graphable numeric time series and can coexist in
  one Log ansys workspace; and
* a *real*, independently supplied fixture exists for release evidence.

Synthetic fixtures are useful regression evidence, but never become a release
``pass`` in this script.  They finish as ``manual_required``.  Missing fixture
slots finish as ``missing_fixture`` instead of silently disappearing from the
matrix.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable
import uuid


os.environ["PX4_LOG_FORMAT_POLICY"] = "multiformat_stable"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PX4_INTERACTIVE_FAST_RENDER_PREVIEW", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from engines.io_engine import LogIOEngine
from storage.parquet_cache import ParquetCacheManager


SCHEMA_VERSION = 1
PROVENANCE_VALUES = frozenset({"real", "synthetic", "unknown"})


@dataclass(frozen=True, slots=True)
class FormatSpec:
    slot: str
    label: str
    expected_reader_id: str
    expected_format_id: str
    suffixes: tuple[str, ...]


FORMAT_SPECS = (
    FormatSpec("px4_ulog", "PX4 ULog", "px4_ulog", "px4_ulog", (".ulg",)),
    FormatSpec("ros1_bag", "ROS 1 bag", "rosbag", "ros1_bag", (".bag",)),
    FormatSpec("ros2_db3", "ROS 2 SQLite3", "rosbag", "ros2_bag", (".db3",)),
    FormatSpec("ros2_mcap", "ROS 2 MCAP", "rosbag", "ros2_bag", (".mcap",)),
    FormatSpec(
        "ardupilot_bin",
        "ArduPilot DataFlash binary",
        "ardupilot",
        "ardupilot_dataflash",
        (".bin",),
    ),
    FormatSpec(
        "ardupilot_log",
        "ArduPilot DataFlash text",
        "ardupilot",
        "ardupilot_dataflash",
        (".log",),
    ),
    FormatSpec(
        "mavlink_tlog",
        "MAVLink telemetry log",
        "ardupilot",
        "mavlink_tlog",
        (".tlog",),
    ),
    FormatSpec("csv", "Delimited CSV", "tabular", "tabular", (".csv",)),
    FormatSpec("json", "JSON records", "tabular", "tabular", (".json",)),
)
FORMAT_BY_SLOT = {spec.slot: spec for spec in FORMAT_SPECS}


@dataclass(frozen=True, slots=True)
class Fixture:
    slot: str
    path: Path
    provenance: str = "unknown"
    notes: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


def _sha256_source(path: Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    digest = hashlib.sha256()
    for member in sorted(
        (candidate for candidate in path.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(path).as_posix(),
    ):
        relative = member.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(member.stat().st_size.to_bytes(8, "big"))
        with member.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest().upper()


def _source_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(member.stat().st_size for member in path.rglob("*") if member.is_file())


def _check(status: str, detail: str = "", **evidence: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": status}
    if detail:
        payload["detail"] = detail
    payload.update(evidence)
    return payload


def _parse_assignment(raw: str, *, option: str) -> tuple[str, str]:
    slot, separator, value = raw.partition("=")
    slot = slot.strip()
    value = value.strip()
    if not separator or not slot or not value:
        raise ValueError(f"{option} must use SLOT=VALUE syntax: {raw!r}")
    if slot not in FORMAT_BY_SLOT:
        expected = ", ".join(FORMAT_BY_SLOT)
        raise ValueError(f"Unknown fixture slot {slot!r}; expected one of: {expected}")
    return slot, value


def _normalise_manifest_entry(
    slot: str,
    raw: Any,
    *,
    base_dir: Path,
) -> Fixture:
    if isinstance(raw, str):
        raw = {"path": raw, "provenance": "unknown"}
    if not isinstance(raw, dict):
        raise ValueError(f"fixtures.{slot} must be an object or path string")
    raw_path = str(raw.get("path", "")).strip()
    if not raw_path:
        raise ValueError(f"fixtures.{slot}.path must not be empty")
    provenance = str(raw.get("provenance", "unknown")).strip().lower()
    if provenance not in PROVENANCE_VALUES:
        allowed = ", ".join(sorted(PROVENANCE_VALUES))
        raise ValueError(f"fixtures.{slot}.provenance must be one of: {allowed}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return Fixture(
        slot=slot,
        path=path.resolve(),
        provenance=provenance,
        notes=str(raw.get("notes", "")).strip(),
    )


def load_fixtures(
    manifest_path: Path | None,
    cli_fixtures: Iterable[str] = (),
    cli_provenance: Iterable[str] = (),
) -> dict[str, Fixture]:
    """Load a strict manifest and apply optional command-line overrides."""

    fixtures: dict[str, Fixture] = {}
    if manifest_path is not None:
        manifest_path = manifest_path.expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            raise ValueError("manifest root must be a JSON object")
        version = payload.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported manifest schema_version={version!r}; expected {SCHEMA_VERSION}"
            )
        raw_fixtures = payload.get("fixtures", {})
        if not isinstance(raw_fixtures, dict):
            raise ValueError("manifest fixtures must be a JSON object")
        unknown = sorted(set(raw_fixtures) - set(FORMAT_BY_SLOT))
        if unknown:
            raise ValueError("Unknown manifest fixture slots: " + ", ".join(unknown))
        for slot, raw in raw_fixtures.items():
            fixtures[slot] = _normalise_manifest_entry(
                slot,
                raw,
                base_dir=manifest_path.parent,
            )

    provenance_overrides: dict[str, str] = {}
    for raw in cli_provenance:
        slot, provenance = _parse_assignment(raw, option="--provenance")
        provenance = provenance.lower()
        if provenance not in PROVENANCE_VALUES:
            allowed = ", ".join(sorted(PROVENANCE_VALUES))
            raise ValueError(f"--provenance value must be one of: {allowed}")
        provenance_overrides[slot] = provenance

    for raw in cli_fixtures:
        slot, raw_path = _parse_assignment(raw, option="--fixture")
        prior = fixtures.get(slot)
        fixtures[slot] = Fixture(
            slot=slot,
            path=Path(raw_path).expanduser().resolve(),
            provenance=provenance_overrides.get(
                slot,
                prior.provenance if prior is not None else "unknown",
            ),
            notes=prior.notes if prior is not None else "",
        )

    for slot, provenance in provenance_overrides.items():
        if slot not in fixtures:
            raise ValueError(f"--provenance supplied without a fixture: {slot}")
        prior = fixtures[slot]
        fixtures[slot] = Fixture(slot, prior.path, provenance, prior.notes)

    resolved_paths: dict[str, str] = {}
    for slot, fixture in fixtures.items():
        key = os.path.normcase(str(fixture.path))
        if key in resolved_paths:
            raise ValueError(
                f"One source cannot satisfy two acceptance slots: "
                f"{resolved_paths[key]} and {slot} -> {fixture.path}"
            )
        resolved_paths[key] = slot
    return fixtures


def _suffix_matches(spec: FormatSpec, source: Path) -> bool:
    return source.is_file() and source.suffix.lower() in spec.suffixes


def _numeric_summary(dataset: Any, *, min_samples: int) -> dict[str, Any]:
    signals: list[dict[str, Any]] = []
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
            finite = np.isfinite(timestamps) & np.isfinite(values)
            finite_count = int(np.count_nonzero(finite))
            if finite_count == 0:
                continue
            finite_x = timestamps[finite]
            finite_y = values[finite]
            signals.append(
                {
                    "topic": topic_name,
                    "signal": signal_name,
                    "finite_samples": finite_count,
                    "start_sec": float(np.min(finite_x)),
                    "end_sec": float(np.max(finite_x)),
                    "min": float(np.min(finite_y)),
                    "max": float(np.max(finite_y)),
                }
            )
    signals.sort(
        key=lambda item: (
            item["finite_samples"] >= min_samples,
            item["finite_samples"],
            item["topic"],
            item["signal"],
        ),
        reverse=True,
    )
    renderable = [item for item in signals if item["finite_samples"] >= min_samples]
    signature_payload = [
        {
            "topic": item["topic"],
            "signal": item["signal"],
            "finite_samples": item["finite_samples"],
            "start_sec": item["start_sec"],
            "end_sec": item["end_sec"],
            "min": item["min"],
            "max": item["max"],
        }
        for item in sorted(signals, key=lambda item: (item["topic"], item["signal"]))
    ]
    signature = hashlib.sha256(
        json.dumps(
            signature_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest().upper()
    return {
        "topic_count": len(dataset.topics),
        "numeric_signal_count": len(signals),
        "renderable_signal_count": len(renderable),
        "selected": renderable[0] if renderable else None,
        "signature": signature,
    }


def _source_variant_check(spec: FormatSpec, source: Path) -> dict[str, Any]:
    if source.is_dir():
        return _check(
            "fail",
            f"{spec.slot} requires a standalone {', '.join(spec.suffixes)} fixture, not a directory",
        )
    if not _suffix_matches(spec, source):
        return _check(
            "fail",
            f"expected suffix {', '.join(spec.suffixes)}, got {source.suffix or '(none)'}",
        )
    return _check("pass", source.suffix.lower())


def _corrupt_fixture(spec: FormatSpec, directory: Path) -> Path:
    suffix = spec.suffixes[0]
    path = directory / f"corrupt_{spec.slot}{suffix}"
    # An empty source is invalid for every supported reader.  The suffix still
    # drives the same user entry point while content-based probing must reject
    # it before a partial dataset can enter the session.
    path.write_bytes(b"")
    return path


def _engine_session_snapshot(
    loaded: dict[str, dict[str, Any]],
) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        sorted(
            (
                slot,
                id(record["result"].dataset),
                record["numeric"]["signature"],
            )
            for slot, record in loaded.items()
        )
    )


def _validate_fixture(
    spec: FormatSpec,
    fixture: Fixture,
    *,
    engine: LogIOEngine,
    min_samples: int,
    check_warm_cache: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    source = fixture.path
    record: dict[str, Any] = {
        "slot": spec.slot,
        "label": spec.label,
        "status": "fail",
        "provenance": fixture.provenance,
        "path": str(source),
        "notes": fixture.notes,
        "expected_reader_id": spec.expected_reader_id,
        "expected_format_id": spec.expected_format_id,
        "checks": {},
    }
    if not source.exists():
        record["status"] = "missing_fixture"
        record["checks"]["fixture"] = _check("missing_fixture", "path does not exist")
        return record, None
    if not source.is_file():
        record["checks"]["fixture"] = _check("fail", "fixture must be a regular file")
        return record, None

    record["checks"]["fixture"] = _check("pass")
    record["checks"]["source_variant"] = _source_variant_check(spec, source)
    record["size_bytes"] = _source_size(source)
    record["sha256"] = _sha256_source(source)

    try:
        probe_started = time.perf_counter()
        reader, probe = engine.detect(str(source))
        probe_ms = (time.perf_counter() - probe_started) * 1000.0
        probe_ok = (
            reader.id == spec.expected_reader_id
            and probe.format_id == spec.expected_format_id
            and probe.confidence > 0
        )
        record["probe"] = {
            "reader_id": reader.id,
            "format_id": probe.format_id,
            "confidence": probe.confidence,
            "reason": probe.reason,
            "elapsed_ms": round(probe_ms, 3),
        }
        record["checks"]["probe"] = _check(
            "pass" if probe_ok else "fail",
            "content probe matched the required reader and format"
            if probe_ok
            else "probe selected a different reader or format",
        )
    except Exception as exc:
        record["checks"]["probe"] = _check(
            "fail", f"{type(exc).__name__}: {exc}"
        )
        return record, None

    try:
        cold_started = time.perf_counter()
        result = engine.load_result(str(source))
        cold_ms = (time.perf_counter() - cold_started) * 1000.0
        numeric = _numeric_summary(result.dataset, min_samples=min_samples)
        load_ok = (
            result.format_id == spec.expected_format_id
            and str(result.dataset.source_format) == spec.expected_format_id
        )
        record["load"] = {
            "format_id": result.format_id,
            "dataset_source_format": str(result.dataset.source_format),
            "reader_id": engine.last_load_info.get("reader_id"),
            "load_type": engine.last_load_info.get("load_type"),
            "elapsed_ms": round(cold_ms, 3),
            "warnings": list(result.warnings),
            **numeric,
        }
        record["checks"]["load"] = _check(
            "pass" if load_ok else "fail",
            "reader produced the expected common dataset"
            if load_ok
            else "loaded result format did not match the acceptance slot",
        )
        numeric_ok = numeric["renderable_signal_count"] > 0
        record["checks"]["numeric_timeseries"] = _check(
            "pass" if numeric_ok else "fail",
            f"requires at least one signal with {min_samples} finite timestamp/value pairs",
            selected=numeric["selected"],
        )
    except Exception as exc:
        record["checks"]["load"] = _check(
            "fail", f"{type(exc).__name__}: {exc}"
        )
        return record, None

    if check_warm_cache:
        try:
            warm_started = time.perf_counter()
            warm_result = engine.load_result(str(source))
            warm_ms = (time.perf_counter() - warm_started) * 1000.0
            warm_numeric = _numeric_summary(warm_result.dataset, min_samples=min_samples)
            warm_type = str(engine.last_load_info.get("load_type", ""))
            warm_ok = (
                "Cache" in warm_type
                and warm_result.format_id == result.format_id
                and warm_numeric["signature"] == numeric["signature"]
            )
            record["warm_load"] = {
                "format_id": warm_result.format_id,
                "load_type": warm_type,
                "elapsed_ms": round(warm_ms, 3),
                **warm_numeric,
            }
            record["checks"]["warm_cache"] = _check(
                "pass" if warm_ok else "fail",
                "cache load preserved numeric structure"
                if warm_ok
                else "warm load was not a cache hit or changed numeric structure",
            )
        except Exception as exc:
            record["checks"]["warm_cache"] = _check(
                "fail", f"{type(exc).__name__}: {exc}"
            )
    else:
        record["checks"]["warm_cache"] = _check(
            "manual_required", "warm-cache validation was explicitly skipped"
        )

    technical_failure = any(
        check.get("status") == "fail" for check in record["checks"].values()
    )
    if technical_failure:
        record["status"] = "fail"
    elif fixture.provenance != "real":
        record["status"] = "manual_required"
        record["release_evidence"] = _check(
            "manual_required",
            "a real captured log is required; synthetic/unknown provenance cannot release this format",
        )
    else:
        record["status"] = "pass"
        record["release_evidence"] = _check("pass", "real fixture declared")
    return record, {"result": result, "numeric": numeric, "fixture": fixture}


def _gui_snapshot(window: Any, workspace: Any) -> dict[str, Any]:
    plot = workspace.first_plot
    return {
        "dataset_keys": tuple(window.loaded_datasets),
        "dataset_ids": tuple(
            (key, id(value)) for key, value in window.loaded_datasets.items()
        ),
        "aircraft_keys": tuple(window.loaded_aircraft_types),
        "metadata_keys": tuple(window.loaded_log_metadata),
        "active": window.active_analysis_log,
        "tree_rows": window.tree_model.invisibleRootItem().rowCount(),
        "plotted": tuple(plot.plotted_signals),
        "signal_cache": tuple(plot.signal_cache),
        "global_range": (workspace.global_min_x, workspace.global_max_x),
    }


def _run_gui_session_check(
    loaded: dict[str, dict[str, Any]],
    corrupt_paths: dict[str, Path],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Apply every parsed dataset and exercise UI transaction isolation."""

    from PySide6.QtWidgets import QApplication
    import pyqtgraph as pg

    from gui.main_window import (
        LogLoadWorker,
        MainWindow,
        Workspace,
        _metadata_from_load_result,
        _preprocess_loaded_dataset,
    )

    pg.setConfigOptions(useOpenGL=False)
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1500, 850)
    window.show()
    app.processEvents()
    workspace = window.tab_widget.currentWidget()
    if not isinstance(workspace, Workspace):
        raise RuntimeError("initial workspace was not created")

    per_slot: dict[str, Any] = {}
    applied_keys: dict[str, str] = {}
    try:
        for slot, loaded_record in loaded.items():
            result = loaded_record["result"]
            fixture: Fixture = loaded_record["fixture"]
            selected = loaded_record["numeric"]["selected"]
            if selected is None:
                per_slot[slot] = _check("fail", "no renderable signal selected")
                continue
            if result.format_id == "px4_ulog":
                _preprocess_loaded_dataset(result.dataset)
                metadata = {
                    "evaluation": {
                        "overall_score": None,
                        "overall_status": "not_run_in_format_acceptance",
                        "overall_status_label": "format acceptance only",
                        "items": [],
                    }
                }
                aircraft_type = "PX4"
            else:
                metadata = _metadata_from_load_result(result)
                aircraft_type = result.format_id

            before_keys = set(window.loaded_datasets)
            payload = {
                "file_path": str(fixture.path),
                "filename": fixture.path.name,
                "dataset": result.dataset,
                "aircraft_type": aircraft_type,
                "metadata": metadata,
                "format_id": result.format_id,
            }
            accepted = bool(window._apply_loaded_log_result(payload))
            after_keys = set(window.loaded_datasets)
            new_keys = sorted(after_keys - before_keys)
            if not accepted or len(new_keys) != 1:
                per_slot[slot] = _check(
                    "fail",
                    "workspace rejected the dataset or source identity was not unique",
                    before=sorted(before_keys),
                    after=sorted(after_keys),
                )
                continue
            source_key = new_keys[0]
            rendered = workspace.first_plot.render_signal(
                source_key,
                selected["topic"],
                selected["signal"],
            )
            app.processEvents()
            if not rendered:
                per_slot[slot] = _check(
                    "fail",
                    f"failed to render {selected['topic']}.{selected['signal']}",
                )
                continue
            applied_keys[slot] = source_key
            per_slot[slot] = _check(
                "pass",
                "dataset committed and selected numeric signal rendered",
                source_key=source_key,
                topic=selected["topic"],
                signal=selected["signal"],
            )

        app.processEvents()
        isolation: dict[str, Any] = {}
        for slot, corrupt_path in corrupt_paths.items():
            before = _gui_snapshot(window, workspace)
            emitted: list[dict[str, Any]] = []
            summaries: list[dict[str, Any]] = []
            worker = LogLoadWorker(
                [str(corrupt_path)],
                existing_names=set(window.loaded_datasets),
            )
            worker.fileLoaded.connect(emitted.append)
            worker.finished.connect(summaries.append)
            # LogLoadWorker prints a traceback for normal parser failures.
            # Corrupt inputs are intentional here; the structured summary is
            # the evidence and repeated tracebacks would obscure gate output.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                worker.run()
            app.processEvents()
            after = _gui_snapshot(window, workspace)
            summary = summaries[0] if len(summaries) == 1 else None
            isolated = (
                not emitted
                and summary is not None
                and int(summary.get("emitted_count", -1)) == 0
                and len(summary.get("failed", [])) == 1
                and after == before
            )
            isolation[slot] = _check(
                "pass" if isolated else "fail",
                "failed source left the existing UI session unchanged"
                if isolated
                else "failed source mutated the UI session or was unexpectedly accepted",
                summary=summary,
            )

        screenshot = output_dir / "mixed_format_acceptance.png"
        pixmap = window.grab()
        screenshot_written = (
            not pixmap.isNull() and pixmap.save(str(screenshot), "PNG")
        )
        expected_count = len(loaded)
        curve_count = len(workspace.first_plot.plotted_signals)
        all_applied = len(applied_keys) == expected_count
        all_isolated = len(isolation) == expected_count and all(
            item["status"] == "pass" for item in isolation.values()
        )
        status = "pass" if all_applied and all_isolated and curve_count == expected_count else "fail"
        return {
            "status": status,
            "scope": "offscreen Qt workspace commit/render and failed-load transaction isolation",
            "loaded_slots": sorted(applied_keys),
            "curve_count": curve_count,
            "expected_curve_count": expected_count,
            "per_slot": per_slot,
            "corrupt_isolation": isolation,
            "screenshot": str(screenshot) if screenshot_written else None,
        }
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def _run_frozen_check(
    executable: Path,
    loaded: dict[str, dict[str, Any]],
    *,
    output_dir: Path,
    min_samples: int,
    timeout_sec: float,
) -> dict[str, Any]:
    executable = executable.expanduser().resolve()
    if not executable.is_file():
        return _check("manual_required", f"frozen executable not found: {executable}")
    if not loaded:
        return _check("missing_fixture", "no loadable fixtures were available")

    output_dir = output_dir.expanduser().resolve()
    run_id = uuid.uuid4().hex
    final_report_path = output_dir / f"frozen_load_report-{run_id}.json"
    sources = [record["fixture"].path.resolve() for record in loaded.values()]
    collision = _acceptance_path_collision(
        (
            ("report_directory", output_dir, False),
            ("final_report", final_report_path, False),
        ),
        sources,
    )
    if collision is not None:
        return _check("fail", collision, run_id=run_id)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _check(
            "fail",
            f"could not create frozen report directory: {type(exc).__name__}: {exc}",
            run_id=run_id,
        )
    if final_report_path.exists():
        return _check(
            "fail",
            f"refusing to reuse an existing frozen report: {final_report_path}",
            run_id=run_id,
        )

    with tempfile.TemporaryDirectory(prefix=f"log-ansys-frozen-{run_id}-") as temp_name:
        temp_root = Path(temp_name).resolve()
        child_report_path = temp_root / f"frozen-child-{run_id}.json"
        cache_dir = temp_root / "cache"
        collision = _acceptance_path_collision(
            (
                ("temporary_directory", temp_root, True),
                ("child_report", child_report_path, False),
                ("cache_directory", cache_dir, True),
            ),
            sources,
        )
        if collision is not None:
            return _check("fail", collision, run_id=run_id)

        command = [
            str(executable),
            "--acceptance-load-report",
            str(child_report_path),
            "--acceptance-cache-dir",
            str(cache_dir),
            "--acceptance-run-id",
            run_id,
            "--acceptance-min-samples",
            str(min_samples),
        ]
        for record in loaded.values():
            command.extend(["--acceptance-load", str(record["fixture"].path)])

        try:
            completed = subprocess.run(
                command,
                cwd=str(executable.parent),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return _check(
                "fail",
                f"frozen acceptance command timed out after {timeout_sec:g}s; rebuild may be required",
                run_id=run_id,
            )
        except OSError as exc:
            return _check(
                "fail",
                f"could not start frozen executable: {exc}",
                run_id=run_id,
            )

        if not child_report_path.is_file():
            return _check(
                "fail",
                "frozen executable did not write the unique requested report; rebuild may be required",
                run_id=run_id,
                exit_code=completed.returncode,
                stdout=completed.stdout[-4000:],
                stderr=completed.stderr[-4000:],
            )
        try:
            # Preserve exactly what the child produced for diagnostics while
            # the temporary report and cache are removed on context exit.
            with child_report_path.open("rb") as source_stream:
                with final_report_path.open("xb") as destination_stream:
                    shutil.copyfileobj(source_stream, destination_stream)
        except OSError as exc:
            return _check(
                "fail",
                f"could not preserve frozen report: {type(exc).__name__}: {exc}",
                run_id=run_id,
            )
        try:
            report = json.loads(child_report_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            return _check(
                "fail",
                f"invalid frozen report: {type(exc).__name__}: {exc}",
                run_id=run_id,
                report=str(final_report_path),
            )
        if not isinstance(report, dict):
            return _check(
                "fail",
                "frozen report root must be an object",
                run_id=run_id,
                report=str(final_report_path),
                exit_code=completed.returncode,
            )

        header_errors = []
        if report.get("schema_version") != SCHEMA_VERSION:
            header_errors.append(
                f"schema_version={report.get('schema_version')!r}, expected {SCHEMA_VERSION}"
            )
        if report.get("run_id") != run_id:
            header_errors.append(
                f"run_id={report.get('run_id')!r}, expected {run_id!r}"
            )
        if report.get("frozen") is not True:
            header_errors.append("frozen must be exactly true")
        if header_errors:
            return _check(
                "fail",
                "untrusted frozen report header: " + "; ".join(header_errors),
                run_id=run_id,
                report=str(final_report_path),
                exit_code=completed.returncode,
            )

    raw_inputs = report.get("inputs", [])
    if not isinstance(raw_inputs, list):
        return _check(
            "fail",
            "frozen report inputs must be a list",
            run_id=run_id,
            report=str(final_report_path),
        )
    per_slot, unmatched_inputs = _map_frozen_inputs(loaded, raw_inputs)
    all_slots_passed = bool(per_slot) and all(
        item["status"] == "pass" for item in per_slot.values()
    )
    passed = (
        completed.returncode == 0
        and report.get("status") == "pass"
        and all_slots_passed
        and not unmatched_inputs
    )
    return {
        "status": "pass" if passed else "fail",
        "detail": "all fixtures loaded inside the frozen process"
        if passed
        else "one or more fixtures failed in the frozen process",
        "executable": str(executable),
        "run_id": run_id,
        "exit_code": completed.returncode,
        "report": str(final_report_path),
        "inputs": raw_inputs,
        "per_slot": per_slot,
        "unmatched_inputs": unmatched_inputs,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _acceptance_path_collision(
    outputs: Iterable[tuple[str, Path, bool]],
    sources: Iterable[Path],
) -> str | None:
    """Reject output trees that could overwrite or delete an input source."""

    canonical_sources = [Path(source).expanduser().resolve() for source in sources]
    for label, raw_output, is_tree in outputs:
        output = Path(raw_output).expanduser().resolve()
        for source in canonical_sources:
            if output == source:
                return f"{label} collides with input source: {source}"
            if source.is_dir() and _is_relative_to(output, source):
                return f"{label} would be created inside input source directory: {source}"
            if is_tree and _is_relative_to(source, output):
                return f"input source would be inside disposable {label}: {source}"
    return None


def _normalised_path_key(raw_path: Any) -> str | None:
    """Return one stable key for source/report path correlation."""

    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    try:
        resolved = Path(text).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    return os.path.normcase(os.path.normpath(str(resolved)))


def _map_frozen_inputs(
    loaded: dict[str, dict[str, Any]],
    inputs: list[Any],
) -> tuple[dict[str, dict[str, Any]], list[Any]]:
    """Map frozen-process input records to fixture slots without status bleed.

    A non-zero frozen process exit is expected when even one input fails.  The
    process-level result therefore cannot be copied to every format.  Each
    expected fixture must have exactly one path-matched record and inherits
    only that record's status.
    """

    expected_by_path: dict[str, str] = {}
    for slot, loaded_record in loaded.items():
        key = _normalised_path_key(loaded_record["fixture"].path)
        if key is not None:
            expected_by_path[key] = slot

    observed_by_slot: dict[str, list[dict[str, Any]]] = {
        slot: [] for slot in loaded
    }
    unmatched: list[Any] = []
    for raw_item in inputs:
        if not isinstance(raw_item, dict):
            unmatched.append(raw_item)
            continue
        key = _normalised_path_key(raw_item.get("path"))
        slot = expected_by_path.get(key) if key is not None else None
        if slot is None:
            unmatched.append(raw_item)
            continue
        observed_by_slot[slot].append(raw_item)

    per_slot: dict[str, dict[str, Any]] = {}
    for slot, loaded_record in loaded.items():
        expected_path = str(loaded_record["fixture"].path.resolve())
        candidates = observed_by_slot.get(slot, [])
        if not candidates:
            per_slot[slot] = _check(
                "fail",
                "frozen report omitted this fixture",
                path=expected_path,
            )
            continue
        if len(candidates) != 1:
            per_slot[slot] = _check(
                "fail",
                f"frozen report contained {len(candidates)} records for this fixture",
                path=expected_path,
                inputs=candidates,
            )
            continue
        item = candidates[0]
        spec = FORMAT_BY_SLOT.get(slot)
        expected = {
            "reader_id": spec.expected_reader_id if spec is not None else None,
            "probe_format_id": spec.expected_format_id if spec is not None else None,
            "format_id": spec.expected_format_id if spec is not None else None,
        }
        observed = {
            "reader_id": item.get("reader_id"),
            "probe_format_id": item.get("probe_format_id"),
            "format_id": item.get("format_id"),
        }
        mismatches = {
            field: {"expected": expected_value, "observed": observed[field]}
            for field, expected_value in expected.items()
            if expected_value is None or observed[field] != expected_value
        }
        item_status = (
            "pass"
            if item.get("status") == "pass" and not mismatches
            else "fail"
        )
        if item.get("status") != "pass":
            detail = str(item.get("error") or "fixture failed inside the frozen process")
        elif mismatches:
            detail = "frozen reader/format identity mismatch"
        elif item_status == "pass":
            detail = "fixture loaded inside the frozen process with expected reader and format"
        per_slot[slot] = _check(
            item_status,
            detail,
            path=expected_path,
            expected=expected,
            observed=observed,
            mismatches=mismatches,
            input=item,
        )
    return per_slot, unmatched


def _status_from_formats(records: Iterable[dict[str, Any]]) -> str:
    statuses = [record["status"] for record in records]
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "missing_fixture" for status in statuses):
        return "incomplete"
    if any(status == "manual_required" for status in statuses):
        return "manual_required"
    return "pass"


def status_exit_code(status: str) -> int:
    if status == "pass":
        return 0
    if status == "fail":
        return 1
    return 2


def run_gate(
    fixtures: dict[str, Fixture],
    *,
    output_dir: Path,
    min_samples: int = 2,
    check_warm_cache: bool = True,
    run_gui: bool = True,
    frozen_executable: Path | None = None,
    require_frozen: bool = False,
    frozen_timeout_sec: float = 300.0,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2 for graph acceptance")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "project_root": str(PROJECT_ROOT),
        "git_revision": _git_revision(),
        "policy": os.environ.get("PX4_LOG_FORMAT_POLICY"),
        "release_rule": (
            "Every required slot needs a real fixture and all probe/load/numeric/cache/"
            "mixed-session/isolation checks. Synthetic or unknown provenance is never a release pass."
        ),
        "formats": {},
    }

    loaded: dict[str, dict[str, Any]] = {}
    cache_parent = output_dir / ".acceptance-cache"
    cache_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="run-", dir=cache_parent) as cache_name:
        engine = LogIOEngine(cache_mgr=ParquetCacheManager(cache_dir=cache_name))
        for spec in FORMAT_SPECS:
            fixture = fixtures.get(spec.slot)
            if fixture is None:
                report["formats"][spec.slot] = {
                    "slot": spec.slot,
                    "label": spec.label,
                    "status": "missing_fixture",
                    "expected_reader_id": spec.expected_reader_id,
                    "expected_format_id": spec.expected_format_id,
                    "checks": {
                        "fixture": _check(
                            "missing_fixture",
                            "no manifest or --fixture entry was supplied",
                        )
                    },
                }
                continue
            format_record, loaded_record = _validate_fixture(
                spec,
                fixture,
                engine=engine,
                min_samples=min_samples,
                check_warm_cache=check_warm_cache,
            )
            report["formats"][spec.slot] = format_record
            if loaded_record is not None and format_record["status"] != "fail":
                loaded[spec.slot] = loaded_record

        before = _engine_session_snapshot(loaded)
        corrupt_dir = output_dir / ".corrupt-fixtures"
        corrupt_dir.mkdir(parents=True, exist_ok=True)
        corrupt_paths: dict[str, Path] = {}
        engine_isolation: dict[str, Any] = {}
        for slot, loaded_record in loaded.items():
            spec = FORMAT_BY_SLOT[slot]
            corrupt_path = _corrupt_fixture(spec, corrupt_dir)
            corrupt_paths[slot] = corrupt_path
            try:
                engine.load_result(str(corrupt_path))
            except Exception as exc:
                unchanged = _engine_session_snapshot(loaded) == before
                engine_isolation[slot] = _check(
                    "pass" if unchanged else "fail",
                    "corrupt source was rejected and existing loaded results remained intact"
                    if unchanged
                    else "corrupt source changed the in-memory acceptance session",
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                engine_isolation[slot] = _check(
                    "fail", "corrupt source was unexpectedly accepted"
                )

    distinct_formats = {
        record["result"].format_id for record in loaded.values()
    }
    enough_formats_to_mix = len(distinct_formats) >= 2
    model_session_passed = (
        enough_formats_to_mix
        and len(loaded) == len(
            [record for record in report["formats"].values() if record["status"] != "missing_fixture"]
        )
        and all(item["status"] == "pass" for item in engine_isolation.values())
    )
    report["cross_format_session"] = {
        "status": "pass" if model_session_passed else (
            "missing_fixture" if not enough_formats_to_mix else "fail"
        ),
        "scope": "one LogIOEngine run, retained common-model datasets, corrupt-load isolation",
        "loaded_slots": sorted(loaded),
        "distinct_format_ids": sorted(distinct_formats),
        "engine_corrupt_isolation": engine_isolation,
    }

    if run_gui and loaded:
        try:
            gui_report = _run_gui_session_check(
                loaded,
                corrupt_paths,
                output_dir=output_dir,
            )
            if gui_report.get("status") == "pass" and not enough_formats_to_mix:
                gui_report["status"] = "missing_fixture"
                gui_report["detail"] = (
                    "at least two distinct format IDs are required to claim a mixed workspace"
                )
        except Exception as exc:
            gui_report = _check("fail", f"{type(exc).__name__}: {exc}")
    elif run_gui:
        gui_report = _check("missing_fixture", "no loadable fixture was available")
    else:
        gui_report = _check(
            "manual_required", "Qt mixed-workspace validation was explicitly skipped"
        )
    report["gui_session"] = gui_report

    for slot, record in report["formats"].items():
        if slot not in loaded:
            continue
        engine_check = engine_isolation.get(slot, _check("fail", "isolation result missing"))
        record["checks"]["corrupt_isolation"] = engine_check
        ui_commit = gui_report.get("per_slot", {}).get(slot)
        ui_isolation = gui_report.get("corrupt_isolation", {}).get(slot)
        if ui_commit is not None and ui_isolation is not None:
            record["checks"]["gui_commit_and_render"] = ui_commit
            record["checks"]["gui_corrupt_isolation"] = ui_isolation
        else:
            record["checks"]["gui_commit_and_render"] = dict(gui_report)

        if any(check.get("status") == "fail" for check in record["checks"].values()):
            record["status"] = "fail"
        elif any(
            check.get("status") == "manual_required"
            for check in record["checks"].values()
        ) or record.get("provenance") != "real":
            record["status"] = "manual_required"
        else:
            record["status"] = "pass"

    if frozen_executable is not None:
        frozen_report = _run_frozen_check(
            frozen_executable,
            loaded,
            output_dir=output_dir,
            min_samples=min_samples,
            timeout_sec=frozen_timeout_sec,
        )
    else:
        frozen_report = _check(
            "manual_required",
            "not run; pass --frozen-exe after building the current source",
        )
    report["packaged_load"] = frozen_report

    frozen_per_slot = frozen_report.get("per_slot")
    if frozen_executable is not None and isinstance(frozen_per_slot, dict):
        for slot in loaded:
            slot_check = frozen_per_slot.get(
                slot,
                _check("fail", "frozen result for this fixture is missing"),
            )
            report["formats"][slot]["checks"]["frozen_load"] = slot_check
            if slot_check.get("status") == "fail":
                report["formats"][slot]["status"] = "fail"
    elif frozen_executable is not None and frozen_report["status"] == "fail":
        # No parseable per-input report exists (timeout, startup failure, or
        # malformed output), so no slot has evidence of a successful packaged
        # load.  This differs from a valid report with only some failed slots.
        for slot in loaded:
            report["formats"][slot]["checks"]["frozen_load"] = dict(frozen_report)
            report["formats"][slot]["status"] = "fail"
    elif require_frozen:
        for slot in loaded:
            report["formats"][slot]["checks"]["frozen_load"] = dict(frozen_report)
            if report["formats"][slot]["status"] == "pass":
                report["formats"][slot]["status"] = "manual_required"

    gate_status = _status_from_formats(report["formats"].values())
    if gate_status == "pass" and report["cross_format_session"]["status"] != "pass":
        gate_status = "fail"
    if gate_status == "pass" and gui_report.get("status") != "pass":
        gate_status = "manual_required" if gui_report.get("status") == "manual_required" else "fail"
    if require_frozen and gate_status == "pass" and frozen_report["status"] != "pass":
        gate_status = "manual_required"

    status_counts: dict[str, int] = {}
    for record in report["formats"].values():
        status = record["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    report["status_counts"] = status_counts
    report["status"] = gate_status
    report["exit_code"] = status_exit_code(gate_status)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Private JSON fixture manifest. Relative paths resolve beside the manifest.",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        default=[],
        metavar="SLOT=PATH",
        help="Add or override one fixture without storing its path in Git.",
    )
    parser.add_argument(
        "--provenance",
        action="append",
        default=[],
        metavar="SLOT=real|synthetic|unknown",
        help="Declare fixture provenance. Only real fixtures can reach release pass.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runtime" / "test-artifacts" / "format-acceptance",
    )
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument(
        "--skip-warm-cache",
        action="store_true",
        help="Skip cold/warm equivalence. The affected slots remain manual_required.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip offscreen MainWindow mixed-session checks. The gate cannot fully pass.",
    )
    parser.add_argument(
        "--frozen-exe",
        type=Path,
        help="Current packaged Log ansys executable to run real load checks inside.",
    )
    parser.add_argument(
        "--require-frozen",
        action="store_true",
        help="Keep the gate incomplete unless --frozen-exe passes.",
    )
    parser.add_argument("--frozen-timeout-sec", type=float, default=300.0)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    report_path = output_dir / "report.json"
    report_path_is_safe = False
    try:
        fixtures = load_fixtures(
            args.manifest,
            cli_fixtures=args.fixture,
            cli_provenance=args.provenance,
        )
        acceptance_inputs = [fixture.path for fixture in fixtures.values()]
        if args.manifest is not None:
            acceptance_inputs.append(args.manifest.expanduser().resolve())
        if args.frozen_exe is not None:
            acceptance_inputs.append(args.frozen_exe.expanduser().resolve())
        collision = _acceptance_path_collision(
            [("acceptance output directory", output_dir, True)],
            acceptance_inputs,
        )
        if collision is not None:
            report = {
                "schema_version": SCHEMA_VERSION,
                "generated_at_utc": _utc_now(),
                "status": "fail",
                "exit_code": 1,
                "error": collision,
            }
        else:
            # Only permit artifact writes after every fixture has been
            # resolved and proven to live outside the complete output tree.
            # The gate creates caches, corrupt fixtures, screenshots, frozen
            # child reports, and the final report below that directory.
            report_path_is_safe = True
            report = run_gate(
                fixtures,
                output_dir=output_dir,
                min_samples=args.min_samples,
                check_warm_cache=not args.skip_warm_cache,
                run_gui=not args.no_gui,
                frozen_executable=args.frozen_exe,
                require_frozen=args.require_frozen,
                frozen_timeout_sec=args.frozen_timeout_sec,
            )
    except Exception as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": _utc_now(),
            "status": "fail",
            "exit_code": 1,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if report_path_is_safe:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report_path_is_safe:
        print(f"REPORT={report_path}")
    else:
        print("REPORT_NOT_WRITTEN=unsafe or unresolved fixture paths")
    return int(report.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
