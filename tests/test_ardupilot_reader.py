from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import readers.ardupilot as ardupilot  # noqa: E402
from readers.ardupilot import (  # noqa: E402
    DATAFLASH_FORMAT_ID,
    MAVLINK_TLOG_FORMAT_ID,
    ArduPilotReader,
    _dataflash_instance_id,
    _extract_timestamp_ns,
    _mavlink_instance_id,
    _numeric_row,
    _rows_to_topic,
)


def _dataflash_fmt_packet(name: bytes = b"IMU\x00") -> bytes:
    # 실제 FMT 레이아웃의 핵심(Type/Length/Name)을 넣어 단순 확장자 판별을 막는다.
    return (
        b"\xA3\x95\x80"
        + bytes((1, 32))
        + name
        + b"Qfff".ljust(16, b"\x00")
        + b"TimeUS,GyrX,GyrY,GyrZ".ljust(64, b"\x00")
    )


def _mavlink_v1_frame(sequence: int) -> bytes:
    # probe는 dialect가 없어도 동작해야 하므로 CRC 의미가 아닌 프레임 경계만 구성한다.
    return bytes((0xFE, 0, sequence & 0xFF, 1, 1, 0, 0, 0))


def test_probe_binary_dataflash_uses_content_not_extension(tmp_path: Path):
    path = tmp_path / "flight.unknown"
    path.write_bytes(_dataflash_fmt_packet() + b"\xA3\x95\x01payload")

    result = ArduPilotReader().probe(path)

    assert result.format_id == DATAFLASH_FORMAT_ID
    assert result.confidence >= 90
    assert "FMT" in result.reason


def test_probe_text_dataflash_overrides_tlog_extension(tmp_path: Path):
    path = tmp_path / "misleading.tlog"
    path.write_text(
        "FMT, 128, 89, FMT, BBnNZ, Type,Length,Name,Format,Columns\n"
        "FMT, 129, 32, IMU, Qfff, TimeUS,GyrX,GyrY,GyrZ\n",
        encoding="ascii",
    )

    result = ArduPilotReader().probe(path)

    assert result.format_id == DATAFLASH_FORMAT_ID
    assert result.confidence >= 90


def test_probe_rejects_random_bin_despite_supported_extension(tmp_path: Path):
    path = tmp_path / "not-a-log.bin"
    path.write_bytes(b"ordinary binary data without a format record")

    result = ArduPilotReader().probe(path)

    assert result.confidence == 0


def test_probe_timestamped_mavlink_frames_despite_bin_extension(tmp_path: Path):
    path = tmp_path / "telemetry.bin"
    payload = b"".join(
        (
            struct.pack(">Q", 1_700_000_000_000_000) + _mavlink_v1_frame(1),
            struct.pack(">Q", 1_700_000_000_010_000) + _mavlink_v1_frame(2),
            struct.pack(">Q", 1_700_000_000_020_000) + _mavlink_v1_frame(3),
        )
    )
    path.write_bytes(payload)

    result = ArduPilotReader().probe(path)

    assert result.format_id == MAVLINK_TLOG_FORMAT_ID
    assert result.confidence >= 90
    assert "MAVLink" in result.reason


def test_timestamp_priority_and_unit_conversion():
    fields = {
        "TimeUS": 2_500_000,
        "TimeMS": 9_999,
        "time_boot_ms": 8_888,
    }

    assert _extract_timestamp_ns(fields) == 2_500_000_000
    assert _extract_timestamp_ns({"TimeMS": 12.5}) == 12_500_000
    assert _extract_timestamp_ns({"time_boot_ms": 7}) == 7_000_000


def test_tlog_packet_timestamp_precedes_payload_clock():
    class Message:
        _timestamp = 123.25

    assert (
        _extract_timestamp_ns(
            {"time_boot_ms": 9_999}, Message(), prefer_message_timestamp=True
        )
        == 123_250_000_000
    )


def test_instance_helpers_are_stable_and_avoid_float_i_false_positive():
    assert _dataflash_instance_id({"I": 2}) == 2
    assert _dataflash_instance_id({"I": 2.0}) == 0
    assert _dataflash_instance_id({"Instance": 3, "I": 1}) == 3
    assert _mavlink_instance_id(1, 42) == 298
    assert _mavlink_instance_id(2, 42) != _mavlink_instance_id(1, 42)


def test_numeric_row_keeps_only_scalar_signals():
    row = _numeric_row(
        {
            "TimeUS": 10,
            "mavpackettype": "IMU",
            "X": 1.5,
            "Count": 2,
            "Healthy": True,
            "Label": "imu0",
            "Samples": [1, 2, 3],
        },
        10_000,
    )

    assert row == {"__source_timestamp_ns": 10_000, "X": 1.5, "Count": 2}


def test_rows_to_topic_preserves_source_time_and_builds_relative_time():
    topic = _rows_to_topic(
        "IMU",
        0,
        [
            {"__source_timestamp_ns": 2_000_000_000, "X": 2.0},
            {"__source_timestamp_ns": 1_000_000_000, "X": 1.0},
        ],
        origin_ns=1_000_000_000,
    )

    assert topic is not None
    assert topic.unique_name == "IMU_0"
    assert topic.dataframe["timestamp_sec"].to_list() == [0.0, 1.0]
    assert topic.dataframe["source_timestamp_ns"].to_list() == [
        1_000_000_000,
        2_000_000_000,
    ]
    assert set(topic.signals) == {"X"}


def test_missing_pymavlink_has_actionable_error(monkeypatch: pytest.MonkeyPatch):
    def missing(_name: str):
        raise ModuleNotFoundError("No module named 'pymavlink'", name="pymavlink")

    monkeypatch.setattr(ardupilot.importlib, "import_module", missing)

    with pytest.raises(RuntimeError, match="pip install pymavlink"):
        ardupilot._require_pymavlink_component("DFReader")

