"""ArduPilot DataFlash 및 MAVLink telemetry 로그 reader."""

from __future__ import annotations

import importlib
import math
import numbers
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import polars as pl

from core.log_model import LogDataset, Signal, TopicInstance
from readers.base import BaseLogReader, LoadResult, LogPath, ProbeResult


DATAFLASH_FORMAT_ID = "ardupilot_dataflash"
MAVLINK_TLOG_FORMAT_ID = "mavlink_tlog"

_PROBE_LIMIT = 256 * 1024
_DATAFLASH_SYNC = b"\xA3\x95"
_DATAFLASH_FMT_HEADER = b"\xA3\x95\x80"
_MAVLINK_MAGICS = frozenset((0x55, 0xFE, 0xFD))
_DATAFLASH_DEFINITION_TYPES = frozenset(("FMT", "FMTU", "UNIT", "MULT"))
_DATAFLASH_SPECIAL_TYPES = frozenset(("PARM", "MSG", "ERR", "EV", "MODE"))

# 이름이 비슷한 시각 필드가 함께 있는 로그가 있으므로 순서 자체가 형식 계약이다.
_TIMESTAMP_SPECS: tuple[tuple[str, int], ...] = (
    ("TimeUS", 1_000),
    ("TimeMS", 1_000_000),
    ("time_boot_ms", 1_000_000),
    ("time_unix_usec", 1_000),
    ("time_usec", 1_000),
    ("time_us", 1_000),
    ("usec", 1_000),
    ("timestamp_us", 1_000),
    ("timestamp_ms", 1_000_000),
    ("TimeS", 1_000_000_000),
    ("time_s", 1_000_000_000),
    ("timestamp_sec", 1_000_000_000),
)
_TIMESTAMP_FIELD_NAMES = frozenset(name for name, _scale in _TIMESTAMP_SPECS) | {
    "timestamp",
    "source_timestamp_ns",
}
_TRANSPORT_FIELD_NAMES = frozenset(
    {
        "mavpackettype",
        "srcSystem",
        "srcComponent",
        "sysid",
        "compid",
    }
)
_TEXT_FMT_RE = re.compile(
    rb"(?mi)^[ \t]*FMT[ \t]*,[ \t]*\d+[ \t]*,[ \t]*\d+[ \t]*,"
    rb"[ \t]*[A-Za-z0-9_]{1,16}[ \t]*,[ \t]*[A-Za-z0-9_]+[ \t]*,"
)


def _clean_text(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).replace("\x00", "").strip()


def _plain_value(value: Any) -> Any:
    """pymavlink/numpy 값을 metadata에 안전한 기본형으로 줄인다."""
    if isinstance(value, bytes):
        return _clean_text(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            result = item()
        except (TypeError, ValueError):
            pass
        else:
            if isinstance(result, (str, int, float, bool)) or result is None:
                return result
    if isinstance(value, (list, tuple)):
        return [_plain_value(item_value) for item_value in value]
    return _clean_text(value)


def _numeric_scalar(value: Any) -> int | float | None:
    """배열/문자열이 signal 열로 스며들지 않게 실제 숫자 scalar만 받는다."""
    if isinstance(value, bool):
        return None
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, numbers.Number):
        try:
            value = item()
        except (TypeError, ValueError):
            return None
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    return float(value)


def _coerce_timestamp_ns(value: Any, scale: int) -> int | None:
    numeric = _numeric_scalar(value)
    if numeric is None:
        return None
    try:
        numeric_float = float(numeric)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric_float) or numeric_float < 0:
        return None
    return int(round(numeric_float * scale))


def _ambiguous_timestamp_ns(value: Any) -> int | None:
    """단위 표기가 없는 timestamp는 크기로만 최소한의 보정을 한다."""
    numeric = _numeric_scalar(value)
    if numeric is None:
        return None
    value_float = float(numeric)
    if not math.isfinite(value_float) or value_float < 0:
        return None
    if value_float >= 1e17:  # 현재 epoch nanoseconds 범위
        return int(round(value_float))
    if value_float >= 1e11:  # epoch microseconds 또는 긴 부팅 microseconds
        return int(round(value_float * 1_000))
    return int(round(value_float * 1_000_000_000))


def _extract_timestamp_ns(
    fields: Mapping[str, Any],
    message: Any | None = None,
    *,
    prefer_message_timestamp: bool = False,
) -> int | None:
    """원본 시각을 ns로 보존하되 DataFlash의 공식 필드 우선순위를 지킨다."""

    def message_timestamp() -> int | None:
        return _coerce_timestamp_ns(getattr(message, "_timestamp", None), 1_000_000_000)

    if prefer_message_timestamp:
        packet_timestamp = message_timestamp()
        if packet_timestamp is not None:
            return packet_timestamp

    for field_name, scale in _TIMESTAMP_SPECS:
        if field_name in fields:
            result = _coerce_timestamp_ns(fields[field_name], scale)
            if result is not None:
                return result

    if "timestamp" in fields:
        result = _ambiguous_timestamp_ns(fields["timestamp"])
        if result is not None:
            return result

    return message_timestamp()


def _message_type(message: Any, fields: Mapping[str, Any] | None = None) -> str:
    getter = getattr(message, "get_type", None)
    if callable(getter):
        try:
            result = _clean_text(getter())
        except Exception:
            result = ""
        if result:
            return result
    if fields:
        result = _clean_text(fields.get("mavpackettype", ""))
        if result:
            return result
    return type(message).__name__


def _message_fields(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        return {str(key): value for key, value in message.items()}

    converter = getattr(message, "to_dict", None)
    if callable(converter):
        try:
            converted = converter()
        except Exception:
            converted = None
        if isinstance(converted, Mapping):
            return {str(key): value for key, value in converted.items()}

    names: Iterable[Any] = getattr(message, "_fieldnames", ()) or ()
    if not names:
        getter = getattr(message, "get_fieldnames", None)
        if callable(getter):
            try:
                names = getter() or ()
            except Exception:
                names = ()
    result: dict[str, Any] = {}
    for raw_name in names:
        name = str(raw_name)
        try:
            result[name] = getattr(message, name)
        except Exception:
            continue
    return result


def _dataflash_instance_id(fields: Mapping[str, Any]) -> int:
    for field_name in ("Instance", "instance", "Inst", "I"):
        value = fields.get(field_name)
        # 짧은 I는 PID 적분항에도 쓰이므로 작은 정수일 때만 센서 instance로 본다.
        if isinstance(value, numbers.Integral) and not isinstance(value, bool):
            candidate = int(value)
            if 0 <= candidate <= 31:
                return candidate
    return 0


def _source_id(message: Any, method_name: str, fields: Mapping[str, Any], fallback: str) -> int:
    getter = getattr(message, method_name, None)
    value: Any = None
    if callable(getter):
        try:
            value = getter()
        except Exception:
            value = None
    if value is None:
        value = fields.get(fallback, 0)
    try:
        return max(0, min(255, int(value)))
    except (TypeError, ValueError, OverflowError):
        return 0


def _mavlink_instance_id(system_id: int, component_id: int) -> int:
    """두 source id를 충돌 없이 TopicInstance의 정수 id 하나로 접는다."""
    return ((int(system_id) & 0xFF) << 8) | (int(component_id) & 0xFF)


def _numeric_row(fields: Mapping[str, Any], source_timestamp_ns: int | None) -> dict[str, Any]:
    row: dict[str, Any] = {"__source_timestamp_ns": source_timestamp_ns}
    for raw_name, raw_value in fields.items():
        name = str(raw_name)
        if (
            not name
            or name.startswith("_")
            or name in _TIMESTAMP_FIELD_NAMES
            or name in _TRANSPORT_FIELD_NAMES
        ):
            continue
        numeric = _numeric_scalar(raw_value)
        if numeric is not None:
            row[name] = numeric
    return row


def _valid_dataflash_fmt_at(data: bytes, offset: int) -> bool:
    # FMT packet은 고정 89 bytes지만 작은 probe fixture도 판별할 수 있게 핵심 헤더만 검증한다.
    if offset < 0 or offset + 9 > len(data):
        return False
    if data[offset : offset + 3] != _DATAFLASH_FMT_HEADER:
        return False
    declared_length = data[offset + 4]
    name = data[offset + 5 : offset + 9].rstrip(b"\x00")
    return 3 <= declared_length <= 255 and bool(name) and all(
        byte == 0x5F or 0x30 <= byte <= 0x39 or 0x41 <= byte <= 0x5A or 0x61 <= byte <= 0x7A
        for byte in name
    )


def _dataflash_binary_evidence(data: bytes) -> tuple[int, str]:
    fmt_offsets: list[int] = []
    search_from = 0
    while len(fmt_offsets) < 16:
        offset = data.find(_DATAFLASH_FMT_HEADER, search_from)
        if offset < 0:
            break
        if _valid_dataflash_fmt_at(data, offset):
            fmt_offsets.append(offset)
        search_from = offset + 1
    if not fmt_offsets:
        return 0, ""
    sync_count = data.count(_DATAFLASH_SYNC)
    confidence = 98 if sync_count >= 2 else 93
    return confidence, f"DataFlash sync와 유효한 FMT 레코드 {len(fmt_offsets)}개 확인"


def _dataflash_text_evidence(data: bytes) -> tuple[int, str]:
    matches = list(_TEXT_FMT_RE.finditer(data))
    if not matches:
        return 0, ""
    confidence = 98 if len(matches) >= 2 else 94
    return confidence, f"DataFlash 텍스트 FMT 선언 {len(matches)}개 확인"


def _mavlink_frame_length(data: bytes, offset: int) -> int | None:
    if offset < 0 or offset + 2 > len(data):
        return None
    magic = data[offset]
    payload_length = data[offset + 1]
    if magic in (0x55, 0xFE):
        frame_length = payload_length + 8
        if offset + frame_length > len(data):
            return None
        # sysid/compid가 모두 0인 구조는 임의 바이너리의 오탐일 가능성이 높다.
        if offset + 5 >= len(data) or (data[offset + 3] == 0 and data[offset + 4] == 0):
            return None
        return frame_length
    if magic == 0xFD:
        if offset + 10 > len(data):
            return None
        signed_length = 13 if data[offset + 2] & 0x01 else 0
        frame_length = payload_length + 12 + signed_length
        if offset + frame_length > len(data):
            return None
        if data[offset + 5] == 0 and data[offset + 6] == 0:
            return None
        return frame_length
    return None


def _plausible_tlog_prefix(prefix: bytes) -> bool:
    if len(prefix) != 8:
        return False
    timestamp_us = struct.unpack(">Q", prefix)[0]
    # tlog 접두 시각은 보통 epoch us지만 boot 기반 기록도 허용한다.
    return 1_000 <= timestamp_us <= 9_000_000_000_000_000_000


def _mavlink_evidence(data: bytes) -> tuple[int, str]:
    frames: list[tuple[int, int, bool]] = []
    offset = 0
    while offset < len(data) and len(frames) < 64:
        if data[offset] not in _MAVLINK_MAGICS:
            offset += 1
            continue
        frame_length = _mavlink_frame_length(data, offset)
        if frame_length is None:
            offset += 1
            continue
        has_prefix = offset >= 8 and _plausible_tlog_prefix(data[offset - 8 : offset])
        frames.append((offset, offset + frame_length, has_prefix))
        offset += frame_length

    if not frames:
        return 0, ""

    linked = 0
    for previous, current in zip(frames, frames[1:]):
        gap = current[0] - previous[1]
        if gap == 0 or (gap == 8 and current[2]):
            linked += 1
    prefixed = sum(1 for _start, _end, has_prefix in frames if has_prefix)

    if len(frames) >= 3 and linked >= 2:
        confidence = 98
    elif len(frames) >= 2 and linked >= 1:
        confidence = 95
    elif prefixed:
        confidence = 86
    elif len(frames) >= 2:
        confidence = 78
    else:
        confidence = 60
    return confidence, f"MAVLink 프레임 {len(frames)}개(연속 {linked}, tlog 시각 접두 {prefixed}) 확인"


def _probe_sample(data: bytes) -> ProbeResult:
    candidates = (
        (*_dataflash_binary_evidence(data), DATAFLASH_FORMAT_ID),
        (*_dataflash_text_evidence(data), DATAFLASH_FORMAT_ID),
        (*_mavlink_evidence(data), MAVLINK_TLOG_FORMAT_ID),
    )
    confidence, reason, format_id = max(candidates, key=lambda item: item[0])
    if confidence <= 0:
        return ProbeResult(0, "ardupilot", "DataFlash 또는 MAVLink 구조를 찾지 못함")
    return ProbeResult(confidence, format_id, reason)


def _require_pymavlink_component(component: str) -> Any:
    try:
        return importlib.import_module(f"pymavlink.{component}")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "ArduPilot 로그를 읽으려면 선택 의존성 'pymavlink'가 필요합니다. "
            "'pip install pymavlink' 후 다시 시도하세요."
        ) from exc


def _close_reader(reader: Any) -> None:
    closer = getattr(reader, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            return
        return
    file_object = getattr(reader, "filehandle", None) or getattr(reader, "f", None)
    closer = getattr(file_object, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass


def _safe_topic_name(raw_name: Any) -> str:
    name = _clean_text(raw_name) or "UNKNOWN"
    return re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_") or "UNKNOWN"


def _special_text(message_type: str, fields: Mapping[str, Any]) -> str:
    if message_type == "MSG":
        for key in ("Message", "message", "Text", "text"):
            if key in fields:
                text = _clean_text(fields[key])
                if text:
                    return text
    preferred: dict[str, tuple[str, ...]] = {
        "ERR": ("Subsys", "ECode"),
        "EV": ("Id", "Event"),
        "MODE": ("Mode", "ModeNum", "Rsn"),
    }
    keys = preferred.get(message_type, tuple(fields))
    parts = [
        f"{key}={_clean_text(fields[key])}"
        for key in keys
        if key in fields and key not in _TIMESTAMP_FIELD_NAMES
    ]
    return ", ".join(parts) or message_type


def _special_record(
    message_type: str,
    fields: Mapping[str, Any],
    source_timestamp_ns: int | None,
) -> dict[str, Any]:
    payload = {
        str(key): _plain_value(value)
        for key, value in fields.items()
        if str(key) not in _TRANSPORT_FIELD_NAMES and not str(key).startswith("_")
    }
    return {
        "type": message_type,
        "level": message_type,
        "text": _special_text(message_type, fields),
        "source_timestamp_ns": source_timestamp_ns,
        "__time_ns": source_timestamp_ns,
        "fields": payload,
    }


def _normalise_records(records: list[dict[str, Any]], origin_ns: int) -> None:
    for record in records:
        basis = record.pop("__time_ns", None)
        record["timestamp_sec"] = (
            (basis - origin_ns) / 1_000_000_000.0 if basis is not None else None
        )
        # 기존 metadata 표가 사용하는 키도 공통 상대 시각으로 유지한다.
        record["timestamp"] = record["timestamp_sec"]


def _rows_to_topic(
    base_name: str,
    instance_id: int,
    rows: list[dict[str, Any]],
    origin_ns: int,
) -> TopicInstance | None:
    normalised_rows: list[dict[str, Any]] = []
    has_source_time = any(row.get("__source_timestamp_ns") is not None for row in rows)
    for row_index, row in enumerate(rows):
        source_timestamp_ns = row.pop("__source_timestamp_ns", None)
        ordered = {
            "timestamp_sec": (
                (source_timestamp_ns - origin_ns) / 1_000_000_000.0
                if source_timestamp_ns is not None
                else (None if has_source_time else float(row_index))
            ),
            "source_timestamp_ns": source_timestamp_ns,
        }
        ordered.update(row)
        normalised_rows.append(ordered)

    if not normalised_rows:
        return None
    dataframe = pl.from_dicts(normalised_rows, infer_schema_length=None, strict=False)
    if "timestamp_sec" in dataframe.columns:
        dataframe = dataframe.sort("timestamp_sec", nulls_last=True, maintain_order=True)
    topic = TopicInstance(base_name=base_name, instance_id=instance_id, dataframe=dataframe)
    topic.signals = {
        column: Signal(name=column, data=dataframe[column])
        for column in dataframe.columns
        if column not in {"timestamp_sec", "source_timestamp_ns"}
        and dataframe.schema[column].is_numeric()
    }
    if not topic.signals:
        return None
    return topic


def _parameter_pair(fields: Mapping[str, Any]) -> tuple[str, Any] | None:
    raw_name = None
    for key in ("Name", "name", "param_id", "ParamId"):
        if key in fields:
            raw_name = fields[key]
            break
    if raw_name is None:
        return None
    name = _clean_text(raw_name)
    if not name:
        return None
    for key in ("Value", "value", "param_value"):
        if key in fields:
            return name, _plain_value(fields[key])
    return None


class ArduPilotReader(BaseLogReader):
    """DataFlash ``.bin/.log`` 및 MAVLink ``.tlog``를 공통 모델로 변환한다."""

    id = "ardupilot"
    version = "1.0"
    extensions = frozenset({".bin", ".log", ".tlog"})

    def probe(self, path: LogPath) -> ProbeResult:
        source = Path(path)
        try:
            if not source.is_file():
                return ProbeResult(0, self.id, "파일이 아니거나 존재하지 않음")
            with source.open("rb") as handle:
                sample = handle.read(_PROBE_LIMIT)
        except OSError as exc:
            return ProbeResult(0, self.id, f"파일을 읽을 수 없음: {exc}")
        return _probe_sample(sample)

    def load(self, path: LogPath) -> LoadResult:
        source = Path(path)
        probe = self.probe(source)
        if probe.confidence <= 0:
            raise ValueError(f"지원되는 ArduPilot 로그 구조가 아닙니다: {source}")
        if probe.format_id == DATAFLASH_FORMAT_ID:
            return self._load_dataflash(source)
        if probe.format_id == MAVLINK_TLOG_FORMAT_ID:
            return self._load_tlog(source)
        raise ValueError(f"알 수 없는 ArduPilot 로그 형식입니다: {probe.format_id}")

    def _load_dataflash(self, source: Path) -> LoadResult:
        dfreader = _require_pymavlink_component("DFReader")
        with source.open("rb") as handle:
            sample = handle.read(_PROBE_LIMIT)
        is_text = _dataflash_text_evidence(sample)[0] > _dataflash_binary_evidence(sample)[0]
        constructor_name = "DFReader_text" if is_text else "DFReader_binary"
        constructor = getattr(dfreader, constructor_name, None)
        if not callable(constructor):
            raise RuntimeError(f"설치된 pymavlink에 {constructor_name}가 없습니다.")

        reader = constructor(str(source))
        topic_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        parameters: dict[str, Any] = {}
        message_counts: Counter[str] = Counter()
        records_by_type: dict[str, list[dict[str, Any]]] = {
            "MSG": [],
            "ERR": [],
            "EV": [],
            "MODE": [],
        }
        messages: list[dict[str, Any]] = []
        all_timestamps: list[int] = []

        try:
            while True:
                message = reader.recv_msg()
                if message is None:
                    break
                fields = _message_fields(message)
                message_type = _safe_topic_name(_message_type(message, fields)).upper()
                message_counts[message_type] += 1
                if message_type in _DATAFLASH_DEFINITION_TYPES:
                    continue
                timestamp_ns = _extract_timestamp_ns(fields, message)
                if timestamp_ns is not None:
                    all_timestamps.append(timestamp_ns)

                if message_type == "PARM":
                    pair = _parameter_pair(fields)
                    if pair is not None:
                        parameters[pair[0]] = pair[1]
                    continue
                if message_type in records_by_type:
                    record = _special_record(message_type, fields, timestamp_ns)
                    records_by_type[message_type].append(record)
                    messages.append(record)
                    continue

                instance_id = _dataflash_instance_id(fields)
                row = _numeric_row(fields, timestamp_ns)
                if len(row) > 1:
                    topic_rows[(message_type, instance_id)].append(row)
        finally:
            _close_reader(reader)

        origin_ns = min(all_timestamps, default=0)
        _normalise_records(messages, origin_ns)
        metadata: dict[str, Any] = {
            "reader": {"id": self.id, "version": self.version},
            "encoding": "text" if is_text else "binary",
            "timestamp_origin_ns": origin_ns,
            "message_counts": dict(sorted(message_counts.items())),
            "text_messages": records_by_type["MSG"],
            "errors": records_by_type["ERR"],
            "events": records_by_type["EV"],
            "mode_changes": records_by_type["MODE"],
        }
        capabilities = {"ardupilot", "timeseries", "parameters", "messages"}
        dataset = LogDataset(
            source_format=DATAFLASH_FORMAT_ID,
            source_path=str(source),
            capabilities=capabilities,
            metadata=metadata,
        )
        for (base_name, instance_id), rows in topic_rows.items():
            topic = _rows_to_topic(base_name, instance_id, rows, origin_ns)
            if topic is not None:
                dataset.add_topic(topic)

        warnings = [] if dataset.topics else ["숫자 시계열 토픽을 찾지 못했습니다."]
        return LoadResult(
            dataset=dataset,
            format_id=DATAFLASH_FORMAT_ID,
            metadata=metadata,
            parameters=parameters,
            messages=messages,
            capabilities=capabilities,
            warnings=warnings,
            source_path=str(source),
        )

    def _load_tlog(self, source: Path) -> LoadResult:
        mavutil = _require_pymavlink_component("mavutil")
        try:
            connection = mavutil.mavlink_connection(
                str(source), notimestamps=False, robust_parsing=True
            )
        except TypeError:
            # 오래된 pymavlink도 읽을 수 있도록 추가 옵션만 제거한다.
            connection = mavutil.mavlink_connection(str(source), notimestamps=False)

        topic_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        topic_sources: dict[str, dict[str, int]] = {}
        parameters: dict[str, Any] = {}
        messages: list[dict[str, Any]] = []
        message_counts: Counter[str] = Counter()
        all_timestamps: list[int] = []

        try:
            while True:
                message = connection.recv_match(blocking=False)
                if message is None:
                    break
                fields = _message_fields(message)
                message_type = _safe_topic_name(_message_type(message, fields)).upper()
                if message_type in {"BAD_DATA", "UNKNOWN"}:
                    continue
                message_counts[message_type] += 1
                system_id = _source_id(message, "get_srcSystem", fields, "sysid")
                component_id = _source_id(message, "get_srcComponent", fields, "compid")
                instance_id = _mavlink_instance_id(system_id, component_id)
                timestamp_ns = _extract_timestamp_ns(
                    fields, message, prefer_message_timestamp=True
                )
                if timestamp_ns is not None:
                    all_timestamps.append(timestamp_ns)

                pair = _parameter_pair(fields) if message_type == "PARAM_VALUE" else None
                if pair is not None:
                    parameters[pair[0]] = pair[1]
                if message_type == "STATUSTEXT":
                    record = _special_record(message_type, fields, timestamp_ns)
                    severity = fields.get("severity")
                    if severity is not None:
                        record["level"] = _plain_value(severity)
                    messages.append(record)

                row = _numeric_row(fields, timestamp_ns)
                if len(row) <= 1:
                    continue
                key = (message_type, instance_id)
                topic_rows[key].append(row)
                unique_name = f"{message_type}_{instance_id}"
                topic_sources[unique_name] = {
                    "srcSystem": system_id,
                    "srcComponent": component_id,
                    "system_id": system_id,
                    "component_id": component_id,
                    "instance_id": instance_id,
                }
        finally:
            _close_reader(connection)

        origin_ns = min(all_timestamps, default=0)
        _normalise_records(messages, origin_ns)
        metadata = {
            "reader": {"id": self.id, "version": self.version},
            "timestamp_origin_ns": origin_ns,
            "message_counts": dict(sorted(message_counts.items())),
            "topic_sources": topic_sources,
            "mavlink_sources": sorted(
                {
                    (source_ids["srcSystem"], source_ids["srcComponent"])
                    for source_ids in topic_sources.values()
                }
            ),
        }
        capabilities = {"ardupilot", "mavlink", "timeseries", "parameters", "messages"}
        dataset = LogDataset(
            source_format=MAVLINK_TLOG_FORMAT_ID,
            source_path=str(source),
            capabilities=capabilities,
            metadata=metadata,
        )
        for (base_name, instance_id), rows in topic_rows.items():
            topic = _rows_to_topic(base_name, instance_id, rows, origin_ns)
            if topic is not None:
                dataset.add_topic(topic)

        warnings = [] if dataset.topics else ["숫자 MAVLink 시계열 토픽을 찾지 못했습니다."]
        return LoadResult(
            dataset=dataset,
            format_id=MAVLINK_TLOG_FORMAT_ID,
            metadata=metadata,
            parameters=parameters,
            messages=messages,
            capabilities=capabilities,
            warnings=warnings,
            source_path=str(source),
        )


__all__ = [
    "ArduPilotReader",
    "DATAFLASH_FORMAT_ID",
    "MAVLINK_TLOG_FORMAT_ID",
]
