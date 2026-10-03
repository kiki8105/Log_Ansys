"""ROS1/ROS2 bag을 공통 로그 모델로 변환하는 reader."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
import sqlite3
from typing import Any

import polars as pl

from core.log_model import LogDataset, Signal, TopicInstance
from readers.base import BaseLogReader, LoadResult, LogPath, ProbeResult


ROS1_MAGIC = b"#ROSBAG V2.0\n"
MCAP_MAGIC = b"\x89MCAP0\r\n"
DEFAULT_MAX_ARRAY_ELEMENTS = 64
_TIMESTAMP_COLUMNS = frozenset({"source_timestamp_ns", "timestamp_sec"})


def _format_hint(path: Path) -> str:
    """탐지 실패 때도 registry 계약을 만족할 형식 힌트를 반환한다."""

    return "ros1_bag" if path.suffix.lower() == ".bag" else "ros2_bag"


def _read_prefix(path: Path, size: int) -> bytes:
    try:
        with path.open("rb") as stream:
            return stream.read(size)
    except OSError:
        return b""


def _has_mcap_footer(path: Path) -> bool:
    try:
        if path.stat().st_size < len(MCAP_MAGIC) * 2:
            return False
        with path.open("rb") as stream:
            stream.seek(-len(MCAP_MAGIC), 2)
            return stream.read(len(MCAP_MAGIC)) == MCAP_MAGIC
    except OSError:
        return False


def _probe_sqlite_rosbag2(path: Path) -> tuple[int, str] | None:
    """SQLite 파일이면 ROS2 핵심 테이블/열을 읽기 전용으로 확인한다."""

    if _read_prefix(path, 16) != b"SQLite format 3\x00":
        return None

    try:
        # WHY: 일반 connect는 손상되거나 빈 경로에 DB를 만들 수 있으므로 probe는 반드시 읽기 전용이다.
        uri = f"{path.resolve().as_uri()}?mode=ro"
        # sqlite3.Connection의 컨텍스트 관리자는 트랜잭션만 끝내므로 Windows 파일 잠금 해제를 위해 closing도 쓴다.
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if not {"messages", "topics"}.issubset(tables):
                return 20, "SQLite 파일이지만 ROS2 messages/topics 테이블이 없습니다."

            topic_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(topics)")
            }
            message_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(messages)")
            }
    except sqlite3.Error as exc:
        return 10, f"SQLite 헤더는 맞지만 스키마를 읽지 못했습니다: {exc}"

    expected_topics = {"id", "name", "type"}
    expected_messages = {"topic_id", "timestamp", "data"}
    if expected_topics.issubset(topic_columns) and expected_messages.issubset(message_columns):
        return 100, "ROS2 rosbag2 SQLite 스키마를 확인했습니다."
    return 45, "ROS2 테이블 이름은 있으나 필수 열 일부가 없습니다."


def _probe_metadata_file(path: Path) -> tuple[int, str]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        return 0, f"metadata.yaml을 읽지 못했습니다: {exc}"

    try:
        import yaml
    except ImportError:
        # WHY: probe 자체는 선택 의존성 없이 동작해야 파일 선택 UI에서 ROS2 폴더를 찾을 수 있다.
        has_root = "rosbag2_bagfile_information:" in text
        has_storage = "storage_identifier:" in text
        if has_root and has_storage:
            return 90, "ROS2 metadata.yaml의 필수 키를 확인했습니다."
        return 0, "ROS2 metadata.yaml 필수 키가 없습니다."

    try:
        document = yaml.safe_load(text)
    except Exception as exc:  # PyYAML 예외 타입을 선택 의존성 밖으로 노출하지 않는다.
        return 0, f"metadata.yaml 구문을 해석하지 못했습니다: {exc}"

    if not isinstance(document, Mapping):
        return 0, "metadata.yaml 최상위 구조가 매핑이 아닙니다."
    info = document.get("rosbag2_bagfile_information")
    if not isinstance(info, Mapping):
        return 0, "rosbag2_bagfile_information 항목이 없습니다."
    storage = str(info.get("storage_identifier", "")).strip().lower()
    paths = info.get("relative_file_paths")
    if storage in {"sqlite3", "mcap"} and isinstance(paths, list):
        return 100, f"ROS2 metadata.yaml({storage}) 구조를 확인했습니다."
    if storage in {"sqlite3", "mcap"}:
        return 85, f"ROS2 metadata.yaml({storage})을 확인했습니다."
    return 55, "ROS2 metadata 구조는 맞지만 저장소 형식을 확인하지 못했습니다."


def _iter_object_fields(value: Any) -> list[tuple[str, Any]]:
    if is_dataclass(value) and not isinstance(value, type):
        return [(field.name, getattr(value, field.name)) for field in fields(value)]
    if hasattr(value, "_asdict"):
        try:
            return list(value._asdict().items())
        except (AttributeError, TypeError, ValueError):
            pass
    try:
        attributes = vars(value)
    except TypeError:
        attributes = None
    if attributes:
        return [
            (str(name), item)
            for name, item in attributes.items()
            if not str(name).startswith("_")
        ]
    slots = getattr(type(value), "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    return [
        (str(name), getattr(value, name))
        for name in slots
        if not str(name).startswith("_") and hasattr(value, name)
    ]


def _array_size(value: Any) -> int | None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    size = getattr(value, "size", None)
    if size is not None and not callable(size):
        try:
            return int(size)
        except (TypeError, ValueError, OverflowError):
            pass
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value)
    return None


def _iter_array(value: Any) -> Any:
    if isinstance(value, memoryview):
        return value.tolist()
    flat = getattr(value, "flat", None)
    if flat is not None:
        return flat
    return value


def _normalise_scalar(value: Any) -> bool | int | float | str | None:
    if isinstance(value, Enum):
        return _normalise_scalar(value.value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
        except (TypeError, ValueError):
            converted = value
        if converted is not value:
            return _normalise_scalar(converted)
    return str(value)


def _flatten_message(
    message: Any,
    *,
    max_array_elements: int = DEFAULT_MAX_ARRAY_ELEMENTS,
) -> dict[str, bool | int | float | str | None]:
    """ROS 메시지의 중첩 필드를 점 표기법의 scalar 열로 펼친다."""

    if max_array_elements < 0:
        raise ValueError("max_array_elements must be non-negative")

    flattened: dict[str, bool | int | float | str | None] = {}
    active_ids: set[int] = set()

    def visit(value: Any, prefix: str, depth: int) -> None:
        if value is None or isinstance(value, (bool, int, float, str, Enum)):
            if prefix:
                flattened[prefix] = _normalise_scalar(value)
            return

        item = getattr(value, "item", None)
        if callable(item) and not isinstance(value, (bytes, bytearray, memoryview)):
            try:
                scalar = item()
            except (TypeError, ValueError):
                scalar = value
            if scalar is not value and not isinstance(scalar, (list, tuple, dict)):
                if prefix:
                    flattened[prefix] = _normalise_scalar(scalar)
                return

        if depth >= 16:
            if prefix:
                flattened[f"{prefix}.omitted"] = "maximum nesting depth"
            return

        array_size = _array_size(value)
        if array_size is not None:
            if array_size > max_array_elements:
                # WHY: 이미지/PointCloud payload를 열 수천 개로 만들면 메모리와 UI가 동시에 고갈된다.
                if prefix:
                    flattened[f"{prefix}.length"] = int(array_size)
                return
            identity = id(value)
            if identity in active_ids:
                return
            active_ids.add(identity)
            try:
                for index, item_value in enumerate(_iter_array(value)):
                    child = f"{prefix}.{index}" if prefix else str(index)
                    visit(item_value, child, depth + 1)
            finally:
                active_ids.remove(identity)
            return

        if isinstance(value, Mapping):
            identity = id(value)
            if identity in active_ids:
                return
            active_ids.add(identity)
            try:
                for name, item_value in value.items():
                    child = f"{prefix}.{name}" if prefix else str(name)
                    visit(item_value, child, depth + 1)
            finally:
                active_ids.remove(identity)
            return

        object_fields = _iter_object_fields(value)
        if object_fields:
            identity = id(value)
            if identity in active_ids:
                return
            active_ids.add(identity)
            try:
                for name, item_value in object_fields:
                    child = f"{prefix}.{name}" if prefix else name
                    visit(item_value, child, depth + 1)
            finally:
                active_ids.remove(identity)
            return

        if prefix:
            flattened[prefix] = _normalise_scalar(value)

    visit(message, "", 0)
    return flattened


def _normalise_topic_name(topic_name: str) -> str:
    normalised = topic_name.strip().strip("/").replace("/", ".")
    return normalised or "root"


def _import_any_reader() -> type[Any]:
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RuntimeError(
            "ROS 로그를 읽으려면 선택 의존성 'rosbags>=0.11,<0.12'가 필요합니다. "
            "rosbags 0.11.x를 설치한 뒤 다시 시도하세요."
        ) from exc
    return AnyReader


def _default_typestore_kwargs(format_id: str) -> dict[str, Any]:
    if format_id != "ros2_bag":
        return {}
    try:
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        return {}
    return {"default_typestore": get_typestore(Stores.LATEST)}


class RosbagReader(BaseLogReader):
    """`rosbags` AnyReader 기반 ROS1/ROS2 로그 reader."""

    id = "rosbag"
    version = "1.0"
    extensions = frozenset({".bag", ".db3", ".mcap", ".yaml"})

    def __init__(self, *, max_array_elements: int = DEFAULT_MAX_ARRAY_ELEMENTS) -> None:
        if max_array_elements < 0:
            raise ValueError("max_array_elements must be non-negative")
        self.max_array_elements = max_array_elements

    def probe(self, path: LogPath) -> ProbeResult:
        source = Path(path)
        format_id = _format_hint(source)
        if not source.exists():
            return ProbeResult(0, format_id, "경로가 존재하지 않습니다.")

        metadata_path: Path | None = None
        if source.is_dir():
            metadata_path = source / "metadata.yaml"
            format_id = "ros2_bag"
        elif source.name.lower() == "metadata.yaml":
            metadata_path = source
            format_id = "ros2_bag"

        if metadata_path is not None:
            if not metadata_path.is_file():
                return ProbeResult(0, format_id, "ROS2 metadata.yaml이 없습니다.")
            confidence, reason = _probe_metadata_file(metadata_path)
            return ProbeResult(confidence, format_id, reason)

        if not source.is_file():
            return ProbeResult(0, format_id, "지원하는 일반 파일이 아닙니다.")

        prefix = _read_prefix(source, max(len(ROS1_MAGIC), 16))
        if prefix.startswith(ROS1_MAGIC):
            return ProbeResult(100, "ros1_bag", "ROS1 bag 매직 헤더를 확인했습니다.")

        sqlite_probe = _probe_sqlite_rosbag2(source)
        if sqlite_probe is not None:
            confidence, reason = sqlite_probe
            return ProbeResult(confidence, "ros2_bag", reason)

        if prefix.startswith(MCAP_MAGIC):
            if _has_mcap_footer(source):
                return ProbeResult(100, "ros2_bag", "MCAP 시작/종료 매직을 확인했습니다.")
            return ProbeResult(70, "ros2_bag", "MCAP 시작 매직은 있으나 종료 매직이 없습니다.")

        return ProbeResult(0, format_id, "지원하는 ROS bag 매직 또는 스키마가 없습니다.")

    def load(self, path: LogPath) -> LoadResult:
        source = Path(path)
        probe = self.probe(source)
        if probe.confidence < 70:
            raise ValueError(f"지원하는 ROS bag으로 확인되지 않았습니다: {probe.reason}")

        reader_source = source.parent if source.name.lower() == "metadata.yaml" else source
        AnyReader = _import_any_reader()
        warnings: list[str] = []
        rows_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
        types_by_topic: dict[str, set[str]] = defaultdict(set)
        failed_connections: set[int] = set()
        metadata: dict[str, Any] = {
            "reader": self.id,
            "reader_version": self.version,
            "original_topics": {},
        }

        try:
            reader_options = _default_typestore_kwargs(probe.format_id)
            with AnyReader([reader_source], **reader_options) as reader:
                message_count = int(reader.message_count)
                start_time_ns = int(reader.start_time) if message_count else None
                end_time_ns = int(reader.end_time) if message_count else None
                metadata.update(
                    {
                        "message_count": message_count,
                        "start_time_ns": start_time_ns,
                        "end_time_ns": end_time_ns,
                        "duration_ns": int(reader.duration) if message_count else 0,
                    }
                )

                for connection in reader.connections:
                    topic_name = str(connection.topic)
                    types_by_topic[topic_name].add(str(connection.msgtype))

                try:
                    iterator = reader.messages()
                    for connection, timestamp_ns, rawdata in iterator:
                        connection_key = id(connection)
                        if connection_key in failed_connections:
                            continue
                        topic_name = str(connection.topic)
                        try:
                            message = reader.deserialize(rawdata, connection.msgtype)
                        except Exception as exc:
                            # WHY: 한 사용자 정의 타입 실패가 다른 정상 토픽까지 막지 않도록 연결 단위로 격리한다.
                            failed_connections.add(connection_key)
                            warnings.append(
                                f"토픽 '{topic_name}'({connection.msgtype})을 해석하지 못해 "
                                f"건너뜁니다: {type(exc).__name__}: {exc}"
                            )
                            continue

                        flattened = _flatten_message(
                            message,
                            max_array_elements=self.max_array_elements,
                        )
                        for reserved_name in _TIMESTAMP_COLUMNS.intersection(flattened):
                            flattened[f"message.{reserved_name}"] = flattened.pop(reserved_name)

                        row: dict[str, Any] = {
                            "source_timestamp_ns": int(timestamp_ns),
                            # WHY: 서로 다른 epoch/시뮬레이션 시계를 동일한 플롯 축에서 다룰 수 있게 bag 시작점을 0초로 둔다.
                            "timestamp_sec": (
                                (int(timestamp_ns) - start_time_ns) / 1_000_000_000.0
                                if start_time_ns is not None
                                else 0.0
                            ),
                        }
                        row.update(flattened)
                        rows_by_topic[topic_name].append(row)
                except Exception as exc:
                    warnings.append(
                        "ROS bag 메시지 순회가 중단되어 읽은 구간까지만 적재했습니다: "
                        f"{type(exc).__name__}: {exc}"
                    )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"ROS bag을 열지 못했습니다('{reader_source}'): {type(exc).__name__}: {exc}"
            ) from exc

        dataset = LogDataset()
        base_name_counts: dict[str, int] = defaultdict(int)
        topic_metadata: dict[str, dict[str, Any]] = {}
        loaded_message_count = 0

        for original_name, rows in rows_by_topic.items():
            if not rows:
                continue
            base_name = _normalise_topic_name(original_name)
            instance_id = base_name_counts[base_name]
            base_name_counts[base_name] += 1
            try:
                dataframe = pl.DataFrame(rows, infer_schema_length=None, strict=False)
            except Exception as exc:
                warnings.append(
                    f"토픽 '{original_name}'의 표를 만들지 못해 건너뜁니다: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            topic = TopicInstance(
                base_name=base_name,
                instance_id=instance_id,
                dataframe=dataframe,
            )
            for column_name, dtype in dataframe.schema.items():
                if column_name not in _TIMESTAMP_COLUMNS and dtype.is_numeric():
                    topic.signals[column_name] = Signal(
                        name=column_name,
                        data=dataframe[column_name],
                    )
            dataset.add_topic(topic)
            loaded_message_count += dataframe.height
            topic_metadata[topic.unique_name] = {
                "original_name": original_name,
                "message_types": sorted(types_by_topic[original_name]),
                "message_count": dataframe.height,
            }

        metadata["loaded_message_count"] = loaded_message_count
        metadata["topics"] = topic_metadata
        metadata["original_topics"] = {
            unique_name: details["original_name"]
            for unique_name, details in topic_metadata.items()
        }

        ros_version = "ros1" if probe.format_id == "ros1_bag" else "ros2"
        return LoadResult(
            dataset=dataset,
            format_id=probe.format_id,
            metadata=metadata,
            parameters={},
            messages=[],
            capabilities={"ros", ros_version, "timeseries"},
            warnings=warnings,
            source_path=str(source),
        )


# 외부 플러그인이 두 표기를 모두 사용해도 같은 구현을 보도록 호환 별칭을 유지한다.
ROSBagReader = RosbagReader


__all__ = [
    "DEFAULT_MAX_ARRAY_ELEMENTS",
    "ROSBagReader",
    "RosbagReader",
    "_flatten_message",
]
