from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from readers.rosbag import (  # noqa: E402
    MCAP_MAGIC,
    ROS1_MAGIC,
    RosbagReader,
    _flatten_message,
)


@dataclass
class _Stamp:
    sec: int
    nanosec: int


@dataclass
class _Header:
    stamp: _Stamp
    frame_id: str


@dataclass
class _Message:
    header: _Header
    enabled: bool
    covariance: list[float]
    payload: bytes


def test_probe_ros1_uses_magic_not_only_extension(tmp_path: Path) -> None:
    path = tmp_path / "recording.dat"
    path.write_bytes(ROS1_MAGIC + b"rest")

    result = RosbagReader().probe(path)

    assert result.confidence == 100
    assert result.format_id == "ros1_bag"


def test_probe_ros2_directory_uses_metadata_structure(tmp_path: Path) -> None:
    bag_dir = tmp_path / "ros2_bag"
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text(
        """rosbag2_bagfile_information:
  version: 9
  storage_identifier: sqlite3
  relative_file_paths:
    - data_0.db3
""",
        encoding="utf-8",
    )

    result = RosbagReader().probe(bag_dir)

    assert result.confidence >= 90
    assert result.format_id == "ros2_bag"


def test_probe_ros2_sqlite_checks_schema(tmp_path: Path) -> None:
    path = tmp_path / "recording.db3"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "CREATE TABLE topics(id INTEGER, name TEXT, type TEXT, serialization_format TEXT)"
        )
        connection.execute(
            "CREATE TABLE messages(id INTEGER, topic_id INTEGER, timestamp INTEGER, data BLOB)"
        )
        connection.commit()

    result = RosbagReader().probe(path)

    assert result.confidence == 100
    assert result.format_id == "ros2_bag"


def test_probe_mcap_checks_start_and_end_magic(tmp_path: Path) -> None:
    path = tmp_path / "recording.mcap"
    path.write_bytes(MCAP_MAGIC + b"payload" + MCAP_MAGIC)

    result = RosbagReader().probe(path)

    assert result.confidence == 100
    assert result.format_id == "ros2_bag"


def test_flatten_nested_scalars_and_summarise_large_payload() -> None:
    message = _Message(
        header=_Header(_Stamp(12, 345), "map"),
        enabled=True,
        covariance=[float(index) for index in range(9)],
        payload=b"x" * 100,
    )

    flattened = _flatten_message(message, max_array_elements=16)

    assert flattened["header.stamp.sec"] == 12
    assert flattened["header.stamp.nanosec"] == 345
    assert flattened["header.frame_id"] == "map"
    assert flattened["enabled"] is True
    assert flattened["covariance.8"] == 8.0
    assert flattened["payload.length"] == 100
    assert "payload.0" not in flattened


def test_load_preserves_receive_timestamps_and_isolates_bad_topic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "recording.bag"
    path.write_bytes(ROS1_MAGIC)
    good = SimpleNamespace(id=1, topic="/imu/data", msgtype="example/Imu")
    bad = SimpleNamespace(id=2, topic="/custom/broken", msgtype="example/Broken")

    class FakeAnyReader:
        def __init__(self, paths, **kwargs):
            assert paths == [path]
            assert kwargs == {}
            self.connections = [good, bad]
            self.message_count = 3
            self.start_time = 1_000_000_000
            self.end_time = 1_500_000_001
            self.duration = self.end_time - self.start_time

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def messages(self):
            yield good, 1_000_000_000, _Message(
                _Header(_Stamp(1, 0), "imu"), True, [1.0], b""
            )
            yield bad, 1_200_000_000, object()
            yield good, 1_500_000_000, _Message(
                _Header(_Stamp(1, 500_000_000), "imu"), False, [2.0], b""
            )

        def deserialize(self, rawdata, msgtype):
            if msgtype == "example/Broken":
                raise LookupError("unknown message definition")
            return rawdata

    monkeypatch.setattr("readers.rosbag._import_any_reader", lambda: FakeAnyReader)

    result = RosbagReader().load(path)

    topic = result.dataset.get_topic("imu.data", 0)
    assert topic is not None
    assert topic.dataframe["source_timestamp_ns"].to_list() == [
        1_000_000_000,
        1_500_000_000,
    ]
    assert topic.dataframe["timestamp_sec"].to_list() == [0.0, 0.5]
    assert "header.stamp.sec" in topic.signals
    assert result.capabilities == {"ros", "ros1", "timeseries"}
    assert len(result.warnings) == 1
    assert "/custom/broken" in result.warnings[0]


@pytest.mark.skipif(
    importlib.util.find_spec("rosbags") is None,
    reason="선택 의존성 rosbags가 설치되지 않았습니다.",
)
def test_rosbags_synthetic_ros1_round_trip(tmp_path: Path) -> None:
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    String = typestore.types["std_msgs/msg/String"]
    bag_path = tmp_path / "synthetic_ros1.bag"

    with Writer(bag_path) as writer:
        connection = writer.add_connection(
            "/chatter",
            String.__msgtype__,
            typestore=typestore,
        )
        message = String(data="hello ros1")
        writer.write(
            connection,
            3_100_000_000,
            typestore.serialize_ros1(message, message.__msgtype__),
        )

    result = RosbagReader().load(bag_path)

    topic = result.dataset.get_topic("chatter", 0)
    assert topic is not None
    assert topic.dataframe["data"].to_list() == ["hello ros1"]
    assert topic.dataframe["source_timestamp_ns"].to_list() == [3_100_000_000]
    assert topic.dataframe["timestamp_sec"].to_list() == [0.0]


@pytest.mark.skipif(
    importlib.util.find_spec("rosbags") is None,
    reason="선택 의존성 rosbags가 설치되지 않았습니다.",
)
def test_rosbags_synthetic_ros2_round_trip(tmp_path: Path) -> None:
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.LATEST)
    String = typestore.types["std_msgs/msg/String"]
    bag_dir = tmp_path / "synthetic_ros2"

    with Writer(bag_dir, version=9) as writer:
        connection = writer.add_connection(
            "/chatter",
            String.__msgtype__,
            typestore=typestore,
        )
        message = String(data="hello")
        writer.write(
            connection,
            4_200_000_000,
            typestore.serialize_cdr(message, message.__msgtype__),
        )

    result = RosbagReader().load(bag_dir)

    topic = result.dataset.get_topic("chatter", 0)
    assert topic is not None
    assert topic.dataframe["data"].to_list() == ["hello"]
    assert topic.dataframe["source_timestamp_ns"].to_list() == [4_200_000_000]
    assert topic.dataframe["timestamp_sec"].to_list() == [0.0]

    standalone_result = RosbagReader().load(next(bag_dir.glob("*.db3")))
    standalone_topic = standalone_result.dataset.get_topic("chatter", 0)
    assert standalone_topic is not None
    assert standalone_topic.dataframe["data"].to_list() == ["hello"]


@pytest.mark.skipif(
    importlib.util.find_spec("rosbags") is None,
    reason="선택 의존성 rosbags가 설치되지 않았습니다.",
)
def test_rosbags_synthetic_standalone_mcap_round_trip(tmp_path: Path) -> None:
    from rosbags.rosbag2 import StoragePlugin, Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.LATEST)
    String = typestore.types["std_msgs/msg/String"]
    bag_dir = tmp_path / "synthetic_mcap"

    with Writer(bag_dir, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        connection = writer.add_connection(
            "/events",
            String.__msgtype__,
            typestore=typestore,
        )
        message = String(data="mcap")
        writer.write(
            connection,
            8_000_000_000,
            typestore.serialize_cdr(message, message.__msgtype__),
        )

    result = RosbagReader().load(next(bag_dir.glob("*.mcap")))

    topic = result.dataset.get_topic("events", 0)
    assert topic is not None
    assert topic.dataframe["data"].to_list() == ["mcap"]
    assert topic.dataframe["source_timestamp_ns"].to_list() == [8_000_000_000]
