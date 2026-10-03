from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from engines.io_engine import (  # noqa: E402
    is_supported_log_source,
    log_file_dialog_filter,
    supported_source_summary,
)
from readers.registry import create_default_registry  # noqa: E402
from readers.release_policy import (  # noqa: E402
    FORMAT_POLICY_ENV,
    MULTIFORMAT_STABLE,
    MULTIFORMAT_PREVIEW,
    ULG_STABLE,
    resolve_format_release_policy,
)


def test_release_policy_defaults_to_multiformat(monkeypatch):
    monkeypatch.delenv(FORMAT_POLICY_ENV, raising=False)

    registry = create_default_registry()

    assert registry.release_policy.name == MULTIFORMAT_STABLE
    assert set(registry.reader_ids) == {"px4_ulog", "tabular", "rosbag", "ardupilot"}
    assert supported_source_summary(registry) == "ULG / ROS / ArduPilot / CSV·JSON"
    assert {".ulg", ".bag", ".db3", ".mcap", ".bin", ".tlog", ".csv", ".json"} <= registry.extensions


def test_ulg_only_policy_remains_available_as_explicit_opt_down(monkeypatch):
    monkeypatch.setenv(FORMAT_POLICY_ENV, ULG_STABLE)

    registry = create_default_registry()

    assert registry.release_policy.name == ULG_STABLE
    assert set(registry.reader_ids) == {"px4_ulog"}
    assert registry.extensions == frozenset({".ulg"})
    assert supported_source_summary(registry) == "ULG"
    assert log_file_dialog_filter(registry) == "Supported Log Files (*.ulg);;All Files (*)"


def test_multiformat_preview_value_remains_backward_compatible(monkeypatch):
    monkeypatch.setenv(FORMAT_POLICY_ENV, MULTIFORMAT_PREVIEW)

    registry = create_default_registry()

    assert registry.release_policy.name == MULTIFORMAT_PREVIEW
    assert set(registry.reader_ids) == {"px4_ulog", "tabular", "rosbag", "ardupilot"}
    assert supported_source_summary(registry) == "ULG / ROS / ArduPilot / CSV·JSON"
    assert {".bag", ".db3", ".mcap", ".bin", ".tlog", ".csv", ".json"} <= registry.extensions


def test_unknown_policy_value_fails_closed():
    policy = resolve_format_release_policy("typo-enable-everything")
    registry = create_default_registry(policy=policy)

    assert policy.name == ULG_STABLE
    assert set(registry.reader_ids) == {"px4_ulog"}


def test_ros2_directory_front_door_obeys_release_policy(tmp_path):
    bag_dir = tmp_path / "ros2_bag"
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n", encoding="utf-8")
    stable = create_default_registry(policy=resolve_format_release_policy(ULG_STABLE))
    preview = create_default_registry(policy=resolve_format_release_policy(MULTIFORMAT_PREVIEW))

    assert not is_supported_log_source(str(bag_dir), stable)
    assert is_supported_log_source(str(bag_dir), preview)
