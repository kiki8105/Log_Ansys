"""다중 로그 형식 reader 공개 API."""

from __future__ import annotations

from importlib import import_module

from readers.base import BaseLogReader, LoadResult, LogPath, ProbeResult
from readers.registry import (
    AmbiguousFormatError,
    AmbiguousLogFormatError,
    DEFAULT_REGISTRY,
    DuplicateReaderError,
    ReaderProbe,
    ReaderRegistry,
    ReaderRegistryError,
    UnsupportedFormatError,
    UnsupportedLogError,
    create_default_registry,
    default_reader_registry,
    get_default_registry,
)
from readers.tabular import TabularReader
from readers.ulog import ULogReader
from readers.release_policy import (
    FORMAT_POLICY_ENV,
    FormatReleasePolicy,
    MULTIFORMAT_PREVIEW,
    ULG_STABLE,
    resolve_format_release_policy,
)


def __getattr__(name: str):
    # WHY: 선택 형식의 무거운 SDK가 없어도 기본 PX4/표 reader import는 성공해야 합니다.
    optional = {
        "ROSBagReader": ("rosbag", "ROSBagReader"),
        "RosbagReader": ("rosbag", "RosbagReader"),
        "ArduPilotReader": ("ardupilot", "ArduPilotReader"),
    }
    if name not in optional:
        raise AttributeError(name)
    module_name, class_name = optional[name]
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, class_name)
    globals()[name] = value
    return value


__all__ = [
    "AmbiguousFormatError",
    "AmbiguousLogFormatError",
    "ArduPilotReader",
    "BaseLogReader",
    "DEFAULT_REGISTRY",
    "DuplicateReaderError",
    "FORMAT_POLICY_ENV",
    "FormatReleasePolicy",
    "LoadResult",
    "LogPath",
    "MULTIFORMAT_PREVIEW",
    "ProbeResult",
    "ROSBagReader",
    "ReaderProbe",
    "ReaderRegistry",
    "ReaderRegistryError",
    "RosbagReader",
    "TabularReader",
    "ULG_STABLE",
    "ULogReader",
    "UnsupportedFormatError",
    "UnsupportedLogError",
    "create_default_registry",
    "default_reader_registry",
    "get_default_registry",
    "resolve_format_release_policy",
]
