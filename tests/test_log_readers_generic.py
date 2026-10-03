from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.log_model import LogDataset  # noqa: E402
from readers.base import BaseLogReader, LoadResult, ProbeResult  # noqa: E402
from readers.registry import (  # noqa: E402
    AmbiguousLogFormatError,
    DuplicateReaderError,
    ReaderRegistry,
    UnsupportedLogError,
    create_default_registry,
)
from readers.tabular import TabularReader  # noqa: E402
from readers.ulog import ULogReader  # noqa: E402


def test_csv_load_normalizes_microseconds_and_only_registers_numeric_signals(tmp_path):
    path = tmp_path / "flight.csv"
    path.write_text(
        "timestamp_us,roll,state,armed\n"
        "2500000,1.5,ready,true\n"
        "2750000,2.5,flying,false\n",
        encoding="utf-8",
    )

    result = TabularReader().load(path)
    topic = result.dataset.get_topic("tabular", 0)

    assert topic.dataframe["timestamp_sec"].to_list() == pytest.approx([0.0, 0.25])
    assert set(topic.signals) == {"roll"}
    assert result.format_id == "tabular"
    assert result.capabilities == {"tabular", "timeseries"}
    assert result.dataset.source_format == "tabular"
    assert result.dataset.source_path == str(path)


@pytest.mark.parametrize("suffix", [".jsonl", ".ndjson"])
def test_json_lines_timestamp_units(tmp_path, suffix):
    path = tmp_path / f"events{suffix}"
    path.write_text(
        '{"timestamp_ms": 1000, "value": 4}\n'
        '{"timestamp_ms": 1250, "value": 7}\n',
        encoding="utf-8",
    )

    result = TabularReader().load(path)
    topic = result.dataset.get_topic("tabular", 0)
    assert topic.dataframe["timestamp_sec"].to_list() == pytest.approx([0.0, 0.25])
    assert set(topic.signals) == {"value"}


def test_missing_timestamp_uses_row_number_and_warning(tmp_path):
    path = tmp_path / "plain.tsv"
    path.write_text("value\tname\n3\ta\n5\tb\n", encoding="utf-8")

    result = TabularReader().load(path)
    topic = result.dataset.get_topic("tabular", 0)
    assert topic.dataframe["timestamp_sec"].to_list() == [0.0, 1.0]
    assert any("행 번호" in warning for warning in result.warnings)


class _FixedReader(BaseLogReader):
    extensions = frozenset({".dummy"})

    def __init__(self, reader_id: str, confidence: int):
        self.id = reader_id
        self.confidence = confidence

    def probe(self, path):
        return ProbeResult(self.confidence, self.id, "fixed")

    def load(self, path):
        return LoadResult(LogDataset(), self.id, source_path=str(path))


class _CountingReader(_FixedReader):
    def __init__(self, reader_id: str, confidence: int, *, definitive: bool = False):
        super().__init__(reader_id, confidence)
        self.definitive_probe = definitive
        self.probe_count = 0

    def probe(self, path):
        self.probe_count += 1
        return super().probe(path)


def test_registry_selects_highest_confidence_and_detect_returns_probe(tmp_path):
    path = tmp_path / "sample.dummy"
    path.write_text("data", encoding="utf-8")
    registry = ReaderRegistry()
    low = registry.register(_FixedReader("low", 20))
    high = registry.register(_FixedReader("high", 90))

    reader, probe = registry.detect(path)
    assert reader is high
    assert probe.confidence == 90
    assert registry.select(path) is high
    assert registry.load(path).format_id == "high"
    assert low is not high


def test_registry_short_circuits_exclusive_magic_match(tmp_path):
    path = tmp_path / "sample.dummy"
    path.write_text("data", encoding="utf-8")
    registry = ReaderRegistry()
    magic = registry.register(_CountingReader("magic", 100, definitive=True))
    expensive = registry.register(_CountingReader("expensive", 100))

    assert registry.select(path) is magic
    assert magic.probe_count == 1
    assert expensive.probe_count == 0

    # Diagnostic probe_all deliberately retains exhaustive/tie visibility.
    probes = registry.probe_all(path)
    assert len(probes) == 2
    assert expensive.probe_count == 1


def test_registry_rejects_ties_duplicates_and_unsupported(tmp_path):
    path = tmp_path / "sample.dummy"
    path.write_text("data", encoding="utf-8")
    registry = ReaderRegistry()
    registry.register(_FixedReader("one", 50))
    registry.register(_FixedReader("two", 50))

    with pytest.raises(AmbiguousLogFormatError):
        registry.detect(path)
    with pytest.raises(DuplicateReaderError):
        registry.register(_FixedReader("one", 99))

    empty = ReaderRegistry()
    empty.register(_FixedReader("none", 0))
    with pytest.raises(UnsupportedLogError):
        empty.detect(path)


def test_default_registry_core_readers_and_extensions():
    registry = create_default_registry(include_optional=False)
    assert set(registry.reader_ids) == {"px4_ulog", "tabular"}
    assert {".ulg", ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".txt"} <= registry.extensions


def test_ulog_reader_wraps_existing_parser(tmp_path, monkeypatch):
    from engines import parser as parser_module

    path = tmp_path / "sample.ulg"
    path.write_bytes(b"ULog\x01\x12\x35\x01")
    expected = LogDataset()
    seen: list[str] = []

    def fake_parse(self, source_path):
        seen.append(source_path)
        return expected

    monkeypatch.setattr(parser_module.ULGParser, "parse", fake_parse)
    reader = ULogReader()
    assert reader.probe(path).confidence == 100

    result = reader.load(path)
    assert seen == [str(path)]
    assert result.dataset is expected
    assert result.format_id == "px4_ulog"
    assert result.capabilities == {"px4", "timeseries"}
    assert expected.source_format == "px4_ulog"


def test_probe_result_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        ProbeResult(101, "bad", "too high")
