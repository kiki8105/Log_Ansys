from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.log_model import LogDataset, Signal, TopicInstance  # noqa: E402
from readers.base import LoadResult  # noqa: E402
from storage.parquet_cache import ParquetCacheManager  # noqa: E402


def _sample_result(source: Path) -> LoadResult:
    dataframe = pl.DataFrame({
        "timestamp_sec": [0.0, 0.1],
        "source_timestamp_ns": [100, 100_000_100],
        "value": [1.0, 2.0],
    })
    topic = TopicInstance("/imu/data", 0, dataframe=dataframe)
    topic.refresh_signals()
    dataset = LogDataset(
        source_format="test_reader",
        source_path=str(source),
        capabilities={"timeseries"},
    )
    dataset.add_topic(topic)
    return LoadResult(
        dataset=dataset,
        format_id="test_reader",
        metadata={"nested": {"value": 1}},
        parameters={"P": 2},
        messages=[{"level": "INFO", "text": "ok"}],
        capabilities={"timeseries"},
        source_path=str(source),
    )


def test_v2_cache_round_trip_uses_manifest_and_safe_topic_filename(tmp_path):
    source = tmp_path / "source.log"
    source.write_bytes(b"sample source")
    manager = ParquetCacheManager(cache_dir=str(tmp_path / "cache"))
    result = _sample_result(source)

    key = manager.save_result(result, str(source), "test_reader", "1.0")
    restored = manager.load_result(str(source), "test_reader", "1.0")

    assert restored is not None
    assert restored.format_id == "test_reader"
    assert restored.dataset.get_topic("/imu/data", 0).dataframe["value"].to_list() == [1.0, 2.0]
    manifest = tmp_path / "cache" / "v2" / key / "manifest.json"
    assert manifest.is_file()
    assert not any("imu" in path.name for path in manifest.parent.glob("*.parquet"))


def test_cache_key_changes_when_source_content_changes(tmp_path):
    source = tmp_path / "source.log"
    source.write_bytes(b"first")
    manager = ParquetCacheManager(cache_dir=str(tmp_path / "cache"))
    first_key = manager.cache_key(str(source), "test_reader", "1.0")

    source.write_bytes(b"second version")
    second_key = manager.cache_key(str(source), "test_reader", "1.0")

    assert first_key != second_key


def test_cache_restores_exact_numeric_signal_allowlist(tmp_path):
    source = tmp_path / "table.csv"
    source.write_text("time_ms,value,state\n0,1.0,ready\n", encoding="utf-8")
    dataframe = pl.DataFrame({
        "time_ms": [0, 100],
        "timestamp_sec": [0.0, 0.1],
        "source_timestamp_ns": [0, 100_000_000],
        "value": [1.0, 2.0],
        "state": ["ready", "flying"],
    })
    topic = TopicInstance("tabular", 0, dataframe=dataframe)
    # Cold parse deliberately excludes the source time column and text column.
    topic.signals = {"value": Signal("value", dataframe["value"])}
    dataset = LogDataset(source_format="tabular", capabilities={"timeseries"})
    dataset.add_topic(topic)
    result = LoadResult(
        dataset=dataset,
        format_id="tabular",
        capabilities={"timeseries"},
        source_path=str(source),
    )
    manager = ParquetCacheManager(cache_dir=str(tmp_path / "cache"))

    manager.save_result(result, str(source), "tabular", "1.0")
    restored = manager.load_result(str(source), "tabular", "1.0")

    restored_topic = restored.dataset.get_topic("tabular", 0)
    assert set(restored_topic.signals) == {"value"}
    assert "state" in restored_topic.dataframe.columns
    assert "time_ms" in restored_topic.dataframe.columns
