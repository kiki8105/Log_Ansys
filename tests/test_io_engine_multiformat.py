from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from engines.io_engine import LogIOEngine, supported_source_extensions  # noqa: E402
from readers.registry import create_default_registry  # noqa: E402
from storage.parquet_cache import ParquetCacheManager  # noqa: E402


def test_io_engine_detects_tabular_and_reuses_v2_cache(tmp_path):
    source = tmp_path / "telemetry.csv"
    source.write_text("time_ms,roll,pitch\n1000,1.0,2.0\n1100,1.5,2.5\n", encoding="utf-8")
    engine = LogIOEngine(
        registry=create_default_registry(include_optional=False),
        cache_mgr=ParquetCacheManager(cache_dir=str(tmp_path / "cache")),
    )

    first = engine.load_result(str(source))
    assert first.format_id == "tabular"
    assert engine.last_load_info["load_type"] == "Full Parsing"
    assert first.dataset.get_topic("tabular", 0).dataframe["timestamp_sec"].to_list() == pytest.approx([0.0, 0.1])
    cold_signals = set(first.dataset.get_topic("tabular", 0).signals)
    assert cold_signals == {"roll", "pitch"}

    second = engine.load_result(str(source))
    assert second.format_id == "tabular"
    assert engine.last_load_info["load_type"] == "Cache Load"
    assert second.dataset.get_topic("tabular", 0).signals["roll"].data.to_list() == [1.0, 1.5]
    assert set(second.dataset.get_topic("tabular", 0).signals) == cold_signals


def test_supported_extensions_cover_requested_log_families():
    extensions = set(supported_source_extensions(create_default_registry(include_optional=True)))
    assert {".ulg", ".bag", ".db3", ".mcap", ".bin", ".log", ".tlog", ".csv", ".json"} <= extensions
