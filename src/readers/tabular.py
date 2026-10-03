"""CSV/JSON 계열 일반 로그 reader."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any

import polars as pl

from core.log_model import LogDataset, Signal, TopicInstance
from readers.base import BaseLogReader, LoadResult, LogPath, ProbeResult


_JSON_SUFFIXES = {".json", ".jsonl", ".ndjson"}
_DELIMITED_SUFFIXES = {".csv", ".tsv", ".txt"}


class TabularReader(BaseLogReader):
    """행 기반 범용 로그를 하나의 ``tabular_0`` topic으로 변환합니다."""

    id = "tabular"
    version = "1.0"
    extensions = frozenset(_JSON_SUFFIXES | _DELIMITED_SUFFIXES)

    def probe(self, path: LogPath) -> ProbeResult:
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix not in self.extensions:
            return ProbeResult(0, self.id, f"지원하지 않는 확장자: {suffix or '(없음)'}")
        if not file_path.is_file():
            return ProbeResult(0, self.id, "파일을 찾을 수 없음")

        try:
            sample = file_path.read_bytes()[:8192]
        except OSError as exc:
            return ProbeResult(0, self.id, f"파일 헤더를 읽을 수 없음: {exc}")
        if not sample.strip():
            return ProbeResult(0, self.id, "빈 파일")

        text = sample.decode("utf-8-sig", errors="replace").lstrip()
        if suffix in _JSON_SUFFIXES:
            looks_json = text.startswith(("{", "["))
            confidence = 95 if looks_json else 65
            reason = "JSON 계열 확장자와 내용 일치" if looks_json else "JSON 계열 확장자"
            return ProbeResult(confidence, self.id, reason)

        delimiter = self._detect_delimiter(text, suffix)
        if suffix == ".txt":
            confidence = 70 if delimiter is not None else 35
        else:
            confidence = 90 if delimiter is not None else 65
        reason = "구분자 기반 표 형식" if delimiter is not None else "표 형식 확장자"
        return ProbeResult(confidence, self.id, reason)

    def load(self, path: LogPath) -> LoadResult:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(file_path)

        dataframe = self._read_dataframe(file_path)
        if dataframe.height == 0:
            raise ValueError(f"표 로그에 데이터 행이 없습니다: {file_path}")

        warnings: list[str] = []
        timestamp_name, timestamp_sec = self._find_timestamp(dataframe)
        if timestamp_sec is None:
            # WHY: 시간 열이 없는 데이터도 기존 시계열 UI에서 열 수 있도록 행 번호를 상대 초로 사용합니다.
            timestamp_sec = pl.Series("timestamp_sec", range(dataframe.height), dtype=pl.Float64)
            warnings.append("시간 열을 찾지 못해 행 번호를 timestamp_sec로 사용했습니다.")
            timestamp_name = None
        dataframe = dataframe.with_columns(
            timestamp_sec.alias("timestamp_sec"),
            (timestamp_sec * 1_000_000_000).round().cast(pl.Int64).alias("source_timestamp_ns"),
        )

        topic = TopicInstance(base_name="tabular", instance_id=0, dataframe=dataframe)
        excluded = {"timestamp_sec", "source_timestamp_ns"}
        if timestamp_name is not None:
            excluded.add(timestamp_name)
        for column_name, dtype in dataframe.schema.items():
            if column_name in excluded or not dtype.is_numeric():
                continue
            topic.signals[column_name] = Signal(column_name, dataframe[column_name])

        if not topic.signals:
            warnings.append("시간 열 이외의 숫자 신호를 찾지 못했습니다.")

        dataset = LogDataset()
        dataset.add_topic(topic)
        metadata = {
            "reader_version": self.version,
            "source_name": file_path.name,
            "row_count": dataframe.height,
            "column_count": dataframe.width,
            "timestamp_source": timestamp_name,
            "topic_name": topic.unique_name,
        }
        return LoadResult(
            dataset=dataset,
            format_id=self.id,
            metadata=metadata,
            capabilities={"tabular", "timeseries"},
            warnings=warnings,
            source_path=str(file_path),
        )

    def _read_dataframe(self, path: Path) -> pl.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".json":
            return self._read_json(path)
        if suffix in {".jsonl", ".ndjson"}:
            return pl.read_ndjson(path, infer_schema_length=10_000)

        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if suffix == ".txt" and text.lstrip().startswith(("{", "[")):
            # WHY: 현장 장비가 JSON Lines에도 .txt를 붙이는 경우가 있어 내용이 명확할 때 확장자보다 우선합니다.
            try:
                return self._read_json(path)
            except (ValueError, json.JSONDecodeError, pl.exceptions.PolarsError):
                return pl.read_ndjson(path, infer_schema_length=10_000)

        separator = self._detect_delimiter(text[:8192], suffix)
        if separator is None:
            separator = "\t" if suffix == ".tsv" else ","
        source: Path | io.StringIO = path
        if separator == " ":
            # WHY: Polars의 CSV separator는 정규식을 받지 않아 가변 공백 TXT를 탭으로 정규화합니다.
            normalized = "\n".join(
                "\t".join(re.split(r"\s+", line.strip()))
                for line in text.splitlines()
                if line.strip()
            )
            source = io.StringIO(normalized)
            separator = "\t"
        return pl.read_csv(
            source,
            separator=separator,
            try_parse_dates=True,
            infer_schema_length=10_000,
            encoding="utf8-lossy",
        )

    @staticmethod
    def _read_json(path: Path) -> pl.DataFrame:
        text = path.read_text(encoding="utf-8-sig")
        if text.lstrip().startswith("{"):
            payload = json.loads(text)
            if isinstance(payload, dict):
                for key in ("records", "rows", "data", "items"):
                    if isinstance(payload.get(key), list):
                        payload = payload[key]
                        break
            if isinstance(payload, dict):
                values = list(payload.values())
                if values and all(isinstance(value, list) for value in values):
                    lengths = {len(value) for value in values}
                    if len(lengths) == 1:
                        return pl.DataFrame(payload)
                # WHY: 단일 JSON 객체도 한 행짜리 로그로 열어 탐색할 수 있게 합니다.
                payload = [payload]
            if not isinstance(payload, list):
                raise ValueError("JSON 최상위 값이 행 배열 또는 객체가 아닙니다.")
            return pl.DataFrame(payload)
        try:
            return pl.read_json(path, infer_schema_length=10_000)
        except pl.exceptions.PolarsError:
            payload = json.loads(text)
            if not isinstance(payload, list):
                raise ValueError("JSON 최상위 값이 행 배열 또는 객체가 아닙니다.")
            return pl.DataFrame(payload)

    @staticmethod
    def _detect_delimiter(sample: str, suffix: str) -> str | None:
        if suffix == ".tsv" and "\t" in sample:
            return "\t"
        lines = [line for line in sample.splitlines() if line.strip()][:20]
        if not lines:
            return None
        try:
            return csv.Sniffer().sniff("\n".join(lines), delimiters=",\t;|").delimiter
        except csv.Error:
            widths = [len(re.split(r"\s+", line.strip())) for line in lines]
            if suffix == ".txt" and widths and min(widths) >= 2 and len(set(widths)) == 1:
                return " "
            return None

    @classmethod
    def _find_timestamp(cls, dataframe: pl.DataFrame) -> tuple[str | None, pl.Series | None]:
        candidates = sorted(
            dataframe.columns,
            key=lambda name: cls._timestamp_priority(name),
            reverse=True,
        )
        for name in candidates:
            if cls._timestamp_priority(name) <= 0:
                break
            converted = cls._timestamp_to_seconds(dataframe[name], name)
            if converted is None:
                continue
            finite_values = [value for value in converted.drop_nulls().to_list() if math.isfinite(value)]
            if not finite_values:
                continue
            origin = finite_values[0]
            return name, (converted - origin).cast(pl.Float64)
        return None, None

    @staticmethod
    def _timestamp_priority(name: str) -> int:
        normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        exact = {
            "timestamp_sec": 120,
            "timestamp_s": 119,
            "time_sec": 118,
            "time_s": 117,
            "elapsed_sec": 116,
            "elapsed_s": 115,
            "timestamp_ns": 114,
            "timestamp_us": 113,
            "timestamp_ms": 112,
            "time_ns": 111,
            "time_us": 110,
            "time_ms": 109,
            "timestamp": 100,
            "time": 95,
            "stamp": 90,
            "datetime": 85,
            "date_time": 84,
            "date": 80,
            "ts": 75,
            "t": 60,
        }
        if normalized in exact:
            return exact[normalized]
        if normalized.endswith(("_timestamp", "_time", "_stamp")):
            return 50
        return 0

    @classmethod
    def _timestamp_to_seconds(cls, series: pl.Series, name: str) -> pl.Series | None:
        dtype = series.dtype
        try:
            if dtype == pl.Date:
                return series.cast(pl.Int32).cast(pl.Float64) * 86_400.0
            if isinstance(dtype, pl.Datetime):
                return series.dt.epoch(time_unit="ns").cast(pl.Float64) / 1e9
            if dtype == pl.Time:
                return series.cast(pl.Int64).cast(pl.Float64) / 1e9
            if dtype == pl.String:
                numeric = series.cast(pl.Float64, strict=False)
                if numeric.len() and numeric.null_count() < numeric.len():
                    return numeric / cls._timestamp_scale(name, numeric)
                parsed = series.str.to_datetime(strict=False)
                return parsed.dt.epoch(time_unit="ns").cast(pl.Float64) / 1e9
            if not dtype.is_numeric():
                return None
            numeric = series.cast(pl.Float64, strict=False)
            scale = cls._timestamp_scale(name, numeric)
            return numeric / scale
        except (TypeError, ValueError, pl.exceptions.PolarsError):
            return None

    @staticmethod
    def _timestamp_scale(name: str, series: pl.Series) -> float:
        normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        tokens = set(normalized.split("_"))
        if tokens & {"ns", "nsec", "nanosecond", "nanoseconds"}:
            return 1e9
        if tokens & {"us", "usec", "microsecond", "microseconds"}:
            return 1e6
        if tokens & {"ms", "msec", "millisecond", "milliseconds"}:
            return 1e3
        if tokens & {"s", "sec", "second", "seconds"}:
            return 1.0

        values = [float(value) for value in series.drop_nulls().to_list() if math.isfinite(float(value))]
        if not values:
            return 1.0
        magnitude = max(abs(value) for value in values)
        if magnitude >= 1e17:
            return 1e9
        if magnitude >= 1e14:
            return 1e6
        if magnitude >= 1e11:
            return 1e3

        deltas = [abs(right - left) for left, right in zip(values, values[1:]) if right != left]
        typical_delta = median(deltas) if deltas else 0.0
        if typical_delta >= 1e8:
            return 1e9
        if typical_delta >= 1e3:
            return 1e6
        return 1.0


__all__ = ["TabularReader"]
