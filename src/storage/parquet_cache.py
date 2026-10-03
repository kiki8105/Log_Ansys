# src/storage/parquet_cache.py

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from core.log_model import LogDataset, TopicInstance


CACHE_SCHEMA_VERSION = 2
SIGNAL_CONTRACT_VERSION = 1


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


class ParquetCacheManager:
    """원본 fingerprint와 reader 버전으로 격리된 Parquet 캐시 관리자."""

    def __init__(self, cache_dir=os.path.join("runtime", "cache")):
        project_root = os.path.abspath(os.path.join(src_dir, "../"))
        self.cache_dir = os.path.abspath(os.path.join(project_root, cache_dir))
        self.v2_dir = os.path.join(self.cache_dir, "v2")
        os.makedirs(self.v2_dir, exist_ok=True)

    @staticmethod
    def _sample_file_hash(path: Path, sample_size: int = 64 * 1024) -> str:
        digest = hashlib.sha256()
        size = path.stat().st_size
        with path.open("rb") as handle:
            digest.update(handle.read(sample_size))
            if size > sample_size:
                handle.seek(max(0, size - sample_size))
                digest.update(handle.read(sample_size))
        return digest.hexdigest()

    def source_descriptor(self, source_path: str) -> Dict[str, Any]:
        source = Path(source_path).expanduser().resolve()
        if source.is_file():
            stat = source.stat()
            return {
                "kind": "file",
                "path": str(source),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sample_sha256": self._sample_file_hash(source),
            }
        if source.is_dir():
            members = []
            for child in sorted(p for p in source.rglob("*") if p.is_file()):
                stat = child.stat()
                members.append({
                    "path": child.relative_to(source).as_posix(),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                })
            return {"kind": "directory", "path": str(source), "members": members}
        raise FileNotFoundError(f"로그 원본을 찾을 수 없습니다: {source}")

    def cache_key(
        self,
        source_path: str,
        reader_id: str,
        reader_version: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source": self.source_descriptor(source_path),
            "reader_id": str(reader_id),
            "reader_version": str(reader_version),
            "options": _json_safe(options or {}),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _v2_path(self, cache_key: str) -> str:
        return os.path.join(self.v2_dir, cache_key)

    def has_result(
        self,
        source_path: str,
        reader_id: str,
        reader_version: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> bool:
        key = self.cache_key(source_path, reader_id, reader_version, options)
        return os.path.isfile(os.path.join(self._v2_path(key), "manifest.json"))

    def save_result(
        self,
        result,
        source_path: str,
        reader_id: str,
        reader_version: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        key = self.cache_key(source_path, reader_id, reader_version, options)
        target_dir = self._v2_path(key)
        existing_manifest_path = os.path.join(target_dir, "manifest.json")
        if os.path.isfile(existing_manifest_path):
            try:
                with open(existing_manifest_path, "r", encoding="utf-8") as handle:
                    existing_manifest = json.load(handle)
                if int(existing_manifest.get("signal_contract_version", -1)) == SIGNAL_CONTRACT_VERSION:
                    return key
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        temp_dir = tempfile.mkdtemp(prefix=f".{key}.", dir=self.v2_dir)
        try:
            topic_records = []
            for unique_name, topic in result.dataset.topics.items():
                topic_file = hashlib.sha256(unique_name.encode("utf-8")).hexdigest()[:24] + ".parquet"
                topic.dataframe.write_parquet(os.path.join(temp_dir, topic_file))
                topic_records.append({
                    "unique_name": unique_name,
                    "base_name": topic.base_name,
                    "instance_id": int(topic.instance_id),
                    "file": topic_file,
                    # Reader의 cold parse에서 선택한 그래프 신호를
                    # 저장해 warm cache가 문자열/원본 시간 열을 새 신호로
                    # 만드는 계약 변경을 막는다.
                    "signal_names": [
                        name
                        for name in topic.signals
                        if name in topic.dataframe.columns
                        and topic.dataframe.schema[name].is_numeric()
                    ],
                    "units": {
                        name: signal.unit
                        for name, signal in topic.signals.items()
                        if signal is not None and getattr(signal, "unit", "")
                    },
                })

            manifest = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "signal_contract_version": SIGNAL_CONTRACT_VERSION,
                "cache_key": key,
                "source": self.source_descriptor(source_path),
                "reader": {"id": str(reader_id), "version": str(reader_version)},
                "format_id": str(result.format_id),
                "capabilities": sorted(str(v) for v in result.capabilities),
                "metadata": _json_safe(result.metadata),
                "parameters": _json_safe(result.parameters),
                "messages": _json_safe(result.messages),
                "warnings": _json_safe(result.warnings),
                "topics": topic_records,
            }
            with open(os.path.join(temp_dir, "manifest.json"), "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)

            # (2026-10-03) 완성된 manifest가 있는 디렉터리만 공개해 중단된
            # 파싱 결과를 다음 실행에서 정상 캐시로 오인하지 않게 합니다.
            if os.path.isdir(target_dir):
                # manifest가 없는 동일 key 디렉터리는 이전 중단 쓰기이므로 교체합니다.
                shutil.rmtree(target_dir, ignore_errors=True)
            os.replace(temp_dir, target_dir)
            return key
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def load_result(
        self,
        source_path: str,
        reader_id: str,
        reader_version: str,
        options: Optional[Dict[str, Any]] = None,
    ):
        from readers.base import LoadResult

        key = self.cache_key(source_path, reader_id, reader_version, options)
        cache_dir = self._v2_path(key)
        manifest_path = os.path.join(cache_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            return None
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
            return None
        # 이 필드가 없는 기존 v2 cache는 signal 목록을 저장하지
        # 않아 cold/warm 트리가 달라진다. 한 번 재파싱해 자체 치유한다.
        if int(manifest.get("signal_contract_version", -1)) != SIGNAL_CONTRACT_VERSION:
            return None

        dataset = LogDataset(
            source_format=manifest.get("format_id", "unknown"),
            source_path=source_path,
            capabilities=manifest.get("capabilities") or (),
            metadata=manifest.get("metadata") or {},
        )
        for record in manifest.get("topics") or []:
            signal_names = record.get("signal_names")
            if not isinstance(signal_names, list):
                return None
            parquet_path = os.path.join(cache_dir, str(record.get("file", "")))
            if not os.path.isfile(parquet_path):
                return None
            dataframe = pl.read_parquet(parquet_path)
            topic = TopicInstance(
                base_name=str(record.get("base_name", "topic")),
                instance_id=int(record.get("instance_id", 0)),
                dataframe=dataframe,
            )
            topic.refresh_signals(included=signal_names)
            for name, unit in (record.get("units") or {}).items():
                if name in topic.signals:
                    topic.signals[name].unit = str(unit)
            dataset.add_topic(topic)

        return LoadResult(
            dataset=dataset,
            format_id=str(manifest.get("format_id", "unknown")),
            metadata=dict(manifest.get("metadata") or {}),
            parameters=dict(manifest.get("parameters") or {}),
            messages=list(manifest.get("messages") or []),
            capabilities=set(manifest.get("capabilities") or ()),
            warnings=list(manifest.get("warnings") or []),
            source_path=str(source_path),
        )

    # 기존 803MB 캐시를 즉시 폐기하지 않고 ULog 전환 기간에 읽습니다.
    @staticmethod
    def _legacy_safe_name(log_filename: str) -> str:
        return str(log_filename).replace(".", "_")

    def _legacy_dir(self, log_filename: str) -> str:
        return os.path.join(self.cache_dir, self._legacy_safe_name(log_filename))

    def save_dataset(self, dataset: LogDataset, log_filename: str):
        log_cache_dir = self._legacy_dir(log_filename)
        os.makedirs(log_cache_dir, exist_ok=True)
        for unique_name, topic in dataset.topics.items():
            topic.dataframe.write_parquet(os.path.join(log_cache_dir, f"{unique_name}.parquet"))

    def load_dataset(self, log_filename: str) -> Optional[LogDataset]:
        log_cache_dir = self._legacy_dir(log_filename)
        if not os.path.isdir(log_cache_dir):
            return None
        dataset = LogDataset(source_format="px4_ulog", capabilities={"px4", "timeseries"})
        for file_name in os.listdir(log_cache_dir):
            if not file_name.endswith(".parquet"):
                continue
            unique_name = file_name[:-8]
            parts = unique_name.rsplit("_", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            dataframe = pl.read_parquet(os.path.join(log_cache_dir, file_name))
            topic = TopicInstance(parts[0], int(parts[1]), dataframe=dataframe)
            topic.refresh_signals()
            dataset.add_topic(topic)
        return dataset

    def is_cached(self, log_filename: str) -> bool:
        return os.path.isdir(self._legacy_dir(log_filename))
