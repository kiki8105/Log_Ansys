# src/engines/io_engine.py

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from readers.base import LoadResult
from readers.registry import ReaderRegistry, default_reader_registry
from readers.release_policy import resolve_format_release_policy
from storage.parquet_cache import ParquetCacheManager


def supported_source_extensions(registry: Optional[ReaderRegistry] = None):
    return tuple(sorted((registry or default_reader_registry()).extensions))


def supported_source_summary(registry: Optional[ReaderRegistry] = None) -> str:
    """현재 출시 정책으로 UI에 노출된 형식만 짧게 표시한다."""

    active_registry = registry or default_reader_registry()
    policy = getattr(active_registry, "release_policy", None) or resolve_format_release_policy()
    return policy.format_summary(active_registry.reader_ids)


def is_supported_log_source(path: str, registry: Optional[ReaderRegistry] = None) -> bool:
    source = Path(path)
    extensions = set(supported_source_extensions(registry))
    if source.is_dir():
        # ROS2 bag 디렉터리도 현재 출시 정책에 rosbag reader가
        # 노출된 경우에만 drag/drop 진입점을 연다.
        return ".yaml" in extensions and (source / "metadata.yaml").is_file()
    return source.is_file() and source.suffix.lower() in extensions


def log_file_dialog_filter(registry: Optional[ReaderRegistry] = None) -> str:
    patterns = " ".join(f"*{suffix}" for suffix in supported_source_extensions(registry))
    return f"Supported Log Files ({patterns});;All Files (*)"


class LogIOEngine:
    """reader 탐지, 형식 중립 적재, fingerprint 캐시를 한 진입점으로 묶습니다."""

    def __init__(
        self,
        registry: Optional[ReaderRegistry] = None,
        cache_mgr: Optional[ParquetCacheManager] = None,
    ):
        self.registry = registry or default_reader_registry()
        self.cache_mgr = cache_mgr or ParquetCacheManager()
        self.last_load_info: Dict[str, str] = {}

    def detect(self, file_path: str):
        return self.registry.detect(file_path)

    def load_result(self, file_path: str, options: Optional[dict] = None) -> LoadResult:
        start_time = time.time()
        reader, probe = self.registry.detect(file_path)
        options = dict(options or {})
        source = Path(file_path)
        cache_source = source.parent if reader.id == "rosbag" and source.name.lower() == "metadata.yaml" else source
        result = self.cache_mgr.load_result(str(cache_source), reader.id, reader.version, options)
        load_type = "Cache Load"

        if result is None and reader.id == "px4_ulog":
            # (2026-10-03) 원본이 보이지 않는 기존 캐시가 있어 basename 기반
            # v1 캐시를 읽기 전용 호환 경로로 유지합니다.
            legacy = self.cache_mgr.load_dataset(os.path.basename(file_path))
            if legacy is not None:
                legacy.source_path = str(file_path)
                legacy.source_format = reader.id
                legacy.capabilities.update({"px4", "timeseries"})
                result = LoadResult(
                    dataset=legacy,
                    format_id=reader.id,
                    metadata={"reader_version": reader.version, "cache_schema": "legacy-v1"},
                    capabilities={"px4", "timeseries"},
                    warnings=["기존 basename 캐시를 읽었습니다. 원본 재파싱 시 v2 캐시로 전환됩니다."],
                    source_path=str(file_path),
                )
                load_type = "Legacy Cache Load"

        if result is None:
            result = reader.load(file_path)
            load_type = "Full Parsing"
            try:
                self.cache_mgr.save_result(
                    result,
                    str(cache_source),
                    reader.id,
                    reader.version,
                    options,
                )
            except Exception as cache_error:
                result.warnings.append(f"캐시 저장 실패: {cache_error}")

        result.dataset.source_format = result.format_id
        result.dataset.source_path = str(file_path)
        result.source_path = str(file_path)
        result.dataset.capabilities.update(result.capabilities)
        result.dataset.metadata.update(result.metadata)
        self.last_load_info = {
            "reader_id": reader.id,
            "reader_version": reader.version,
            "probe_reason": probe.reason,
            "load_type": load_type,
        }
        print(
            f"\n[{load_type}] {os.path.basename(file_path)} | "
            f"reader={reader.id} | {time.time() - start_time:.4f} 초"
        )
        return result

    def load(self, file_path: str):
        """기존 호출부와의 호환을 위해 dataset만 반환합니다."""
        return self.load_result(file_path).dataset


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(src_dir, "../"))
    test_file = os.path.join(project_root, "data", "raw", "sample.ulg")
    io_engine = LogIOEngine()
    loaded = io_engine.load_result(test_file)
    loaded.dataset.print_summary()
