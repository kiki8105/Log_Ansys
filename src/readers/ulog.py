"""기존 PX4 ULog 파서를 공통 reader 계약에 연결합니다."""

from __future__ import annotations

from pathlib import Path

from readers.base import BaseLogReader, LoadResult, LogPath, ProbeResult


class ULogReader(BaseLogReader):
    """``ULGParser``의 동작은 유지하면서 registry에서 사용할 어댑터를 제공합니다."""

    id = "px4_ulog"
    version = "1.0"
    extensions = frozenset({".ulg"})
    definitive_probe = True

    def probe(self, path: LogPath) -> ProbeResult:
        file_path = Path(path)
        if not file_path.is_file():
            return ProbeResult(0, self.id, "파일을 찾을 수 없음")
        try:
            with file_path.open("rb") as stream:
                magic = stream.read(4)
        except OSError as exc:
            return ProbeResult(0, self.id, f"파일 헤더를 읽을 수 없음: {exc}")

        if magic == b"ULog":
            return ProbeResult(100, self.id, "PX4 ULog magic 일치")
        if file_path.suffix.lower() == ".ulg":
            return ProbeResult(60, self.id, ".ulg 확장자이나 ULog magic 불일치")
        return ProbeResult(0, self.id, "ULog 형식이 아님")

    def load(self, path: LogPath) -> LoadResult:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(file_path)

        # WHY: 기존 ULGParser를 단일 구현점으로 유지해야 GUI와 reader 경로의 변환 결과가 갈라지지 않습니다.
        from engines.parser import ULGParser

        dataset = ULGParser().parse(str(file_path))
        metadata = {
            "reader_version": self.version,
            "source_name": file_path.name,
            "topic_count": len(dataset.topics),
        }
        return LoadResult(
            dataset=dataset,
            format_id=self.id,
            metadata=metadata,
            capabilities={"px4", "timeseries"},
            source_path=str(file_path),
        )


__all__ = ["ULogReader"]
