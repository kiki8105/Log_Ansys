"""로그 형식별 reader가 공유하는 최소 계약."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar


LogPath = str | PathLike[str]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """reader가 파일 형식을 식별한 결과."""

    confidence: int
    format_id: str
    reason: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise TypeError("confidence must be a number between 0 and 100")
        if not 0 <= self.confidence <= 100:
            raise ValueError("confidence must be between 0 and 100")
        if not self.format_id.strip():
            raise ValueError("format_id must not be empty")


@dataclass(slots=True)
class LoadResult:
    """형식별 데이터를 공통 ``LogDataset`` 모델과 부가 정보로 묶은 결과."""

    dataset: Any
    format_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    messages: list[Any] = field(default_factory=list)
    capabilities: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)
    source_path: str = ""

    def __post_init__(self) -> None:
        if not self.format_id.strip():
            raise ValueError("format_id must not be empty")
        if self.source_path:
            self.source_path = str(Path(self.source_path))
        # WHY: 기존 GUI는 LogDataset만 전달받으므로 결과 봉투의 핵심 출처 정보도 dataset에 투영합니다.
        for name, value in (
            ("source_format", self.format_id),
            ("source_path", self.source_path),
            ("capabilities", set(self.capabilities)),
        ):
            try:
                setattr(self.dataset, name, value)
            except (AttributeError, TypeError):
                pass
        try:
            dataset_metadata = getattr(self.dataset, "metadata", None)
            if isinstance(dataset_metadata, dict):
                dataset_metadata.update(self.metadata)
            else:
                setattr(self.dataset, "metadata", dict(self.metadata))
        except (AttributeError, TypeError):
            pass


class BaseLogReader(ABC):
    """모든 로그 reader가 구현해야 하는 형식 탐지/적재 인터페이스."""

    id: ClassVar[str] = ""
    version: ClassVar[str] = "1.0"
    extensions: ClassVar[frozenset[str]] = frozenset()
    # Readers may opt in only when confidence=100 comes from an exclusive
    # file signature (not merely an extension).  The registry can then avoid
    # expensive probes for unrelated formats without weakening normal tie
    # detection.
    definitive_probe: ClassVar[bool] = False

    @abstractmethod
    def probe(self, path: LogPath) -> ProbeResult:
        """파일을 변경하지 않고 지원 가능성을 0~100 점으로 반환합니다."""

    @abstractmethod
    def load(self, path: LogPath) -> LoadResult:
        """파일을 읽어 공통 데이터 모델로 반환합니다."""

    @classmethod
    def normalized_extensions(cls) -> frozenset[str]:
        """대소문자/점 표기 차이가 registry 필터에 새지 않도록 정규화합니다."""

        return frozenset(
            suffix if suffix.startswith(".") else f".{suffix}"
            for raw in cls.extensions
            if (suffix := str(raw).strip().lower())
        )


__all__ = ["BaseLogReader", "LoadResult", "LogPath", "ProbeResult"]
