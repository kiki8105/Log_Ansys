"""로그 reader 등록, 자동 탐지, 적재 진입점."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from threading import RLock
from typing import Iterator

from readers.base import BaseLogReader, LoadResult, LogPath, ProbeResult
from readers.release_policy import (
    FormatReleasePolicy,
    MULTIFORMAT_PREVIEW,
    resolve_format_release_policy,
)


class ReaderRegistryError(RuntimeError):
    """reader registry 공통 오류."""


class DuplicateReaderError(ReaderRegistryError):
    """같은 ID의 reader를 중복 등록했을 때 발생합니다."""


class UnsupportedLogError(ReaderRegistryError):
    """어떤 reader도 입력을 지원한다고 판단하지 못했을 때 발생합니다."""


class AmbiguousLogFormatError(ReaderRegistryError):
    """최고 confidence가 같은 reader가 둘 이상일 때 발생합니다."""


# 초기 호출부가 사용하던 짧은 이름도 함께 제공합니다.
UnsupportedFormatError = UnsupportedLogError
AmbiguousFormatError = AmbiguousLogFormatError


@dataclass(frozen=True, slots=True)
class ReaderProbe:
    reader: BaseLogReader
    result: ProbeResult


@dataclass(frozen=True, slots=True)
class _LazyReader:
    module_name: str
    class_name: str


class ReaderRegistry:
    """명시적으로 등록된 reader 중 probe 점수가 가장 높은 구현을 선택합니다."""

    def __init__(self, *, release_policy: FormatReleasePolicy | None = None) -> None:
        self._readers: dict[str, BaseLogReader] = {}
        self._lazy_readers: list[_LazyReader] = []
        self._lazy_loaded: set[_LazyReader] = set()
        self._lazy_import_errors: dict[str, str] = {}
        self._lock = RLock()
        self.release_policy = release_policy

    def register(
        self,
        reader: BaseLogReader | type[BaseLogReader],
        *,
        replace: bool = False,
    ) -> BaseLogReader:
        if isinstance(reader, type):
            if not issubclass(reader, BaseLogReader):
                raise TypeError("reader class must inherit BaseLogReader")
            reader = reader()
        if not isinstance(reader, BaseLogReader):
            raise TypeError("reader must be a BaseLogReader instance")
        reader_id = str(reader.id).strip()
        if not reader_id:
            raise ValueError("reader.id must not be empty")

        with self._lock:
            if reader_id in self._readers and not replace:
                raise DuplicateReaderError(f"reader ID가 이미 등록되어 있습니다: {reader_id}")
            self._readers[reader_id] = reader
        return reader

    def unregister(self, reader_id: str) -> BaseLogReader:
        self._load_lazy_readers()
        with self._lock:
            try:
                return self._readers.pop(reader_id)
            except KeyError as exc:
                raise KeyError(f"등록되지 않은 reader: {reader_id}") from exc

    def add_lazy_reader(self, module_name: str, class_name: str) -> None:
        """선택 기능의 모듈은 실제 탐지 시점까지 import하지 않습니다."""

        spec = _LazyReader(module_name, class_name)
        with self._lock:
            if spec not in self._lazy_readers:
                self._lazy_readers.append(spec)

    @property
    def readers(self) -> tuple[BaseLogReader, ...]:
        self._load_lazy_readers()
        with self._lock:
            return tuple(self._readers.values())

    @property
    def reader_ids(self) -> tuple[str, ...]:
        return tuple(reader.id for reader in self.readers)

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset(
            suffix
            for reader in self.readers
            for suffix in reader.normalized_extensions()
        )

    @property
    def lazy_import_errors(self) -> dict[str, str]:
        self._load_lazy_readers()
        with self._lock:
            return dict(self._lazy_import_errors)

    def get(self, reader_id: str) -> BaseLogReader:
        self._load_lazy_readers()
        with self._lock:
            try:
                return self._readers[reader_id]
            except KeyError as exc:
                raise KeyError(f"등록되지 않은 reader: {reader_id}") from exc

    def probe_all(self, path: LogPath) -> list[ReaderProbe]:
        probes: list[ReaderProbe] = []
        for reader in self.readers:
            probes.append(self._probe_reader(reader, path))
        return sorted(probes, key=lambda item: (-item.result.confidence, item.reader.id))

    def probe(self, path: LogPath) -> ProbeResult:
        """선택될 reader의 probe 결과를 반환하며 동점/미지원은 명시적으로 거부합니다."""

        return self._winner(path).result

    def detect(self, path: LogPath) -> tuple[BaseLogReader, ProbeResult]:
        winner = self._winner(path)
        return winner.reader, winner.result

    def select_reader(self, path: LogPath) -> BaseLogReader:
        return self._winner(path).reader

    # 호출부에서 자연스럽게 쓸 수 있는 짧은 별칭입니다.
    select = select_reader

    def load(
        self,
        path: LogPath,
        reader_id: str | None = None,
        *,
        format_id: str | None = None,
    ) -> LoadResult:
        if reader_id is not None and format_id is not None and reader_id != format_id:
            raise ValueError("reader_id와 format_id를 동시에 다른 값으로 지정할 수 없습니다.")
        requested_id = reader_id if reader_id is not None else format_id
        reader = self.get(requested_id) if requested_id is not None else self.select_reader(path)
        result = reader.load(path)
        if not isinstance(result, LoadResult):
            raise TypeError(f"{reader.id}.load() must return LoadResult")
        return result

    def _winner(self, path: LogPath) -> ReaderProbe:
        probes: list[ReaderProbe] = []
        for reader in self.readers:
            probe = self._probe_reader(reader, path)
            probes.append(probe)
            if reader.definitive_probe and probe.result.confidence == 100:
                return probe
        probes.sort(key=lambda item: (-item.result.confidence, item.reader.id))
        if not probes or probes[0].result.confidence <= 0:
            reasons = "; ".join(
                f"{item.reader.id}: {item.result.reason}" for item in probes
            )
            detail = f" ({reasons})" if reasons else ""
            raise UnsupportedLogError(f"지원하는 로그 형식을 찾지 못했습니다{detail}")

        best_score = probes[0].result.confidence
        winners = [item for item in probes if item.result.confidence == best_score]
        if len(winners) > 1:
            ids = ", ".join(item.reader.id for item in winners)
            raise AmbiguousLogFormatError(
                f"로그 형식을 하나로 결정할 수 없습니다(confidence={best_score}): {ids}"
            )
        return winners[0]

    @staticmethod
    def _probe_reader(reader: BaseLogReader, path: LogPath) -> ReaderProbe:
        try:
            result = reader.probe(path)
            if not isinstance(result, ProbeResult):
                raise TypeError("probe() must return ProbeResult")
        except Exception as exc:  # 개별 선택 기능의 고장이 다른 형식 탐지를 막아서는 안 됩니다.
            result = ProbeResult(0, reader.id, f"probe 실패: {type(exc).__name__}: {exc}")
        return ReaderProbe(reader, result)

    def _load_lazy_readers(self) -> None:
        with self._lock:
            pending = [spec for spec in self._lazy_readers if spec not in self._lazy_loaded]
            for spec in pending:
                self._lazy_loaded.add(spec)
                label = f"{spec.module_name}:{spec.class_name}"
                try:
                    module = import_module(spec.module_name)
                    reader_class = getattr(module, spec.class_name)
                    self.register(reader_class)
                except Exception as exc:
                    # WHY: ROS/ArduPilot 선택 의존성이 빠져도 PX4·표 로그 기능은 계속 시작되어야 합니다.
                    self._lazy_import_errors[label] = f"{type(exc).__name__}: {exc}"

    def __iter__(self) -> Iterator[BaseLogReader]:
        return iter(self.readers)

    def __len__(self) -> int:
        return len(self.readers)


def create_default_registry(
    *,
    include_optional: bool | None = None,
    policy: FormatReleasePolicy | None = None,
) -> ReaderRegistry:
    """출시 정책에서 허용한 reader만 등록한다.

    ``include_optional``은 기존 테스트/플러그인 호출부용 호환 인자다.
    새 제품 코드는 인자를 생략해 중앙 출시 정책을 따른다.
    """

    from readers.tabular import TabularReader
    from readers.ulog import ULogReader

    if policy is not None and include_optional is not None:
        raise ValueError("policy와 include_optional을 동시에 지정할 수 없습니다.")
    if policy is None:
        if include_optional is None:
            policy = resolve_format_release_policy()
        elif include_optional:
            policy = resolve_format_release_policy(MULTIFORMAT_PREVIEW)
        else:
            # 기존 include_optional=False 호출부는 표 로그 reader를
            # core reader로 사용해 왔다. 명시적 테스트/내부 주입에서만
            # 그 의미를 유지한다. 기본 앱은 별도 인자 없이 다중 포맷
            # 출시 정책을 사용한다.
            policy = FormatReleasePolicy(
                name="legacy_core",
                enabled_reader_ids=frozenset({"px4_ulog", "tabular"}),
                preview_reader_ids=frozenset({"tabular"}),
            )

    registry = ReaderRegistry(release_policy=policy)
    if policy.enables("px4_ulog"):
        registry.register(ULogReader())
    if policy.enables("tabular"):
        registry.register(TabularReader())
    if policy.enables("rosbag"):
        package_name = __package__ or "readers"
        registry.add_lazy_reader(f"{package_name}.rosbag", "ROSBagReader")
    if policy.enables("ardupilot"):
        package_name = __package__ or "readers"
        registry.add_lazy_reader(f"{package_name}.ardupilot", "ArduPilotReader")
    return registry


DEFAULT_REGISTRY = create_default_registry()


def get_default_registry() -> ReaderRegistry:
    return DEFAULT_REGISTRY


def default_reader_registry() -> ReaderRegistry:
    """애플리케이션 공용 기본 registry를 반환합니다."""

    return DEFAULT_REGISTRY


__all__ = [
    "AmbiguousFormatError",
    "AmbiguousLogFormatError",
    "DEFAULT_REGISTRY",
    "DuplicateReaderError",
    "ReaderProbe",
    "ReaderRegistry",
    "ReaderRegistryError",
    "UnsupportedFormatError",
    "UnsupportedLogError",
    "create_default_registry",
    "default_reader_registry",
    "get_default_registry",
]
