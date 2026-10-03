"""로그 형식을 실제 제품 UI에 노출하는 중앙 출시 정책.

Reader 구현과 제품 출시 여부를 분리한다. ROS/ArduPilot/표 로그 reader는
코드와 테스트에 계속 남겨 두되, ULG 수용 테스트가 완료되기 전에는
기본 배포 UI에서 도달할 수 없게 한다.

개발 미리보기는 프로그램 시작 전에 다음 환경 변수를 설정해 명시적으로
활성화한다::

    PX4_LOG_FORMAT_POLICY=multiformat_preview
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping


FORMAT_POLICY_ENV = "PX4_LOG_FORMAT_POLICY"
ULG_STABLE = "ulg_stable"
MULTIFORMAT_PREVIEW = "multiformat_preview"

_ALL_READER_IDS = frozenset({"px4_ulog", "tabular", "rosbag", "ardupilot"})
_DISPLAY_ORDER = ("px4_ulog", "rosbag", "ardupilot", "tabular")
_DISPLAY_LABELS = {
    "px4_ulog": "ULG",
    "rosbag": "ROS",
    "ardupilot": "ArduPilot",
    "tabular": "CSV·JSON",
}


@dataclass(frozen=True, slots=True)
class FormatReleasePolicy:
    """현재 실행에서 사용자에게 노출할 reader 집합."""

    name: str
    enabled_reader_ids: frozenset[str]
    preview_reader_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        unknown = set(self.enabled_reader_ids) - set(_ALL_READER_IDS)
        if unknown:
            raise ValueError(f"알 수 없는 reader ID: {', '.join(sorted(unknown))}")
        if "px4_ulog" not in self.enabled_reader_ids:
            raise ValueError("PX4 ULog reader는 모든 출시 정책에 포함되어야 합니다.")
        if not self.preview_reader_ids <= self.enabled_reader_ids:
            raise ValueError("preview reader는 enabled reader에 포함되어야 합니다.")

    def enables(self, reader_id: str) -> bool:
        return str(reader_id) in self.enabled_reader_ids

    def format_summary(self, reader_ids=None) -> str:
        enabled = set(self.enabled_reader_ids if reader_ids is None else reader_ids)
        labels = [
            _DISPLAY_LABELS[reader_id]
            for reader_id in _DISPLAY_ORDER
            if reader_id in enabled
        ]
        return " / ".join(labels) if labels else "ULG"


def resolve_format_release_policy(
    value: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> FormatReleasePolicy:
    """정책 이름을 결정한다. 오타/미정 값은 안전하게 ULG-only로 닫힌다."""

    source = os.environ if environ is None else environ
    raw = value if value is not None else source.get(FORMAT_POLICY_ENV, ULG_STABLE)
    normalized = str(raw or ULG_STABLE).strip().lower().replace("-", "_")

    if normalized in {MULTIFORMAT_PREVIEW, "preview", "all"}:
        preview = _ALL_READER_IDS - {"px4_ulog"}
        return FormatReleasePolicy(
            name=MULTIFORMAT_PREVIEW,
            enabled_reader_ids=_ALL_READER_IDS,
            preview_reader_ids=frozenset(preview),
        )

    # Unknown values deliberately fail closed instead of accidentally exposing
    # formats whose real-log/frozen-EXE acceptance gate has not passed yet.
    return FormatReleasePolicy(
        name=ULG_STABLE,
        enabled_reader_ids=frozenset({"px4_ulog"}),
    )


__all__ = [
    "FORMAT_POLICY_ENV",
    "FormatReleasePolicy",
    "MULTIFORMAT_PREVIEW",
    "ULG_STABLE",
    "resolve_format_release_policy",
]
