# src/core/log_model.py

from dataclasses import dataclass, field
from typing import Dict, Any, Iterable, Optional, Set
import numpy as np
import polars as pl

from core.lod_pyramid import LODPyramid

# ---------------------------------------------------------
# 1. Signal (가장 작은 단위의 데이터 선)
# ---------------------------------------------------------
@dataclass
class Signal:
    """
    하나의 시계열 데이터 흐름을 의미합니다. (예: sensor_gyro_0의 'x'축 데이터)
    PlotJuggler에서 선 하나를 그릴 때 이 Signal 객체 하나가 사용됩니다.

    * @dataclass: 파이썬에서 데이터를 담는 상자(클래스)를 쉽게 만들게 해주는 기능입니다.
    """
    name: str              # 신호의 이름 (예: 'x', 'y', 'z', 'roll', 'pitch')
    data: pl.Series        # 실제 데이터 값들이 들어있는 1차원 배열 (Polars Series 사용)
    unit: str = ""         # 단위 (예: 'rad/s', 'm/s^2') - 추후 메타데이터 활용용
    _lod_pyramid: Optional[LODPyramid] = field(default=None, repr=False, compare=False)

    def ensure_lod(self, timestamps: np.ndarray, values: np.ndarray) -> LODPyramid:
        """LOD 피라미드를 요청 시점에 1회 빌드하고 캐시한다. 소스 데이터가 변하면 재빌드한다."""
        if self._lod_pyramid is None or not self._lod_pyramid.matches(timestamps, values):
            self._lod_pyramid = LODPyramid.build(timestamps, values)
        return self._lod_pyramid

    def get_lod(self) -> Optional[LODPyramid]:
        return self._lod_pyramid

    def invalidate_lod(self) -> None:
        self._lod_pyramid = None


# ---------------------------------------------------------
# 2. TopicInstance (하나의 센서 또는 모듈 단위)
# ---------------------------------------------------------
@dataclass
class TopicInstance:
    """
    특정 Topic의 개별 인스턴스(multi_id)를 의미합니다.
    예: sensor_gyro가 3개 있다면, TopicInstance도 0번, 1번, 2번 3개가 만들어집니다.
    """
    base_name: str         # 원본 Topic 이름 (예: 'sensor_gyro')
    instance_id: int       # 인스턴스 ID (예: 0)
    
    # 이 Topic이 가진 여러 Signal들을 이름(key)으로 찾을 수 있게 모아둔 사전(Dictionary)
    # field(default_factory=dict)는 객체가 생성될 때 빈 사전을 자동으로 만들어주라는 뜻입니다.
    signals: Dict[str, Signal] = field(default_factory=dict) 
    
    # 이 Topic의 전체 데이터를 표 형태로 통째로 가지고 있을 DataFrame
    dataframe: pl.DataFrame = None 

    @property
    def unique_name(self) -> str:
        """
        이름과 ID를 합쳐 고유한 식별자를 반환하는 기능입니다.
        데이터 지옥(충돌)을 막아주는 핵심 역할입니다. (예: 'sensor_gyro_0')
        """
        return f"{self.base_name}_{self.instance_id}"

    def refresh_signals(
        self,
        excluded: Iterable[str] = ("timestamp", "timestamp_sec", "source_timestamp_ns"),
        *,
        included: Optional[Iterable[str]] = None,
        numeric_only: bool = True,
    ):
        """DataFrame을 단일 기준으로 삼아 그래프 신호 인덱스를 다시 맞춘다.

        공통 그래프는 수치 scalar 열만 받는다. ``included``는 reader가
        최초 파싱 때 선택한 신호 목록을 cache 복원 후에도 그대로
        유지하기 위한 allowlist다.
        """
        excluded_names = set(excluded)
        included_names = None if included is None else set(included)
        if self.dataframe is None:
            self.signals = {}
            return
        self.signals = {
            col_name: Signal(name=col_name, data=self.dataframe[col_name])
            for col_name in self.dataframe.columns
            if col_name not in excluded_names
            and (included_names is None or col_name in included_names)
            and (
                not numeric_only
                or self.dataframe.schema[col_name].is_numeric()
            )
        }


# ---------------------------------------------------------
# 3. LogDataset (비행 로그 전체를 관리하는 최상위 지휘소)
# ---------------------------------------------------------
class LogDataset:
    """
    하나의 로그 source를 형식과 무관하게 메모리에 담는 최상위 데이터 모델입니다.
    Registry(등록소) 역할도 겸합니다.
    """
    def __init__(
        self,
        *,
        source_format: str = "unknown",
        source_path: str = "",
        capabilities: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # 전체 로그 안의 모든 TopicInstance들을 고유 이름으로 저장하는 사전
        self.topics: Dict[str, TopicInstance] = {}
        # (2026-10-03) 파서 형식과 기능을 데이터셋에 보존해 PX4 전용 분석을
        # 다른 로그에 잘못 적용하지 않도록 합니다.
        self.source_format = str(source_format or "unknown")
        self.source_path = str(source_path or "")
        self.capabilities: Set[str] = set(capabilities or ())
        self.metadata: Dict[str, Any] = dict(metadata or {})

    def add_topic(self, topic: TopicInstance):
        """새로운 TopicInstance를 데이터셋에 등록합니다."""
        self.topics[topic.unique_name] = topic
        
    def get_topic(self, base_name: str, instance_id: int = 0) -> TopicInstance:
        """
        원하는 Topic 데이터를 안전하게 꺼내옵니다.
        예: dataset.get_topic('sensor_gyro', 1) -> 1번 자이로 데이터 반환
        """
        unique_name = f"{base_name}_{instance_id}"
        return self.topics.get(unique_name, None)

    def print_summary(self):
        """현재 데이터셋에 어떤 데이터들이 들어왔는지 요약해서 보여줍니다."""
        print("=== Log Dataset Summary ===")
        print(f"Total Topics: {len(self.topics)}")
        for name, topic in self.topics.items():
            # 각 토픽당 몇 개의 필드(Signal)가 있는지 출력
            print(f" - {name} (Signals: {len(topic.signals)})")
