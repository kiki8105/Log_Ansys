# src/engines/time_sync.py

import os
import sys
import numpy as np
from scipy.interpolate import interp1d
import polars as pl

# 상위 폴더(src) 인식용 코드
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from core.log_model import TopicInstance

class TimeSyncEngine:
    """
    서로 다른 샘플링 주파수(Hz)를 가진 여러 Topic의 데이터를
    사용자가 지정한 기준 주파수(Master Hz)의 단일 시간축으로 강제 정렬(Resampling)하는 엔진입니다.
    """
    def __init__(self, target_hz: float = 100.0):
        self.target_hz = target_hz
        self.dt = 1.0 / target_hz  # 데이터 간의 시간 간격 (예: 100Hz면 0.01초)

    def sync_signals(self, signal_requests: list) -> pl.DataFrame:
        """
        요청받은 신호들을 마스터 시간축에 맞춰 보간(Interpolate)하여 하나의 DataFrame으로 반환합니다.
        
        :param signal_requests: [(topic_객체, "신호이름"), ...] 형태의 리스트
        """
        if not signal_requests:
            print("동기화할 신호가 없습니다.")
            return None

        print(f"\n[Time Sync] {self.target_hz}Hz 공통 시간축 정렬을 시작합니다...")

        # 1. 모든 신호가 공통으로 존재하는 시간 구간(교집합) 찾기
        start_times = []
        end_times = []
        
        for topic_inst, col_name in signal_requests:
            # 해당 토픽의 시간 배열 가져오기 (결측치 제거)
            time_array = topic_inst.dataframe["timestamp_sec"].drop_nulls().to_numpy()
            if len(time_array) > 0:
                start_times.append(time_array[0])
                end_times.append(time_array[-1])

        # 모든 센서가 켜져 있는 가장 늦은 시작 시간 ~ 가장 이른 종료 시간
        global_start = max(start_times)
        global_end = min(end_times)

        # 2. 마스터 시간축(Master Timeline) 생성 (예: 10.0초, 10.01초, 10.02초 ...)
        master_time = np.arange(global_start, global_end, self.dt)
        
        # 결과를 담을 딕셔너리 (마스터 시간축 먼저 넣기)
        synced_data = {"master_time_sec": master_time}

        # 3. 각 신호별로 보간(Interpolation) 수행
        for topic_inst, col_name in signal_requests:
            raw_time = topic_inst.dataframe["timestamp_sec"].to_numpy()
            raw_value = topic_inst.signals[col_name].data.to_numpy()
            
            # SciPy를 이용한 선형 보간기(Linear Interpolator) 생성
            # bounds_error=False, fill_value="extrapolate" : 범위 끝부분 에러 방지
            interpolator = interp1d(raw_time, raw_value, kind='linear', bounds_error=False, fill_value="extrapolate")
            
            # 마스터 시간축에 맞춰 데이터 재계산 (Resampling)
            resampled_value = interpolator(master_time)
            
            # 열 이름이 겹치지 않게 '토픽명_신호명' 형태로 저장
            new_col_name = f"{topic_inst.unique_name}__{col_name}"
            synced_data[new_col_name] = resampled_value
            print(f" - {new_col_name} 동기화 완료")

        # 4. 분석하기 편하도록 Polars DataFrame으로 변환하여 반환
        result_df = pl.DataFrame(synced_data)
        print("[Time Sync] 동기화 및 병합 완료!")
        
        return result_df

# === 단독 테스트 실행 ===
if __name__ == "__main__":
    from engines.io_engine import LogIOEngine
    
    project_root = os.path.abspath(os.path.join(src_dir, "../"))
    test_file = os.path.join(project_root, "data", "raw", "sample.ulg") 
    
    # 1. 초고속 캐시 로딩 (io_engine 활용)
    io_engine = LogIOEngine()
    dataset = io_engine.load(test_file)
    
    if dataset:
        # 2. 동기화할 신호 고르기 (서로 주파수가 다른 attitude와 local_position)
        attitude_topic = dataset.get_topic("vehicle_attitude", 0)
        position_topic = dataset.get_topic("vehicle_local_position", 0)
        
        if attitude_topic and position_topic:
            # 3. Time Sync 엔진 가동 (50Hz로 통일해보기)
            sync_engine = TimeSyncEngine(target_hz=50.0)
            
            requests = [
                (attitude_topic, "q[0]"),          # 쿼터니언 W (대략 250Hz)
                (position_topic, "z")              # 고도 Z (대략 50Hz)
            ]
            
            synced_df = sync_engine.sync_signals(requests)
            
            print("\n[동기화된 데이터 결과 미리보기 (상위 5행)]")
            print(synced_df.head(5))