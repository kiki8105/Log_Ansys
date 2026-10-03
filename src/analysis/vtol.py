# src/analysis/vtol.py

import os
import sys
import numpy as np
import polars as pl

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from core.log_model import LogDataset
from engines.time_sync import TimeSyncEngine
from engines.math_engine import MathEngine

class VTOLAnalyzer:
    """
    VTOL 기체의 비행 로그를 분석하는 클래스입니다.
    핵심: 천이 구간(Transition) 추출 및 고도 강하량 분석
    """
    def __init__(self, dataset: LogDataset):
        self.dataset = dataset
        self.time_sync = TimeSyncEngine(target_hz=50.0)

    def analyze_transition_performance(self):
        print("\n[Analysis] VTOL 천이 구간(Transition) 성능 분석을 시작합니다...")
        
        status_topic = self.dataset.get_topic("vehicle_status", 0)
        local_pos = self.dataset.get_topic("vehicle_local_position", 0)
        
        if not status_topic or not local_pos:
            print("  -> 에러: vehicle_status 또는 local_position 로그가 없습니다.")
            return None

        # 1. 상태(Status)와 고도(Z) 데이터 동기화
        try:
            requests = [
                (status_topic, "in_transition_mode"), # 천이 중이면 1(True), 아니면 0(False)
                (local_pos, "z")
            ]
            synced_df = self.time_sync.sync_signals(requests)
            
            # Polars 데이터프레임에서 Numpy 배열로 추출
            # (시간 배열, 천이 상태 배열, 고도 배열)
            time_sec = synced_df["master_time_sec"].to_numpy()
            in_transition = synced_df["vehicle_status_0__in_transition_mode"].fill_null(strategy="forward").to_numpy()
            
            # NED 좌표계 Z축을 양수 고도(Altitude)로 변환
            raw_z = synced_df["vehicle_local_position_0__z"].fill_null(strategy="forward").to_numpy()
            altitude = MathEngine.ned_z_to_altitude(raw_z)
            
            # 2. 천이 구간 추출 로직 (Phase Segmentation)
            # in_transition_mode 가 1(True)인 데이터의 인덱스만 추출
            transition_indices = np.where(in_transition == 1)[0]
            
            if len(transition_indices) == 0:
                print("  -> [안내] 이 로그에는 천이(Transition) 구간이 존재하지 않습니다. (순수 고정익 또는 멀티콥터 비행)")
                return None
                
            # 천이 시작 시간과 종료 시간 파악
            start_idx = transition_indices[0]
            end_idx = transition_indices[-1]
            
            start_time = time_sec[start_idx]
            end_time = time_sec[end_idx]
            duration = end_time - start_time
            
            print(f"  -> 천이 구간 감지: {start_time:.1f}초 ~ {end_time:.1f}초 (소요 시간: {duration:.2f}초)")
            
            # 3. 고도 강하량(Altitude Loss) 분석
            alt_during_transition = altitude[start_idx:end_idx+1]
            alt_at_start = alt_during_transition[0]
            min_alt = np.min(alt_during_transition)
            
            # (시작 고도) - (천이 중 최저 고도) = 강하량
            alt_loss = alt_at_start - min_alt
            # 고도가 오히려 상승한 경우(음수)는 0으로 처리
            alt_loss = max(0.0, alt_loss) 
            
            print(f"  -> 천이 중 최대 고도 강하량: {alt_loss:.2f} 미터")
            
            # 4. 성능 자동 평가
            if alt_loss < 2.0:
                print("  -> [평가] EXCELLENT: 고도 처짐 없이 완벽한 천이가 이루어졌습니다.")
            elif alt_loss < 5.0:
                print("  -> [평가] GOOD: 안정적인 천이입니다.")
            else:
                print("  -> [평가] WARNING: 천이 중 고도 처짐이 심합니다. Transition Airspeed 세팅이나 Pusher 모터 응답성을 확인하세요.")

        except Exception as e:
            print(f"  -> VTOL 분석 중 오류 발생: {e}")

# === 테스트 ===
if __name__ == "__main__":
    from engines.io_engine import LogIOEngine
    project_root = os.path.abspath(os.path.join(src_dir, "../"))
    test_file = os.path.join(project_root, "data", "raw", "sample.ulg") 
    
    io_engine = LogIOEngine()
    my_dataset = io_engine.load(test_file)
    if my_dataset:
        vtol_analyzer = VTOLAnalyzer(my_dataset)
        vtol_analyzer.analyze_transition_performance()