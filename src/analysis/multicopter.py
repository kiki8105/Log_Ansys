# src/analysis/multicopter.py

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
from engines.dsp_engine import DSPEngine

class MulticopterAnalyzer:
    """
    멀티콥터 비행 로그를 자동으로 분석하여 제어 성능 지표를 뽑아내는 클래스입니다.
    """
    def __init__(self, dataset: LogDataset):
        self.dataset = dataset
        self.time_sync = TimeSyncEngine(target_hz=50.0) # 제어 분석용 50Hz 정렬
        self.dsp = DSPEngine()

    def analyze_attitude_tracking(self):
        """
        [목표] Roll, Pitch, Yaw의 목표값(Setpoint)과 실제값(Actual)의 오차(RMSE) 계산
        """
        print("\n[Analysis] 자세 제어 추종성(Attitude Tracking) 분석을 시작합니다...")
        
        # 1. 필요한 Topic 가져오기
        att_actual = self.dataset.get_topic("vehicle_attitude", 0)
        att_sp = self.dataset.get_topic("vehicle_attitude_setpoint", 0)
        
        if not att_actual or not att_sp:
            print("  -> 경고: 자세 제어 관련 로그(Topic)가 존재하지 않습니다.")
            return None

        # 2. Time Sync 엔진으로 두 데이터의 시간축을 강제 정렬
        requests = [
            (att_sp, "roll_body"), (att_actual, "roll_body") # Roll 
            # (실무에서는 쿼터니언을 오일러각으로 변환하는 과정이 선행되나, 
            # 최신 PX4 ULG 일부나 특정 튜닝에서는 euler 각이 함께 로깅되기도 합니다.
            # 이 예제에서는 구조를 보여주기 위해 가상의 roll_body 필드가 있다고 가정합니다.
            # 추후 Math 엔진에서 쿼터니언 변환 수식을 추가할 예정입니다.)
        ]
        
        # ⚠️ 만약 ULG에 'roll_body'가 없다면 에러가 날 수 있으므로,
        # 분석 엔진이 다운되지 않도록 예외 처리를 해야 합니다.
        try:
            # 여기서는 예시로 q[0] (Quaternion W)의 추종성을 비교해봅니다.
            requests = [
                (att_sp, "q_d[0]"),   # 목표 쿼터니언 W
                (att_actual, "q[0]")  # 실제 쿼터니언 W
            ]
            synced_df = self.time_sync.sync_signals(requests)
            
            # 3. RMSE (Root Mean Square Error) 오차율 계산
            sp_val = synced_df["vehicle_attitude_setpoint_0__q_d[0]"].to_numpy()
            act_val = synced_df["vehicle_attitude_0__q[0]"].to_numpy()
            
            # 수식: RMSE = sqrt( mean( (목표 - 실제)^2 ) )
            error = sp_val - act_val
            rmse = np.sqrt(np.mean(error**2))
            
            print(f"  -> Q[0] Tracking RMSE (오차율): {rmse:.6f}")
            
            # 자동 코멘트 평가 시스템 (Rule-based)
            if rmse < 0.05:
                print("  -> [평가] EXCELLENT: 자세 제어기 튜닝이 매우 잘 되어 있습니다.")
            elif rmse < 0.1:
                print("  -> [평가] GOOD: 비행에 문제없는 수준입니다.")
            else:
                print("  -> [평가] WARNING: 추종 오차가 큽니다. PID Gain(P, D) 재튜닝을 권장합니다.")
                
            return rmse
            
        except Exception as e:
            print(f"  -> 분석 실패 (해당 필드가 로그에 없음): {e}")
            return None

# === 단독 테스트 실행 ===
if __name__ == "__main__":
    from engines.io_engine import LogIOEngine
    
    project_root = os.path.abspath(os.path.join(src_dir, "../"))
    test_file = os.path.join(project_root, "data", "raw", "sample.ulg") 
    
    io_engine = LogIOEngine()
    my_dataset = io_engine.load(test_file)
    
    if my_dataset:
        mc_analyzer = MulticopterAnalyzer(my_dataset)
        mc_analyzer.analyze_attitude_tracking()