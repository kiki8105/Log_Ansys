# src/analysis/fixedwing.py

import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from core.log_model import LogDataset
from engines.time_sync import TimeSyncEngine
from engines.math_engine import MathEngine  # 새로 만든 수학 엔진 가져오기

class FixedWingAnalyzer:
    def __init__(self, dataset: LogDataset):
        self.dataset = dataset
        self.time_sync = TimeSyncEngine(target_hz=50.0) 

    def analyze_tecs_performance(self):
        print("\n[Analysis] 고정익 TECS(고도/속도 제어) 성능 분석을 시작합니다...")
        
        tecs_status = self.dataset.get_topic("tecs_status", 0)
        local_pos = self.dataset.get_topic("vehicle_local_position", 0)
        
        if not tecs_status:
            print("  -> 에러: tecs_status 로그가 없습니다.")
            return None

        # --- [1] 고도 추종 성능 분석 ---
        try:
            print("\n  [1] 고도 추종 성능 (Altitude Tracking)")
            
            # Fallback 로직: tecs에 altitude_filtered가 없으면 local_position.z를 융합(Fusion)
            if "altitude_filtered" in tecs_status.signals:
                requests_alt = [(tecs_status, "altitude_sp"), (tecs_status, "altitude_filtered")]
                synced_alt = self.time_sync.sync_signals(requests_alt)
                act_alt = synced_alt["tecs_status_0__altitude_filtered"].fill_null(strategy="forward").to_numpy()
            elif local_pos and "z" in local_pos.signals:
                # 💡 여기가 핵심: 서로 완전히 다른 두 센서 토픽을 하나의 시간축으로 묶어버립니다.
                print("  -> 안내: tecs_status에 고도 필드가 없어 vehicle_local_position 토픽을 융합합니다.")
                requests_alt = [(tecs_status, "altitude_sp"), (local_pos, "z")]
                synced_alt = self.time_sync.sync_signals(requests_alt)
                
                # Math Engine을 사용하여 NED 좌표계의 z값을 양수 고도(Altitude)로 변환
                raw_z = synced_alt["vehicle_local_position_0__z"].fill_null(strategy="forward").to_numpy()
                act_alt = MathEngine.ned_z_to_altitude(raw_z)
            else:
                raise ValueError("고도를 추정할 수 있는 데이터가 로그에 없습니다.")

            # 목표 고도 데이터 추출
            sp_alt = synced_alt["tecs_status_0__altitude_sp"].fill_null(strategy="forward").to_numpy()
            
            # RMSE 계산
            error_alt = sp_alt - act_alt
            rmse_alt = np.sqrt(np.mean(error_alt**2))
            print(f"  -> Altitude RMSE (고도 오차): {rmse_alt:.3f} 미터")
            
            if rmse_alt < 2.0:
                print("  -> [평가] EXCELLENT: TECS 고도 제어가 안정적입니다.")
            elif rmse_alt < 5.0:
                print("  -> [평가] GOOD: 고도 추종이 양호합니다.")
            else:
                print("  -> [평가] WARNING: 고도 추종 오차가 큽니다. TECS 게인을 점검하세요.")

        except Exception as e:
            print(f"  -> 고도 분석 실패: {e}")

        # --- [2] 대기속도 추종 성능 분석 (기존 유지) ---
        try:
            print("\n  [2] 대기속도 추종 성능 (Airspeed Tracking)")
            requests_spd = [
                (tecs_status, "true_airspeed_sp"),
                (tecs_status, "true_airspeed_filtered")
            ]
            synced_spd = self.time_sync.sync_signals(requests_spd)
            
            sp_spd = synced_spd["tecs_status_0__true_airspeed_sp"].fill_null(strategy="forward").to_numpy()
            act_spd = synced_spd["tecs_status_0__true_airspeed_filtered"].fill_null(strategy="forward").to_numpy()
            
            error_spd = sp_spd - act_spd
            rmse_spd = np.sqrt(np.mean(error_spd**2))
            print(f"  -> Airspeed RMSE (속도 오차): {rmse_spd:.3f} m/s")
            
            if rmse_spd < 1.0:
                print("  -> [평가] EXCELLENT: 대기속도 유지 능력이 우수합니다.")
            else:
                print("  -> [평가] GOOD / WARNING: 속도 편차를 확인하세요.")

        except Exception as e:
            print(f"  -> 속도 분석 실패: {e}")

# === 테스트 ===
if __name__ == "__main__":
    from engines.io_engine import LogIOEngine
    project_root = os.path.abspath(os.path.join(src_dir, "../"))
    test_file = os.path.join(project_root, "data", "raw", "sample.ulg") 
    
    io_engine = LogIOEngine()
    my_dataset = io_engine.load(test_file)
    if my_dataset:
        fw_analyzer = FixedWingAnalyzer(my_dataset)
        fw_analyzer.analyze_tecs_performance()