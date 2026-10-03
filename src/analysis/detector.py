# src/analysis/detector.py

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from core.log_model import LogDataset

# 우리가 만든 분석기들 가져오기
from analysis.multicopter import MulticopterAnalyzer
from analysis.fixedwing import FixedWingAnalyzer
from analysis.vtol import VTOLAnalyzer

class FlightTypeDetector:
    """
    로그를 스캔하여 기체 타입을 자동 판별하고,
    알맞은 분석 엔진을 자동으로 매칭하여 실행하는 지휘관 클래스입니다.
    """
    def __init__(self, dataset: LogDataset):
        self.dataset = dataset
        self.airframe_type = "UNKNOWN"
        self.is_vtol = False

    def detect(self):
        """기체 타입을 판별합니다."""
        status_topic = self.dataset.get_topic("vehicle_status", 0)
        
        if not status_topic:
            print("[Detector] 에러: vehicle_status 토픽이 없어 기체 타입을 판별할 수 없습니다.")
            return "UNKNOWN"

        try:
            # 1. VTOL 여부 확인 (1=True, 0=False)
            if "is_vtol" in status_topic.signals:
                vtol_flag = status_topic.signals["is_vtol"].data[0] # 첫 번째 데이터만 확인해도 됨
                if vtol_flag == 1:
                    self.is_vtol = True

            # 2. 기본 기체 타입 확인 (1: Rotary wing, 2: Fixed wing)
            if "vehicle_type" in status_topic.signals:
                v_type = status_topic.signals["vehicle_type"].data[0]
                
                if self.is_vtol:
                    self.airframe_type = "VTOL"
                elif v_type == 1:
                    self.airframe_type = "MULTICOPTER"
                elif v_type == 2:
                    self.airframe_type = "FIXED_WING"
                else:
                    self.airframe_type = f"OTHER_{v_type}"
                    
            print(f"\n[Detector] 자동 판별 완료: 이 로그는 '{self.airframe_type}' 기체의 비행 로그입니다.")
            return self.airframe_type
            
        except Exception as e:
            print(f"[Detector] 판별 중 오류 발생: {e}")
            return "UNKNOWN"

    def run_auto_analysis(self):
        """판별된 기체에 맞는 분석기를 자동으로 실행합니다."""
        frame = self.detect()
        
        print("-" * 50)
        
        if frame == "VTOL":
            # VTOL은 천이(Transition)와 멀티콥터(Hover), 고정익(Cruise) 특성을 다 가지므로 모두 실행
            vtol_ana = VTOLAnalyzer(self.dataset)
            vtol_ana.analyze_transition_performance()
            
            fw_ana = FixedWingAnalyzer(self.dataset)
            fw_ana.analyze_tecs_performance()
            
            mc_ana = MulticopterAnalyzer(self.dataset)
            mc_ana.analyze_attitude_tracking()
            
        elif frame == "FIXED_WING":
            fw_ana = FixedWingAnalyzer(self.dataset)
            fw_ana.analyze_tecs_performance()
            
        elif frame == "MULTICOPTER":
            mc_ana = MulticopterAnalyzer(self.dataset)
            mc_ana.analyze_attitude_tracking()
            
        else:
            print("[Detector] 지원하지 않는 기체 타입이거나 판별에 실패하여 분석을 건너뜁니다.")
            
        print("-" * 50)
        print("[Detector] 자동 분석 파이프라인이 종료되었습니다.")


# === 단독 통합 테스트 실행 ===
if __name__ == "__main__":
    from engines.io_engine import LogIOEngine
    project_root = os.path.abspath(os.path.join(src_dir, "../"))
    test_file = os.path.join(project_root, "data", "raw", "sample.ulg") 
    
    io_engine = LogIOEngine()
    my_dataset = io_engine.load(test_file)
    
    if my_dataset:
        # 단 한 줄로 전체 분석 자동화 수행!
        detector = FlightTypeDetector(my_dataset)
        detector.run_auto_analysis()
