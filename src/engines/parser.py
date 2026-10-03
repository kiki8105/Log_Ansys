# src/engines/parser.py

import os
import sys
import polars as pl
from pyulog import ULog

# 단독으로 이 스크립트를 실행할 때, 상위 폴더인 'src'를 인식하게 만드는 코드입니다.
# (이 코드가 없으면 core.log_model을 찾지 못해 에러가 발생합니다.)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# 앞서 우리가 뼈대로 만들었던 데이터 모델들을 불러옵니다.
from core.log_model import LogDataset, TopicInstance, Signal

class ULGParser:
    """
    .ulg 파일을 읽어 우리가 설계한 LogDataset 형태로 변환하는 엔진입니다.
    """
    def __init__(self):
        pass

    def parse(self, file_path: str) -> LogDataset:
        """
        메인 파싱 함수: 파일을 읽고 데이터 모델을 조립합니다.
        """
        print(f"[{os.path.basename(file_path)}] 로그 분석을 시작합니다...")
        
        # 1. 빈 전체 데이터셋 지휘소(LogDataset) 생성
        dataset = LogDataset(
            source_format="px4_ulog",
            source_path=file_path,
            capabilities={"px4", "timeseries"},
        )
        
        # 2. pyulog를 통해 ULG 파일 해독
        ulog = ULog(file_path)
        
        # 3. 로그 안의 모든 Topic 블록 순회
        for data in ulog.data_list:
            base_name = data.name        # 예: 'sensor_gyro'
            instance_id = data.multi_id  # 예: 0 또는 1
            
            # Polars DataFrame으로 변환 및 timestamp(마이크로초 -> 초) 정규화
            df = pl.DataFrame(data.data)
            df = df.with_columns(
                (pl.col("timestamp") / 1e6).alias("timestamp_sec"),
                (pl.col("timestamp").cast(pl.UInt64) * 1000).alias("source_timestamp_ns"),
            )
            
            # 4. 우리가 설계한 규격화된 박스(TopicInstance) 생성
            topic_inst = TopicInstance(
                base_name=base_name,
                instance_id=instance_id,
                dataframe=df
            )
            
            # 5. DataFrame 안의 각 열(column)을 Signal 선으로 분리하여 저장 (timestamp는 제외)
            for col_name in df.columns:
                if col_name not in ["timestamp", "timestamp_sec", "source_timestamp_ns"]:
                    # 각 열의 데이터를 뽑아 Signal 객체로 만들어 Topic에 담습니다.
                    signal = Signal(name=col_name, data=df[col_name])
                    topic_inst.signals[col_name] = signal
            
            # 6. 완성된 Topic 상자를 전체 데이터셋에 등록
            dataset.add_topic(topic_inst)
            
        print("[완료] ULG 파싱 및 데이터 모델 맵핑 성공!")
        return dataset

# === 단독 테스트 실행 부분 ===
if __name__ == "__main__":
    # 프로젝트의 최상위 폴더 경로를 자동으로 찾습니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    
    # data/raw 폴더 안에 넣은 실제 파일 이름을 적어주세요.
    log_filename = "sample.ulg" # <--- 준비하신 파일 이름으로 변경하세요!
    
    test_file = os.path.join(project_root, "data", "raw", log_filename)
    
    if not os.path.exists(test_file):
        print(f"에러: {test_file} 파일을 찾을 수 없습니다. data/raw 폴더에 파일이 있는지 확인해주세요.")
    else:
        # 우리가 만든 파서 엔진 객체 생성
        parser = ULGParser()
        
        # 파싱 시작 (LogDataset 박스에 데이터 채워넣기)
        my_log_data = parser.parse(test_file)
        
        # 전체 데이터셋 요약 출력
        print("\n")
        my_log_data.print_summary()
        
        # 'vehicle_attitude' 0번 센서의 데이터가 정상적으로 들어왔는지 확인
        attitude = my_log_data.get_topic("vehicle_attitude", 0)
        if attitude:
            print("\n[성공] vehicle_attitude_0 의 보유 신호(Signal) 목록:")
            print(list(attitude.signals.keys()))
