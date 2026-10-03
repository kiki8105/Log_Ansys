# 다중 로그 포맷 아키텍처

작성일: 2026-10-03

## 목표와 범위

기존 PX4 ULog 분석 기능을 유지하면서 다음 입력을 같은 탐색·그래프 UI에서 연다.

- PX4 ULog (`.ulg`)
- ROS1 bag (`.bag`)
- ROS2 bag (`metadata.yaml` 디렉터리, `.db3`, `.mcap`)
- ArduPilot DataFlash (`.bin`, text `.log`)
- MAVLink telemetry (`.tlog`)
- 일반 표 로그 (`.csv`, `.tsv`, `.json`, `.jsonl`, `.ndjson`, 구조화 `.txt`)

1차 목표는 포맷 공통의 토픽/숫자 신호 탐색, 시계열 그래프, 캐시다. PX4 자동평가를 이름이 비슷한 ROS·ArduPilot 데이터에 억지로 적용하지 않는다. 플랫폼별 의미 매핑과 자동평가는 별도 단계로 확장한다.

## 데이터 흐름

```text
파일 또는 ROS2 bag 디렉터리
        │
        ▼
ReaderRegistry ── 각 reader의 probe(content + extension)
        │
        ├─ ULogReader
        ├─ RosbagReader
        ├─ ArduPilotReader
        └─ TabularReader
        │
        ▼
LoadResult
  ├─ LogDataset → TopicInstance → Signal
  ├─ metadata / parameters / messages / warnings
  └─ capabilities (px4, ros, ardupilot, timeseries ...)
        │
        ├─ fingerprint + reader version 기반 Parquet cache
        ├─ 공통 트리/검색/그래프/CSV export
        └─ capability가 일치하는 전처리·분석기만 실행
```

## 공통 시간 규약

- 모든 topic DataFrame은 UI 호환용 `timestamp_sec`를 가진다.
- 원본 정밀 시각은 가능한 경우 `source_timestamp_ns`로 보존한다.
- `timestamp_sec`는 각 스트림 또는 로그 원점 대비 상대 초다. 큰 epoch 값을 바로 부동소수점 초로 변환하지 않는다.
- ROS는 1차적으로 bag receive time을 사용한다. 메시지의 `header.stamp`는 평탄화된 별도 필드로 남긴다.
- ingest 단계에서 서로 다른 샘플률을 강제 resample하지 않는다.

## 포맷 감지 원칙

확장자는 후보를 좁히는 신호일 뿐 최종 판정이 아니다.

- ULog: `ULog` magic
- ROS1: `#ROSBAG V2.0` header
- ROS2 SQLite: SQLite header와 rosbag schema
- MCAP: MCAP magic
- DataFlash binary: packet sync와 `FMT` record
- DataFlash text: 유효한 `FMT,` 선언
- MAVLink tlog: MAVLink v1/v2 frame marker와 parser 검증
- CSV/JSON: delimiter/header 또는 JSON 구조 sample parse

각 reader는 0~100 confidence와 판정 이유를 반환한다. 동률인 고신뢰 reader가 생기면 자동 추측 대신 명시적 오류로 사용자 선택 지점을 남긴다.

## 캐시

기존 basename 캐시는 같은 이름의 다른 로그가 충돌하고 원본 변경을 감지하지 못한다. v2 캐시는 다음 값을 해시한다.

- 절대 source 경로
- 파일 크기·mtime·앞/뒤 sample hash 또는 ROS2 디렉터리 member 목록
- reader ID와 reader version
- cache schema version
- import options

토픽 파일명은 토픽 문자열을 직접 쓰지 않고 SHA-256 축약값을 사용한다. 원본 토픽명과 instance는 `manifest.json`에 기록한다. 임시 디렉터리를 완성한 뒤 공개해 중단된 쓰기를 정상 캐시로 오인하지 않는다. 기존 ULog Parquet 캐시는 원본 보존이 확인될 때까지 읽기 호환으로 유지한다.

## 의존성과 배포 판단

- `pyulog`: 기존 PX4 reader
- `rosbags`: ROS 설치 없이 ROS1/ROS2 SQLite3/MCAP 읽기
- `pymavlink`: DataFlash 및 MAVLink tlog 읽기
- Polars/표준 라이브러리: CSV/JSON 계열

`pymavlink`의 DataFlash 코드가 포함된 EXE를 외부 배포할 때는 해당 라이선스 의무를 릴리스 체크리스트에서 확인한다. PyInstaller는 필요한 MAVLink dialect만 우선 포함하고 실제 샘플 로그를 frozen EXE에서 여는 통합 테스트를 수행한다.

## 단계와 재검토 지점

1. reader 경계, 공통 LoadResult, 안전한 캐시, ULog 회귀 호환
2. ROS/ArduPilot/표 로그의 탐색·그래프
3. semantic alias (`attitude.roll`, `position.z`, `gps.fix` 등)
4. ArduPilot 및 ROS별 분석 프로파일
5. 대용량 Image/PointCloud의 지연 로딩과 topic 선택 import

로그가 수십 GB 규모로 늘거나 Image/PointCloud 분석 요구가 생기면 현재의 전체 수치 필드 메모리 적재를 재검토한다. 그 시점에는 `inspect()`와 선택적 `load()`를 분리하고 chunk/streaming cache를 도입한다.
