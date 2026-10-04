# Log ansys

Windows 데스크톱에서 비행 로그의 시계열, 2D/3D 경로와 자동 평가를 확인하는
PySide6 기반 분석 도구입니다. 기본 실행에서 PX4 ULog, ROS1/ROS2,
ArduPilot DataFlash/MAVLink 및 CSV/JSON 형식 reader를 함께 제공합니다.

## 빠른 시작

검증 기준 환경은 Windows와 Python 3.14.5입니다.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src\main.py
```

기본 실행은 PX4 ULG, ROS1 bag, ROS2 SQLite3/MCAP, ArduPilot DataFlash,
MAVLink TLOG, CSV/JSON 계열을 같은 탐색·그래프 UI에 적재합니다. 문제 격리나
ULG 전용 검증이 필요한 경우에만 실행 전에 정책을 축소합니다.

```powershell
$env:PX4_LOG_FORMAT_POLICY = "ulg_stable"
python src\main.py
```

다중 포맷 모드에서 각 로그의
`timestamp_sec`는 원점 대비 상대 초이며, 서로 다른 형식의 신호를 한 패널에
겹칠 때 원본 로그를 범례로 구분합니다. PX4 자동 평가는 PX4 capability가 없는
ROS/ArduPilot 로그에 적용하지 않습니다.

## 검증

```powershell
$env:PYTHONPATH = "src"
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest -q
python src\main.py --smoke-test
```

비공개 ULog를 이용한 그래프 승인 검사는 원본을 저장소 밖에 둔 채 실행합니다.

```powershell
python scripts\validate_ulg_graph_acceptance.py --cache-cycles 3 `
  "C:\path\to\validation_A.ulg" "C:\path\to\validation_B.ulg"
```

같은 창에서 PX4·ArduPilot·ROS 로그를 함께 여는 계약은 각 원본 경로를
명시해 별도로 검증합니다. 원본 경로는 스크립트나 Git 추적 파일에 저장되지 않지만,
검증 시 생성되는 ignored `runtime/` 로컬 보고서에는 재현성을 위해 기록됩니다.

```powershell
python scripts\validate_mixed_format_workspace.py `
  --ulg "C:\path\to\px4.ulg" `
  --ardupilot "C:\path\to\ardupilot.tlog" `
  --ros1 "C:\path\to\ros1.bag" `
  --ros2 "C:\path\to\rosbag2.db3"
```

핵심 9개 형식 변형의 실제 원본 존재와 probe, cold/warm 적재, 숫자
신호, 혼합 workspace, 손상 파일 격리를 하나의 JSON 매트릭스로 검증하려면
[다중 포맷 실로그 승인 게이트](docs/MULTIFORMAT_ACCEPTANCE_GATE.md)를 사용합니다. 실제 원본이
없는 형식은 통과로 처리하지 않고 `missing_fixture`로 보고합니다. ROS 2 bag
디렉터리/`metadata.yaml`과 `.tsv`, `.txt`, `.jsonl`, `.ndjson`은 reader 지원 대상이지만
아직 이 핵심 9개 릴리스 게이트의 독립 승인 슬롯에 포함되지 않습니다.

승인 항목과 정량 기준은 [ULG 그래프 승인 계획](docs/ULG_GRAPH_ACCEPTANCE_PLAN.md),
reader·캐시 경계는 [다중 포맷 아키텍처](docs/MULTIFORMAT_ARCHITECTURE.md)를
참조하십시오.

## Windows 빌드

```powershell
python -m pip install -r requirements-build.txt
python build.py --clean
```

기본 결과는 `dist/Log ansys/Log ansys.exe`를 포함한 onedir 번들입니다. 빌드 스크립트는 완료 후
frozen executable의 핵심 import smoke test를 자동 실행합니다.

## 데이터와 저장소 규칙

- 원본 로그는 `data/raw/`, 로컬 지도는 `data/maps/`에 두며 Git에 포함하지 않습니다.
- 캐시·보고서·스크린샷은 `runtime/`, 빌드 산출물은 `build/`와 `dist/`에 생성됩니다.
- 실제 로그 경로, 위치, 해시와 평가 결과는 공개 문서가 아닌 로컬 결과물에만 둡니다.
- 샘플을 추가할 때는 재배포 권한이 확인된 소형 합성 데이터만 `tests/fixtures/`에 둡니다.

자세한 디렉터리 정책은 [저장 구조](docs/STORAGE_LAYOUT.md)를 참조하십시오.

## 분석 결과의 한계

이 프로그램의 점수와 권고는 정비·감항·비행 안전 승인을 대신하지 않습니다.
결측 신호, 잘못된 단위, 기체별 설정, 비행 단계와 환경 조건에 따라 결과가 달라질
수 있으므로 원시 신호와 비행 맥락을 함께 검토해야 합니다. 두 개의 검증 로그만으로
임계값의 일반적 타당성을 확정하지 않습니다.

위성 지도 기능은 외부 타일 공급자에게 네트워크 요청을 보내므로 요청 좌표와 IP가
해당 공급자에게 전달될 수 있습니다. 민감한 비행 위치를 다룰 때는 기능 사용 여부와
공급자 약관을 먼저 확인하십시오.

## 아이콘과 라이선스

공개 기본 자산은 중립적인 `assets/icons/default_app.svg` 하나입니다. 사용자 로고와
Windows 실행 아이콘, splash 이미지는 로컬에서 교체할 수 있으며 Git에는 포함되지
않습니다. 자세한 방법은 `assets/icons/README.md`와 `assets/splash/README.md`를
참조하십시오.

프로젝트 자체 라이선스는 아직 지정하지 않았습니다. 명시적인 라이선스가 추가되기
전까지 공개 복제·재배포 권한이 부여된 것으로 간주하지 마십시오. PySide6와
pymavlink 등 제3자 구성요소의 배포 의무는 별도로 확인해야 합니다.
