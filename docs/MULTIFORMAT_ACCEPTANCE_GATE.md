# 다중 포맷 실로그 승인 게이트

`scripts/validate_format_acceptance.py`는 지원 표기만 있고 실제 원본은 없는
상태를 통과로 잘못 판정하지 않기 위한 릴리스 게이트다. 지원되는 모든
입력 변형이 아닌 다음 **핵심 9개 변형**을
독립적으로 검증한다.

| 슬롯 | 필요 원본 | 기대 내부 포맷 |
|---|---|---|
| `px4_ulog` | PX4 `.ulg` | `px4_ulog` |
| `ros1_bag` | ROS 1 `.bag` | `ros1_bag` |
| `ros2_db3` | ROS 2 SQLite3 `.db3` | `ros2_bag` |
| `ros2_mcap` | ROS 2 `.mcap` | `ros2_bag` |
| `ardupilot_bin` | DataFlash binary `.bin` | `ardupilot_dataflash` |
| `ardupilot_log` | DataFlash text `.log` | `ardupilot_dataflash` |
| `mavlink_tlog` | MAVLink telemetry `.tlog` | `mavlink_tlog` |
| `csv` | `.csv` | `tabular` |
| `json` | JSON records `.json` | `tabular` |

현재 reader가 열 수 있지만 아직 이 릴리스 게이트에 별도 슬롯이 없는 변형은
다음과 같다.

- ROS 2 bag 디렉터리와 `metadata.yaml` 진입점
- 표 로그 `.tsv`, `.txt`, `.jsonl`, `.ndjson`

이 변형은 독립 실원본 게이트가 추가되기 전까지 `core 9` 통과를 근거로
릴리스 승인되었다고 표현하지 않는다.

## 판정 계층

각 형식에 대해 다음을 순서대로 확인한다.

1. 확장자가 아닌 내용 기반 `probe`가 예상 reader와 format ID를 선택하는지
2. cold parse가 공통 `LogDataset`을 생성하는지
3. 최소 2개의 유한한 시간/값 쌍을 갖는 숫자 신호가 있는지
4. warm cache가 실제 cache hit이며 숫자 구조가 cold parse와 같은지
5. 하나의 `LogIOEngine` 실행과 하나의 Qt workspace에 모든 원본이 함께 존재하며
   선택 신호를 표시하는지
6. 손상 파일이 거부되고 기존 dataset, tree, 곡선, 전역 시간 범위가 변하지 않는지
7. 선택 실행 시 같은 원본을 packaged EXE 프로세스 내부에서 적재하는지

offscreen Qt 검사는 세션 트랜잭션과 신호 렌더 계약을 확인하지만, 실제
GPU, DPI, 물리 마우스 조작의 수동 승인을 대체하지 않는다.

## 비공개 원본 manifest

실제 로그 경로와 해시는 Git에 추적하지 않는다. 아래 형식의 manifest를
`runtime/format-acceptance-fixtures.json`에 저장하고 원본은 `data/raw/`나 저장소
밖의 읽기 전용 경로에 둔다.

```json
{
  "schema_version": 1,
  "fixtures": {
    "px4_ulog": {"path": "D:/private/px4.ulg", "provenance": "real"},
    "ros1_bag": {"path": "D:/private/ros1.bag", "provenance": "real"},
    "ros2_db3": {"path": "D:/private/ros2.db3", "provenance": "real"},
    "ros2_mcap": {"path": "D:/private/ros2.mcap", "provenance": "real"},
    "ardupilot_bin": {"path": "D:/private/ardupilot.bin", "provenance": "real"},
    "ardupilot_log": {"path": "D:/private/ardupilot.log", "provenance": "real"},
    "mavlink_tlog": {"path": "D:/private/telemetry.tlog", "provenance": "real"},
    "csv": {"path": "D:/private/export.csv", "provenance": "real"},
    "json": {"path": "D:/private/export.json", "provenance": "real"}
  }
}
```

`provenance` 값은 `real`, `synthetic`, `unknown` 중 하나다. 합성 원본은 reader
회귀 테스트를 통과할 수 있지만 해당 포맷의 릴리스 상태는 항상
`manual_required`다. 원본이 없으면 `missing_fixture`로 보고한다.

## 실행

먼저 reader 회귀와 혼합 workspace 경로만 점검하려면 8개 합성 포맷 원본을
생성할 수 있다. `--ulg`는 복사하지 않고 제공된 실제 ULG 경로를 manifest에만
참조한다. 출력 디렉터리는 새 경로이거나 비어 있어야 한다.

```powershell
python scripts\generate_synthetic_acceptance_fixtures.py `
  --output-dir runtime\synthetic-format-fixtures `
  --ulg "C:\private\flight.ulg"

python scripts\validate_format_acceptance.py `
  --manifest runtime\synthetic-format-fixtures\manifest.json
```

이 조합이 모두 동작해도 합성 ROS·ArduPilot·CSV·JSON은 `manual_required`로
남는다. 릴리스 승인은 아래의 실제 원본 manifest로 다시 실행해야 한다.

```powershell
python scripts\validate_format_acceptance.py `
  --manifest runtime\format-acceptance-fixtures.json
```

특정 원본을 Git에 경로를 남기지 않고 즉시 지정할 수도 있다.

```powershell
python scripts\validate_format_acceptance.py `
  --fixture "px4_ulog=C:\private\flight.ulg" `
  --provenance "px4_ulog=real"
```

현재 소스로 새로 빌드한 EXE의 실제 parser 의존성까지 검증할 때는 다음과 같이
실행한다.

```powershell
python scripts\validate_format_acceptance.py `
  --manifest runtime\format-acceptance-fixtures.json `
  --frozen-exe "dist\Log ansys\Log ansys.exe" `
  --require-frozen
```

기존 EXE가 `--acceptance-load-report` 진입점을 포함하지 않으면 창을 열고 timeout이
발생할 수 있으므로, 반드시 현재 소스를 다시 빌드한 뒤 실행한다.

## 결과 계약

결과는 기본적으로 `runtime/test-artifacts/format-acceptance/report.json`에
저장된다.

| 상태 | 의미 | 종료 코드 |
|---|---|---:|
| `pass` | 9개 슬롯의 실제 원본과 필수 자동 항목이 모두 통과 | 0 |
| `fail` | 제공된 원본의 탐지, 적재, 신호, cache, 세션 또는 격리 계약 실패 | 1 |
| `incomplete` | 하나 이상의 원본 슬롯이 없음 | 2 |
| `manual_required` | 기술 검사는 통과했지만 합성/출처 미상 원본이거나 필수 실화면 검사를 생략 | 2 |

게이트가 만드는 캐시, 손상 fixture, 스크린샷과 보고서는 모두 ignored
`runtime/`에만 남으며 원본을 수정하지 않는다.
