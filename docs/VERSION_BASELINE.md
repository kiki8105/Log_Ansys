# 공개 기준본

기준일: 2026-10-03

## 결론

공개 기준본은 이 저장소의 현재 `main` 브랜치다. 저장소 이름은 `Log_Ansys`,
사용자에게 표시되는 프로그램 이름은 `Log ansys`다.

- 실행 진입점: `src/main.py`
- GUI 기준본: `src/gui/main_window.py`
- Windows 빌드 정의: `build.py`, `Log_ansys.spec`
- 기본 출시 정책: `multiformat_stable`
- ULG 전용 문제 격리 정책: `ulg_stable`
- 이전 스크립트 호환 정책: `multiformat_preview`

이 공개 이력은 회사 로고, 회사명, 특정 기체 치수, 운영 위치, 실제 비행 로그,
로컬 캐시와 개인 경로를 포함하지 않는 정제된 스냅샷에서 시작한다.

## 기준본 관리 원칙

1. 수동 복제 파일을 소스 폴더에 두지 않고 Git 이력으로 변경을 관리한다.
2. 실제 로그·지도·캐시·빌드 결과는 `.gitignore` 규칙에 따라 공개 저장소에서 제외한다.
3. 기본 아이콘은 중립적인 공개 SVG만 포함한다. 사용자 로고와 Windows `.ico`,
   splash 이미지는 로컬에서 교체할 수 있다.
4. 릴리스 전 전체 자동 테스트와 frozen executable smoke test를 다시 실행한다.
5. 검증 결과와 알려진 제한은 `docs/ULG_GRAPH_ACCEPTANCE_PLAN.md`에 기록한다.
