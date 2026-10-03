# 저장소 및 로컬 데이터 구조

소스, 입력 데이터, 파생 데이터와 배포 산출물을 다음처럼 분리한다.

```text
Log_Ansys/
├─ src/                 애플리케이션 소스
├─ tests/               자동화 테스트와 소형 공개 fixture
├─ config/              분석 규칙과 기본 설정
├─ assets/              재배포 가능한 중립 자산
│  ├─ icons/            기본 SVG와 사용자 아이콘 안내
│  └─ splash/           사용자 splash 안내
├─ docs/                아키텍처와 검증 문서
├─ scripts/             재현 가능한 검증 도구
├─ data/                로컬 입력 자료(Git 제외)
│  ├─ raw/              ULog, ROS bag, DataFlash 등 원본 로그
│  └─ maps/             KML/KMZ 지도 자료
├─ runtime/             캐시·로그·검증 산출물(Git 제외)
├─ build/               PyInstaller 임시 산출물(Git 제외)
└─ dist/                배포 산출물(Git 제외)
```

## 운영 원칙

1. 실제 로그와 지도 파일은 Git에 올리지 않는다.
2. 캐시는 원본이 아니며 재생성 가능해야 한다.
3. 배포본은 `dist/`, 중간 빌드 파일은 `build/`에만 둔다.
4. 공개 기본 아이콘은 `assets/icons/default_app.svg`다.
5. Settings → Icons에서 선택한 사용자 자산 경로는 로컬 설정에만 저장한다.
6. Windows 빌드 아이콘은 `assets/icons/custom_app.ico`, 선택 splash는
   `assets/splash/custom_splash.png`로 교체할 수 있으며 기본 저장소에는 포함하지 않는다.
7. 포맷별 캐시는 원본 fingerprint, reader ID·버전, schema·옵션을 키에 포함한다.
8. `copy.py`, `*_old.py` 같은 수동 복제본은 저장소에 두지 않는다.
