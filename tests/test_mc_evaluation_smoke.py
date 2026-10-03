from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_mc_log import run_mc_validation  # noqa: E402


def test_mc_evaluation_smoke_from_env():
    log_path = os.environ.get("PX4_MC_LOG", "").strip()
    if not log_path:
        pytest.skip("PX4_MC_LOG 환경변수가 없어 MC 로그 검증을 건너뜁니다.")

    result = run_mc_validation(
        log_path=Path(log_path).expanduser().resolve(),
        thresholds_path=PROJECT_ROOT / "config" / "thresholds.yaml",
        strict_airframe=True,
    )

    assert result["detected_airframe"] == "MULTICOPTER"
    summary = result["summary"]
    assert summary["item_count"] > 0
    assert summary["overall_status"] in {"good", "warning", "problem", "unavailable"}
