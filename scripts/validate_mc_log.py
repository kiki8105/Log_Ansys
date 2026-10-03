from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.evaluator import AutoEvaluationEngine  # noqa: E402
from engines.io_engine import LogIOEngine  # noqa: E402
from analysis.detector import FlightTypeDetector  # noqa: E402


MC_ALIASES = {"MULTICOPTER", "MC", "MULTI_COPTER", "MULTIROTOR", "ROTARY_WING"}


def normalize_airframe(value: Any) -> str:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    return "MULTICOPTER" if text in MC_ALIASES else text


def ensure_mc_thresholds_loaded(evaluator: AutoEvaluationEngine) -> None:
    cfg = evaluator.cfg if isinstance(evaluator.cfg, dict) else {}
    metrics_airframe = cfg.get("metrics_airframe", {})
    weights_airframe = cfg.get("weights_airframe", {})
    missing = []
    if "MULTICOPTER" not in metrics_airframe:
        missing.append("metrics_airframe.MULTICOPTER")
    if "MULTICOPTER" not in weights_airframe:
        missing.append("weights_airframe.MULTICOPTER")
    if missing:
        raise RuntimeError(
            "MC 전용 threshold 설정을 찾지 못했습니다: "
            + ", ".join(missing)
            + "\nconfig/thresholds.yaml에 MULTICOPTER 항목이 있어야 합니다."
        )


def summarize_result(result: dict) -> dict:
    summary = result.get("summary", {}) if isinstance(result, dict) else {}
    return {
        "aircraft_type": result.get("aircraft_type"),
        "overall_status": result.get("overall_status"),
        "overall_status_label": result.get("overall_status_label"),
        "overall_score": result.get("overall_score"),
        "common_score": result.get("common_score"),
        "airframe_score": result.get("airframe_score"),
        "item_count": len(result.get("items", []) or []),
        "good": summary.get("good", 0),
        "warning": summary.get("warning", 0),
        "problem": summary.get("problem", 0),
        "unavailable": summary.get("unavailable", 0),
        "top_issues": result.get("top_issues", [])[:5],
    }


def run_mc_validation(
    log_path: Path,
    thresholds_path: Path,
    strict_airframe: bool = True,
    min_score: float | None = None,
) -> dict:
    if not log_path.is_file():
        raise FileNotFoundError(f"MC ULog 파일을 찾지 못했습니다: {log_path}")
    if not thresholds_path.is_file():
        raise FileNotFoundError(f"threshold 설정 파일을 찾지 못했습니다: {thresholds_path}")

    io_engine = LogIOEngine()
    dataset = io_engine.load(str(log_path))
    if dataset is None:
        raise RuntimeError("로그 로드 결과가 비어 있습니다. 파일 손상 또는 parser 오류 가능성이 있습니다.")

    detector = FlightTypeDetector(dataset)
    detected = normalize_airframe(detector.detect())
    if strict_airframe and detected != "MULTICOPTER":
        raise RuntimeError(
            f"이 로그는 MC로 판정되지 않았습니다. detected={detected!r}\n"
            "MC 검증에는 vehicle_status.vehicle_type이 rotary-wing 계열인 로그가 필요합니다.\n"
            "비MC 로그로 엔진만 테스트하려면 --allow-non-mc 옵션을 사용하세요."
        )

    evaluator = AutoEvaluationEngine(thresholds_path=str(thresholds_path))
    ensure_mc_thresholds_loaded(evaluator)

    result = evaluator.evaluate(
        dataset,
        file_name=log_path.name,
        aircraft_type="MULTICOPTER",
    )
    if not isinstance(result, dict) or not result.get("items"):
        raise RuntimeError("평가 결과가 비어 있습니다. evaluator.evaluate() 출력 구조를 확인해야 합니다.")

    summary = summarize_result(result)
    if min_score is not None:
        score = summary.get("overall_score")
        if score is None or float(score) < float(min_score):
            raise RuntimeError(f"MC 평가 점수가 기준보다 낮습니다. score={score}, min_score={min_score}")

    return {
        "log_path": str(log_path),
        "thresholds_path": str(thresholds_path),
        "detected_airframe": detected,
        "summary": summary,
        "raw_result": result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MC(Multicopter) ULog 자동평가 스모크 검증 스크립트",
    )
    parser.add_argument(
        "--log",
        default=os.environ.get("PX4_MC_LOG", ""),
        help="검증할 MC .ulg 파일 경로입니다. 생략하면 PX4_MC_LOG 환경변수를 사용합니다.",
    )
    parser.add_argument(
        "--thresholds",
        default=str(PROJECT_ROOT / "config" / "thresholds.yaml"),
        help="MC threshold가 포함된 YAML 파일 경로입니다.",
    )
    parser.add_argument(
        "--allow-non-mc",
        action="store_true",
        help="로그가 MC로 자동 판정되지 않아도 MULTICOPTER threshold로 평가를 강제 실행합니다.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="선택 사항입니다. overall_score가 이 값보다 낮으면 실패 처리합니다.",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="평가 요약과 원본 결과를 JSON으로 저장할 경로입니다.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.log:
        print(
            "[SKIP] MC 로그 경로가 지정되지 않았습니다.\n"
            "       사용 예: python scripts/validate_mc_log.py --log C:\\logs\\MC_SAMPLE.ulg\n"
            "       또는: set PX4_MC_LOG=C:\\logs\\MC_SAMPLE.ulg"
        )
        return 0

    try:
        output = run_mc_validation(
            log_path=Path(args.log).expanduser().resolve(),
            thresholds_path=Path(args.thresholds).expanduser().resolve(),
            strict_airframe=not args.allow_non_mc,
            min_score=args.min_score,
        )
    except Exception as exc:
        print(f"[FAIL] MC 검증 실패: {exc}")
        return 2

    summary = output["summary"]
    print("[OK] MC 자동평가 스모크 검증 완료")
    print(f"  - detected_airframe: {output['detected_airframe']}")
    print(f"  - overall_status: {summary.get('overall_status_label')} ({summary.get('overall_status')})")
    print(f"  - overall_score: {summary.get('overall_score')}")
    print(f"  - common_score / airframe_score: {summary.get('common_score')} / {summary.get('airframe_score')}")
    print(
        "  - items good/warning/problem/unavailable: "
        f"{summary.get('good')}/{summary.get('warning')}/{summary.get('problem')}/{summary.get('unavailable')}"
    )
    for idx, issue in enumerate(summary.get("top_issues", []), start=1):
        print(f"  - top_issue_{idx}: {issue}")

    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"[OK] JSON 보고서 저장: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
