"""Tests for the FW depth expansion: airframe-aware attitude loop,
TECS altitude/airspeed inference, and FW parameter mappings.

Exercises the inference functions and recommender end-to-end against the
production YAML configs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.parameter_recommender import ParameterRecommender  # noqa: E402
from analysis.tuning_inference import TuningInferenceEngine  # noqa: E402


@pytest.fixture(scope="module")
def production_config():
    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML not installed")
    cfg_path = PROJECT_ROOT / "config" / "parameter_recommendations.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def tuning_rules():
    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML not installed")
    rules_path = PROJECT_ROOT / "config" / "tuning_rules.yaml"
    with open(rules_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def inference(tuning_rules):
    return TuningInferenceEngine(tuning_rules)


# ---------------------------------------------------------------------------
# Airframe-aware attitude loop (FW uses TC semantics)
# ---------------------------------------------------------------------------

def test_fw_attitude_loop_overshoot_emits_tc_increase(inference):
    feats = {
        "overshoot": 35.0,    # > problem 30
        "rise_time": 0.5,
        "settling_time": 1.0,
        "attitude_mae_deg": 1.0,
        "saturation_ratio": 0.0,
        "vibration_severity": 0.0,
        "innovation_ratio": 0.0,
        "sample_count": 1000,
        "maneuver_count": 3.0,
    }
    issue, _ = inference.infer_attitude_loop_issue("roll", feats, airframe="FIXED_WING")
    assert issue["severity"] == "problem"
    # FW: overshoot → slower → TC increase
    assert "attitude TC increase candidate" in issue["tuning_candidates"]
    assert "attitude P decrease candidate" not in issue["tuning_candidates"]


def test_fw_attitude_loop_high_mae_emits_tc_decrease(inference):
    feats = {
        "overshoot": 5.0,
        "rise_time": 0.5,
        "settling_time": 1.0,
        "attitude_mae_deg": 10.0,
        "saturation_ratio": 0.0,
        "vibration_severity": 0.0,
        "innovation_ratio": 0.0,
        "sample_count": 1000,
        "maneuver_count": 3.0,
    }
    issue, _ = inference.infer_attitude_loop_issue("pitch", feats, airframe="FIXED_WING")
    assert issue["severity"] == "problem"
    # FW: needs faster response → TC decrease
    assert "attitude TC decrease candidate" in issue["tuning_candidates"]
    assert "attitude P increase candidate" not in issue["tuning_candidates"]


def test_mc_attitude_still_uses_p_semantics(inference):
    """Regression: MC airframe must keep P-gain candidate texts."""
    feats = {
        "overshoot": 35.0,
        "rise_time": 0.5,
        "settling_time": 1.0,
        "attitude_mae_deg": 1.0,
        "saturation_ratio": 0.0,
        "vibration_severity": 0.0,
        "innovation_ratio": 0.0,
        "sample_count": 1000,
        "maneuver_count": 3.0,
    }
    issue, _ = inference.infer_attitude_loop_issue("roll", feats, airframe="MULTICOPTER")
    assert "attitude P decrease candidate" in issue["tuning_candidates"]
    assert "attitude TC increase candidate" not in issue["tuning_candidates"]


# ---------------------------------------------------------------------------
# TECS altitude response inference
# ---------------------------------------------------------------------------

def test_tecs_altitude_high_mae_is_problem(inference):
    feats = {"altitude_mae_m": 12.0, "altitude_max_abs_error_m": 22.0, "sample_count": 500}
    issue, _ = inference.infer_tecs_altitude_response_issue(feats)
    assert issue["severity"] == "problem"
    assert "FW_T_ALT_TC decrease candidate" in issue["tuning_candidates"]
    assert "FW_T_CLMB_MAX increase candidate" in issue["tuning_candidates"]


def test_tecs_altitude_normal(inference):
    feats = {"altitude_mae_m": 1.5, "altitude_max_abs_error_m": 4.0, "sample_count": 500}
    issue, _ = inference.infer_tecs_altitude_response_issue(feats)
    assert issue["severity"] == "normal"
    assert issue["tuning_candidates"] == []


def test_tecs_altitude_insufficient_samples_unavailable(inference):
    feats = {"altitude_mae_m": 1.0, "altitude_max_abs_error_m": 2.0, "sample_count": 5}
    issue, _ = inference.infer_tecs_altitude_response_issue(feats)
    assert issue["severity"] == "unavailable"


# ---------------------------------------------------------------------------
# TECS airspeed response inference
# ---------------------------------------------------------------------------

def test_tecs_airspeed_problem_mae(inference):
    feats = {
        "airspeed_mae_mps": 6.0,
        "airspeed_max_abs_error_mps": 10.0,
        "sample_count": 500,
        "source": "tecs_status",
    }
    issue, _ = inference.infer_tecs_airspeed_response_issue(feats)
    assert issue["severity"] == "problem"
    assert "FW_T_TAS_TC decrease candidate" in issue["tuning_candidates"]


def test_tecs_airspeed_fallback_residual_only(inference):
    """When tecs_status is absent, residual RMS drives severity and confidence drops."""
    feats = {
        "airspeed_mae_mps": None,
        "airspeed_residual_rms_mps": 2.0,   # above warn 1.5
        "sample_count": 500,
        "source": "airspeed_vs_groundspeed_fallback",
    }
    issue, _ = inference.infer_tecs_airspeed_response_issue(feats)
    assert issue["severity"] == "warning"
    assert issue["confidence"] == 55.0    # fallback path discount


def test_tecs_airspeed_unavailable(inference):
    feats = {"sample_count": 0}
    issue, _ = inference.infer_tecs_airspeed_response_issue(feats)
    assert issue["severity"] == "unavailable"


# ---------------------------------------------------------------------------
# YAML mapping coverage for FW + VTOL forward-flight reuse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "airframe,issue_id,candidate,param,current",
    [
        # FW rate loop
        ("FIXED_WING", "rate_loop_roll", "P decrease candidate", "FW_RR_P", "0.10"),
        ("FIXED_WING", "rate_loop_pitch", "P increase candidate", "FW_PR_P", "0.10"),
        ("FIXED_WING", "rate_loop_yaw", "FF increase candidate", "FW_YR_FF", "0.5"),
        ("FIXED_WING", "rate_loop_roll", "I increase candidate", "FW_RR_I", "0.10"),
        # FW attitude loop (TC semantics)
        ("FIXED_WING", "attitude_loop_roll", "attitude TC increase candidate", "FW_R_TC", "0.4"),
        ("FIXED_WING", "attitude_loop_pitch", "attitude TC decrease candidate", "FW_P_TC", "0.4"),
        # FW TECS altitude
        ("FIXED_WING", "tecs_altitude_response", "FW_T_ALT_TC decrease candidate", "FW_T_ALT_TC", "8.0"),
        ("FIXED_WING", "tecs_altitude_response", "FW_T_CLMB_MAX increase candidate", "FW_T_CLMB_MAX", "5.0"),
        ("FIXED_WING", "tecs_altitude_response", "FW_T_SINK_MIN increase candidate", "FW_T_SINK_MIN", "2.0"),
        # FW TECS airspeed
        ("FIXED_WING", "tecs_airspeed_response", "FW_T_TAS_TC decrease candidate", "FW_T_TAS_TC", "5.0"),
        ("FIXED_WING", "tecs_airspeed_response", "FW_T_SPDWEIGHT increase candidate", "FW_T_SPDWEIGHT", "1.0"),
        ("FIXED_WING", "tecs_airspeed_response", "FW_T_THR_DAMP decrease candidate", "FW_T_THR_DAMP", "0.5"),
        # VTOL reuses FW TECS mappings
        ("VTOL", "tecs_altitude_response", "FW_T_ALT_TC decrease candidate", "FW_T_ALT_TC", "8.0"),
        ("VTOL", "tecs_airspeed_response", "FW_T_TAS_TC decrease candidate", "FW_T_TAS_TC", "5.0"),
    ],
)
def test_yaml_mapping_produces_recommendation(production_config, airframe, issue_id, candidate, param, current):
    rec = ParameterRecommender(production_config)
    issue = {
        "issue_id": issue_id,
        "severity": "warning",
        "confidence": 80.0,
        "airframe": airframe,
        "tuning_candidates": [candidate],
        "metric_value": {},
    }
    out = rec.recommend(issue, {param: current}, airframe)
    assert len(out) == 1, f"expected recommendation for ({airframe}, {issue_id}, {candidate}) -> {param}"
    assert out[0]["param"] == param


def test_fw_yaw_attitude_no_mapping_yields_empty(production_config):
    """FW yaw attitude is intentionally unmapped (rudder mix logic in PX4)."""
    rec = ParameterRecommender(production_config)
    issue = {
        "issue_id": "attitude_loop_yaw",
        "severity": "warning",
        "confidence": 80.0,
        "airframe": "FIXED_WING",
        "tuning_candidates": ["attitude TC increase candidate"],
        "metric_value": {},
    }
    out = rec.recommend(issue, {"FW_R_TC": "0.4"}, "FIXED_WING")
    assert out == []


def test_fw_full_round_trip_attitude_overshoot(inference, production_config):
    """End-to-end: infer FW attitude overshoot, then recommend the FW_R_TC change.

    Under the conservative policy:
        - severity_delta problem = 1.15 (compressed from 1.25)
        - risk_tier = low (stabilizing direction for TC-increase) -> tier mult 1.0
        - max_delta_percent_per_step = 10% -> cap forces ratio to 1.10
        - current 0.4 -> suggested 0.44
    """
    rec = ParameterRecommender(production_config)
    feats = {
        "overshoot": 35.0,
        "rise_time": 0.5,
        "settling_time": 1.0,
        "attitude_mae_deg": 1.0,
        "saturation_ratio": 0.0,
        "vibration_severity": 0.0,
        "innovation_ratio": 0.0,
        "sample_count": 1000,
        "maneuver_count": 3.0,
    }
    issue, _ = inference.infer_attitude_loop_issue("roll", feats, airframe="FIXED_WING")
    out = rec.recommend(issue, {"FW_R_TC": "0.4"}, "FIXED_WING")
    assert len(out) == 1
    item = out[0]
    assert item["param"] == "FW_R_TC"
    assert item["advisory_only"] is False
    assert item["suggested"] == pytest.approx(0.44, rel=1e-3)
    assert item["delta_percent"] == pytest.approx(10.0, abs=1e-2)
    assert any("캡" in note for note in item["safety_notes"])
    # Staged-apply guidance must accompany every numeric suggestion.
    assert item["staged_apply"]
    assert any("단계" in note for note in item["safety_notes"])
