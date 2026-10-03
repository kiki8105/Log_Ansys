"""Tests for the MC depth expansion: attitude loop, rate-MAX, acc/tilt limits,
FF candidate addition, and the corresponding parameter_recommendations.yaml
mappings.

These tests exercise the inference functions and the recommender end-to-end
without requiring a real PX4 log file.
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
# FF candidate addition to rate loop
# ---------------------------------------------------------------------------

def test_rate_loop_emits_ff_candidate_on_lag_without_overshoot(inference):
    """rise_slow + no overshoot should trigger FF candidate."""
    feats = {
        "overshoot": 5.0,        # below warn 18
        "rise_time": 1.4,        # above problem 1.2
        "settling_time": 1.0,
        "steady_state_error": 0.05,
        "oscillation_index": 5.0,
        "dominant_frequency": 0.0,
        "saturation_ratio": 0.0,
        "vibration_severity": 0.0,
        "control_effort_variation": 0.0,
        "innovation_ratio": 0.0,
        "sample_count": 1000,
        "maneuver_count": 3.0,
    }
    issue, _ = inference.infer_rate_loop_issue("roll", feats)
    assert "FF increase candidate" in issue["tuning_candidates"]


def test_rate_loop_no_ff_when_overshoot_present(inference):
    """rise_slow with overshoot_hi shouldn't emit FF (FF would worsen overshoot)."""
    feats = {
        "overshoot": 40.0,
        "rise_time": 1.4,
        "settling_time": 2.0,
        "steady_state_error": 0.05,
        "oscillation_index": 5.0,
        "dominant_frequency": 0.0,
        "saturation_ratio": 0.0,
        "vibration_severity": 0.0,
        "control_effort_variation": 0.0,
        "innovation_ratio": 0.0,
        "sample_count": 1000,
        "maneuver_count": 3.0,
    }
    issue, _ = inference.infer_rate_loop_issue("roll", feats)
    assert "FF increase candidate" not in issue["tuning_candidates"]


# ---------------------------------------------------------------------------
# Attitude loop inference
# ---------------------------------------------------------------------------

def test_attitude_loop_high_mae_emits_p_increase(inference):
    feats = {
        "overshoot": 5.0,
        "rise_time": 0.5,
        "settling_time": 1.0,
        "attitude_mae_deg": 10.0,   # above problem 8
        "saturation_ratio": 0.0,
        "vibration_severity": 0.0,
        "innovation_ratio": 0.0,
        "sample_count": 1000,
        "maneuver_count": 3.0,
    }
    issue, _ = inference.infer_attitude_loop_issue("roll", feats)
    assert issue["severity"] == "problem"
    assert "attitude P increase candidate" in issue["tuning_candidates"]


def test_attitude_loop_high_overshoot_emits_p_decrease(inference):
    feats = {
        "overshoot": 35.0,  # above problem 30
        "rise_time": 0.5,
        "settling_time": 1.0,
        "attitude_mae_deg": 1.0,
        "saturation_ratio": 0.0,
        "vibration_severity": 0.0,
        "innovation_ratio": 0.0,
        "sample_count": 1000,
        "maneuver_count": 3.0,
    }
    issue, _ = inference.infer_attitude_loop_issue("pitch", feats)
    assert issue["severity"] == "problem"
    assert "attitude P decrease candidate" in issue["tuning_candidates"]


def test_attitude_loop_normal_response_returns_normal(inference):
    feats = {
        "overshoot": 8.0,
        "rise_time": 0.5,
        "settling_time": 1.0,
        "attitude_mae_deg": 1.5,
        "saturation_ratio": 0.0,
        "vibration_severity": 0.0,
        "innovation_ratio": 0.0,
        "sample_count": 1000,
        "maneuver_count": 3.0,
    }
    issue, _ = inference.infer_attitude_loop_issue("yaw", feats)
    assert issue["severity"] == "normal"


def test_attitude_loop_suppresses_p_when_saturated(inference):
    feats = {
        "overshoot": 35.0,
        "rise_time": 0.5,
        "settling_time": 1.0,
        "attitude_mae_deg": 1.0,
        "saturation_ratio": 25.0,   # >= 8.0 -> sat_hi
        "vibration_severity": 0.0,
        "innovation_ratio": 0.0,
        "sample_count": 1000,
        "maneuver_count": 3.0,
    }
    issue, _ = inference.infer_attitude_loop_issue("roll", feats)
    # P-change should NOT be among candidates; saturation-first should.
    assert "attitude P decrease candidate" not in issue["tuning_candidates"]
    assert "actuator authority check first" in issue["tuning_candidates"]


# ---------------------------------------------------------------------------
# Rate-MAX saturation inference
# ---------------------------------------------------------------------------

def test_rate_limit_problem_emits_max_increase_and_attitude_p_decrease(inference):
    feats = {
        "rate_max_param": "MC_ROLLRATE_MAX",
        "rate_max_value": 220.0,
        "saturation_ratio_percent": 22.0,
        "max_abs_rate_setpoint": 215.0,
        "sample_count": 1000,
        "param_present": True,
    }
    issue, _ = inference.infer_rate_limit_issue("roll", feats)
    assert issue["severity"] == "problem"
    assert "rate MAX increase candidate" in issue["tuning_candidates"]
    assert "attitude P decrease candidate" in issue["tuning_candidates"]


def test_rate_limit_param_absent_returns_unavailable(inference):
    feats = {
        "rate_max_param": "MC_ROLLRATE_MAX",
        "rate_max_value": None,
        "saturation_ratio_percent": None,
        "max_abs_rate_setpoint": 150.0,
        "sample_count": 1000,
        "param_present": False,
    }
    issue, _ = inference.infer_rate_limit_issue("roll", feats)
    assert issue["severity"] == "unavailable"


# ---------------------------------------------------------------------------
# Acc/Tilt limit inference
# ---------------------------------------------------------------------------

def test_acc_limit_worst_axis_drives_severity(inference):
    acc_metrics = {
        "horizontal": {"param_present": True, "saturation_ratio_percent": 20.0, "param": "MPC_ACC_HOR_MAX", "limit": 5.0, "max_cmd": 6.0},
        "up": {"param_present": True, "saturation_ratio_percent": 2.0, "param": "MPC_ACC_UP_MAX", "limit": 4.0, "max_cmd": 3.0},
        "down": {"param_present": False, "saturation_ratio_percent": None, "param": "MPC_ACC_DOWN_MAX", "limit": None, "max_cmd": 1.0},
    }
    issue, _ = inference.infer_acc_limit_issue(acc_metrics)
    assert issue["severity"] == "problem"
    assert "MPC_ACC_HOR_MAX increase candidate" in issue["tuning_candidates"]


def test_tilt_limit_normal(inference):
    tilt_metrics = {"param_present": True, "saturation_ratio_percent": 1.0, "limit_deg": 45.0, "max_tilt_deg": 30.0, "sample_count": 1000}
    issue, _ = inference.infer_tilt_limit_issue(tilt_metrics)
    assert issue["severity"] == "normal"


def test_tilt_limit_warning(inference):
    tilt_metrics = {"param_present": True, "saturation_ratio_percent": 8.0, "limit_deg": 45.0, "max_tilt_deg": 44.5, "sample_count": 1000}
    issue, _ = inference.infer_tilt_limit_issue(tilt_metrics)
    assert issue["severity"] == "warning"
    assert "MPC_TILTMAX_AIR increase candidate" in issue["tuning_candidates"]


# ---------------------------------------------------------------------------
# Recommender mapping coverage (production YAML)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "issue_id,candidate,param,current",
    [
        ("rate_loop_roll", "FF increase candidate", "MC_ROLLRATE_FF", "0.1"),
        ("rate_loop_pitch", "FF increase candidate", "MC_PITCHRATE_FF", "0.1"),
        ("rate_loop_yaw", "FF increase candidate", "MC_YAWRATE_FF", "0.1"),
        ("attitude_loop_roll", "attitude P decrease candidate", "MC_ROLL_P", "8.0"),
        ("attitude_loop_pitch", "attitude P increase candidate", "MC_PITCH_P", "6.0"),
        ("attitude_loop_yaw", "attitude P decrease candidate", "MC_YAW_P", "3.0"),
        ("rate_limit_saturation_roll", "rate MAX increase candidate", "MC_ROLLRATE_MAX", "220"),
        ("rate_limit_saturation_pitch", "rate MAX increase candidate", "MC_PITCHRATE_MAX", "220"),
        ("rate_limit_saturation_yaw", "rate MAX increase candidate", "MC_YAWRATE_MAX", "200"),
        ("acceleration_limit_saturation", "MPC_ACC_HOR_MAX increase candidate", "MPC_ACC_HOR_MAX", "5.0"),
        ("acceleration_limit_saturation", "MPC_ACC_UP_MAX increase candidate", "MPC_ACC_UP_MAX", "4.0"),
        ("acceleration_limit_saturation", "MPC_ACC_DOWN_MAX increase candidate", "MPC_ACC_DOWN_MAX", "3.0"),
        ("tilt_limit_saturation", "MPC_TILTMAX_AIR increase candidate", "MPC_TILTMAX_AIR", "45.0"),
    ],
)
def test_yaml_mapping_produces_recommendation(production_config, issue_id, candidate, param, current):
    rec = ParameterRecommender(production_config)
    issue = {
        "issue_id": issue_id,
        "severity": "warning",
        "confidence": 80.0,
        "airframe": "MULTICOPTER",
        "tuning_candidates": [candidate],
        "metric_value": {"vibration_rms": 1.0},
    }
    out = rec.recommend(issue, {param: current}, "MULTICOPTER")
    assert len(out) == 1, f"expected recommendation for ({issue_id}, {candidate}) -> {param}"
    assert out[0]["param"] == param


def test_rate_limit_proposes_both_max_increase_and_p_decrease(production_config):
    """A real rate-limit-saturation issue suggests two parameters at once."""
    rec = ParameterRecommender(production_config)
    issue = {
        "issue_id": "rate_limit_saturation_roll",
        "severity": "problem",
        "confidence": 85.0,
        "airframe": "MULTICOPTER",
        "tuning_candidates": ["rate MAX increase candidate", "attitude P decrease candidate"],
        "metric_value": {},
    }
    params = {"MC_ROLLRATE_MAX": "220", "MC_ROLL_P": "8.0"}
    out = rec.recommend(issue, params, "MULTICOPTER")
    names = {r["param"] for r in out}
    assert "MC_ROLLRATE_MAX" in names
    assert "MC_ROLL_P" in names
