"""Tests for the VTOL depth expansion:
 - infer_vtol_transition_issue using real sample_count / transition_count
   (replacing the prior hardcoded 400 / 2.0)
 - VTOL transition parameter mappings (VT_F_TRANS_THR, VT_F_TRANS_DUR)
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
# Confidence: real observation counts replace the prior hardcoded 400 / 2.0
# ---------------------------------------------------------------------------

def test_zero_transitions_caps_confidence_at_40(inference):
    metrics = {
        "duration": 25.0,   # problem
        "altitude_loss": 5.0,
        "innovation_ratio": 3.0,
        "saturation_ratio": 0.0,
        "vibration_severity": 0.0,
        "sample_count": 0,
        "transition_count": 0,
    }
    issue, _ = inference.infer_vtol_transition_issue(metrics)
    assert issue["severity"] == "problem"
    assert issue["confidence"] <= 40.0, "0-transition logs must cap confidence at 40"


def test_two_transitions_yields_normal_confidence(inference):
    metrics = {
        "duration": 20.0,
        "altitude_loss": 10.0,
        "innovation_ratio": 3.0,
        "saturation_ratio": 0.0,
        "vibration_severity": 0.0,
        "sample_count": 800,
        "transition_count": 2,
    }
    issue, _ = inference.infer_vtol_transition_issue(metrics)
    # Sufficient maneuver count -> confidence should not be artificially capped low.
    assert issue["confidence"] >= 60.0


def test_single_transition_still_capped(inference):
    metrics = {
        "duration": 20.0,
        "altitude_loss": 10.0,
        "innovation_ratio": 3.0,
        "saturation_ratio": 0.0,
        "vibration_severity": 0.0,
        "sample_count": 400,
        "transition_count": 1,   # < 2 -> still under-observed
    }
    issue, _ = inference.infer_vtol_transition_issue(metrics)
    # Only one transition: insufficient evidence to extrapolate, confidence stays moderate.
    assert issue["confidence"] <= 70.0


# ---------------------------------------------------------------------------
# Candidate emission produces parameter-oriented texts the YAML can map
# ---------------------------------------------------------------------------

def test_long_duration_emits_throttle_and_duration_candidates(inference):
    metrics = {
        "duration": 25.0,   # problem (>24)
        "altitude_loss": 5.0,
        "innovation_ratio": 3.0,
        "saturation_ratio": 0.0,
        "vibration_severity": 0.0,
        "sample_count": 800,
        "transition_count": 3,
    }
    issue, _ = inference.infer_vtol_transition_issue(metrics)
    assert "VT_F_TRANS_THR increase candidate" in issue["tuning_candidates"]
    assert "VT_F_TRANS_DUR increase candidate" in issue["tuning_candidates"]


def test_altitude_loss_emits_throttle_candidate(inference):
    metrics = {
        "duration": 10.0,   # normal
        "altitude_loss": 20.0,   # problem (>18)
        "innovation_ratio": 3.0,
        "saturation_ratio": 0.0,
        "vibration_severity": 0.0,
        "sample_count": 800,
        "transition_count": 3,
    }
    issue, _ = inference.infer_vtol_transition_issue(metrics)
    assert "VT_F_TRANS_THR increase candidate" in issue["tuning_candidates"]


def test_innovation_first_blocks_gain_recommendations(inference, production_config):
    """High innovation should emit estimator-first; recommender must then return []."""
    metrics = {
        "duration": 25.0,
        "altitude_loss": 20.0,
        "innovation_ratio": 15.0,   # problem (>10)
        "saturation_ratio": 0.0,
        "vibration_severity": 0.0,
        "sample_count": 800,
        "transition_count": 3,
    }
    issue, _ = inference.infer_vtol_transition_issue(metrics)
    assert "estimator / sensor check first" in issue["tuning_candidates"]
    rec = ParameterRecommender(production_config)
    out = rec.recommend(
        issue,
        {"VT_F_TRANS_THR": "0.8", "VT_F_TRANS_DUR": "5.0"},
        "VTOL",
    )
    assert out == []   # confounder-first blocks all gain changes


# ---------------------------------------------------------------------------
# YAML mapping coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "candidate,param,current",
    [
        # VT_F_TRANS_THR/DUR are advisory_only in the production yaml:
        # transition envelope expansion depends on motor/thrust headroom
        # that single-log analysis cannot verify.
        ("VT_F_TRANS_THR increase candidate", "VT_F_TRANS_THR", "0.8"),
        ("VT_F_TRANS_DUR increase candidate", "VT_F_TRANS_DUR", "5.0"),
    ],
)
def test_vtol_transition_yaml_emits_advisory_only(production_config, candidate, param, current):
    rec = ParameterRecommender(production_config)
    issue = {
        "issue_id": "vtol_transition_tuning",
        "severity": "warning",
        "confidence": 80.0,
        "airframe": "VTOL",
        "tuning_candidates": [candidate],
        "metric_value": {},
    }
    out = rec.recommend(issue, {param: current}, "VTOL")
    assert len(out) == 1
    item = out[0]
    assert item["param"] == param
    assert item["advisory_only"] is True
    assert item["suggested"] is None
    assert item["risk_tier"] == "high"
    assert any("ADVISORY" in note for note in item["safety_notes"])


def test_vt_f_trans_thr_advisory_does_not_clamp(production_config):
    """advisory_only actions emit no number, so range-clamp logic does not run."""
    rec = ParameterRecommender(production_config)
    issue = {
        "issue_id": "vtol_transition_tuning",
        "severity": "problem",
        "confidence": 80.0,
        "airframe": "VTOL",
        "tuning_candidates": ["VT_F_TRANS_THR increase candidate"],
        "metric_value": {},
    }
    out = rec.recommend(issue, {"VT_F_TRANS_THR": "0.95"}, "VTOL")
    assert len(out) == 1
    assert out[0]["suggested"] is None
    assert out[0]["clamped"] is False
    assert out[0]["advisory_only"] is True


def test_vtol_round_trip_long_duration(inference, production_config):
    """End-to-end: infer long-duration transition then recommend params."""
    metrics = {
        "duration": 25.0,
        "altitude_loss": 5.0,
        "innovation_ratio": 3.0,
        "saturation_ratio": 0.0,
        "vibration_severity": 0.0,
        "sample_count": 800,
        "transition_count": 3,
    }
    issue, _ = inference.infer_vtol_transition_issue(metrics)
    rec = ParameterRecommender(production_config)
    params = {"VT_F_TRANS_THR": "0.8", "VT_F_TRANS_DUR": "5.0"}
    out = rec.recommend(issue, params, "VTOL")
    names = {r["param"] for r in out}
    assert "VT_F_TRANS_THR" in names
    assert "VT_F_TRANS_DUR" in names
