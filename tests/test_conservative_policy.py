"""Regression tests for the conservative-tuning policy added on top of the
ParameterRecommender baseline.

Each test pins one rule of the safety contract:
    - max per-step Δ cap (10%)
    - confidence floor (70%) returns advisory row, not empty
    - risk_tier multiplier shrinks raw severity_delta
    - advisory_only actions emit text guidance with suggested=None
    - VTOL fallback recommendations carry the explicit MC-fallback safety note
    - every numeric suggestion carries a staged-apply guide
    - gate failure on a numeric action returns advisory row (not silent drop)
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


@pytest.fixture(scope="module")
def production_config():
    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML not installed")
    cfg_path = PROJECT_ROOT / "config" / "parameter_recommendations.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _mc_p_decrease_issue(severity="warning", confidence=80.0):
    return {
        "issue_id": "rate_loop_roll",
        "severity": severity,
        "confidence": confidence,
        "airframe": "MULTICOPTER",
        "tuning_candidates": ["P decrease candidate"],
        "metric_value": {},
    }


# ---------------------------------------------------------------------------
# Cap / confidence / staged-apply
# ---------------------------------------------------------------------------

def test_max_delta_cap_is_ten_percent(production_config):
    """problem severity_delta×0.85 = -15%; cap should clip to -10%."""
    rec = ParameterRecommender(production_config)
    issue = _mc_p_decrease_issue(severity="problem", confidence=85.0)
    out = rec.recommend(issue, {"MC_ROLLRATE_P": "0.20"}, "MULTICOPTER")
    assert len(out) == 1
    item = out[0]
    # Raw -15% -> tier low (1.0) -> -15% -> cap -> -10% -> 0.20 * 0.9 = 0.18.
    assert item["suggested"] == pytest.approx(0.18, rel=1e-3)
    assert item["delta_percent"] == pytest.approx(-10.0, abs=1e-2)
    assert any("캡" in note for note in item["safety_notes"])


def test_warning_within_cap_passes_through_untouched(production_config):
    """warning severity_delta = 0.92 (-8%) is inside the 10% cap, so no clip."""
    rec = ParameterRecommender(production_config)
    issue = _mc_p_decrease_issue(severity="warning", confidence=85.0)
    out = rec.recommend(issue, {"MC_ROLLRATE_P": "0.20"}, "MULTICOPTER")
    assert len(out) == 1
    assert out[0]["suggested"] == pytest.approx(0.184, rel=1e-3)
    assert out[0]["delta_percent"] == pytest.approx(-8.0, abs=1e-2)
    assert not any("캡" in note for note in out[0]["safety_notes"])


def test_confidence_below_70_emits_advisory_row(production_config):
    """Below the 70% floor: row is emitted but `suggested` is None."""
    rec = ParameterRecommender(production_config)
    issue = _mc_p_decrease_issue(severity="problem", confidence=65.0)
    out = rec.recommend(issue, {"MC_ROLLRATE_P": "0.20"}, "MULTICOPTER")
    assert len(out) == 1
    item = out[0]
    assert item["suggested"] is None
    assert item["delta_percent"] is None
    assert any("신뢰도" in note for note in item["safety_notes"])
    assert item["rationale"]   # rationale text still reaches the pilot


def test_staged_apply_guide_attached_to_numeric_suggestion(production_config):
    rec = ParameterRecommender(production_config)
    issue = _mc_p_decrease_issue(severity="warning", confidence=85.0)
    out = rec.recommend(issue, {"MC_ROLLRATE_P": "0.20"}, "MULTICOPTER")
    assert len(out) == 1
    item = out[0]
    assert item["staged_apply"]
    # First-step value should sit between current and final.
    assert 0.184 < 0.192 < 0.20  # sanity ordering
    # 1단계 표기 포함
    assert any("단계" in note for note in item["safety_notes"])


# ---------------------------------------------------------------------------
# risk_tier multiplier
# ---------------------------------------------------------------------------

def test_medium_tier_shrinks_raw_delta(production_config):
    """MC_ROLLRATE_P 'P increase candidate' is risk_tier=medium in yaml.
    Raw warning ratio 1.08 -> medium (0.7×) -> 1.056 -> ~+5.6% (within cap)."""
    rec = ParameterRecommender(production_config)
    issue = {
        "issue_id": "rate_loop_roll",
        "severity": "warning",
        "confidence": 85.0,
        "airframe": "MULTICOPTER",
        "tuning_candidates": ["P increase candidate"],
        "metric_value": {},
    }
    out = rec.recommend(issue, {"MC_ROLLRATE_P": "0.20"}, "MULTICOPTER")
    assert len(out) == 1
    item = out[0]
    assert item["risk_tier"] == "medium"
    # 0.20 * (1.0 + 0.08 * 0.7) = 0.20 * 1.056 = 0.2112
    assert item["suggested"] == pytest.approx(0.2112, rel=1e-3)
    assert item["delta_percent"] == pytest.approx(5.6, abs=1e-1)
    # A tier-shrink note is recorded so the pilot knows the rule was softened.
    assert any("risk_tier" in note for note in item["safety_notes"])


def test_low_tier_keeps_full_delta(production_config):
    """P decrease (low tier) keeps the raw -8% delta after tier multiplication."""
    rec = ParameterRecommender(production_config)
    issue = _mc_p_decrease_issue(severity="warning", confidence=85.0)
    out = rec.recommend(issue, {"MC_ROLLRATE_P": "0.20"}, "MULTICOPTER")
    item = out[0]
    assert item["risk_tier"] == "low"
    # No tier-shrink note when low tier = 1.0×
    assert not any("risk_tier" in note for note in item["safety_notes"])


# ---------------------------------------------------------------------------
# advisory_only behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "issue_id, candidate, param, current",
    [
        ("rate_loop_roll", "D increase candidate", "MC_ROLLRATE_D", "0.003"),
        ("rate_loop_pitch", "D increase candidate", "MC_PITCHRATE_D", "0.003"),
        ("rate_limit_saturation_roll", "rate MAX increase candidate", "MC_ROLLRATE_MAX", "220"),
        ("tilt_limit_saturation", "MPC_TILTMAX_AIR increase candidate", "MPC_TILTMAX_AIR", "45.0"),
        ("acceleration_limit_saturation", "MPC_ACC_DOWN_MAX increase candidate", "MPC_ACC_DOWN_MAX", "3.0"),
    ],
)
def test_risky_actions_are_advisory_only(production_config, issue_id, candidate, param, current):
    """D-term increases and envelope expansions emit text guidance only —
    no `suggested` number is produced regardless of severity/confidence."""
    rec = ParameterRecommender(production_config)
    issue = {
        "issue_id": issue_id,
        "severity": "problem",
        "confidence": 90.0,
        "airframe": "MULTICOPTER",
        "tuning_candidates": [candidate],
        "metric_value": {"vibration_rms": 1.0},   # below D gate
    }
    out = rec.recommend(issue, {param: current}, "MULTICOPTER")
    assert len(out) == 1
    item = out[0]
    assert item["advisory_only"] is True
    assert item["suggested"] is None
    assert item["risk_tier"] == "high"
    assert any("ADVISORY" in note for note in item["safety_notes"])
    assert item["rationale"]


def test_fw_tecs_envelope_expansions_are_advisory(production_config):
    rec = ParameterRecommender(production_config)
    issue = {
        "issue_id": "tecs_altitude_response",
        "severity": "problem",
        "confidence": 85.0,
        "airframe": "FIXED_WING",
        "tuning_candidates": [
            "FW_T_CLMB_MAX increase candidate",
            "FW_T_SINK_MIN increase candidate",
        ],
        "metric_value": {},
    }
    out = rec.recommend(
        issue,
        {"FW_T_CLMB_MAX": "5.0", "FW_T_SINK_MIN": "2.0"},
        "FIXED_WING",
    )
    assert {r["param"] for r in out} == {"FW_T_CLMB_MAX", "FW_T_SINK_MIN"}
    for item in out:
        assert item["advisory_only"] is True
        assert item["suggested"] is None


def test_fw_spdweight_is_advisory(production_config):
    """FW_T_SPDWEIGHT changes throttle↔pitch priority — too consequential
    to derive from a single log."""
    rec = ParameterRecommender(production_config)
    issue = {
        "issue_id": "tecs_airspeed_response",
        "severity": "problem",
        "confidence": 85.0,
        "airframe": "FIXED_WING",
        "tuning_candidates": ["FW_T_SPDWEIGHT increase candidate"],
        "metric_value": {},
    }
    out = rec.recommend(issue, {"FW_T_SPDWEIGHT": "1.0"}, "FIXED_WING")
    assert len(out) == 1
    assert out[0]["advisory_only"] is True
    assert out[0]["suggested"] is None


# ---------------------------------------------------------------------------
# VTOL fallback safety note
# ---------------------------------------------------------------------------

def test_vtol_fallback_recommendation_carries_explicit_warning(production_config):
    """VTOL inherits MC mapping for rate_loop_*; every fallback row must
    carry the [VTOL FALLBACK] safety note so the pilot does not mistake the
    MC-derived value for a VTOL-validated one."""
    rec = ParameterRecommender(production_config)
    issue = {
        "issue_id": "rate_loop_roll",
        "severity": "warning",
        "confidence": 85.0,
        "airframe": "VTOL",
        "tuning_candidates": ["P decrease candidate"],
        "metric_value": {},
    }
    out = rec.recommend(issue, {"MC_ROLLRATE_P": "0.15"}, "VTOL")
    assert len(out) == 1
    item = out[0]
    assert item["suggested"] is not None
    assert any("VTOL FALLBACK" in note for note in item["safety_notes"])


def test_vtol_explicit_mapping_does_not_get_fallback_note(production_config):
    """tecs_altitude_response has an explicit VTOL entry — fallback note
    must NOT be attached because no fallback occurred."""
    rec = ParameterRecommender(production_config)
    issue = {
        "issue_id": "tecs_altitude_response",
        "severity": "warning",
        "confidence": 85.0,
        "airframe": "VTOL",
        "tuning_candidates": ["FW_T_ALT_TC decrease candidate"],
        "metric_value": {},
    }
    out = rec.recommend(issue, {"FW_T_ALT_TC": "8.0"}, "VTOL")
    assert len(out) == 1
    assert not any("VTOL FALLBACK" in note for note in out[0]["safety_notes"])


# ---------------------------------------------------------------------------
# Gate failure on numeric action -> advisory row (not silent drop)
# ---------------------------------------------------------------------------

def test_confounder_first_still_silences_everything(production_config):
    """Confounder-first text (e.g. estimator/sensor check first) must block
    every gain change — even advisory-only ones — by collapsing the result."""
    rec = ParameterRecommender(production_config)
    issue = {
        "issue_id": "rate_loop_roll",
        "severity": "problem",
        "confidence": 85.0,
        "airframe": "MULTICOPTER",
        "tuning_candidates": [
            "estimator / sensor check first",
            "P decrease candidate",
            "D increase candidate",
        ],
        "metric_value": {},
    }
    out = rec.recommend(
        issue,
        {"MC_ROLLRATE_P": "0.15", "MC_ROLLRATE_D": "0.003"},
        "MULTICOPTER",
    )
    assert out == []
