from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.parameter_recommender import ParameterRecommender  # noqa: E402


@pytest.fixture
def base_config():
    """Minimal MC roll/pitch/yaw mappings used by the depth-phase recommender.

    Mirrors the production parameter_recommendations.yaml but keeps a single
    candidate per axis so test expectations stay explicit.
    """
    return {
        "safety": {
            "max_delta_percent_per_step": 25.0,
            "min_confidence_to_recommend": 50.0,
            "block_when_candidates_contain": [
                "actuator authority check first",
                "hardware vibration mitigation first",
                "estimator / sensor check first",
            ],
        },
        "mappings": [
            {
                "airframe": "MULTICOPTER",
                "issue_id": "rate_loop_roll",
                "candidates": {
                    "P decrease candidate": [
                        {
                            "param": "MC_ROLLRATE_P",
                            "risk_tier": "low",
                            "severity_delta": {"warning": 0.85, "problem": 0.75},
                            "range": [0.01, 0.5],
                            "rationale": "test rationale",
                            "verification": "test verification",
                        }
                    ],
                    # D gate path: numeric (advisory_only off) so the gate
                    # block path is testable here. Production yaml flips
                    # this to advisory_only=true (verified by test_fw_expansion).
                    "D increase candidate": [
                        {
                            "param": "MC_ROLLRATE_D",
                            "risk_tier": "low",
                            "severity_delta": {"warning": 1.15, "problem": 1.20},
                            "range": [0.0, 0.01],
                            "rationale": "D test",
                            "verification": "D verification",
                            "gate": {"require_metric_below": {"vibration_rms": 3.5}},
                        }
                    ],
                    # Used to exercise the ±25% cap (severity_delta exceeds cap).
                    "P aggressive candidate": [
                        {
                            "param": "MC_ROLLRATE_P",
                            "risk_tier": "low",
                            "severity_delta": {"warning": 1.5, "problem": 2.0},
                            "range": [0.01, 0.5],
                            "rationale": "cap test",
                            "verification": "cap verification",
                        }
                    ],
                },
            },
        ],
    }


def _make_issue(**overrides):
    issue = {
        "issue_id": "rate_loop_roll",
        "severity": "warning",
        "confidence": 80.0,
        "airframe": "MULTICOPTER",
        "tuning_candidates": ["P decrease candidate"],
        "metric_value": {"vibration_rms": 2.0},
    }
    issue.update(overrides)
    return issue


def test_recommends_p_decrease_within_cap(base_config):
    rec = ParameterRecommender(base_config)
    out = rec.recommend(_make_issue(), {"MC_ROLLRATE_P": "0.15"}, "MULTICOPTER")
    assert len(out) == 1
    item = out[0]
    assert item["param"] == "MC_ROLLRATE_P"
    assert item["current"] == pytest.approx(0.15)
    # warning severity -> 0.85x -> 0.1275
    assert item["suggested"] == pytest.approx(0.1275, rel=1e-3)
    assert item["delta_percent"] == pytest.approx(-15.0, abs=1e-2)
    assert item["clamped"] is False


def test_problem_severity_uses_problem_ratio(base_config):
    rec = ParameterRecommender(base_config)
    out = rec.recommend(
        _make_issue(severity="problem", confidence=85.0),
        {"MC_ROLLRATE_P": "0.20"},
        "MULTICOPTER",
    )
    # problem -> 0.75x -> 0.15
    assert out[0]["suggested"] == pytest.approx(0.15, rel=1e-3)
    assert out[0]["delta_percent"] == pytest.approx(-25.0, abs=1e-2)


def test_low_confidence_emits_advisory_row_without_number(base_config):
    """Below safety floor confidence: row is emitted (so the pilot still sees
    rationale/verification) but `suggested` is None and a safety note explains
    the missing number."""
    rec = ParameterRecommender(base_config)
    out = rec.recommend(
        _make_issue(confidence=49.0),
        {"MC_ROLLRATE_P": "0.15"},
        "MULTICOPTER",
    )
    assert len(out) == 1
    item = out[0]
    assert item["param"] == "MC_ROLLRATE_P"
    assert item["suggested"] is None
    assert item["delta_percent"] is None
    assert any("신뢰도" in note for note in item["safety_notes"])
    # Rationale must still reach the pilot so they understand the candidate.
    assert item["rationale"] == "test rationale"


def test_normal_severity_blocks_recommendation(base_config):
    rec = ParameterRecommender(base_config)
    out = rec.recommend(
        _make_issue(severity="normal", confidence=90.0),
        {"MC_ROLLRATE_P": "0.15"},
        "MULTICOPTER",
    )
    assert out == []


def test_confounder_first_blocks_all_gain_changes(base_config):
    rec = ParameterRecommender(base_config)
    issue = _make_issue(
        tuning_candidates=[
            "actuator authority check first",
            "P decrease candidate",
        ]
    )
    out = rec.recommend(issue, {"MC_ROLLRATE_P": "0.15"}, "MULTICOPTER")
    assert out == []


def test_d_gate_blocks_numeric_suggestion_when_vibration_high(base_config):
    """When gate (vibration ≥ threshold) fails, the recommender must NOT emit
    a `suggested` number, but should still surface the rationale + safety
    note so the pilot understands why the D-term value was withheld."""
    rec = ParameterRecommender(base_config)
    issue = _make_issue(
        tuning_candidates=["D increase candidate"],
        metric_value={"vibration_rms": 4.0},  # >= 3.5 threshold
    )
    out = rec.recommend(issue, {"MC_ROLLRATE_D": "0.003"}, "MULTICOPTER")
    assert len(out) == 1
    item = out[0]
    assert item["param"] == "MC_ROLLRATE_D"
    assert item["suggested"] is None
    assert any("게이트" in note for note in item["safety_notes"])


def test_d_gate_passes_when_vibration_low(base_config):
    rec = ParameterRecommender(base_config)
    issue = _make_issue(
        tuning_candidates=["D increase candidate"],
        metric_value={"vibration_rms": 2.0},
    )
    out = rec.recommend(issue, {"MC_ROLLRATE_D": "0.003"}, "MULTICOPTER")
    assert len(out) == 1
    assert out[0]["param"] == "MC_ROLLRATE_D"


def test_missing_parameter_skipped(base_config):
    rec = ParameterRecommender(base_config)
    out = rec.recommend(_make_issue(), {}, "MULTICOPTER")
    assert out == []


def test_max_delta_cap_applied(base_config):
    rec = ParameterRecommender(base_config)
    issue = _make_issue(
        severity="problem",
        confidence=85.0,
        tuning_candidates=["P aggressive candidate"],
    )
    out = rec.recommend(issue, {"MC_ROLLRATE_P": "0.10"}, "MULTICOPTER")
    assert len(out) == 1
    # problem severity asks for 2.0x (+100%), cap forces +25% -> 0.125
    assert out[0]["suggested"] == pytest.approx(0.125, rel=1e-3)
    assert out[0]["delta_percent"] == pytest.approx(25.0, abs=1e-2)
    assert any("캡" in s for s in out[0]["safety_notes"])


def test_range_clamp_applied(base_config):
    rec = ParameterRecommender(base_config)
    issue = _make_issue(
        severity="problem",
        confidence=85.0,
        tuning_candidates=["P aggressive candidate"],
    )
    # Current already near upper range; even after cap, clamped to 0.5.
    out = rec.recommend(issue, {"MC_ROLLRATE_P": "0.45"}, "MULTICOPTER")
    assert len(out) == 1
    assert out[0]["suggested"] == pytest.approx(0.5, rel=1e-6)
    assert out[0]["clamped"] is True
    assert any("range" in s for s in out[0]["safety_notes"])


def test_duplicate_param_kept_once(base_config):
    rec = ParameterRecommender(base_config)
    issue = _make_issue(
        tuning_candidates=["P decrease candidate", "P aggressive candidate"],
    )
    out = rec.recommend(issue, {"MC_ROLLRATE_P": "0.15"}, "MULTICOPTER")
    # MC_ROLLRATE_P only appears once even though two candidates reference it.
    params = [r["param"] for r in out]
    assert params.count("MC_ROLLRATE_P") == 1


def test_unknown_airframe_returns_empty(base_config):
    rec = ParameterRecommender(base_config)
    out = rec.recommend(_make_issue(), {"MC_ROLLRATE_P": "0.15"}, "FIXED_WING")
    assert out == []


def test_unknown_issue_id_returns_empty(base_config):
    rec = ParameterRecommender(base_config)
    out = rec.recommend(_make_issue(issue_id="unknown_issue"), {"MC_ROLLRATE_P": "0.15"}, "MULTICOPTER")
    assert out == []


def test_string_parameter_value_is_parsed(base_config):
    rec = ParameterRecommender(base_config)
    # Real logs deliver parameter values as strings via _metadata_value_to_text.
    out = rec.recommend(_make_issue(), {"MC_ROLLRATE_P": "0.150"}, "MULTICOPTER")
    assert len(out) == 1
    assert out[0]["current"] == pytest.approx(0.15)


def test_empty_config_safe(base_config):
    rec = ParameterRecommender(None)
    out = rec.recommend(_make_issue(), {"MC_ROLLRATE_P": "0.15"}, "MULTICOPTER")
    assert out == []


def test_yaml_config_loads(base_config):
    """Smoke test: production YAML config parses and produces some output."""
    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML not installed")
    cfg_path = PROJECT_ROOT / "config" / "parameter_recommendations.yaml"
    assert cfg_path.is_file()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rec = ParameterRecommender(cfg)
    out = rec.recommend(_make_issue(), {"MC_ROLLRATE_P": "0.15"}, "MULTICOPTER")
    assert len(out) >= 1
    assert out[0]["param"] == "MC_ROLLRATE_P"
