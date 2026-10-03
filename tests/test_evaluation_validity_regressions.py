"""High-confidence regression tests for evaluator signal semantics.

The expected values below are calculated directly from small independent
fixtures.  They intentionally exercise the failure modes found in two private
VTOL validation logs: waypoint distance labelled as cross-track error, uncorrected
TAS-ground-speed comparison, degree/radian cruise filtering, percentile-derived
PWM limits, and altitude gain counted as transition loss.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.evaluator import AutoEvaluationEngine  # noqa: E402
from core.log_model import LogDataset, TopicInstance  # noqa: E402


def _dataset(**frames: pl.DataFrame) -> LogDataset:
    dataset = LogDataset(source_format="px4_ulog", capabilities={"px4", "timeseries"})
    for base_name, frame in frames.items():
        dataset.add_topic(TopicInstance(base_name=base_name, instance_id=0, dataframe=frame))
    return dataset


def _item(items, key: str) -> dict:
    return next(row for row in items if row.get("key") == key)


def _engine_for(airframe: str) -> AutoEvaluationEngine:
    engine = AutoEvaluationEngine()
    engine._eval_airframe_ctx = airframe
    engine._eval_parameters = {}
    engine._eval_messages = []
    engine._eval_firmware = {}
    return engine


def test_navigation_prefers_npfg_signed_track_error_over_large_position_proxies():
    ts = np.arange(0.0, 101.0)
    dataset = _dataset(
        npfg_status=pl.DataFrame({"timestamp_sec": ts, "signed_track_error": np.where(ts % 2 == 0, -2.0, 2.0)}),
        position_controller_status=pl.DataFrame({"timestamp_sec": ts, "xtrack_error": np.full(ts.size, 50.0)}),
        vehicle_status=pl.DataFrame({"timestamp_sec": ts, "nav_state": np.full(ts.size, 3.0)}),
        vehicle_local_position=pl.DataFrame({
            "timestamp_sec": ts,
            "x": np.full(ts.size, 1000.0),
            "y": np.full(ts.size, 1000.0),
            "xy_valid": np.ones(ts.size),
            "v_xy_valid": np.ones(ts.size),
        }),
        vehicle_local_position_setpoint=pl.DataFrame({
            "timestamp_sec": ts,
            "x": np.zeros(ts.size),
            "y": np.zeros(ts.size),
        }),
    )
    engine = _engine_for("VTOL")
    items = engine._eval_navigation_path_tracking(dataset, "VTOL", parameters={"NAV_ACC_RAD": 10.0})
    path = _item(items, "navigation_path_tracking_error")

    # Independent reference: mean(abs(+/-2)) == 2 m.  The 50 m controller
    # signal and 1414 m position-to-waypoint distance must not be selected.
    assert path["value"] == pytest.approx(2.0)
    assert path["evidence_values"]["path_p95_m"] == pytest.approx(2.0)
    assert path["fallback_used"] is False
    assert path["used_signals"][0] == "npfg_status_0.signed_track_error"


def test_navigation_does_not_score_position_to_setpoint_distance_as_cross_track():
    ts = np.arange(0.0, 31.0)
    dataset = _dataset(
        vehicle_status=pl.DataFrame({"timestamp_sec": ts, "nav_state": np.full(ts.size, 3.0)}),
        vehicle_local_position=pl.DataFrame({"timestamp_sec": ts, "x": np.full(ts.size, 50.0), "y": np.zeros(ts.size)}),
        vehicle_local_position_setpoint=pl.DataFrame({"timestamp_sec": ts, "x": np.zeros(ts.size), "y": np.zeros(ts.size)}),
    )
    path = _item(
        _engine_for("FIXED_WING")._eval_navigation_path_tracking(dataset, "FIXED_WING", parameters={}),
        "navigation_path_tracking_error",
    )
    assert path["status"] == "unavailable"
    assert path["value"] is None
    assert "future waypoint" in path["reason"]


def test_navigation_uses_explicit_controller_xtrack_only_as_secondary_source():
    ts = np.arange(0.0, 41.0)
    dataset = _dataset(
        position_controller_status=pl.DataFrame({"timestamp_sec": ts, "xtrack_error": np.full(ts.size, 3.0)}),
        vehicle_status=pl.DataFrame({"timestamp_sec": ts, "nav_state": np.full(ts.size, 3.0)}),
    )
    path = _item(
        _engine_for("FIXED_WING")._eval_navigation_path_tracking(dataset, "FIXED_WING", parameters={}),
        "navigation_path_tracking_error",
    )
    assert path["value"] == pytest.approx(3.0)
    assert path["fallback_used"] is True
    assert path["used_signals"][0] == "position_controller_status_0.xtrack_error"


def _wind_fixture(include_wind: bool = True) -> LogDataset:
    ts = np.arange(0.0, 40.0)
    frames = {
        "airspeed_validated": pl.DataFrame({"timestamp_sec": ts, "true_airspeed_m_s": np.full(ts.size, 10.0)}),
        "vehicle_gps_position": pl.DataFrame({
            "timestamp_sec": ts,
            "vel_n_m_s": np.full(ts.size, 15.0),
            "vel_e_m_s": np.zeros(ts.size),
            "vel_d_m_s": np.zeros(ts.size),
        }),
        "vehicle_status": pl.DataFrame({
            "timestamp_sec": ts,
            "nav_state": np.full(ts.size, 3.0),
            "vehicle_type": np.full(ts.size, 2.0),
            "in_transition_mode": np.zeros(ts.size),
        }),
        "vehicle_local_position": pl.DataFrame({"timestamp_sec": ts, "vz": np.zeros(ts.size)}),
        # These are explicitly degree-valued MathEngine fields.  Ten and five
        # degrees must pass the 20/12 degree cruise filter.
        "vehicle_attitude": pl.DataFrame({
            "timestamp_sec": ts,
            "roll_euler": np.full(ts.size, 10.0),
            "pitch_euler": np.full(ts.size, 5.0),
        }),
    }
    if include_wind:
        frames["wind"] = pl.DataFrame({
            "timestamp_sec": ts,
            "windspeed_north": np.full(ts.size, 5.0),
            "windspeed_east": np.zeros(ts.size),
        })
    return _dataset(**frames)


def test_airspeed_metrics_use_gps_minus_wind_and_degree_cruise_filter():
    engine = _engine_for("FIXED_WING")
    items = engine._eval_airframe(_wind_fixture(), "FIXED_WING", parameters={})
    consistency = _item(items, "airspeed_consistency")
    scale = _item(items, "airspeed_scale_appropriateness")

    # Independent vector reference: |(15,0,0) - (5,0,0)| == 10 m/s.
    assert consistency["value"] == pytest.approx(0.0, abs=1e-12)
    assert scale["value"] == pytest.approx(0.0, abs=1e-12)
    assert scale["fallback_used"] is False
    assert scale["sample_count"] >= 30
    assert scale["evidence_values"]["air_reference_mean_mps"] == pytest.approx(10.0)
    assert "wind_0.windspeed_north" in scale["used_signals"]


def test_airspeed_metrics_are_unavailable_without_wind_instead_of_scoring_raw_ground_speed():
    items = _engine_for("FIXED_WING")._eval_airframe(_wind_fixture(include_wind=False), "FIXED_WING", parameters={})
    assert _item(items, "airspeed_consistency")["status"] == "unavailable"
    assert _item(items, "airspeed_scale_appropriateness")["status"] == "unavailable"


def test_pwm_percentiles_are_not_treated_as_physical_saturation_limits():
    ts = np.arange(0.0, 101.0)
    dataset = _dataset(
        actuator_outputs=pl.DataFrame({"timestamp_sec": ts, "output[0]": np.linspace(1000.0, 2000.0, ts.size)}),
        actuator_armed=pl.DataFrame({"timestamp_sec": ts, "armed": np.ones(ts.size)}),
        vehicle_land_detected=pl.DataFrame({"timestamp_sec": ts, "landed": np.zeros(ts.size)}),
    )
    result = _engine_for("UNKNOWN").evaluate(dataset, aircraft_type="UNKNOWN")
    saturation = _item(result["items"], "controller_saturation")
    assert saturation["status"] == "unavailable"
    assert saturation["value"] is None
    assert "PWM observed percentiles" in saturation["reason"]


def test_normalized_motor_saturation_uses_armed_airborne_time_and_inclusive_boundary():
    ts = np.arange(0.0, 101.0)
    control = np.full(ts.size, 0.5)
    control[:8] = 1.0  # 8 seconds of the 100 weighted seconds == exactly 8%.
    dataset = _dataset(
        actuator_motors=pl.DataFrame({"timestamp_sec": ts, "control[0]": control}),
        actuator_armed=pl.DataFrame({"timestamp_sec": ts, "armed": np.ones(ts.size)}),
        vehicle_land_detected=pl.DataFrame({"timestamp_sec": ts, "landed": np.zeros(ts.size)}),
    )
    saturation = _item(_engine_for("UNKNOWN").evaluate(dataset, aircraft_type="UNKNOWN")["items"], "controller_saturation")
    assert saturation["value"] == pytest.approx(8.0)
    assert saturation["status"] == "good"  # warn boundary is inclusive by contract.
    assert saturation["evidence_values"]["upper_saturation_threshold"] == pytest.approx(0.98)


def test_score_low_threshold_boundaries_are_stable():
    engine = _engine_for("UNKNOWN")
    assert engine._score_low(8.0, 8.0, 20.0)[1] == "good"
    assert engine._score_low(np.nextafter(8.0, np.inf), 8.0, 20.0)[1] == "warning"
    assert engine._score_low(20.0, 8.0, 20.0)[1] == "warning"
    assert engine._score_low(np.nextafter(20.0, np.inf), 8.0, 20.0)[1] == "problem"


@pytest.mark.parametrize(
    "z_values, expected_loss",
    [
        ([0.0, -5.0, -10.0, -10.0], 0.0),  # NED z decreases: aircraft climbed.
        ([0.0, 5.0, 7.0, 7.0], 2.0),       # During t=1..2, NED z grows 5 -> 7: 2 m loss.
    ],
)
def test_transition_altitude_loss_counts_only_downward_excursion(z_values, expected_loss):
    ts = np.arange(0.0, 4.0)
    dataset = _dataset(
        vehicle_status=pl.DataFrame({
            "timestamp_sec": ts,
            "nav_state": np.full(ts.size, 3.0),
            "vehicle_type": np.full(ts.size, 2.0),
            "in_transition_mode": [0.0, 1.0, 1.0, 0.0],
        }),
        vehicle_local_position=pl.DataFrame({"timestamp_sec": ts, "z": z_values}),
    )
    items = _engine_for("VTOL")._eval_airframe(dataset, "VTOL", parameters={})
    loss = _item(items, "transition_altitude_loss")
    assert loss["value"] == pytest.approx(expected_loss)
    assert loss["evidence_values"]["altitude_convention"] == "NED down-positive"
