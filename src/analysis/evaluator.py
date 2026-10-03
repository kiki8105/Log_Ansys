import copy
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

try:
    import yaml
except Exception:
    yaml = None

try:
    from .feature_extractor import ControlFeatureExtractor
    from .tuning_inference import TuningInferenceEngine
    from .recommendation_engine import RecommendationEngine
    from .parameter_recommender import ParameterRecommender
except Exception:
    ControlFeatureExtractor = None
    TuningInferenceEngine = None
    RecommendationEngine = None
    ParameterRecommender = None


DEFAULT_THRESHOLDS = {
    "overall": {"good_score_min": 80.0, "warning_score_min": 60.0},
    "segments": {"hover": {"min_samples": 80, "hspd_max": 2.0, "vz_max": 0.5}},
    "weights": {
        "vibration": 1.0,
        "attitude_tracking_error": 1.0,
        "angular_tracking_error": 1.0,
        "innovation_fail_ratio": 1.0,
        "controller_saturation": 1.0,
        "gps_quality": 0.8,
        "communication_quality": 0.8,
        "failsafe_event_count": 0.6,
        "mission_stability": 0.4,
        "altitude_tracking_error": 0.6,
        "hover_altitude_hold_error": 1.0,
        "tecs_altitude_tracking_error": 1.0,
        "tecs_airspeed_tracking_error": 1.0,
        "tecs_energy_balance_error": 0.9,
        "airspeed_consistency": 0.9,
        "airspeed_scale_appropriateness": 0.8,
        "navigation_path_tracking_error": 0.5,
        "navigation_path_tracking_reliability": 0.3,
        "throttle_pitch_coupling": 0.8,
        "rpm_stability": 0.8,
        "rotor_harmonic_vibration": 0.8,
        "control_coupling_index": 0.8,
        "transition_duration": 1.0,
        "transition_altitude_loss": 1.0,
    },
    "metrics": {
        "vibration": {"warn": 3.0, "problem": 6.0, "unit": "m/s^2 RMS", "suggestion": "Check propeller/rotor balance and IMU isolation."},
        "attitude_tracking_error": {"warn": 5.0, "problem": 12.0, "unit": "deg MAE", "suggestion": "Retune attitude loop gains."},
        "angular_tracking_error": {"warn": 0.18, "problem": 0.45, "unit": "rad/s MAE", "suggestion": "Retune rate loop gains."},
        "innovation_fail_ratio": {"warn": 3.0, "problem": 10.0, "unit": "% duration", "suggestion": "Check estimator tuning and sensor quality."},
        "controller_saturation": {"warn": 8.0, "problem": 20.0, "unit": "% duration", "suggestion": "Review control margin and allocation."},
        "gps_quality": {"warn_score_min": 80.0, "problem_score_min": 60.0, "suggestion": "Check GNSS visibility and multipath."},
        "communication_quality": {"warn_score_min": 80.0, "problem_score_min": 60.0, "unit": "score", "suggestion": "Review telemetry packet loss, radio buffer health, and GCS link quality."},
        "failsafe_event_count": {"warn": 1.0, "problem": 3.0, "unit": "회", "suggestion": "Review failsafe trigger source and safety action configuration."},
        "mission_stability": {"warn_score_min": 80.0, "problem_score_min": 60.0, "unit": "score", "suggestion": "Review unexpected mode divergence and mission-phase mode switching stability."},
        "altitude_tracking_error": {"warn": 3.0, "problem": 8.0, "unit": "m MAE", "suggestion": "Review vertical control tuning.", "applicable_modes": ["AltCtl", "PosCtl", "Mission", "Loiter", "RTL", "PositionSlow", "AltitudeCruise", "Descend", "Offboard", "Takeoff", "Land", "FollowTarget", "PrecisionLand", "Orbit", "VTOLTakeoff"], "excluded_modes": ["Manual", "Stabilized", "Acro"]},
        "hover_altitude_hold_error": {"warn": 1.5, "problem": 4.0, "unit": "m MAE", "suggestion": "Tune hover altitude controller.", "applicable_modes": ["AltCtl", "PosCtl", "Loiter", "RTL", "Offboard", "Takeoff", "Land", "PrecisionLand", "VTOLTakeoff"], "excluded_modes": ["Manual", "Stabilized", "Acro"]},
        "tecs_altitude_tracking_error": {"warn": 4.0, "problem": 10.0, "unit": "m MAE", "suggestion": "Tune TECS height loop and energy balance.", "applicable_modes": ["Mission", "Loiter", "RTL", "Offboard", "Takeoff", "Land", "FollowTarget", "Orbit", "VTOLTakeoff", "AltitudeCruise", "Descend"], "excluded_modes": ["Manual", "Stabilized", "Acro"]},
        "tecs_airspeed_tracking_error": {"warn": 2.0, "problem": 5.0, "unit": "m/s MAE", "suggestion": "Tune TECS speed loop and airspeed calibration.", "applicable_modes": ["Mission", "Loiter", "RTL", "Offboard", "Takeoff", "Land", "FollowTarget", "Orbit", "VTOLTakeoff", "AltitudeCruise", "Descend"], "excluded_modes": ["Manual", "Stabilized", "Acro"]},
        "tecs_energy_balance_error": {"warn": 10.0, "problem": 25.0, "unit": "m eq MAE", "suggestion": "Tune TECS total energy balance (pitch/throttle split).", "applicable_modes": ["Mission", "Loiter", "RTL", "Offboard", "Takeoff", "Land", "FollowTarget", "Orbit", "VTOLTakeoff", "AltitudeCruise", "Descend"], "excluded_modes": ["Manual", "Stabilized", "Acro"]},
        "airspeed_consistency": {"warn": 2.5, "problem": 6.0, "unit": "m/s residual RMS", "suggestion": "Check pitot calibration and wind estimation consistency.", "applicable_modes": ["PosCtl", "Mission", "Loiter", "RTL", "Offboard", "Takeoff", "Land", "FollowTarget", "Orbit", "VTOLTakeoff", "AltitudeCruise", "Descend"], "excluded_modes": ["Manual", "Stabilized", "Acro"]},
        "airspeed_scale_appropriateness": {"warn": 0.5, "problem": 2.0, "unit": "m/s gap", "suggestion": "순항 평균 TAS와 GPS velocity-minus-wind 기준 속도의 차이가 ±0.5 m/s 이내인지 확인. 초과하면 wind estimate와 pitot 상태를 확인한 뒤 ASPD_SCALE_n을 보수적으로 검토.", "applicable_modes": ["PosCtl", "Mission", "Loiter", "RTL", "Offboard", "Takeoff", "Land", "FollowTarget", "Orbit", "VTOLTakeoff", "AltitudeCruise", "Descend"], "excluded_modes": ["Manual", "Stabilized", "Acro"]},
        "navigation_path_tracking_error": {"warn": 10.0, "problem": 25.0, "unit": "m MAE", "suggestion": "Review NPFG/path tuning together with roll tracking.", "applicable_modes": ["PosCtl", "Mission", "Loiter", "RTL", "Offboard", "Takeoff", "Land", "FollowTarget", "PrecisionLand", "Orbit", "VTOLTakeoff"], "excluded_modes": ["Manual", "Stabilized", "Acro"]},
        "navigation_path_tracking_reliability": {"warn_score_min": 80.0, "problem_score_min": 60.0, "unit": "score", "suggestion": "Review NAV_ACC_RAD and estimator/local-position validity together with path tail error.", "applicable_modes": ["PosCtl", "Mission", "Loiter", "RTL", "Offboard", "Takeoff", "Land", "FollowTarget", "PrecisionLand", "Orbit", "VTOLTakeoff"], "excluded_modes": ["Manual", "Stabilized", "Acro"], "warn_p95_ratio": 0.8, "problem_p95_ratio": 1.2, "xy_valid_warn": 98.0, "xy_valid_problem": 95.0},
        "throttle_pitch_coupling": {"warn": 60.0, "problem": 80.0, "unit": "corr %", "suggestion": "Review TECS and pitch/throttle loop decoupling."},
        "rpm_stability": {"warn": 5.0, "problem": 12.0, "unit": "% CV", "suggestion": "Inspect rotor governor/RPM control stability."},
        "rotor_harmonic_vibration": {"warn": 12.0, "problem": 22.0, "unit": "dB peak/median", "suggestion": "Inspect rotor harmonics and structural resonance."},
        "control_coupling_index": {"warn": 35.0, "problem": 60.0, "unit": "corr %", "suggestion": "R/P/Y 채널 간 결합(cross-axis coupling)이 큽니다. mixer/모터 매핑, IMU 정렬, CG 균형을 점검하세요."},
        "transition_duration": {"warn": 12.0, "problem": 25.0, "unit": "s", "suggestion": "Tune transition airspeed/thrust schedule."},
        "transition_altitude_loss": {"warn": 8.0, "problem": 20.0, "unit": "m", "suggestion": "Reduce transition altitude dip with pitch/thrust tuning."},
    },
}

DEFAULT_TUNING_GUIDANCE = {
    "vibration": {
        "warning": {
            "likely_causes": ["Minor prop/rotor imbalance", "IMU mounting isolation degradation"],
            "tuning_actions": ["Check prop balance and motor mount torque", "Inspect IMU damping and frame resonance points"],
            "verification_checklist": ["Re-run hover and compare RMS axis trend", "Confirm dominant FFT peak reduced"],
        },
        "problem": {
            "likely_causes": ["Severe resonance or damaged prop/rotor", "Loose mechanical structure near IMU path"],
            "tuning_actions": ["Replace/repair rotating parts", "Rework structural stiffness and isolation before gain tuning"],
            "verification_checklist": ["Short safety flight with vibration watch", "Verify no clipping/saturation in accel signal"],
        },
    },
    "attitude_tracking_error": {
        "warning": {
            "likely_causes": ["Attitude loop gain slightly low", "Rate loop not fully matched to airframe inertia"],
            "tuning_actions": ["Increase attitude P conservatively", "Retune rate loop bandwidth and filter settings"],
            "verification_checklist": ["Step response overshoot < baseline", "Reduced MAE on roll/pitch/yaw"],
        },
        "problem": {
            "likely_causes": ["Attitude/rate loop instability or saturation", "Sensor delay/noise affecting loop closure"],
            "tuning_actions": ["Re-tune from stable baseline gains", "Inspect sensor timing and filtering"],
            "verification_checklist": ["No unstable oscillation in aggressive maneuver", "Controller outputs remain inside margin"],
        },
    },
    "angular_tracking_error": {
        "warning": {
            "likely_causes": ["Rate loop gain or D-term filtering mismatch", "Actuator response lag"],
            "tuning_actions": ["Adjust rate P/D with noise-aware filter tuning", "Check actuator update rate and latency"],
            "verification_checklist": ["Rate MAE/RMS improved on same profile", "Noise amplification not increased"],
        },
        "problem": {
            "likely_causes": ["Rate loop underdamped/overdamped significantly", "Hardware response bottleneck"],
            "tuning_actions": ["Return to known-safe gains and retune incrementally", "Check motor/servo dynamics and power limits"],
            "verification_checklist": ["No sustained oscillation under command steps", "Rate setpoint tracking recovers"],
        },
    },
    "innovation_fail_ratio": {
        "warning": {
            "likely_causes": ["Estimator tuning sensitivity high", "Intermittent sensor quality degradation"],
            "tuning_actions": ["Review EKF noise params and innovation gates", "Validate IMU/GNSS/baro consistency"],
            "verification_checklist": ["Fail ratio decreases on repeat flight", "Estimator status flags stable"],
        },
        "problem": {
            "likely_causes": ["Estimator unhealthy for mission profile", "Major sensor inconsistency or timing issue"],
            "tuning_actions": ["Perform full sensor health check and calibration", "Revisit EKF configuration by airframe"],
            "verification_checklist": ["No persistent innovation fail spikes", "Position/attitude estimate remains coherent"],
        },
    },
    "controller_saturation": {
        "warning": {
            "likely_causes": ["Control authority margin low in some phase", "Trim/weight balance suboptimal"],
            "tuning_actions": ["Reduce aggressive command profile", "Review trim, CG and mixer allocation"],
            "verification_checklist": ["Saturation duration reduced", "Tracking preserved after adjustment"],
        },
        "problem": {
            "likely_causes": ["Insufficient control authority", "Actuator limits reached frequently"],
            "tuning_actions": ["Redesign allocation/actuator sizing", "Tune gains to avoid frequent clipping"],
            "verification_checklist": ["No long saturation plateaus", "Failsafe or mode anomaly not triggered by control loss"],
        },
    },
    "gps_quality": {
        "warning": {
            "likely_causes": ["GPS fix/satellite geometry is marginal", "EKF GPS quality checks intermittently fail"],
            "tuning_actions": ["Check antenna placement, sky visibility, and multipath environment", "Review estimator GPS quality metrics before tightening navigation tuning"],
            "verification_checklist": ["GPS fail-bit ratio decreases on repeat flight", "Position accuracy P95 stays within target range"],
        },
        "problem": {
            "likely_causes": ["GPS quality is insufficient for reliable navigation", "Estimator reports persistent GPS consistency failure"],
            "tuning_actions": ["Resolve GNSS installation/visibility issues before mission tuning", "Inspect antenna, power, EMI, and spoof/jam related logs"],
            "verification_checklist": ["No persistent GPS fail bits remain", "Horizontal/vertical accuracy recovers to expected range"],
        },
    },
    "communication_quality": {
        "warning": {
            "likely_causes": ["Telemetry packet loss or radio buffer pressure is intermittent", "Link margin is adequate only for part of the flight envelope"],
            "tuning_actions": ["Reduce unnecessary telemetry load and confirm antenna orientation", "Check radio placement, baud/rate settings, and link environment"],
            "verification_checklist": ["Packet loss and overrun counters stop increasing abnormally", "Link quality remains stable through the same route/profile"],
        },
        "problem": {
            "likely_causes": ["Telemetry transport is unreliable for the flown mission profile", "Radio or network path is saturated or experiencing repeated errors"],
            "tuning_actions": ["Fix link reliability before relying on offboard/telemetry-dependent operations", "Inspect radio hardware, data rate, buffering, and interference sources"],
            "verification_checklist": ["Packet loss/error counters improve clearly on reflight", "Failsafe or connection-loss events are no longer observed"],
        },
    },
    "failsafe_event_count": {
        "warning": {
            "likely_causes": ["Recoverable failsafe or connection-loss event occurred during flight", "Operational safety margin is narrower than desired"],
            "tuning_actions": ["Review failsafe trigger source and confirm action parameters are appropriate", "Reproduce under controlled conditions and verify whether the event is repeatable"],
            "verification_checklist": ["No repeated recoverable failsafe in similar profile", "Trigger source is identified from messages / status flags"],
        },
        "problem": {
            "likely_causes": ["Failsafe events are frequent for the flight time", "Critical failsafe or termination-level event occurred"],
            "tuning_actions": ["Resolve root cause before further mission flights", "Review RC/data-link/battery/offboard safety configuration and hardware health together"],
            "verification_checklist": ["No critical failsafe remains in repeat test", "Event rate per hour returns to acceptable range"],
        },
    },
    "mission_stability": {
        "warning": {
            "likely_causes": ["Actual nav/mode state diverges intermittently from user intention", "Mission-phase mode transitions are not fully stable"],
            "tuning_actions": ["Review why nav_state diverged from intended mode", "Check mission/offboard/VTOL phase logic before retest"],
            "verification_checklist": ["nav_state divergence ratio decreases on repeat flight", "No unnecessary mode toggling remains in the same mission profile"],
        },
        "problem": {
            "likely_causes": ["Unexpected mode divergence persists for a meaningful portion of flight", "Mission-phase control logic repeatedly leaves intended mode"],
            "tuning_actions": ["Resolve mission logic / estimator / mode-transition cause before further mission flights", "Correlate mode changes with messages, command source, and failsafe triggers"],
            "verification_checklist": ["Mission profile completes without repeated unintended mode changes", "nav_state follows user intention consistently"],
        },
    },
    "tecs_altitude_tracking_error": {
        "warning": {
            "likely_causes": ["TECS height response time constant too slow or too aggressive", "Pitch loop tracking limits TECS altitude correction"],
            "tuning_actions": ["Review altitude response parameters such as FW_T_ALT_TC / FW_T_TIME_CONST", "Check pitch damping and throttle damping before changing multiple TECS gains together"],
            "verification_checklist": ["Altitude MAE reduced in climb/cruise", "Pitch tracking and throttle response improved without new oscillation"],
        },
        "problem": {
            "likely_causes": ["TECS parameters do not match the airframe energy dynamics", "Altitude/airspeed sensing or pitch tracking prevents TECS from closing the loop"],
            "tuning_actions": ["Retune TECS from a conservative baseline after confirming pitch controller health", "Validate altitude and airspeed sources before changing TECS gains"],
            "verification_checklist": ["Recovery after altitude command changes", "No prolonged altitude divergence or pitch-limit driven stall risk"],
        },
    },
    "tecs_airspeed_tracking_error": {
        "warning": {
            "likely_causes": ["TECS speed response or speed weighting is not well matched", "Airspeed scale / pitot installation bias remains"],
            "tuning_actions": ["Review FW_T_TAS_TC and FW_T_SPDWEIGHT together with throttle/pitch split", "Recheck pitot scale and installation before tightening TECS gains"],
            "verification_checklist": ["Airspeed MAE reduced in cruise segments", "Throttle oscillation and altitude sacrifice not increased"],
        },
        "problem": {
            "likely_causes": ["Airspeed sensing is unreliable or badly scaled", "TECS speed loop is unstable for the flown envelope"],
            "tuning_actions": ["Fix airspeed sensing reliability and scale first", "Retune speed loop with conservative bounds and verify altitude/energy tradeoff"],
            "verification_checklist": ["No persistent speed divergence", "Energy management remains stable while altitude remains controllable"],
        },
    },
    "airspeed_scale_appropriateness": {
        "warning": {
            "likely_causes": ["순항 평균 TAS와 GPS-minus-wind 기준 속도의 차이가 ±0.5 m/s 허용 범위를 벗어남", "pitot scale 또는 wind estimate 품질 영향 가능성"],
            "tuning_actions": ["wind estimate와 pitot 상태를 먼저 확인한 뒤 ASPD_SCALE_n을 보수적으로 검토", "한 번에 full-ratio 보정보다 소폭 조정 후 동일 조건 비행으로 재검증"],
            "verification_checklist": ["재비행 후 TAS-air-reference 평균 차이가 ±0.5 m/s 이내로 수렴", "stall/overspeed 경향이 악화되지 않음"],
        },
        "problem": {
            "likely_causes": ["TAS와 GPS-minus-wind 기준 평균 차이가 ±2 m/s 이상으로 크게 벌어짐", "pitot 설치, 튜브 막힘, wind estimate 또는 기수 주변 유동 교란 가능성"],
            "tuning_actions": ["ASPD_SCALE_n을 차이 방향으로 단계 조정 (full-ratio 추정값 참고, 1차 조정은 ±6% 이내로 보수적)", "스케일 보정만으로 차이가 안 줄면 pitot 설치/튜브 상태 점검 후 재비행"],
            "verification_checklist": ["재비행 후 TAS-air-reference 차이가 단조 감소", "속도 관련 다른 평가 항목(airspeed_consistency 등)도 함께 개선"],
        },
    },
    "navigation_path_tracking_error": {
        "warning": {
            "likely_causes": ["NPFG/path tuning is slightly aggressive or too soft", "Roll tracking or wind compensation limits lateral path holding"],
            "tuning_actions": ["Review NPFG_PERIOD with the observed path overshoot/lag", "Check roll setpoint tracking before tightening path-control parameters"],
            "verification_checklist": ["2D flight path overshoot/lag reduces on repeat run", "No new oscillation is introduced in turns"],
        },
        "problem": {
            "likely_causes": ["Path-control tuning does not match the airframe roll dynamics", "Wind/disturbance rejection is not sufficient for the mission profile"],
            "tuning_actions": ["Stabilize roll tracking before aggressive NPFG tuning", "Retune path-control parameters from a conservative baseline"],
            "verification_checklist": ["Large cross-track error or orbit wobble is reduced", "Path tracking improves without unstable turn behavior"],
        },
    },
    "navigation_path_tracking_reliability": {
        "warning": {
            "likely_causes": ["P95 tail error is large relative to NAV_ACC_RAD", "local position validity drops intermittently during path tracking"],
            "tuning_actions": ["Compare NAV_ACC_RAD with actual path tail error and review whether the acceptance radius is overly strict", "Check estimator/GPS/local-position validity before interpreting path tuning only from 2D shape"],
            "verification_checklist": ["path P95/reference ratio improves on repeat run", "xy_valid or v_xy_valid stays close to 100% in the same mission segment"],
        },
        "problem": {
            "likely_causes": ["NAV_ACC_RAD is much tighter than the observed path tail error", "position estimator validity is not stable enough to trust precise path tracking conclusions"],
            "tuning_actions": ["Fix estimator/local-position validity first if xy_valid is dropping", "If operation allows it, review NAV_ACC_RAD against actual vehicle/path capability instead of tuning NPFG blindly"],
            "verification_checklist": ["P95/reference ratio falls back inside the accepted range", "path-validity related flags remain stable in repeat logs"],
        },
    },
    "transition_duration": {
        "warning": {
            "likely_causes": ["Transition schedule conservative", "Insufficient acceleration margin"],
            "tuning_actions": ["Tune transition airspeed/thrust schedule", "Review transition trigger logic"],
            "verification_checklist": ["Transition time reduced with stable attitude", "No added altitude dip"],
        },
        "problem": {
            "likely_causes": ["Transition control design mismatch", "Thrust/pitch strategy inadequate"],
            "tuning_actions": ["Rework transition control envelope", "Validate mode switching conditions"],
            "verification_checklist": ["Consistent transition completion", "No repeated mode toggling anomalies"],
        },
    },
    "transition_altitude_loss": {
        "warning": {
            "likely_causes": ["Pitch/thrust blend during transition suboptimal"],
            "tuning_actions": ["Increase transition thrust support", "Refine pitch target profile in transition"],
            "verification_checklist": ["Altitude loss reduced on repeat transition", "Settle time acceptable post-transition"],
        },
        "problem": {
            "likely_causes": ["Transition strategy causes significant energy drop"],
            "tuning_actions": ["Retune transition controller with safety margin", "Limit transition at low-energy states"],
            "verification_checklist": ["No hazardous altitude dip", "Post-transition control recovers quickly"],
        },
    },
}

DEFAULT_PRIORITY_CHECKS = {
    "vibration": ["vibration spectrum", "actuator output saturation", "rate tracking error"],
    "attitude_tracking_error": ["rate tracking error", "actuator output saturation", "innovation spike"],
    "angular_tracking_error": ["rate tracking error", "actuator output saturation", "vibration spectrum"],
    "innovation_fail_ratio": ["innovation spike", "sensor offset", "mode transition logic impact"],
    "controller_saturation": ["actuator output saturation", "rate tracking error", "TECS response"],
    "gps_quality": ["fix_type / satellites_used", "gps_check_fail_flags", "pos_horiz_accuracy / pos_vert_accuracy"],
    "communication_quality": ["telemetry packet loss", "buffer overruns / rxerrors", "radio txbuf / noise margin"],
    "failsafe_event_count": ["vehicle_status failsafe edge", "logged_messages cause text", "absolute event count"],
    "mission_stability": ["nav_state_user_intention divergence ratio", "mode switch pattern", "mission phase consistency"],
    "tecs_altitude_tracking_error": ["altitude vs altitude setpoint", "pitch tracking / pitch limits", "TECS throttle response"],
    "tecs_airspeed_tracking_error": ["airspeed vs groundspeed", "pitch/throttle energy split", "airspeed scale / wind consistency"],
    "tecs_energy_balance_error": ["altitude+airspeed simultaneous trend", "pitch energy balance response", "throttle total energy response"],
    "airspeed_scale_appropriateness": ["Airspeed vs GPS-minus-wind reference", "순항 구간 TAS 평균 vs air-relative 기준 평균", "ASPD_SCALE_n 설정값"],
    "navigation_path_tracking_error": ["2D flight path", "roll tracking during turns", "NPFG/path tuning"],
    "navigation_path_tracking_reliability": ["path P95 vs NAV_ACC_RAD", "vehicle_local_position.xy_valid/v_xy_valid", "GPS/estimator local-position quality"],
    "transition_duration": ["mode transition logic impact", "actuator output saturation", "rate tracking error"],
    "transition_altitude_loss": ["mode transition logic impact", "TECS response", "airspeed consistency"],
}

DEFAULT_ADVISOR_REGISTRY = {
    "schema_version": "1.0",
    "registry_name": "Log ansys Advisor Registry (Legacy Fallback)",
    "ui_defaults": {
        "cell_template": "{severity_label} | {confidence_text}{candidate_part}",
        "severity_labels": {
            "normal": "정상",
            "warning": "경고",
            "problem": "문제",
            "unavailable": "N/A",
        },
        "tooltip_section_labels": {
            "likely_causes": "Likely Causes",
            "software_actions": "Software Actions",
            "hardware_actions": "Hardware Actions",
            "verification_checklist": "Verification Checklist",
            "candidates": "Candidates",
            "advisor_hint": "Advisor Hint",
            "issue_id": "issue_id",
            "severity": "Severity",
            "confidence": "Confidence",
            "evidence": "Evidence",
        },
        "empty_texts": {
            "tuning_issue_unavailable": "튜닝 이슈를 만들 수 있을 만큼 유효한 근거가 아직 충분하지 않습니다.",
            "priority_checks_unavailable": "우선 확인 항목이 아직 정리되지 않았습니다. evidence와 verification checklist를 먼저 확인하세요.",
            "advisor_description": "Advisor 설명이 비어 있습니다. evidence, likely causes, tuning actions를 우선 확인하세요.",
            "cell_unavailable": "근거부족",
        },
    },
    "advisors": [
        {
            "advisor_id": "advisor_roll_rate_loop",
            "display_name": "Roll Rate Loop",
            "issue_ids": ["rate_loop_roll"],
            "axis": "roll",
            "scope": "rate_loop",
            "target_airframes": ["MULTICOPTER", "VTOL"],
            "required_topics": ["vehicle_rates_setpoint.roll", "vehicle_angular_velocity.xyz[0]"],
            "severity": {"warn": 0.0, "problem": 0.0, "rule": "legacy-rate-loop"},
            "summary": {
                "normal": "rate tracking error, actuator saturation, P/D balance",
                "warning": "rate tracking error, actuator saturation, P/D balance",
                "problem": "rate tracking error, actuator saturation, P/D balance",
            },
            "tooltip": {
                "title": "Roll Rate Loop Advisor",
                "normal": "Roll axis rate-loop tuning assessment.",
                "warning": "Roll axis rate-loop tuning assessment.",
                "problem": "Roll axis rate-loop tuning assessment.",
            },
            "priority_checks": ["rate tracking error", "actuator saturation", "P/D balance"],
            "description": "Roll axis rate-loop tuning assessment",
        },
        {
            "advisor_id": "advisor_pitch_rate_loop",
            "display_name": "Pitch Rate Loop",
            "issue_ids": ["rate_loop_pitch"],
            "axis": "pitch",
            "scope": "rate_loop",
            "target_airframes": ["MULTICOPTER", "VTOL"],
            "required_topics": ["vehicle_rates_setpoint.pitch", "vehicle_angular_velocity.xyz[1]"],
            "severity": {"warn": 0.0, "problem": 0.0, "rule": "legacy-rate-loop"},
            "summary": {
                "normal": "rate tracking error, damping, saturation",
                "warning": "rate tracking error, damping, saturation",
                "problem": "rate tracking error, damping, saturation",
            },
            "tooltip": {
                "title": "Pitch Rate Loop Advisor",
                "normal": "Pitch axis rate-loop tuning assessment.",
                "warning": "Pitch axis rate-loop tuning assessment.",
                "problem": "Pitch axis rate-loop tuning assessment.",
            },
            "priority_checks": ["rate tracking error", "damping", "saturation"],
            "description": "Pitch axis rate-loop tuning assessment",
        },
        {
            "advisor_id": "advisor_yaw_rate_loop",
            "display_name": "Yaw Rate Loop",
            "issue_ids": ["rate_loop_yaw"],
            "axis": "yaw",
            "scope": "rate_loop",
            "target_airframes": ["MULTICOPTER", "VTOL"],
            "required_topics": ["vehicle_rates_setpoint.yaw", "vehicle_angular_velocity.xyz[2]"],
            "severity": {"warn": 0.0, "problem": 0.0, "rule": "legacy-rate-loop"},
            "summary": {
                "normal": "yaw damping, rate response, saturation",
                "warning": "yaw damping, rate response, saturation",
                "problem": "yaw damping, rate response, saturation",
            },
            "tooltip": {
                "title": "Yaw Rate Loop Advisor",
                "normal": "Yaw axis rate-loop tuning assessment.",
                "warning": "Yaw axis rate-loop tuning assessment.",
                "problem": "Yaw axis rate-loop tuning assessment.",
            },
            "priority_checks": ["yaw damping", "rate response", "saturation"],
            "description": "Yaw axis rate-loop tuning assessment",
        },
        {
            "advisor_id": "advisor_vtol_transition_tuning",
            "display_name": "VTOL Transition",
            "issue_ids": ["vtol_transition_tuning"],
            "axis": "transition",
            "scope": "transition",
            "target_airframes": ["VTOL"],
            "required_topics": ["vehicle_status.in_transition_mode", "vehicle_local_position.z"],
            "severity": {"warn": 0.0, "problem": 0.0, "rule": "legacy-transition"},
            "summary": {
                "normal": "transition throttle schedule, mode transition logic, estimator/sensor check",
                "warning": "transition throttle schedule, mode transition logic, estimator/sensor check",
                "problem": "transition throttle schedule, mode transition logic, estimator/sensor check",
            },
            "tooltip": {
                "title": "VTOL Transition Advisor",
                "normal": "VTOL transition tuning assessment.",
                "warning": "VTOL transition tuning assessment.",
                "problem": "VTOL transition tuning assessment.",
            },
            "priority_checks": ["transition throttle schedule", "mode transition logic", "estimator/sensor check"],
            "description": "VTOL transition tuning assessment",
        },
    ],
}


class AutoEvaluationEngine:
    STATUS_LABEL = {"good": "정상", "warning": "경고", "problem": "문제", "unavailable": "평가 불가", "excluded": "평가 제외"}
    AIRFRAME_ALIASES = {
        "MULTICOPTER": "MULTICOPTER",
        "MC": "MULTICOPTER",
        "MULTI_COPTER": "MULTICOPTER",
        "MULTI_ROTOR": "MULTICOPTER",
        "MULTIROTOR": "MULTICOPTER",
        "ROTARY_WING": "MULTICOPTER",
        "FIXED_WING": "FIXED_WING",
        "FIXED-WING": "FIXED_WING",
        "FIXED WING": "FIXED_WING",
        "FW": "FIXED_WING",
        "VTOL": "VTOL",
        "HELICOPTER": "HELICOPTER",
        "HELI": "HELICOPTER",
    }
    NAV_STATE_LABELS = {
        0: "Manual",
        1: "AltCtl",
        2: "PosCtl",
        3: "Mission",
        4: "Loiter",
        5: "RTL",
        6: "PositionSlow",
        8: "AltitudeCruise",
        10: "Acro",
        12: "Descend",
        13: "Termination",
        14: "Offboard",
        15: "Stabilized",
        17: "Takeoff",
        18: "Land",
        19: "FollowTarget",
        20: "PrecisionLand",
        21: "Orbit",
        22: "VTOLTakeoff",
    }
    PATH_RELEVANT_NAV_STATES = {2, 3, 4, 5, 14, 17, 18, 19, 20, 21, 22}
    SETPOINT_EXCLUDED_MODE_LABELS = ["Manual", "Stabilized", "Acro"]
    GENERIC_SETPOINT_APPLICABLE_MODE_LABELS = [
        "AltCtl", "PosCtl", "Mission", "Loiter", "RTL", "PositionSlow",
        "AltitudeCruise", "Descend", "Offboard", "Takeoff", "Land",
        "FollowTarget", "PrecisionLand", "Orbit", "VTOLTakeoff",
    ]
    PATH_SETPOINT_APPLICABLE_MODE_LABELS = [
        "PosCtl", "Mission", "Loiter", "RTL", "Offboard", "Takeoff",
        "Land", "FollowTarget", "PrecisionLand", "Orbit", "VTOLTakeoff",
    ]
    FORWARD_FLIGHT_APPLICABLE_MODE_LABELS = [
        "PosCtl", "Mission", "Loiter", "RTL", "Offboard", "Takeoff",
        "Land", "FollowTarget", "Orbit", "VTOLTakeoff", "AltitudeCruise", "Descend",
    ]
    SAFETY_CRITICAL_MODE_LABELS = {"RTL", "Land", "Takeoff", "VTOLTakeoff", "PrecisionLand"}
    GPS_CHECK_FAIL_LABELS = {
        0: "GPS fix",
        1: "Min sat count",
        2: "Max PDOP",
        3: "Horizontal error",
        4: "Vertical error",
        5: "Speed error",
        6: "Horizontal drift",
        7: "Vertical drift",
        8: "Horizontal speed",
        9: "Vertical speed",
        10: "Spoofed",
        11: "Jammed",
    }
    FAILSAFE_MESSAGE_PATTERNS = {
        "rc_loss": [r"\brc\b.*loss", r"manual control.*lost", r"rc signal.*lost"],
        "data_link_loss": [r"data ?link.*lost", r"datalink.*lost", r"gcs connection.*lost", r"telemetry.*lost"],
        "offboard_loss": [r"offboard.*lost", r"offboard.*failsafe"],
        "battery": [r"battery.*(low|critical|emergency)", r"low battery", r"critical battery"],
        "geofence": [r"geofence"],
        "termination": [r"terminate", r"termination"],
        "motor_failure": [r"motor failure", r"failure detector"],
    }

    def __init__(self, thresholds_path: Optional[str] = None):
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if not thresholds_path:
            thresholds_path = os.path.join(self.repo_root, "config", "thresholds.yaml")
        self.cfg = self._load_thresholds(thresholds_path)
        self.tuning_rules = self._load_yaml_file(os.path.join(self.repo_root, "config", "tuning_rules.yaml"), default={})
        self.metrics_catalog = self._load_yaml_file(os.path.join(self.repo_root, "config", "metrics_catalog.yaml"), default={})
        self.eval_profile = self._load_yaml_file(
            os.path.join(self.repo_root, "config", "profiles", "general_controller_tuning.yaml"),
            default={},
        )
        self.advisor_registry, self.advisor_registry_info = self._load_advisor_registry(
            os.path.join(self.repo_root, "config", "advisor_registry.yaml")
        )
        self.advisor_registry_index = self._build_advisor_registry_index(self.advisor_registry)
        self.parameter_recommendations_config = self._load_yaml_file(
            os.path.join(self.repo_root, "config", "parameter_recommendations.yaml"),
            default={},
        )
        self.feature_extractor = ControlFeatureExtractor(self.tuning_rules) if ControlFeatureExtractor is not None else None
        self.tuning_inference = TuningInferenceEngine(self.tuning_rules) if TuningInferenceEngine is not None else None
        self.recommendation_engine = RecommendationEngine() if RecommendationEngine is not None else None
        self.parameter_recommender = (
            ParameterRecommender(self.parameter_recommendations_config)
            if ParameterRecommender is not None
            else None
        )

    @staticmethod
    def _deep_update(dst: dict, src: dict) -> dict:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                AutoEvaluationEngine._deep_update(dst[k], v)
            else:
                dst[k] = v
        return dst

    def _load_thresholds(self, path: str) -> dict:
        cfg = copy.deepcopy(DEFAULT_THRESHOLDS)
        if not path or not os.path.isfile(path) or yaml is None:
            return cfg
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                self._deep_update(cfg, loaded)
        except Exception:
            pass
        return cfg

    @staticmethod
    def _load_yaml_file(path: str, default=None):
        if default is None:
            default = {}
        if not path or not os.path.isfile(path) or yaml is None:
            return copy.deepcopy(default)
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
        return copy.deepcopy(default)

    def _load_advisor_registry(self, path: str) -> Tuple[dict, dict]:
        # Explicit safe fallback when file is missing or YAML parser unavailable.
        if (not path) or (not os.path.isfile(path)) or (yaml is None):
            fallback_reg, fb_info = self._validate_advisor_registry(copy.deepcopy(DEFAULT_ADVISOR_REGISTRY))
            return fallback_reg, {
                "loaded": False,
                "fallback": True,
                "reason": "file_missing_or_yaml_unavailable",
                "errors": fb_info.get("errors", []) if isinstance(fb_info, dict) else [],
                "advisor_count": len(fallback_reg.get("advisors", [])) if isinstance(fallback_reg, dict) else 0,
            }

        loaded = self._load_yaml_file(path, default={})
        validated, info = self._validate_advisor_registry(loaded)
        if (not isinstance(validated, dict)) or (not isinstance(validated.get("advisors", []), list)) or (len(validated.get("advisors", [])) == 0):
            # Hard fallback to in-code default
            fallback_reg, fb_info = self._validate_advisor_registry(copy.deepcopy(DEFAULT_ADVISOR_REGISTRY))
            return fallback_reg, {
                "loaded": False,
                "fallback": True,
                "reason": "invalid_or_empty_registry",
                "errors": (info.get("errors", []) if isinstance(info, dict) else []) + (fb_info.get("errors", []) if isinstance(fb_info, dict) else []),
                "advisor_count": len(fallback_reg.get("advisors", [])) if isinstance(fallback_reg, dict) else 0,
            }
        return validated, info

    def _validate_advisor_registry(self, registry: dict) -> Tuple[dict, dict]:
        valid_axis = {"roll", "pitch", "yaw", "transition", "generic", "acceleration", "tilt"}
        valid_scope = {
            "rate_loop",
            "attitude_loop",
            "rate_limit",
            "acc_limit",
            "tilt_limit",
            "transition",
            "control",
            "estimation",
            "actuation",
            "energy",
            "airframe",
            "all",
            "forward_flight",
        }
        errors = []
        out = copy.deepcopy(registry if isinstance(registry, dict) else {})
        ui_defaults = out.get("ui_defaults", {}) if isinstance(out.get("ui_defaults", {}), dict) else {}
        severity_labels = ui_defaults.get("severity_labels", {}) if isinstance(ui_defaults.get("severity_labels", {}), dict) else {}
        tooltip_labels = ui_defaults.get("tooltip_section_labels", {}) if isinstance(ui_defaults.get("tooltip_section_labels", {}), dict) else {}
        empty_texts = ui_defaults.get("empty_texts", {}) if isinstance(ui_defaults.get("empty_texts", {}), dict) else {}
        normalized_ui_defaults = {
            "cell_template": str(ui_defaults.get("cell_template", "{severity_label} | {confidence_text}{candidate_part}")).strip()
            or "{severity_label} | {confidence_text}{candidate_part}",
            "severity_labels": {
                "normal": str(severity_labels.get("normal", "정상")).strip() or "정상",
                "warning": str(severity_labels.get("warning", "경고")).strip() or "경고",
                "problem": str(severity_labels.get("problem", "문제")).strip() or "문제",
                "unavailable": str(severity_labels.get("unavailable", "N/A")).strip() or "N/A",
            },
            "tooltip_section_labels": {
                "likely_causes": str(tooltip_labels.get("likely_causes", "Likely Causes")).strip() or "Likely Causes",
                "software_actions": str(tooltip_labels.get("software_actions", "Software Actions")).strip() or "Software Actions",
                "hardware_actions": str(tooltip_labels.get("hardware_actions", "Hardware Actions")).strip() or "Hardware Actions",
                "verification_checklist": str(tooltip_labels.get("verification_checklist", "Verification Checklist")).strip() or "Verification Checklist",
                "candidates": str(tooltip_labels.get("candidates", "Candidates")).strip() or "Candidates",
                "advisor_hint": str(tooltip_labels.get("advisor_hint", "Advisor Hint")).strip() or "Advisor Hint",
                "issue_id": str(tooltip_labels.get("issue_id", "issue_id")).strip() or "issue_id",
                "severity": str(tooltip_labels.get("severity", "Severity")).strip() or "Severity",
                "confidence": str(tooltip_labels.get("confidence", "Confidence")).strip() or "Confidence",
                "evidence": str(tooltip_labels.get("evidence", "Evidence")).strip() or "Evidence",
            },
            "empty_texts": {
                "tuning_issue_unavailable": str(empty_texts.get("tuning_issue_unavailable", "튜닝 이슈를 만들 수 있을 만큼 유효한 근거가 아직 충분하지 않습니다.")).strip() or "튜닝 이슈를 만들 수 있을 만큼 유효한 근거가 아직 충분하지 않습니다.",
                "priority_checks_unavailable": str(empty_texts.get("priority_checks_unavailable", "우선 확인 항목이 아직 정리되지 않았습니다. evidence와 verification checklist를 먼저 확인하세요.")).strip() or "우선 확인 항목이 아직 정리되지 않았습니다. evidence와 verification checklist를 먼저 확인하세요.",
                "advisor_description": str(empty_texts.get("advisor_description", "Advisor 설명이 비어 있습니다. evidence, likely causes, tuning actions를 우선 확인하세요.")).strip() or "Advisor 설명이 비어 있습니다. evidence, likely causes, tuning actions를 우선 확인하세요.",
                "cell_unavailable": str(empty_texts.get("cell_unavailable", "근거부족")).strip() or "근거부족",
            },
        }
        advisors = out.get("advisors", [])
        if not isinstance(advisors, list):
            advisors = []
        normalized = []

        for idx, adv in enumerate(advisors):
            if not isinstance(adv, dict):
                errors.append(f"advisor[{idx}] is not a dict")
                continue
            aid = str(adv.get("advisor_id", "")).strip()
            if not aid:
                errors.append(f"advisor[{idx}] missing advisor_id")
                continue

            disp = str(adv.get("display_name", aid)).strip() or aid
            issue_ids = adv.get("issue_ids", [])
            if isinstance(issue_ids, str):
                issue_ids = [issue_ids]
            if not isinstance(issue_ids, list):
                issue_ids = []
            issue_ids = [str(x).strip() for x in issue_ids if str(x).strip()]
            if not issue_ids:
                # Legacy fallback linkage
                issue_ids = [aid.replace("advisor_", "")]

            axis = str(adv.get("axis", "generic")).strip().lower()
            scope = str(adv.get("scope", "all")).strip().lower()
            if axis not in valid_axis:
                errors.append(f"{aid}: invalid axis '{axis}', fallback to generic")
                axis = "generic"
            if scope not in valid_scope:
                errors.append(f"{aid}: invalid scope '{scope}', fallback to all")
                scope = "all"

            target_airframes = adv.get("target_airframes", [])
            if isinstance(target_airframes, str):
                target_airframes = [target_airframes]
            if not isinstance(target_airframes, list):
                target_airframes = []
            target_airframes = [str(x).strip().upper() for x in target_airframes if str(x).strip()]
            if not target_airframes:
                target_airframes = ["MULTICOPTER", "FIXED_WING", "VTOL", "HELICOPTER"]
                errors.append(f"{aid}: missing target_airframes, applied default")

            required_topics = adv.get("required_topics", [])
            if isinstance(required_topics, str):
                required_topics = [required_topics]
            if not isinstance(required_topics, list):
                required_topics = []
            required_topics = [str(x).strip() for x in required_topics if str(x).strip()]
            if not required_topics:
                errors.append(f"{aid}: missing required_topics")

            severity = adv.get("severity", {}) if isinstance(adv.get("severity", {}), dict) else {}
            severity.setdefault("warn", 0.0)
            severity.setdefault("problem", 0.0)
            severity.setdefault("rule", "registry-default")

            summary = adv.get("summary", {}) if isinstance(adv.get("summary", {}), dict) else {}
            summary.setdefault("normal", disp)
            summary.setdefault("warning", disp)
            summary.setdefault("problem", disp)

            tooltip = adv.get("tooltip", {}) if isinstance(adv.get("tooltip", {}), dict) else {}
            tooltip.setdefault("title", disp)
            tooltip.setdefault("normal", f"{disp} advisor")
            tooltip.setdefault("warning", f"{disp} advisor")
            tooltip.setdefault("problem", f"{disp} advisor")

            normalized.append(
                {
                    **adv,
                    "advisor_id": aid,
                    "display_name": disp,
                    "issue_ids": issue_ids,
                    "axis": axis,
                    "scope": scope,
                    "target_airframes": target_airframes,
                    "required_topics": required_topics,
                    "severity": severity,
                    "summary": summary,
                    "tooltip": tooltip,
                    "priority_checks": adv.get("priority_checks", []) if isinstance(adv.get("priority_checks", []), list) else [],
                    "description": str(adv.get("description", "")).strip(),
                }
            )

        out["ui_defaults"] = normalized_ui_defaults
        out["advisors"] = normalized
        info = {
            "loaded": True,
            "fallback": False,
            "errors": errors,
            "advisor_count": len(normalized),
        }
        return out, info

    @staticmethod
    def _build_advisor_registry_index(registry: dict) -> Dict[str, List[dict]]:
        idx: Dict[str, List[dict]] = {}
        if not isinstance(registry, dict):
            return idx
        for adv in registry.get("advisors", []) if isinstance(registry.get("advisors", []), list) else []:
            if not isinstance(adv, dict):
                continue
            priority = adv.get("priority", 999)
            try:
                priority = int(priority)
            except Exception:
                priority = 999
            for issue_id in adv.get("issue_ids", []) if isinstance(adv.get("issue_ids", []), list) else []:
                iid = str(issue_id).strip()
                if not iid:
                    continue
                idx.setdefault(iid, []).append((priority, adv))
        for k in list(idx.keys()):
            idx[k] = [x[1] for x in sorted(idx[k], key=lambda t: t[0])]
        return idx

    def _advisor_meta_for_issue(self, issue_id: str, airframe: str) -> Optional[dict]:
        candidates = self.advisor_registry_index.get(str(issue_id).strip(), [])
        if not candidates:
            return None
        af = str(airframe or "").upper().strip()
        for adv in candidates:
            tars = adv.get("target_airframes", [])
            if not isinstance(tars, list):
                continue
            if af in [str(x).upper().strip() for x in tars]:
                return adv
        return candidates[0] if candidates else None

    @staticmethod
    def _first_nonempty_text(values) -> str:
        for value in values or []:
            text = str(value or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def _compact_evidence_text(text: str, limit: int = 120) -> str:
        raw = " ".join(str(text or "").strip().split())
        if not raw:
            return ""
        if len(raw) <= limit:
            return raw
        return raw[: max(0, limit - 3)].rstrip() + "..."

    @staticmethod
    def _is_weak_advisor_text(text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return True
        low = raw.lower()
        weak_texts = {
            "--",
            "-",
            "n/a",
            "priority checks unavailable",
            "advisor detail unavailable",
            "튜닝 이슈 정보 없음",
            "튜닝 이슈를 만들 수 있을 만큼 유효한 근거가 아직 충분하지 않습니다.",
            "우선 확인 항목이 아직 정리되지 않았습니다. evidence와 verification checklist를 먼저 확인하세요.",
            "advisor 설명이 비어 있습니다. evidence, likely causes, tuning actions를 우선 확인하세요.",
        }
        if low in weak_texts:
            return True
        if "tuning assessment" in low or "advisor detail unavailable" in low:
            return True
        if "," in raw and len(raw.split()) <= 8 and "." not in raw and "?" not in raw:
            return True
        return False

    def _build_issue_summary_text(self, issue: dict) -> str:
        if not isinstance(issue, dict):
            return ""
        severity = str(issue.get("severity", "normal")).strip().lower()
        priority_checks = issue.get("priority_checks", []) if isinstance(issue.get("priority_checks", []), list) else []
        likely_causes = issue.get("likely_causes", []) if isinstance(issue.get("likely_causes", []), list) else []
        evidence = self._compact_evidence_text(issue.get("evidence", ""), limit=96)

        primary_check = self._first_nonempty_text(priority_checks)
        primary_cause = self._first_nonempty_text(likely_causes)
        anchor = primary_check or primary_cause or evidence

        if severity == "problem":
            if anchor:
                return f"{anchor} 중심으로 우선 점검이 필요합니다."
            return "문제 등급입니다. 현재 로그 근거를 기준으로 원인 분리와 우선 점검이 필요합니다."
        if severity == "warning":
            if anchor:
                return f"{anchor} 경향이 보여 추가 점검이 필요합니다."
            return "경고 등급입니다. 현재 로그 근거를 바탕으로 추가 점검이 필요합니다."
        if anchor:
            return f"{anchor} 기준으로 추세만 계속 확인하세요."
        return "현재는 큰 이상보다 추세 확인 단계입니다."

    def _build_issue_advisor_description(self, issue: dict) -> str:
        if not isinstance(issue, dict):
            return ""
        severity = str(issue.get("severity", "normal")).strip().lower()
        likely_causes = issue.get("likely_causes", []) if isinstance(issue.get("likely_causes", []), list) else []
        priority_checks = issue.get("priority_checks", []) if isinstance(issue.get("priority_checks", []), list) else []
        software_actions = issue.get("software_actions", []) if isinstance(issue.get("software_actions", []), list) else []
        hardware_actions = issue.get("hardware_actions", []) if isinstance(issue.get("hardware_actions", []), list) else []
        tuning_actions = issue.get("tuning_actions", []) if isinstance(issue.get("tuning_actions", []), list) else []
        evidence = self._compact_evidence_text(issue.get("evidence", ""))

        primary_check = self._first_nonempty_text(priority_checks)
        primary_cause = self._first_nonempty_text(likely_causes)
        primary_action = self._first_nonempty_text(software_actions) or self._first_nonempty_text(hardware_actions) or self._first_nonempty_text(tuning_actions)

        if severity == "problem":
            if primary_check and primary_action:
                return f"{primary_check}를 우선 확인하고, {primary_action}"
            if primary_check and primary_cause:
                return f"{primary_check}와 {primary_cause}를 기준으로 원인 분리가 필요합니다."
            if evidence:
                return f"문제 등급입니다. 현재 근거는 {evidence}"
            return "문제 등급입니다. evidence와 likely causes를 기준으로 우선 원인 분리가 필요합니다."
        if severity == "warning":
            if primary_check and primary_action:
                return f"{primary_check}를 확인하면서 {primary_action}"
            if primary_cause:
                return f"{primary_cause} 가능성을 먼저 확인하세요."
            if evidence:
                return f"경고 등급입니다. 현재 근거는 {evidence}"
            return "경고 등급입니다. evidence와 priority checks를 기준으로 보수적으로 점검하세요."
        if primary_check:
            return f"{primary_check} 기준으로 추세 점검을 이어가세요."
        if evidence:
            return f"현재는 큰 이상보다 추세 확인 단계입니다. 관측 근거: {evidence}"
        return "현재는 큰 이상보다 추세 확인 단계입니다."

    def _build_issue_tooltip_hint(self, issue: dict) -> str:
        if not isinstance(issue, dict):
            return ""
        severity = str(issue.get("severity", "normal")).strip().lower()
        priority_checks = issue.get("priority_checks", []) if isinstance(issue.get("priority_checks", []), list) else []
        tuning_actions = issue.get("tuning_actions", []) if isinstance(issue.get("tuning_actions", []), list) else []
        software_actions = issue.get("software_actions", []) if isinstance(issue.get("software_actions", []), list) else []
        hardware_actions = issue.get("hardware_actions", []) if isinstance(issue.get("hardware_actions", []), list) else []

        primary_check = self._first_nonempty_text(priority_checks)
        primary_action = self._first_nonempty_text(tuning_actions) or self._first_nonempty_text(software_actions) or self._first_nonempty_text(hardware_actions)

        if severity == "problem":
            if primary_check and primary_action:
                return f"{primary_check}를 먼저 확인한 뒤 {primary_action}"
            if primary_check:
                return f"{primary_check}를 먼저 확인하세요."
        elif severity == "warning":
            if primary_check and primary_action:
                return f"{primary_check}를 확인하면서 {primary_action}"
            if primary_check:
                return f"{primary_check}를 기준으로 추가 점검하세요."
        return self._build_issue_summary_text(issue)

    def _apply_registry_to_issue(self, issue: dict, airframe: str) -> dict:
        if not isinstance(issue, dict):
            return issue
        issue_id = str(issue.get("issue_id", "")).strip()
        if not issue_id:
            return issue
        adv = self._advisor_meta_for_issue(issue_id, airframe)
        if not isinstance(adv, dict):
            issue["registry_source"] = "legacy_fallback"
            return issue

        sev = str(issue.get("severity", "normal")).strip().lower()
        if sev not in ("normal", "warning", "problem"):
            sev = "normal"
        ui_defaults = self.advisor_registry.get("ui_defaults", {}) if isinstance(self.advisor_registry, dict) else {}
        if not isinstance(ui_defaults, dict):
            ui_defaults = {}
        issue["registry_source"] = "registry"
        issue["advisor_id"] = str(adv.get("advisor_id", "")).strip()
        issue["advisor_display_name"] = str(adv.get("display_name", issue.get("title", issue_id))).strip()
        try:
            issue["advisor_priority"] = int(adv.get("priority", 999))
        except Exception:
            issue["advisor_priority"] = 999
        issue["axis"] = str(adv.get("axis", issue.get("axis", ""))).strip()
        issue["scope"] = str(adv.get("scope", issue.get("scope", ""))).strip()
        issue["advisor_required_topics"] = adv.get("required_topics", [])
        issue["advisor_description"] = str(adv.get("description", issue.get("advisor_description", ""))).strip()
        existing_priority_checks = issue.get("priority_checks", []) if isinstance(issue.get("priority_checks", []), list) else []
        registry_priority_checks = adv.get("priority_checks", []) if isinstance(adv.get("priority_checks", []), list) else []
        issue["priority_checks"] = registry_priority_checks if registry_priority_checks else existing_priority_checks

        summary_map = adv.get("summary", {}) if isinstance(adv.get("summary", {}), dict) else {}
        issue["summary_text"] = str(summary_map.get(sev, summary_map.get("normal", issue.get("summary_text", "")))).strip()

        tip_map = adv.get("tooltip", {}) if isinstance(adv.get("tooltip", {}), dict) else {}
        issue["tooltip_title"] = str(tip_map.get("title", issue.get("advisor_display_name", ""))).strip()
        issue["tooltip_template"] = str(tip_map.get(sev, tip_map.get("normal", issue.get("tooltip_template", "")))).strip()
        issue["ui_cell_template"] = str(ui_defaults.get("cell_template", "")).strip()
        issue["ui_severity_labels"] = copy.deepcopy(ui_defaults.get("severity_labels", {})) if isinstance(ui_defaults.get("severity_labels", {}), dict) else {}
        issue["ui_tooltip_section_labels"] = copy.deepcopy(ui_defaults.get("tooltip_section_labels", {})) if isinstance(ui_defaults.get("tooltip_section_labels", {}), dict) else {}
        issue["ui_empty_texts"] = copy.deepcopy(ui_defaults.get("empty_texts", {})) if isinstance(ui_defaults.get("empty_texts", {}), dict) else {}

        if self._is_weak_advisor_text(issue.get("summary_text", "")):
            issue["summary_text"] = self._build_issue_summary_text(issue)
        if self._is_weak_advisor_text(issue.get("advisor_description", "")):
            issue["advisor_description"] = self._build_issue_advisor_description(issue)
        if self._is_weak_advisor_text(issue.get("tooltip_template", "")):
            issue["tooltip_template"] = self._build_issue_tooltip_hint(issue)
        return issue

    def _apply_registry_to_axis_advisor(self, advisor: dict, issue: dict, airframe: str) -> dict:
        if not isinstance(advisor, dict):
            advisor = {}
        if not isinstance(issue, dict):
            return advisor
        issue_id = str(issue.get("issue_id", "")).strip()
        adv = self._advisor_meta_for_issue(issue_id, airframe)
        if not isinstance(adv, dict):
            return advisor
        merged = dict(advisor)
        merged["advisor_id"] = str(adv.get("advisor_id", "")).strip()
        merged["display_name"] = str(adv.get("display_name", issue.get("title", issue_id))).strip()
        merged["axis"] = str(adv.get("axis", merged.get("axis", ""))).strip()
        merged["scope"] = str(adv.get("scope", "")).strip()
        merged["summary_text"] = str(issue.get("summary_text", "")).strip()
        merged["tooltip_template"] = str(issue.get("tooltip_template", "")).strip()
        merged["priority_checks"] = adv.get("priority_checks", [])
        return merged

    def _metric_cfg(self, metric_key: str, airframe: Optional[str] = None) -> dict:
        base = copy.deepcopy(self.cfg.get("metrics", {}).get(metric_key, {}))
        af = str(airframe or getattr(self, "_eval_airframe_ctx", "") or "").upper().strip()
        # Preferred: top-level metrics_airframe map
        af_cfg = self.cfg.get("metrics_airframe", {}).get(af, {}) if af else {}
        if isinstance(af_cfg, dict):
            override = af_cfg.get(metric_key)
            if isinstance(override, dict):
                self._deep_update(base, override)
        # Also allow per-metric embedded override map
        embedded = base.get("airframe_overrides")
        if isinstance(embedded, dict) and af and isinstance(embedded.get(af), dict):
            self._deep_update(base, embedded.get(af, {}))
        base.pop("airframe_overrides", None)
        return base

    def _metric_weight(self, metric_key: str, airframe: Optional[str] = None) -> float:
        w = float(self.cfg.get("weights", {}).get(metric_key, 1.0))
        af = str(airframe or getattr(self, "_eval_airframe_ctx", "") or "").upper().strip()
        if af:
            ow = self.cfg.get("weights_airframe", {}).get(af, {}).get(metric_key, None)
            if ow is not None:
                try:
                    w = float(ow)
                except Exception:
                    pass
        return w

    @classmethod
    def _normalize_airframe(cls, aircraft_type: str) -> str:
        key = str(aircraft_type or "").strip().upper()
        key = key.replace("-", "_").replace(" ", "_")
        return cls.AIRFRAME_ALIASES.get(key, key or "UNKNOWN")

    @staticmethod
    def _find_topic(dataset, prefixes) -> Optional[str]:
        if dataset is None or not hasattr(dataset, "topics"):
            return None
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        for p in prefixes:
            if p in dataset.topics:
                return p
        for p in prefixes:
            for topic in dataset.topics.keys():
                if topic.startswith(p):
                    return topic
        return None

    @staticmethod
    def _find_signal(df: pl.DataFrame, candidates: List[str]) -> Optional[str]:
        cols = set(df.columns)
        for c in candidates:
            if c in cols:
                return c
        return None

    @staticmethod
    def _with_timestamp_sec(df: pl.DataFrame) -> pl.DataFrame:
        if "timestamp_sec" in df.columns:
            return df
        if "timestamp" in df.columns:
            return df.with_columns((pl.col("timestamp").cast(pl.Float64) / pl.lit(1_000_000.0)).alias("timestamp_sec"))
        return df

    @staticmethod
    def _to_float(arr) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float64)
        return x[np.isfinite(x)]

    @staticmethod
    def _score_low(value: float, warn: float, problem: float) -> Tuple[float, str]:
        warn = max(float(warn), 1e-9)
        problem = max(float(problem), warn + 1e-9)
        v = float(value)
        if v <= warn:
            return max(80.0, 100.0 - 20.0 * v / warn), "good"
        if v <= problem:
            return max(40.0, 80.0 - 40.0 * (v - warn) / (problem - warn)), "warning"
        return max(5.0, 40.0 - 35.0 * (v - problem) / max(problem, 1e-9)), "problem"

    @staticmethod
    def _status_from_score(score: Optional[float], good_min: float = 80.0, warn_min: float = 60.0) -> str:
        if score is None:
            return "unavailable"
        if score >= good_min:
            return "good"
        if score >= warn_min:
            return "warning"
        return "problem"

    @staticmethod
    def _count_rising_edges(x: np.ndarray, thr: float = 0.5) -> int:
        if x.size <= 1:
            return int(x.size == 1 and x[0] > thr)
        b = x > thr
        return int(np.sum((~b[:-1]) & b[1:]))

    @staticmethod
    def _confidence(sample_count: int, required_topics: int, found_topics: int, quality: float = 1.0) -> float:
        sf = min(1.0, float(max(sample_count, 0)) / 200.0)
        tf = float(found_topics) / float(max(required_topics, 1))
        q = min(1.0, max(0.0, float(quality)))
        return round(max(0.0, min(100.0, 100.0 * sf * tf * q)), 1)

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, float(value)))

    @classmethod
    def _semantic_factor(cls, reference_class: Optional[str]) -> float:
        mapping = {
            "px4_explicit": 1.00,
            "px4_configured_threshold": 0.95,
            "product_default_threshold": 0.85,
            "heuristic_inference": 0.70,
            "derived_from_px4": 0.88,
            "mixed_reference": 0.90,
        }
        return float(mapping.get(str(reference_class or "").strip(), 0.80))

    @classmethod
    def _confidence_details(
        cls,
        sample_count: int,
        required_topics: int,
        found_topics: int,
        *,
        quality: float = 1.0,
        mode_factor: float = 1.0,
        reference_class: Optional[str] = None,
        fallback_penalty: float = 0.0,
    ) -> Tuple[float, dict]:
        sample_factor = cls._clamp(float(max(sample_count, 0)) / 200.0)
        topic_factor = cls._clamp(float(max(found_topics, 0)) / float(max(required_topics, 1)))
        quality_factor = cls._clamp(quality)
        mode_factor = cls._clamp(mode_factor)
        semantic_factor = cls._semantic_factor(reference_class)
        fallback_factor = cls._clamp(1.0 - max(0.0, float(fallback_penalty)))
        total = round(
            max(
                0.0,
                min(
                    100.0,
                    100.0 * sample_factor * topic_factor * quality_factor * mode_factor * semantic_factor * fallback_factor,
                ),
            ),
            1,
        )
        return total, {
            "sample_factor": round(sample_factor, 3),
            "topic_factor": round(topic_factor, 3),
            "quality_factor": round(quality_factor, 3),
            "mode_factor": round(mode_factor, 3),
            "semantic_factor": round(semantic_factor, 3),
            "fallback_penalty": round(max(0.0, float(fallback_penalty)), 3),
            "fallback_factor": round(fallback_factor, 3),
            "formula": "100 * sample * topic * quality * mode * semantic * fallback",
        }

    @staticmethod
    def _score_high(value: float, warn_min: float, problem_min: float) -> Tuple[float, str]:
        warn_min = float(warn_min)
        problem_min = min(float(problem_min), warn_min - 1e-9)
        v = float(value)
        if v >= warn_min:
            score = 80.0 + 20.0 * (v - warn_min) / max(100.0 - warn_min, 1e-9)
            return min(100.0, max(80.0, score)), "good"
        if v >= problem_min:
            score = 40.0 + 40.0 * (v - problem_min) / max(warn_min - problem_min, 1e-9)
            return min(80.0, max(40.0, score)), "warning"
        score = 40.0 * v / max(problem_min, 1e-9)
        return max(5.0, min(40.0, score)), "problem"

    @staticmethod
    def _topic_root(name: str) -> str:
        txt = str(name or "").strip()
        if not txt:
            return ""
        return txt.split(".", 1)[0].strip()

    @classmethod
    def _missing_topics_from_required(cls, required_topics, found_topics) -> List[str]:
        found = [cls._topic_root(x) for x in list(found_topics or []) if cls._topic_root(x)]
        missing = []
        for req in list(required_topics or []):
            req_txt = str(req or "").strip()
            if not req_txt:
                continue
            options = []
            for part in req_txt.split("|"):
                root = cls._topic_root(part)
                if root:
                    options.append(root)
            if options and not any(any(ft == opt or ft.startswith(opt) for ft in found) for opt in options):
                missing.append(req_txt)
        return missing

    def _extract_px4_version(self, firmware) -> dict:
        info = firmware if isinstance(firmware, dict) else {}
        relevant = []
        version_hint = ""
        for key, value in info.items():
            key_txt = str(key or "").strip()
            val_txt = str(value or "").strip()
            if not key_txt and not val_txt:
                continue
            low = f"{key_txt} {val_txt}".lower()
            if any(token in low for token in ("ver_sw", "version", "git", "hash", "branch", "px4", "release", "build")):
                relevant.append(f"{key_txt}={val_txt}")
            if (not version_hint) and val_txt.lower().startswith("v") and any(ch.isdigit() for ch in val_txt):
                version_hint = val_txt
        if not version_hint:
            for text in relevant:
                parts = text.split("=", 1)
                cand = parts[1].strip() if len(parts) > 1 else text
                if cand.lower().startswith("v") and any(ch.isdigit() for ch in cand):
                    version_hint = cand
                    break
        summary = version_hint or (relevant[0] if relevant else "Unknown")
        return {
            "summary": summary,
            "version_hint": version_hint,
            "firmware_lines": relevant[:8],
        }

    def _flight_time_sec(self, dataset) -> float:
        if dataset is None or not hasattr(dataset, "topics"):
            return 0.0
        t_min = None
        t_max = None
        for topic in dataset.topics.values():
            try:
                df = self._with_timestamp_sec(topic.dataframe)
            except Exception:
                continue
            if "timestamp_sec" not in df.columns or df.height <= 0:
                continue
            arr = self._to_float(df["timestamp_sec"].to_numpy())
            if arr.size <= 0:
                continue
            cur_min = float(np.min(arr))
            cur_max = float(np.max(arr))
            t_min = cur_min if t_min is None else min(t_min, cur_min)
            t_max = cur_max if t_max is None else max(t_max, cur_max)
        if t_min is None or t_max is None:
            return 0.0
        return max(0.0, float(t_max - t_min))

    def _extract_nav_state_series(self, dataset) -> Tuple[Optional[pl.DataFrame], Optional[str], Optional[str]]:
        topic = self._find_topic(dataset, "vehicle_status")
        if not topic:
            return None, None, None
        try:
            df = self._with_timestamp_sec(dataset.topics[topic].dataframe)
            if "timestamp_sec" not in df.columns:
                return None, topic, None
            sig = self._find_signal(df, ["nav_state", "nav_state_user_intention", "main_state"])
            if not sig:
                return None, topic, None
            out = df.select([
                pl.col("timestamp_sec").cast(pl.Float64),
                pl.col(sig).cast(pl.Float64).alias("nav_state"),
            ]).drop_nulls().sort("timestamp_sec")
            if out.height <= 0:
                return None, topic, sig
            return out, topic, sig
        except Exception:
            return None, topic, None

    @classmethod
    def _nav_state_label(cls, raw) -> str:
        try:
            nav_int = int(round(float(raw)))
        except Exception:
            return ""
        return cls.NAV_STATE_LABELS.get(nav_int, f"State{nav_int}")

    @classmethod
    def _nav_state_labels(cls, nav_values) -> List[str]:
        if nav_values is None:
            nav_iter = []
        else:
            nav_iter = list(nav_values)
        labels = []
        for raw in nav_iter:
            label = cls._nav_state_label(raw)
            if not label:
                continue
            if label not in labels:
                labels.append(label)
        return labels

    @classmethod
    def _normalize_mode_labels(cls, modes) -> List[str]:
        if modes is None:
            raw_modes = []
        elif isinstance(modes, str):
            raw_modes = [modes]
        else:
            raw_modes = list(modes)
        known = {str(v).strip().lower(): str(v).strip() for v in cls.NAV_STATE_LABELS.values()}
        normalized = []
        for raw in raw_modes:
            text = str(raw or "").strip()
            if not text:
                continue
            if re.fullmatch(r"-?\d+", text):
                text = cls._nav_state_label(int(text))
            else:
                text = known.get(text.lower(), text)
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    def _metric_mode_policy(
        self,
        metric_key: str,
        *,
        airframe: Optional[str] = None,
        default_applicable=None,
        default_excluded=None,
    ) -> Tuple[List[str], List[str]]:
        cfg = self._metric_cfg(metric_key, airframe=airframe)
        if default_applicable is None:
            default_applicable = self.GENERIC_SETPOINT_APPLICABLE_MODE_LABELS
        if default_excluded is None:
            default_excluded = self.SETPOINT_EXCLUDED_MODE_LABELS
        applicable = cfg.get("applicable_modes", default_applicable)
        excluded = cfg.get("excluded_modes", default_excluded)
        return self._normalize_mode_labels(applicable), self._normalize_mode_labels(excluded)

    @staticmethod
    def _time_weights_from_timestamps(timestamps) -> np.ndarray:
        ts = np.asarray(timestamps, dtype=np.float64)
        if ts.size <= 0:
            return np.asarray([], dtype=np.float64)
        if ts.size == 1:
            return np.asarray([0.0], dtype=np.float64)
        diffs = np.diff(ts)
        good = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        default_dt = float(np.median(good)) if good.size else 0.0
        weights = np.zeros(ts.shape, dtype=np.float64)
        fill = diffs.astype(np.float64, copy=True)
        fill[~np.isfinite(fill) | (fill <= 0.0)] = default_dt
        weights[:-1] = fill
        return weights

    @staticmethod
    def _weighted_mean(values, weights) -> Optional[float]:
        v = np.asarray(values, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        mask = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
        if int(np.sum(mask)) <= 0:
            mask = np.isfinite(v)
            if int(np.sum(mask)) <= 0:
                return None
            return float(np.mean(v[mask]))
        total_w = float(np.sum(w[mask]))
        if total_w <= 1e-9:
            return None
        return float(np.sum(v[mask] * w[mask]) / total_w)

    @staticmethod
    def _intervals_from_mask(timestamps, mask):
        """Return list of (start_sec, end_sec) tuples for each contiguous True run in mask."""
        if timestamps is None or mask is None:
            return []
        ts = np.asarray(timestamps, dtype=np.float64)
        m = np.asarray(mask, dtype=bool)
        n = int(min(ts.size, m.size))
        if n == 0:
            return []
        ts = ts[:n]
        m = m[:n]
        if not bool(m.any()):
            return []
        diffs = np.diff(np.r_[False, m, False].astype(np.int8))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0] - 1
        return [(float(ts[s]), float(ts[e])) for s, e in zip(starts, ends)]

    def _inject_mode_time_intervals(self, mode_results, timestamps, mode_labels, evaluated_mask):
        """In-place: enrich each mode_results entry with a `time_intervals` list of (start,end) tuples."""
        if not isinstance(mode_results, list) or not mode_results:
            return
        try:
            labels_arr = np.asarray(mode_labels, dtype=object)
            mask_arr = np.asarray(evaluated_mask, dtype=bool)
        except Exception:
            return
        for entry in mode_results:
            if not isinstance(entry, dict):
                continue
            mode_name = str(entry.get("mode", "") or "").strip()
            if not mode_name:
                entry.setdefault("time_intervals", [])
                continue
            try:
                sub_mask = mask_arr & (labels_arr == mode_name)
                entry["time_intervals"] = self._intervals_from_mask(timestamps, sub_mask)
            except Exception:
                entry.setdefault("time_intervals", [])

    def _mode_results_for_metric(
        self,
        mode_labels,
        evaluated_mask,
        metric_values,
        weights,
        *,
        warn: float,
        problem: float,
        unit: str,
        score_direction: str = "low",
    ) -> List[dict]:
        labels = np.asarray(mode_labels, dtype=object)
        mask = np.asarray(evaluated_mask, dtype=bool)
        values = np.asarray(metric_values, dtype=np.float64)
        row_weights = np.asarray(weights, dtype=np.float64)
        total_eval_time = float(np.sum(row_weights[mask])) if row_weights.size else 0.0
        results = []
        unique_modes = []
        for label in labels[mask]:
            text = str(label or "").strip()
            if text and text not in unique_modes:
                unique_modes.append(text)
        for mode in unique_modes:
            mode_mask = mask & (labels == mode)
            mode_value = self._weighted_mean(values[mode_mask], row_weights[mode_mask])
            if mode_value is None:
                continue
            if score_direction == "high":
                score, status = self._score_high(mode_value, warn, problem)
            else:
                score, status = self._score_low(mode_value, warn, problem)
            results.append({
                "mode": mode,
                "status": status,
                "status_label": self.STATUS_LABEL.get(status, status),
                "score": round(float(score), 1),
                "value": float(mode_value),
                "unit": unit,
                "time_sec": round(float(np.sum(row_weights[mode_mask])), 3),
                "evaluated_ratio_pct": round((100.0 * float(np.sum(row_weights[mode_mask])) / total_eval_time), 3) if total_eval_time > 1e-9 else 0.0,
                "sample_count": int(np.sum(mode_mask)),
            })
        results.sort(key=lambda item: item.get("time_sec", 0.0), reverse=True)
        return results

    def _mode_results_for_rms_metric(
        self,
        mode_labels,
        evaluated_mask,
        metric_values,
        weights,
        *,
        warn: float,
        problem: float,
        unit: str,
    ) -> List[dict]:
        labels = np.asarray(mode_labels, dtype=object)
        mask = np.asarray(evaluated_mask, dtype=bool)
        values = np.asarray(metric_values, dtype=np.float64)
        row_weights = np.asarray(weights, dtype=np.float64)
        total_eval_time = float(np.sum(row_weights[mask])) if row_weights.size else 0.0
        results = []
        unique_modes = []
        for label in labels[mask]:
            text = str(label or "").strip()
            if text and text not in unique_modes:
                unique_modes.append(text)
        for mode in unique_modes:
            mode_mask = mask & (labels == mode)
            v = values[mode_mask]
            w = row_weights[mode_mask]
            good = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
            if int(np.sum(good)) > 0:
                mode_value = float(np.sqrt(np.sum((v[good] ** 2) * w[good]) / max(np.sum(w[good]), 1e-9)))
            else:
                v = v[np.isfinite(v)]
                if v.size <= 0:
                    continue
                mode_value = float(np.sqrt(np.mean(v * v)))
            score, status = self._score_low(mode_value, warn, problem)
            mode_time = float(np.sum(row_weights[mode_mask])) if row_weights.size else 0.0
            results.append({
                "mode": mode,
                "status": status,
                "status_label": self.STATUS_LABEL.get(status, status),
                "score": round(float(score), 1),
                "value": float(mode_value),
                "unit": unit,
                "time_sec": round(mode_time, 3),
                "evaluated_ratio_pct": round((100.0 * mode_time / total_eval_time), 3) if total_eval_time > 1e-9 else 0.0,
                "sample_count": int(np.sum(mode_mask)),
            })
        results.sort(key=lambda item: item.get("time_sec", 0.0), reverse=True)
        return results

    def _mode_results_for_gap_metric(
        self,
        mode_labels,
        evaluated_mask,
        lhs_values,
        rhs_values,
        weights,
        *,
        warn: float,
        problem: float,
        unit: str,
    ) -> List[dict]:
        labels = np.asarray(mode_labels, dtype=object)
        mask = np.asarray(evaluated_mask, dtype=bool)
        lhs = np.asarray(lhs_values, dtype=np.float64)
        rhs = np.asarray(rhs_values, dtype=np.float64)
        row_weights = np.asarray(weights, dtype=np.float64)
        total_eval_time = float(np.sum(row_weights[mask])) if row_weights.size else 0.0
        results = []
        unique_modes = []
        for label in labels[mask]:
            text = str(label or "").strip()
            if text and text not in unique_modes:
                unique_modes.append(text)
        for mode in unique_modes:
            mode_mask = mask & (labels == mode)
            l = lhs[mode_mask]
            r = rhs[mode_mask]
            w = row_weights[mode_mask]
            good = np.isfinite(l) & np.isfinite(r) & np.isfinite(w) & (w > 0.0)
            if int(np.sum(good)) > 0:
                lhs_mean = float(np.sum(l[good] * w[good]) / max(np.sum(w[good]), 1e-9))
                rhs_mean = float(np.sum(r[good] * w[good]) / max(np.sum(w[good]), 1e-9))
            else:
                l = l[np.isfinite(l)]
                r = r[np.isfinite(r)]
                n = min(l.size, r.size)
                if n <= 0:
                    continue
                lhs_mean = float(np.mean(l[:n]))
                rhs_mean = float(np.mean(r[:n]))
            mode_value = abs(lhs_mean - rhs_mean)
            score, status = self._score_low(mode_value, warn, problem)
            mode_time = float(np.sum(row_weights[mode_mask])) if row_weights.size else 0.0
            results.append({
                "mode": mode,
                "status": status,
                "status_label": self.STATUS_LABEL.get(status, status),
                "score": round(float(score), 1),
                "value": float(mode_value),
                "unit": unit,
                "time_sec": round(mode_time, 3),
                "evaluated_ratio_pct": round((100.0 * mode_time / total_eval_time), 3) if total_eval_time > 1e-9 else 0.0,
                "sample_count": int(np.sum(mode_mask)),
            })
        results.sort(key=lambda item: item.get("time_sec", 0.0), reverse=True)
        return results

    @staticmethod
    def _status_rank_for_summary(status: Optional[str]) -> int:
        key = str(status or "").strip().lower()
        if key in ("problem", "critical", "error"):
            return 3
        if key in ("warning", "warn"):
            return 2
        if key in ("good", "normal", "ok"):
            return 1
        return 0

    @classmethod
    def _is_safety_critical_mode(cls, mode_name: str) -> bool:
        text = str(mode_name or "").strip()
        if not text:
            return False
        if text in cls.SAFETY_CRITICAL_MODE_LABELS:
            return True
        return "transition" in text.lower()

    def _summarize_mode_verdict(self, overall_status: str, mode_results) -> dict:
        overall_status = str(overall_status or "").strip().lower() or "unavailable"
        filtered = [
            result for result in list(mode_results or [])
            if isinstance(result, dict) and self._status_rank_for_summary(result.get("status")) > 0
        ]
        summary = {
            "overall_average_status": overall_status,
            "overall_average_status_label": self.STATUS_LABEL.get(overall_status, overall_status),
            "worst_mode_status": "",
            "worst_mode_status_label": "",
            "problem_modes": [],
            "warning_modes": [],
            "safety_critical_problem_modes": [],
            "safety_critical_issue_modes": [],
            "mode_attention_flag": False,
            "mode_attention_reason": "",
            "interpretation_lines": [],
            "final_status": overall_status,
            "final_status_label": self.STATUS_LABEL.get(overall_status, overall_status),
        }
        if not filtered:
            return summary

        worst = max(filtered, key=lambda item: self._status_rank_for_summary(item.get("status")))
        worst_status = str(worst.get("status", "") or "").strip().lower()
        summary["worst_mode_status"] = worst_status
        summary["worst_mode_status_label"] = self.STATUS_LABEL.get(worst_status, worst_status)

        for result in filtered:
            entry = {
                "mode": str(result.get("mode", "") or "").strip(),
                "status": str(result.get("status", "") or "").strip().lower(),
                "status_label": self.STATUS_LABEL.get(result.get("status"), result.get("status")),
                "time_sec": float(result.get("time_sec", 0.0) or 0.0),
                "value": None if result.get("value") is None else float(result.get("value")),
                "unit": str(result.get("unit", "") or "").strip(),
                "evaluated_ratio_pct": float(result.get("evaluated_ratio_pct", 0.0) or 0.0),
                "sample_count": int(result.get("sample_count", 0) or 0),
            }
            if entry["status"] == "problem":
                summary["problem_modes"].append(entry)
                if self._is_safety_critical_mode(entry["mode"]):
                    summary["safety_critical_problem_modes"].append(entry)
            elif entry["status"] == "warning":
                summary["warning_modes"].append(entry)
            if entry["status"] in ("warning", "problem") and self._is_safety_critical_mode(entry["mode"]):
                summary["safety_critical_issue_modes"].append(entry)

        final_status = overall_status
        final_status_label = self.STATUS_LABEL.get(overall_status, overall_status)
        mode_attention_flag = False
        mode_attention_reason = ""
        interpretation_lines = []

        if overall_status == "good" and worst_status == "problem":
            final_status = "warning"
            final_status_label = "정상 / 일부 모드 문제"
            mode_attention_flag = True
            mode_attention_reason = "overall_good_but_problem_modes"
        elif overall_status == "good" and worst_status == "warning":
            final_status = "warning"
            final_status_label = "정상 / 일부 모드 경고"
            mode_attention_flag = True
            mode_attention_reason = "overall_good_but_warning_modes"
        elif overall_status == "warning" and worst_status == "problem":
            final_status = "problem"
            final_status_label = "경고 평균 / 일부 모드 문제"
            mode_attention_flag = True
            mode_attention_reason = "overall_warning_with_problem_modes"
        elif overall_status in ("problem",):
            if worst_status in ("problem", "warning"):
                mode_attention_flag = True
                mode_attention_reason = "overall_problem_with_mode_issues"

        problem_names = [item["mode"] for item in summary["problem_modes"] if item.get("mode")]
        warning_names = [item["mode"] for item in summary["warning_modes"] if item.get("mode")]
        safety_names = [item["mode"] for item in summary["safety_critical_problem_modes"] if item.get("mode")]

        if mode_attention_reason == "overall_good_but_problem_modes":
            interpretation_lines.extend([
                "전체 시간 가중 평균 기준으로는 정상입니다.",
                f"다만 {', '.join(problem_names[:5])} 구간에서 문제 수준이 확인되었습니다.",
                "적용 시간이 긴 다른 비행모드가 평균을 끌어올렸을 수 있으므로, 문제 모드를 별도로 확대 확인해야 합니다.",
            ])
        elif mode_attention_reason == "overall_good_but_warning_modes":
            interpretation_lines.extend([
                "전체 시간 가중 평균 기준으로는 정상입니다.",
                f"다만 {', '.join(warning_names[:5])} 구간에서 경고 수준이 확인되었습니다.",
                "평균값만 보면 묻힐 수 있으므로 모드별 그래프를 함께 확인하는 것이 좋습니다.",
            ])
        elif mode_attention_reason == "overall_warning_with_problem_modes":
            interpretation_lines.extend([
                "전체 평균은 경고 수준입니다.",
                f"하지만 {', '.join(problem_names[:5])} 구간은 문제 수준으로 더 나쁩니다.",
                "짧은 문제 구간이라도 별도로 확대 확인해야 합니다.",
            ])
        elif overall_status == "problem" and problem_names:
            interpretation_lines.extend([
                "전체 평균 기준으로도 문제 수준입니다.",
                f"모드별로도 {', '.join(problem_names[:5])} 구간에서 문제 수준이 확인됩니다.",
            ])

        if safety_names:
            interpretation_lines.append(f"{', '.join(safety_names[:5])} 구간은 안전상 중요한 비행모드이므로 지속시간이 짧아도 별도 확인이 필요합니다.")

        summary.update({
            "final_status": final_status,
            "final_status_label": final_status_label,
            "mode_attention_flag": bool(mode_attention_flag),
            "mode_attention_reason": mode_attention_reason,
            "interpretation_lines": interpretation_lines,
        })
        return summary

    def _build_mode_filtered_context(
        self,
        dataset,
        frame: Optional[pl.DataFrame],
        metric_key: str,
        *,
        valid_mask=None,
        metric_values=None,
        airframe: Optional[str] = None,
        applicable_modes=None,
        excluded_modes=None,
        warn: Optional[float] = None,
        problem: Optional[float] = None,
        unit: str = "",
        score_direction: str = "low",
    ) -> dict:
        total_log_time_sec = float(self._flight_time_sec(dataset))
        base = {
            "evaluation_strategy": "mode_filtered_nav_state_segments",
            "total_log_time_sec": round(total_log_time_sec, 3),
            "evaluated_time_sec": 0.0,
            "excluded_time_sec": round(total_log_time_sec, 3),
            "applied_ratio_pct": 0.0,
            "applicable_modes": self._normalize_mode_labels(applicable_modes),
            "excluded_modes": [],
            "excluded_reasons": [],
            "mode_results": [],
            "evaluated_modes": [],
            "not_evaluated_reason": "",
            "fallback_used": False,
            "fallback_source": "",
            "fallback_reason": "",
            "nav_state_topic": None,
            "nav_state_signal": None,
            "status_hint": None,
            "mode_factor": 0.0,
            "sample_count": 0,
            "weights": np.asarray([], dtype=np.float64),
            "evaluated_mask": np.asarray([], dtype=bool),
            "applicable_mask": np.asarray([], dtype=bool),
            "joined_frame": frame,
        }
        if frame is None or frame.height <= 0:
            base["status_hint"] = "unavailable"
            base["not_evaluated_reason"] = "평가에 사용할 기본 시계열이 없어 비행모드별 분리 평가를 할 수 없습니다."
            return base

        nav_state_series, nav_state_topic, nav_state_sig = self._extract_nav_state_series(dataset)
        base["nav_state_topic"] = nav_state_topic
        base["nav_state_signal"] = nav_state_sig
        if nav_state_series is None or nav_state_topic is None or nav_state_sig is None:
            base["status_hint"] = "unavailable"
            base["not_evaluated_reason"] = "vehicle_status.nav_state가 없어 비행모드별 구간 분리 평가를 할 수 없습니다."
            return base

        try:
            joined = self._with_timestamp_sec(frame).sort("timestamp_sec").join_asof(nav_state_series, on="timestamp_sec", strategy="nearest")
        except Exception:
            base["status_hint"] = "unavailable"
            base["not_evaluated_reason"] = "vehicle_status.nav_state를 평가 시계열과 결합하지 못했습니다."
            return base
        if joined.height <= 0 or "nav_state" not in joined.columns:
            base["status_hint"] = "unavailable"
            base["not_evaluated_reason"] = "비행모드 정보가 평가 대상 시계열에 매칭되지 않았습니다."
            return base

        timestamps = np.asarray(joined["timestamp_sec"].to_numpy(), dtype=np.float64)
        weights = self._time_weights_from_timestamps(timestamps)
        mode_labels = np.asarray([self._nav_state_label(x) for x in joined["nav_state"].to_numpy()], dtype=object)
        applicable = self._normalize_mode_labels(applicable_modes)
        excluded = self._normalize_mode_labels(excluded_modes)
        if not applicable:
            applicable, excluded = self._metric_mode_policy(
                metric_key,
                airframe=airframe,
                default_applicable=applicable_modes,
                default_excluded=excluded_modes,
            )
        valid = np.asarray(valid_mask if valid_mask is not None else np.ones(joined.height, dtype=bool), dtype=bool)
        values = np.asarray(metric_values if metric_values is not None else np.full(joined.height, np.nan), dtype=np.float64)
        valid &= np.isfinite(weights)
        if metric_values is not None:
            valid &= np.isfinite(values)
        has_mode = np.asarray([bool(str(x or "").strip()) for x in mode_labels], dtype=bool)
        applicable_set = set(applicable)
        excluded_set = set(excluded)
        applicable_mode_mask = np.asarray([label in applicable_set for label in mode_labels], dtype=bool)
        excluded_mode_mask = np.asarray([label in excluded_set for label in mode_labels], dtype=bool)
        other_mode_mask = has_mode & ~applicable_mode_mask & ~excluded_mode_mask
        missing_mode_mask = ~has_mode
        invalid_in_applicable_mask = applicable_mode_mask & ~valid
        evaluated_mask = applicable_mode_mask & valid
        evaluated_time_sec = float(np.sum(weights[evaluated_mask])) if weights.size else 0.0
        excluded_time_sec = max(0.0, total_log_time_sec - evaluated_time_sec) if total_log_time_sec > 0.0 else float(np.sum(weights[~evaluated_mask])) if weights.size else 0.0
        applied_ratio_pct = (100.0 * evaluated_time_sec / total_log_time_sec) if total_log_time_sec > 1e-9 else 0.0

        observed_excluded_modes = []
        for mask in (excluded_mode_mask, other_mode_mask):
            for label in mode_labels[mask]:
                text = str(label or "").strip()
                if text and text not in observed_excluded_modes:
                    observed_excluded_modes.append(text)

        excluded_reasons = []
        if int(np.sum(excluded_mode_mask)) > 0:
            excluded_reasons.append(f"{', '.join(excluded)} 구간은 자동 추종 평가 대상이 아니므로 제외했습니다.")
        if int(np.sum(other_mode_mask)) > 0:
            excluded_reasons.append("적용 대상이 아닌 비행모드는 점수 산정에서 제외했습니다.")
        if int(np.sum(invalid_in_applicable_mask)) > 0:
            excluded_reasons.append("적용 가능한 비행모드 안에서도 setpoint 또는 actual 값이 비어 있거나 NaN인 구간은 제외했습니다.")
        if int(np.sum(missing_mode_mask)) > 0:
            excluded_reasons.append("비행모드가 기록되지 않은 구간은 점수 산정에서 제외했습니다.")

        status_hint = None
        not_evaluated_reason = ""
        if int(np.sum(applicable_mode_mask)) <= 0:
            status_hint = "excluded"
            not_evaluated_reason = "이 로그에는 이 항목을 평가할 자동 비행모드 구간이 없습니다."
        elif int(np.sum(evaluated_mask)) <= 0:
            status_hint = "unavailable"
            not_evaluated_reason = "적용 가능한 비행모드는 있었지만 유효한 setpoint/actual 샘플이 없어 평가할 수 없습니다."

        mode_results = []
        if metric_values is not None and warn is not None and problem is not None and int(np.sum(evaluated_mask)) > 0:
            mode_results = self._mode_results_for_metric(
                mode_labels,
                evaluated_mask,
                values,
                weights,
                warn=float(warn),
                problem=float(problem),
                unit=unit,
                score_direction=score_direction,
            )
            self._inject_mode_time_intervals(mode_results, timestamps, mode_labels, evaluated_mask)

        evaluated_modes = []
        for label in mode_labels[evaluated_mask]:
            text = str(label or "").strip()
            if text and text not in evaluated_modes:
                evaluated_modes.append(text)

        base.update({
            "evaluation_strategy": "mode_filtered_nav_state_segments",
            "total_log_time_sec": round(total_log_time_sec, 3),
            "evaluated_time_sec": round(evaluated_time_sec, 3),
            "excluded_time_sec": round(excluded_time_sec, 3),
            "applied_ratio_pct": round(applied_ratio_pct, 3),
            "applicable_modes": applicable,
            "excluded_modes": observed_excluded_modes,
            "excluded_reasons": excluded_reasons,
            "mode_results": mode_results,
            "evaluated_modes": evaluated_modes,
            "not_evaluated_reason": not_evaluated_reason,
            "status_hint": status_hint,
            "mode_factor": self._clamp(applied_ratio_pct / 100.0 if applied_ratio_pct > 0.0 else 0.0),
            "sample_count": int(np.sum(evaluated_mask)),
            "weights": weights,
            "evaluated_mask": evaluated_mask,
            "applicable_mask": applicable_mode_mask,
            "joined_frame": joined,
        })
        return base

    def _classify_logged_messages(self, messages) -> dict:
        counts = {k: 0 for k in self.FAILSAFE_MESSAGE_PATTERNS.keys()}
        matched_examples = {k: [] for k in self.FAILSAFE_MESSAGE_PATTERNS.keys()}
        for item in list(messages or []):
            if isinstance(item, dict):
                text = str(item.get("text", "") or "").strip()
            else:
                text = str(item or "").strip()
            if not text:
                continue
            low = text.lower()
            for label, patterns in self.FAILSAFE_MESSAGE_PATTERNS.items():
                if any(re.search(pattern, low) for pattern in patterns):
                    counts[label] += 1
                    if len(matched_examples[label]) < 3 and text not in matched_examples[label]:
                        matched_examples[label].append(text)
        return {"counts": counts, "examples": matched_examples}

    def _join_ts(self, df_a: pl.DataFrame, sig_a: str, df_b: pl.DataFrame, sig_b: str) -> Optional[pl.DataFrame]:
        df_a = self._with_timestamp_sec(df_a)
        df_b = self._with_timestamp_sec(df_b)
        if "timestamp_sec" not in df_a.columns or "timestamp_sec" not in df_b.columns:
            return None
        try:
            a = df_a.select([pl.col("timestamp_sec").cast(pl.Float64), pl.col(sig_a).cast(pl.Float64).alias("actual")]).drop_nulls().sort("timestamp_sec")
            b = df_b.select([pl.col("timestamp_sec").cast(pl.Float64), pl.col(sig_b).cast(pl.Float64).alias("setpoint")]).drop_nulls().sort("timestamp_sec")
            if a.height == 0 or b.height == 0:
                return None
            m = a.join_asof(b, on="timestamp_sec", strategy="nearest").drop_nulls(subset=["actual", "setpoint"])
            return m if m.height > 0 else None
        except Exception:
            return None

    def _extract_series(self, dataset, topic_candidates, signal_candidates, alias: str) -> Tuple[Optional[pl.DataFrame], Optional[str], Optional[str]]:
        topic = self._find_topic(dataset, topic_candidates)
        if not topic:
            return None, None, None
        try:
            df = self._with_timestamp_sec(dataset.topics[topic].dataframe)
            if "timestamp_sec" not in df.columns:
                return None, topic, None
            sig = self._find_signal(df, signal_candidates)
            if not sig:
                return None, topic, None
            out = df.select([
                pl.col("timestamp_sec").cast(pl.Float64),
                pl.col(sig).cast(pl.Float64).alias(alias),
            ]).drop_nulls().sort("timestamp_sec")
            if out.height <= 0:
                return None, topic, sig
            return out, topic, sig
        except Exception:
            return None, topic, None

    def _extract_ground_speed_series(self, dataset, alias: str = "ground_speed") -> Tuple[Optional[pl.DataFrame], Optional[str], List[str]]:
        search_specs = [
            {
                "topics": ["vehicle_gps_position", "sensor_gps"],
                "speed_signals": ["vel_m_s", "ground_speed_mag", "speed_m_s", "speed", "ground_speed"],
                "vector_pairs": [("vel_n_m_s", "vel_e_m_s"), ("vel_n", "vel_e")],
            },
            {
                "topics": ["vehicle_local_position"],
                "speed_signals": ["ground_speed", "ground_speed_xy", "ground_speed_mag", "speed"],
                "vector_pairs": [("vx", "vy"), ("vel_x", "vel_y")],
            },
        ]
        for spec in search_specs:
            topic = self._find_topic(dataset, spec["topics"])
            if not topic:
                continue
            try:
                df = self._with_timestamp_sec(dataset.topics[topic].dataframe)
                if "timestamp_sec" not in df.columns:
                    continue
                sig = self._find_signal(df, spec["speed_signals"])
                if sig:
                    out = df.select([
                        pl.col("timestamp_sec").cast(pl.Float64),
                        pl.col(sig).cast(pl.Float64).alias(alias),
                    ]).drop_nulls().sort("timestamp_sec")
                    if out.height > 0:
                        return out, topic, [f"{topic}.{sig}"]
                for sig_x, sig_y in spec["vector_pairs"]:
                    if sig_x in df.columns and sig_y in df.columns:
                        out = df.select([
                            pl.col("timestamp_sec").cast(pl.Float64),
                            ((pl.col(sig_x).cast(pl.Float64) ** 2 + pl.col(sig_y).cast(pl.Float64) ** 2).sqrt()).alias(alias),
                        ]).drop_nulls().sort("timestamp_sec")
                        if out.height > 0:
                            return out, topic, [f"{topic}.{sig_x}", f"{topic}.{sig_y}"]
            except Exception:
                continue
        return None, None, []

    def _extract_wind_corrected_speed_series(
        self,
        dataset,
        alias: str = "air_relative_speed",
    ) -> Tuple[Optional[pl.DataFrame], Optional[str], Optional[str], List[str], List[str]]:
        """Build a TAS-comparable velocity magnitude from GPS velocity minus wind.

        GPS ground speed alone is not an airspeed reference.  This routine requires
        both horizontal GPS velocity components and a PX4 wind estimate, then uses
        ``|v_ground - v_wind|`` (including GPS vertical velocity when available).
        For VTOL logs, only settled fixed-wing samples are retained.
        """
        gps_topic = self._find_topic(dataset, ["vehicle_gps_position", "sensor_gps"])
        wind_topic = self._find_topic(dataset, ["wind", "estimator_wind", "airspeed_wind"])
        if not gps_topic or not wind_topic:
            return None, gps_topic, wind_topic, [], [x for x in [gps_topic, wind_topic] if x]

        try:
            gps_df = self._with_timestamp_sec(dataset.topics[gps_topic].dataframe)
            wind_df = self._with_timestamp_sec(dataset.topics[wind_topic].dataframe)
            gps_n = self._find_signal(gps_df, ["vel_n_m_s", "vel_n", "velocity_north"])
            gps_e = self._find_signal(gps_df, ["vel_e_m_s", "vel_e", "velocity_east"])
            gps_d = self._find_signal(gps_df, ["vel_d_m_s", "vel_d", "velocity_down"])
            wind_n = self._find_signal(wind_df, ["windspeed_north", "wind_north", "wind_n"])
            wind_e = self._find_signal(wind_df, ["windspeed_east", "wind_east", "wind_e"])
            if not all([gps_n, gps_e, wind_n, wind_e]):
                return None, gps_topic, wind_topic, [], [gps_topic, wind_topic]

            gps_cols = [
                pl.col("timestamp_sec").cast(pl.Float64),
                pl.col(gps_n).cast(pl.Float64).alias("gps_vn"),
                pl.col(gps_e).cast(pl.Float64).alias("gps_ve"),
            ]
            if gps_d:
                gps_cols.append(pl.col(gps_d).cast(pl.Float64).alias("gps_vd"))
            gps = gps_df.select(gps_cols).drop_nulls(subset=["gps_vn", "gps_ve"]).sort("timestamp_sec")
            wind = wind_df.select([
                pl.col("timestamp_sec").cast(pl.Float64),
                pl.col("timestamp_sec").cast(pl.Float64).alias("wind_timestamp_sec"),
                pl.col(wind_n).cast(pl.Float64).alias("wind_vn"),
                pl.col(wind_e).cast(pl.Float64).alias("wind_ve"),
            ]).drop_nulls(subset=["wind_vn", "wind_ve"]).sort("timestamp_sec")
            if gps.height <= 0 or wind.height <= 0:
                return None, gps_topic, wind_topic, [], [gps_topic, wind_topic]

            joined = gps.join_asof(wind, on="timestamp_sec", strategy="nearest")
            # Do not silently extrapolate a stale wind estimate over a long gap.
            joined = joined.filter(
                (pl.col("timestamp_sec") - pl.col("wind_timestamp_sec")).abs() <= pl.lit(5.0)
            )
            vertical_sq = pl.col("gps_vd") ** 2 if "gps_vd" in joined.columns else pl.lit(0.0)
            joined = joined.with_columns(
                (
                    (pl.col("gps_vn") - pl.col("wind_vn")) ** 2
                    + (pl.col("gps_ve") - pl.col("wind_ve")) ** 2
                    + vertical_sq
                ).sqrt().alias(alias)
            )

            used_signals = [
                f"{gps_topic}.{gps_n}",
                f"{gps_topic}.{gps_e}",
                f"{wind_topic}.{wind_n}",
                f"{wind_topic}.{wind_e}",
            ]
            if gps_d:
                used_signals.append(f"{gps_topic}.{gps_d}")
            found_topics = [gps_topic, wind_topic]

            if str(getattr(self, "_eval_airframe_ctx", "")).upper() == "VTOL":
                status_topic = self._find_topic(dataset, "vehicle_status")
                if not status_topic:
                    return None, gps_topic, wind_topic, used_signals, found_topics
                status_df = self._with_timestamp_sec(dataset.topics[status_topic].dataframe)
                vehicle_type_sig = self._find_signal(status_df, ["vehicle_type"])
                transition_sig = self._find_signal(status_df, ["in_transition_mode"])
                if not vehicle_type_sig or not transition_sig:
                    return None, gps_topic, wind_topic, used_signals, found_topics + [status_topic]
                phase = status_df.select([
                    pl.col("timestamp_sec").cast(pl.Float64),
                    pl.col(vehicle_type_sig).cast(pl.Float64).alias("vehicle_type_phase"),
                    pl.col(transition_sig).cast(pl.Float64).alias("in_transition_phase"),
                ]).drop_nulls().sort("timestamp_sec")
                joined = joined.join_asof(phase, on="timestamp_sec", strategy="nearest").filter(
                    (pl.col("vehicle_type_phase") == pl.lit(2.0))
                    & (pl.col("in_transition_phase") <= pl.lit(0.5))
                )
                used_signals.extend([
                    f"{status_topic}.{vehicle_type_sig}",
                    f"{status_topic}.{transition_sig}",
                ])
                found_topics.append(status_topic)

            out = joined.select([
                pl.col("timestamp_sec").cast(pl.Float64),
                pl.col("timestamp_sec").cast(pl.Float64).alias("air_reference_timestamp_sec"),
                pl.col(alias).cast(pl.Float64),
            ]).drop_nulls().sort("timestamp_sec")
            if out.height <= 0:
                return None, gps_topic, wind_topic, used_signals, found_topics
            return out, gps_topic, wind_topic, used_signals, found_topics
        except Exception:
            return None, gps_topic, wind_topic, [], [x for x in [gps_topic, wind_topic] if x]

    def _eval_navigation_path_tracking(self, dataset, airframe: str, parameters=None) -> List[dict]:
        """Evaluate true PX4 cross-track error without a waypoint-distance proxy."""
        cfg = self._metric_cfg("navigation_path_tracking_error", airframe=airframe)
        reliability_cfg = self._metric_cfg("navigation_path_tracking_reliability", airframe=airframe)
        required_path = [
            "npfg_status.signed_track_error|position_controller_status.xtrack_error",
            "vehicle_status.nav_state",
        ]

        path_topic = self._find_topic(dataset, "npfg_status")
        path_sig = None
        source_priority = ""
        if path_topic:
            path_df_raw = self._with_timestamp_sec(dataset.topics[path_topic].dataframe)
            path_sig = self._find_signal(path_df_raw, ["signed_track_error"])
            if path_sig:
                source_priority = "primary_npfg"
        if not path_sig:
            candidate = self._find_topic(dataset, "position_controller_status")
            if candidate:
                candidate_df = self._with_timestamp_sec(dataset.topics[candidate].dataframe)
                candidate_sig = self._find_signal(candidate_df, ["xtrack_error", "cross_track_error"])
                if candidate_sig:
                    path_topic = candidate
                    path_sig = candidate_sig
                    path_df_raw = candidate_df
                    source_priority = "secondary_controller_status"

        if not path_topic or not path_sig:
            reason = (
                "Missing an explicit PX4 cross-track signal (npfg_status.signed_track_error or "
                "position_controller_status.xtrack_error). Position-to-setpoint distance is not "
                "used because it can be distance to a future waypoint rather than cross-track error."
            )
            return [
                self._item(
                    "navigation_path_tracking_error", "Navigation Path Tracking Error", None, "unavailable", None, cfg["unit"],
                    reason, cfg["suggestion"], scope="airframe", segment="path",
                    required_topics=required_path, found_topics=[], sample_count=0,
                    reference_class="px4_explicit", signal_semantics="px4_explicit",
                    threshold_source=str(cfg.get("threshold_source", "PX4 explicit cross-track signal + product threshold")),
                    threshold_rationale="Only a field whose PX4 semantics explicitly represent cross-track error is scored.",
                    mode_gate="nav_state_applicable_modes",
                    normalization_basis="time-weighted mean absolute signed cross-track error",
                    not_evaluated_reason=reason,
                ),
                self._item(
                    "navigation_path_tracking_reliability", "Navigation Path Tracking Reliability", None, "unavailable", None,
                    str(reliability_cfg.get("unit", "score")), reason,
                    str(reliability_cfg.get("suggestion", "Review path-tail error and local-position validity.")),
                    scope="airframe", segment="path", required_topics=required_path,
                    found_topics=[], sample_count=0, reference_class="mixed_reference",
                    signal_semantics="px4_explicit", not_evaluated_reason=reason,
                ),
            ]

        try:
            path_frame = path_df_raw.select([
                pl.col("timestamp_sec").cast(pl.Float64),
                pl.col(path_sig).cast(pl.Float64).abs().alias("path_error_abs"),
            ]).drop_nulls(subset=["path_error_abs"]).sort("timestamp_sec")
        except Exception:
            path_frame = None

        local_topic = self._find_topic(dataset, "vehicle_local_position")
        xy_valid_sig = None
        vxy_valid_sig = None
        if path_frame is not None and local_topic:
            try:
                local_df = self._with_timestamp_sec(dataset.topics[local_topic].dataframe)
                xy_valid_sig = self._find_signal(local_df, ["xy_valid"])
                vxy_valid_sig = self._find_signal(local_df, ["v_xy_valid"])
                validity_cols = [pl.col("timestamp_sec").cast(pl.Float64)]
                if xy_valid_sig:
                    validity_cols.append(pl.col(xy_valid_sig).cast(pl.Float64).alias("xy_valid"))
                if vxy_valid_sig:
                    validity_cols.append(pl.col(vxy_valid_sig).cast(pl.Float64).alias("v_xy_valid"))
                if len(validity_cols) > 1:
                    validity = local_df.select(validity_cols).sort("timestamp_sec")
                    path_frame = path_frame.join_asof(validity, on="timestamp_sec", strategy="nearest")
            except Exception:
                xy_valid_sig = None
                vxy_valid_sig = None

        if path_frame is None or path_frame.height <= 0:
            reason = f"No finite samples in {path_topic}.{path_sig}."
            return [
                self._item(
                    "navigation_path_tracking_error", "Navigation Path Tracking Error", None, "unavailable", None, cfg["unit"],
                    reason, cfg["suggestion"], scope="airframe", segment="path",
                    required_topics=required_path, found_topics=[path_topic], sample_count=0,
                    used_signals=[f"{path_topic}.{path_sig}"], reference_class="px4_explicit",
                    signal_semantics="px4_explicit", not_evaluated_reason=reason,
                ),
                self._item(
                    "navigation_path_tracking_reliability", "Navigation Path Tracking Reliability", None, "unavailable", None,
                    str(reliability_cfg.get("unit", "score")), reason,
                    str(reliability_cfg.get("suggestion", "Review path-tail error and local-position validity.")),
                    scope="airframe", segment="path", required_topics=required_path,
                    found_topics=[path_topic], sample_count=0, used_signals=[f"{path_topic}.{path_sig}"],
                    reference_class="mixed_reference", signal_semantics="px4_explicit",
                    not_evaluated_reason=reason,
                ),
            ]

        err = np.asarray(path_frame["path_error_abs"].to_numpy(), dtype=np.float64)
        applicable_modes, excluded_modes = self._metric_mode_policy(
            "navigation_path_tracking_error", airframe=airframe,
            default_applicable=self.PATH_SETPOINT_APPLICABLE_MODE_LABELS,
            default_excluded=self.SETPOINT_EXCLUDED_MODE_LABELS,
        )
        mode_ctx = self._build_mode_filtered_context(
            dataset, path_frame, "navigation_path_tracking_error",
            valid_mask=np.isfinite(err), metric_values=err, airframe=airframe,
            applicable_modes=applicable_modes, excluded_modes=excluded_modes,
            warn=cfg["warn"], problem=cfg["problem"], unit=cfg["unit"],
        )
        status_topic = mode_ctx.get("nav_state_topic")
        found_topics = [x for x in [path_topic, status_topic, local_topic if (xy_valid_sig or vxy_valid_sig) else None] if x]
        used_signals = [f"{path_topic}.{path_sig}"]
        if status_topic and mode_ctx.get("nav_state_signal"):
            used_signals.append(f"{status_topic}.{mode_ctx['nav_state_signal']}")
        if local_topic and xy_valid_sig:
            used_signals.append(f"{local_topic}.{xy_valid_sig}")
        if local_topic and vxy_valid_sig:
            used_signals.append(f"{local_topic}.{vxy_valid_sig}")
        source_note = (
            "npfg_status.signed_track_error (primary)"
            if source_priority == "primary_npfg"
            else "position_controller_status.xtrack_error (explicit secondary source)"
        )

        if mode_ctx.get("status_hint"):
            reason = mode_ctx.get("not_evaluated_reason") or "No applicable automatic path-following segment."
            common_kwargs = dict(
                scope="airframe", segment="path", required_topics=required_path,
                found_topics=found_topics, sample_count=int(mode_ctx.get("sample_count", 0)),
                used_signals=used_signals, signal_semantics="px4_explicit",
                mode_gate="nav_state_applicable_modes",
                mode_gate_reason="Only nav_state modes where path tracking is meaningful are scored.",
                evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                applicable_modes=mode_ctx.get("applicable_modes"),
                excluded_modes=mode_ctx.get("excluded_modes"),
                excluded_reasons=mode_ctx.get("excluded_reasons"),
                evaluated_modes=mode_ctx.get("evaluated_modes"),
                not_evaluated_reason=reason,
            )
            return [
                self._item(
                    "navigation_path_tracking_error", "Navigation Path Tracking Error", None,
                    mode_ctx["status_hint"], None, cfg["unit"], reason, cfg["suggestion"],
                    reference_class="px4_explicit", normalization_basis="time-weighted mean absolute signed cross-track error",
                    **common_kwargs,
                ),
                self._item(
                    "navigation_path_tracking_reliability", "Navigation Path Tracking Reliability", None,
                    mode_ctx["status_hint"], None, str(reliability_cfg.get("unit", "score")), reason,
                    str(reliability_cfg.get("suggestion", "Review path-tail error and local-position validity.")),
                    reference_class="mixed_reference", normalization_basis="P95/reference ratio and XY-validity cross-check",
                    **common_kwargs,
                ),
            ]

        eval_mask = np.asarray(mode_ctx.get("evaluated_mask", []), dtype=bool)
        weights = np.asarray(mode_ctx.get("weights", []), dtype=np.float64)
        used_count = int(np.sum(eval_mask))
        finite_eval = err[eval_mask]
        finite_eval = finite_eval[np.isfinite(finite_eval)]
        mae = self._weighted_mean(err[eval_mask], weights[eval_mask]) if used_count else None
        if mae is None or finite_eval.size <= 0:
            reason = "No finite explicit cross-track samples remained after mode filtering."
            return [
                self._item(
                    "navigation_path_tracking_error", "Navigation Path Tracking Error", None, "unavailable", None, cfg["unit"],
                    reason, cfg["suggestion"], scope="airframe", segment="path",
                    required_topics=required_path, found_topics=found_topics, sample_count=0,
                    used_signals=used_signals, reference_class="px4_explicit", signal_semantics="px4_explicit",
                    not_evaluated_reason=reason,
                ),
                self._item(
                    "navigation_path_tracking_reliability", "Navigation Path Tracking Reliability", None, "unavailable", None,
                    str(reliability_cfg.get("unit", "score")), reason,
                    str(reliability_cfg.get("suggestion", "Review path-tail error and local-position validity.")),
                    scope="airframe", segment="path", required_topics=required_path,
                    found_topics=found_topics, sample_count=0, used_signals=used_signals,
                    reference_class="mixed_reference", signal_semantics="px4_explicit",
                    not_evaluated_reason=reason,
                ),
            ]

        p95 = float(np.percentile(finite_eval, 95))
        p99 = float(np.percentile(finite_eval, 99))
        path_score, path_status = self._score_low(mae, cfg["warn"], cfg["problem"])
        mode_labels = np.asarray([
            self._nav_state_label(x) for x in mode_ctx["joined_frame"]["nav_state"].to_numpy()
        ], dtype=object)
        mode_results = self._mode_results_for_metric(
            mode_labels, eval_mask, err, weights,
            warn=cfg["warn"], problem=cfg["problem"], unit=cfg["unit"],
        )
        timestamps = np.asarray(mode_ctx["joined_frame"]["timestamp_sec"].to_numpy(), dtype=np.float64)
        self._inject_mode_time_intervals(mode_results, timestamps, mode_labels, eval_mask)
        path_conf, path_conf_breakdown = self._confidence_details(
            used_count, 2, len([x for x in [path_topic, status_topic] if x]),
            quality=1.0, mode_factor=max(0.35, float(mode_ctx.get("mode_factor", 0.0))),
            reference_class="px4_explicit",
            fallback_penalty=0.05 if source_priority != "primary_npfg" else 0.0,
        )
        path_item = self._item(
            "navigation_path_tracking_error", "Navigation Path Tracking Error", path_score, path_status,
            mae, cfg["unit"], f"MAE={mae:.2f}m / P95={p95:.2f}m / P99={p99:.2f}m / source={source_note}",
            cfg["suggestion"], scope="airframe", segment="path",
            required_topics=required_path, found_topics=found_topics, sample_count=used_count,
            used_signals=used_signals, confidence=path_conf, confidence_breakdown=path_conf_breakdown,
            evidence=f"MAE warn/problem={float(cfg['warn']):.1f}/{float(cfg['problem']):.1f} m",
            reference_class="px4_explicit", signal_semantics="px4_explicit",
            threshold_source=str(cfg.get("threshold_source", "PX4 explicit cross-track signal + product threshold")),
            threshold_rationale="The score uses an explicit cross-track field; waypoint setpoint distance is excluded.",
            reference_docs=cfg.get("reference_docs", []),
            mode_gate="nav_state_applicable_modes",
            mode_gate_reason="Only nav_state modes where path tracking is meaningful are scored.",
            normalization_basis="time-weighted mean absolute signed cross-track error",
            fallback_used=source_priority != "primary_npfg",
            fallback_source="" if source_priority == "primary_npfg" else source_note,
            fallback_reason="" if source_priority == "primary_npfg" else "NPFG signed_track_error was unavailable; an explicit controller xtrack field was used.",
            evaluation_strategy=mode_ctx.get("evaluation_strategy"),
            total_log_time_sec=mode_ctx.get("total_log_time_sec"),
            evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
            excluded_time_sec=mode_ctx.get("excluded_time_sec"),
            applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
            applicable_modes=mode_ctx.get("applicable_modes"),
            excluded_modes=mode_ctx.get("excluded_modes"),
            excluded_reasons=mode_ctx.get("excluded_reasons"),
            evaluated_modes=mode_ctx.get("evaluated_modes"), mode_results=mode_results,
            evidence_values={
                "path_mae_m": round(float(mae), 3), "path_p95_m": round(p95, 3),
                "path_p99_m": round(p99, 3), "samples_used": used_count,
            },
        )

        reliability_scores = []
        reliability_parts = []
        reliability_evidence = {"path_p95_m": round(p95, 3), "path_p99_m": round(p99, 3), "samples_used": used_count}
        reliability_context = []
        ref_radius, ref_name = self._parameter_float(parameters, ["NAV_ACC_RAD"])
        if ref_radius is not None and ref_radius > 1e-6:
            ref_ratio = float(p95 / ref_radius)
            ratio_score, _ = self._score_low(
                ref_ratio,
                float(reliability_cfg.get("warn_p95_ratio", 0.8)),
                float(reliability_cfg.get("problem_p95_ratio", 1.2)),
            )
            reliability_scores.append(ratio_score)
            reliability_parts.append(f"P95/NAV_ACC_RAD={ref_ratio:.2f}")
            reliability_evidence.update({"reference_radius_m": round(float(ref_radius), 3), "path_ref_ratio": round(ref_ratio, 3)})
            reliability_context.append(f"{ref_name or 'NAV_ACC_RAD'}={float(ref_radius):.3f} m")

        valid_ratio = None
        joined_frame = mode_ctx.get("joined_frame")
        if joined_frame is not None and ("xy_valid" in joined_frame.columns or "v_xy_valid" in joined_frame.columns):
            validity = np.ones(joined_frame.height, dtype=bool)
            if "xy_valid" in joined_frame.columns:
                values = np.asarray(joined_frame["xy_valid"].to_numpy(), dtype=np.float64)
                validity &= np.isfinite(values) & (values > 0.5)
            if "v_xy_valid" in joined_frame.columns:
                values = np.asarray(joined_frame["v_xy_valid"].to_numpy(), dtype=np.float64)
                validity &= np.isfinite(values) & (values > 0.5)
            valid_ratio = 100.0 * float(np.mean(validity[eval_mask])) if used_count else None
        if valid_ratio is not None:
            validity_score, _ = self._score_high(
                valid_ratio,
                float(reliability_cfg.get("xy_valid_warn", 98.0)),
                float(reliability_cfg.get("xy_valid_problem", 95.0)),
            )
            reliability_scores.append(validity_score)
            reliability_parts.append(f"xy_valid={valid_ratio:.1f}%")
            reliability_evidence["xy_valid_ratio_pct"] = round(valid_ratio, 3)

        if reliability_scores:
            reliability_score = float(min(reliability_scores))
            reliability_status = self._status_from_score(
                reliability_score,
                good_min=float(reliability_cfg.get("warn_score_min", 80.0)),
                warn_min=float(reliability_cfg.get("problem_score_min", 60.0)),
            )
            corroborator_count = int(ref_radius is not None and ref_radius > 1e-6) + int(valid_ratio is not None)
            reliability_conf, reliability_breakdown = self._confidence_details(
                used_count, 3, 2 + int(valid_ratio is not None),
                quality=0.95 if corroborator_count >= 2 else 0.80,
                mode_factor=max(0.35, float(mode_ctx.get("mode_factor", 0.0))),
                reference_class="mixed_reference", fallback_penalty=0.0 if corroborator_count >= 2 else 0.15,
            )
            reliability_item = self._item(
                "navigation_path_tracking_reliability", "Navigation Path Tracking Reliability",
                reliability_score, reliability_status, reliability_score,
                str(reliability_cfg.get("unit", "score")), " / ".join(reliability_parts),
                str(reliability_cfg.get("suggestion", "Review path-tail error and local-position validity.")),
                scope="airframe", segment="path",
                required_topics=required_path + ["vehicle_local_position.xy_valid|vehicle_local_position.v_xy_valid"],
                found_topics=found_topics, sample_count=used_count, used_signals=used_signals,
                confidence=reliability_conf, confidence_breakdown=reliability_breakdown,
                parameter_context=reliability_context, reference_class="mixed_reference",
                signal_semantics="px4_explicit",
                threshold_source=str(reliability_cfg.get("threshold_source", "NAV_ACC_RAD and local-position validity cross-check")),
                threshold_rationale="Reliability is separated from cross-track magnitude and uses independent tail/validity checks.",
                mode_gate="nav_state_applicable_modes",
                normalization_basis="minimum score from P95/reference ratio and XY-validity ratio",
                fallback_used=corroborator_count < 2,
                fallback_source="single corroborator" if corroborator_count < 2 else "",
                fallback_reason="Only one independent reliability check was available." if corroborator_count < 2 else "",
                evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                applicable_modes=mode_ctx.get("applicable_modes"), excluded_modes=mode_ctx.get("excluded_modes"),
                excluded_reasons=mode_ctx.get("excluded_reasons"), evaluated_modes=mode_ctx.get("evaluated_modes"),
                evidence_values=reliability_evidence,
            )
        else:
            reason = "NAV_ACC_RAD and local-position XY validity were unavailable; the cross-track score is shown without a reliability verdict."
            reliability_item = self._item(
                "navigation_path_tracking_reliability", "Navigation Path Tracking Reliability", None, "unavailable", None,
                str(reliability_cfg.get("unit", "score")), reason,
                str(reliability_cfg.get("suggestion", "Review path-tail error and local-position validity.")),
                scope="airframe", segment="path",
                required_topics=required_path + ["vehicle_local_position.xy_valid|vehicle_local_position.v_xy_valid"],
                found_topics=found_topics, sample_count=used_count, used_signals=used_signals,
                reference_class="mixed_reference", signal_semantics="px4_explicit",
                not_evaluated_reason=reason,
            )
        return [path_item, reliability_item]

    def _find_rpm_series(self, dataset) -> Tuple[Optional[pl.DataFrame], Optional[str], Optional[str]]:
        if dataset is None or not hasattr(dataset, "topics"):
            return None, None, None
        best = None
        best_score = -1
        for topic_name, topic_data in dataset.topics.items():
            try:
                df = self._with_timestamp_sec(topic_data.dataframe)
            except Exception:
                continue
            if "timestamp_sec" not in df.columns:
                continue
            candidates = [c for c in df.columns if c not in ("timestamp", "timestamp_sec") and "rpm" in c.lower()]
            if not candidates:
                continue
            for sig in candidates:
                s = 0
                sl = sig.lower()
                tl = str(topic_name).lower()
                if "main" in sl or "rotor" in sl:
                    s += 4
                if "rpm" in tl:
                    s += 2
                if "esc" in tl:
                    s += 1
                if s > best_score:
                    best_score = s
                    best = (df, str(topic_name), str(sig))
        if best is None:
            return None, None, None
        df, topic_name, sig = best
        try:
            out = df.select([
                pl.col("timestamp_sec").cast(pl.Float64),
                pl.col(sig).cast(pl.Float64).alias("rpm"),
            ]).drop_nulls().sort("timestamp_sec")
            if out.height <= 0:
                return None, topic_name, sig
            return out, topic_name, sig
        except Exception:
            return None, topic_name, sig

    @staticmethod
    def _parameter_float(parameters, names) -> Tuple[Optional[float], Optional[str]]:
        if not isinstance(parameters, dict):
            return None, None
        if isinstance(names, str):
            names = [names]
        for name in names:
            key = str(name or "").strip()
            if not key or key not in parameters:
                continue
            try:
                return float(parameters[key]), key
            except Exception:
                try:
                    return float(str(parameters[key]).strip()), key
                except Exception:
                    continue
        return None, None

    @staticmethod
    def _format_param_context(name: str, value: Optional[float], description: str) -> str:
        if value is None:
            return ""
        return f"{name}={value:.3f} ({description})"

    def _build_tecs_parameter_context(self, parameters) -> List[str]:
        lines = []
        alt_tc, alt_name = self._parameter_float(parameters, ["FW_T_ALT_TC", "FW_T_TIME_CONST"])
        tas_tc, tas_name = self._parameter_float(parameters, ["FW_T_TAS_TC"])
        spd_weight, spd_weight_name = self._parameter_float(parameters, ["FW_T_SPDWEIGHT"])
        pitch_damp, pitch_damp_name = self._parameter_float(parameters, ["FW_T_PTCH_DAMP"])
        thr_damp, thr_damp_name = self._parameter_float(parameters, ["FW_T_THR_DAMPING", "FW_T_THR_DAMP"])
        for name, value, desc in (
            (alt_name or "FW_T_ALT_TC/FW_T_TIME_CONST", alt_tc, "고도 응답 시간상수"),
            (tas_name or "FW_T_TAS_TC", tas_tc, "속도 응답 시간상수"),
            (spd_weight_name or "FW_T_SPDWEIGHT", spd_weight, "pitch의 속도 우선 vs 고도 우선 가중치"),
            (pitch_damp_name or "FW_T_PTCH_DAMP", pitch_damp, "pitch 에너지 분배 damping"),
            (thr_damp_name or "FW_T_THR_DAMPING/FW_T_THR_DAMP", thr_damp, "throttle total-energy damping"),
        ):
            line = self._format_param_context(name, value, desc)
            if line:
                lines.append(line)
        return lines

    def _resolve_vtol_transition_thresholds(self, parameters, warn_threshold, problem_threshold):
        base_warn = float(warn_threshold)
        base_problem = float(problem_threshold)
        effective_warn = base_warn
        effective_problem = base_problem

        timeout_value, timeout_name = self._parameter_float(parameters, ["VT_TRANS_TIMEOUT"])
        open_loop_value, open_loop_name = self._parameter_float(parameters, ["VT_F_TR_OL_TM"])
        min_time_value, min_time_name = self._parameter_float(parameters, ["VT_TRANS_MIN_TM"])

        parameter_context = []
        if timeout_value is not None:
            parameter_context.append(
                f"{timeout_name or 'VT_TRANS_TIMEOUT'}={timeout_value:.3f} s (전환 제한 시간, 초과 시 abort/quadchute 판단 기준)"
            )
        if open_loop_value is not None:
            parameter_context.append(
                f"{open_loop_name or 'VT_F_TR_OL_TM'}={open_loop_value:.3f} s (front transition open-loop 참고 시간)"
            )
        if min_time_value is not None:
            parameter_context.append(
                f"{min_time_name or 'VT_TRANS_MIN_TM'}={min_time_value:.3f} s (front transition minimum 유지 참고 시간)"
            )

        if timeout_value is not None and np.isfinite(timeout_value) and timeout_value > 0.0:
            effective_problem = min(base_problem, float(timeout_value))
            warn_candidate = effective_problem * 0.7
            reference_floor = max(
                [
                    float(v)
                    for v in (open_loop_value, min_time_value)
                    if v is not None and np.isfinite(v) and float(v) > 0.0
                ] or [0.0]
            )
            if reference_floor > 0.0:
                warn_candidate = max(warn_candidate, min(reference_floor, effective_problem * 0.95))
            effective_warn = min(base_warn, warn_candidate)
            if effective_warn >= effective_problem:
                effective_warn = max(effective_problem * 0.85, effective_problem - 0.5)
            threshold_source = "PX4 VT_TRANS_TIMEOUT hard limit + product default transition thresholds"
            threshold_rationale = (
                "실제 로그 파라미터 VT_TRANS_TIMEOUT을 hard limit로 사용하되, "
                "제품 기본 threshold가 더 엄격하면 그 값을 유지합니다."
            )
            normalization_basis = (
                f"전환 이벤트별 지속시간을 계산하고 최대 지속시간을 점수화합니다. "
                f"problem 기준은 min(기본 {base_problem:.2f}s, {timeout_name or 'VT_TRANS_TIMEOUT'} {float(timeout_value):.2f}s), "
                f"warn 기준은 그 problem 기준의 70%를 기본 warn {base_warn:.2f}s와 비교해 더 엄격한 값을 사용합니다."
            )
        else:
            threshold_source = "Product default transition thresholds (PX4 transition timeout parameter unavailable in log)"
            threshold_rationale = "전환 제한 파라미터를 로그에서 확인할 수 없어서 제품 기본 transition duration threshold를 사용합니다."
            normalization_basis = (
                f"전환 이벤트별 지속시간을 계산하고 최대 지속시간을 점수화합니다. "
                f"warn/problem 기준은 기본값 {base_warn:.2f}s / {base_problem:.2f}s 입니다."
            )

        evidence_values = {
            "transition_warn_threshold_sec": float(effective_warn),
            "transition_problem_threshold_sec": float(effective_problem),
        }
        if timeout_value is not None and np.isfinite(timeout_value):
            evidence_values["vt_trans_timeout_sec"] = float(timeout_value)
        if open_loop_value is not None and np.isfinite(open_loop_value):
            evidence_values["transition_open_loop_time_sec"] = float(open_loop_value)
        if min_time_value is not None and np.isfinite(min_time_value):
            evidence_values["transition_min_time_sec"] = float(min_time_value)

        return {
            "warn": float(effective_warn),
            "problem": float(effective_problem),
            "parameter_context": parameter_context,
            "threshold_source": threshold_source,
            "threshold_rationale": threshold_rationale,
            "normalization_basis": normalization_basis,
            "evidence_values": evidence_values,
        }

    def _guidance(self, metric_key: str, status: str) -> Tuple[List[str], List[str], List[str], List[str]]:
        sev = str(status or "").lower()
        if sev not in ("warning", "problem"):
            return [], [], [], []
        cfg_guidance = self.cfg.get("tuning_guidance", {})
        metric_cfg = cfg_guidance.get(metric_key, {}) if isinstance(cfg_guidance, dict) else {}
        sev_cfg = metric_cfg.get(sev, {}) if isinstance(metric_cfg, dict) else {}
        if not isinstance(sev_cfg, dict):
            sev_cfg = {}
        if not sev_cfg:
            sev_cfg = DEFAULT_TUNING_GUIDANCE.get(metric_key, {}).get(sev, {}) or {}
        likely = sev_cfg.get("likely_causes", [])
        actions = sev_cfg.get("tuning_actions", [])
        priority = sev_cfg.get("priority_checks", [])
        checklist = sev_cfg.get("verification_checklist", [])
        if not priority:
            priority = DEFAULT_PRIORITY_CHECKS.get(metric_key, [])
        return list(likely or []), list(priority or []), list(actions or []), list(checklist or [])

    def _item(
        self,
        key,
        name,
        score,
        status,
        value,
        unit,
        reason,
        suggestion,
        *,
        scope="common",
        segment="all",
        is_core=True,
        required_topics=None,
        found_topics=None,
        sample_count=0,
        confidence=None,
        evidence=None,
        used_signals=None,
        airframe=None,
        likely_causes=None,
        priority_checks=None,
        tuning_actions=None,
        verification_checklist=None,
        parameter_context=None,
        reference_class=None,
        signal_semantics=None,
        threshold_source=None,
        threshold_rationale=None,
        reference_docs=None,
        mode_gate=None,
        mode_gate_reason=None,
        normalization_basis=None,
        fallback_used=False,
        fallback_source=None,
        fallback_reason=None,
        confidence_breakdown=None,
        evaluated_modes=None,
        applicable_modes=None,
        excluded_modes=None,
        evaluation_strategy=None,
        total_log_time_sec=None,
        evaluated_time_sec=None,
        excluded_time_sec=None,
        applied_ratio_pct=None,
        excluded_reasons=None,
        mode_results=None,
        not_evaluated_reason=None,
        evidence_values=None,
    ):
        required_topics = list(required_topics or [])
        found_topics = [str(x) for x in list(found_topics or []) if str(x).strip()]
        # preserve order while removing duplicates
        found_topics = list(dict.fromkeys(found_topics))
        used_signals = [str(x) for x in list(used_signals or []) if str(x).strip()]
        used_signals = list(dict.fromkeys(used_signals))
        metric_cfg = self._metric_cfg(key, airframe=airframe)
        if reference_class is None:
            reference_class = metric_cfg.get("reference_class", "heuristic_inference")
        if signal_semantics is None:
            signal_semantics = metric_cfg.get("signal_semantics", reference_class)
        if threshold_source is None:
            threshold_source = metric_cfg.get("threshold_source", "")
        if threshold_rationale is None:
            threshold_rationale = metric_cfg.get("threshold_rationale", "")
        if reference_docs is None:
            reference_docs = metric_cfg.get("reference_docs", [])
        if mode_gate is None:
            mode_gate = metric_cfg.get("mode_gate", "")
        if normalization_basis is None:
            normalization_basis = metric_cfg.get("normalization_basis", "")
        missing_topics = self._missing_topics_from_required(required_topics, found_topics)
        mode_verdict = self._summarize_mode_verdict(status, mode_results)
        overall_average_status = str(mode_verdict.get("overall_average_status", status) or status)
        overall_average_status_label = str(mode_verdict.get("overall_average_status_label", self.STATUS_LABEL.get(overall_average_status, overall_average_status)) or self.STATUS_LABEL.get(overall_average_status, overall_average_status))
        final_status = str(mode_verdict.get("final_status", status) or status)
        final_status_label = str(mode_verdict.get("final_status_label", self.STATUS_LABEL.get(final_status, final_status)) or self.STATUS_LABEL.get(final_status, final_status))

        if confidence is None:
            quality = 1.0 if status not in ("unavailable", "excluded") else (0.35 if status == "excluded" else 0.2)
            confidence, confidence_breakdown = self._confidence_details(
                sample_count,
                max(1, len(required_topics)),
                max(1 if found_topics else 0, len(found_topics)),
                quality=quality,
                mode_factor=1.0,
                reference_class=reference_class,
                fallback_penalty=0.15 if fallback_used else 0.0,
            )
        elif confidence_breakdown is None:
            confidence_breakdown = {
                "formula": "provided_by_metric",
                "value": round(float(confidence), 1),
            }
        if likely_causes is None or tuning_actions is None or verification_checklist is None:
            gl, gp, ga, gc = self._guidance(key, status)
            if likely_causes is None:
                likely_causes = gl
            if priority_checks is None:
                priority_checks = gp
            if tuning_actions is None:
                tuning_actions = ga
            if verification_checklist is None:
                verification_checklist = gc
        return {
            "key": key, "name": name, "score": None if score is None else round(float(score), 1),
            "status": final_status, "status_label": final_status_label, "severity": final_status,
            "overall_average_status": overall_average_status,
            "overall_average_status_label": overall_average_status_label,
            "worst_mode_status": str(mode_verdict.get("worst_mode_status", "") or ""),
            "worst_mode_status_label": str(mode_verdict.get("worst_mode_status_label", "") or ""),
            "problem_modes": copy.deepcopy(mode_verdict.get("problem_modes", [])),
            "warning_modes": copy.deepcopy(mode_verdict.get("warning_modes", [])),
            "safety_critical_problem_modes": copy.deepcopy(mode_verdict.get("safety_critical_problem_modes", [])),
            "safety_critical_issue_modes": copy.deepcopy(mode_verdict.get("safety_critical_issue_modes", [])),
            "mode_attention_flag": bool(mode_verdict.get("mode_attention_flag", False)),
            "mode_attention_reason": str(mode_verdict.get("mode_attention_reason", "") or ""),
            "mode_interpretation_lines": copy.deepcopy(mode_verdict.get("interpretation_lines", [])),
            "value": None if value is None else float(value), "metric_value": None if value is None else float(value),
            "unit": unit, "reason": reason, "evidence": (reason if evidence is None else evidence), "suggestion": suggestion, "recommendation": suggestion,
            "scope": scope, "segment": segment, "is_core": bool(is_core),
            "required_topics": required_topics, "found_topics": found_topics, "missing_topics": missing_topics, "sample_count": int(sample_count),
            "used_signals": used_signals,
            "px4_version": getattr(self, "_eval_px4_version_summary", "Unknown"),
            "px4_version_details": copy.deepcopy(getattr(self, "_eval_px4_version_details", {})),
            "reference_class": reference_class,
            "signal_semantics": signal_semantics,
            "threshold_source": threshold_source,
            "threshold_rationale": threshold_rationale,
            "reference_docs": list(reference_docs or []),
            "mode_gate": mode_gate,
            "mode_gate_reason": mode_gate_reason or "",
            "normalization_basis": normalization_basis or "",
            "fallback_used": bool(fallback_used),
            "fallback_source": "" if not fallback_source else str(fallback_source),
            "fallback_reason": "" if not fallback_reason else str(fallback_reason),
            "evaluated_modes": list(evaluated_modes or []),
            "applicable_modes": list(applicable_modes or []),
            "excluded_modes": list(excluded_modes or []),
            "evaluation_strategy": "" if not evaluation_strategy else str(evaluation_strategy),
            "total_log_time_sec": None if total_log_time_sec is None else float(total_log_time_sec),
            "evaluated_time_sec": None if evaluated_time_sec is None else float(evaluated_time_sec),
            "excluded_time_sec": None if excluded_time_sec is None else float(excluded_time_sec),
            "applied_ratio_pct": None if applied_ratio_pct is None else float(applied_ratio_pct),
            "excluded_reasons": list(excluded_reasons or []),
            "mode_results": copy.deepcopy(mode_results if isinstance(mode_results, list) else []),
            "not_evaluated_reason": "" if not not_evaluated_reason else str(not_evaluated_reason),
            "evidence_values": copy.deepcopy(evidence_values if isinstance(evidence_values, dict) else {}),
            "likely_causes": list(likely_causes or []),
            "priority_checks": list(priority_checks or []),
            "tuning_actions": list(tuning_actions or []),
            "verification_checklist": list(verification_checklist or []),
            "parameter_context": list(parameter_context or []),
            "confidence_breakdown": copy.deepcopy(confidence_breakdown if isinstance(confidence_breakdown, dict) else {}),
            "confidence": float(confidence), "metric_confidence": float(confidence),
            "weight": float(self._metric_weight(key, airframe=airframe)),
        }
    def _eval_common(self, dataset) -> List[dict]:
        items = []
        # Vibration
        c = self._metric_cfg("vibration"); t = self._find_topic(dataset, ["sensor_combined", "vehicle_acceleration", "vehicle_imu"])
        if not t:
            items.append(self._item("vibration", "Vibration", None, "unavailable", None, c["unit"], "Acceleration topic not found.", c["suggestion"], required_topics=["sensor_combined|vehicle_acceleration|vehicle_imu"]))
        else:
            df = dataset.topics[t].dataframe
            cols = [self._find_signal(df, [f"accelerometer_m_s2[{i}]", f"accel_m_s2[{i}]", f"xyz[{i}]", f"accel[{i}]"]) for i in range(3)]
            rms = []; n = 0
            for cc in cols:
                v = self._to_float(df[cc].to_numpy()) if cc else np.array([])
                n = max(n, int(v.size))
                if v.size == 0: rms.append(np.nan); continue
                v = v - np.median(v); rms.append(float(np.sqrt(np.mean(v * v))))
            if np.isfinite(np.nanmax(rms)):
                value = float(np.nanmax(rms)); score, status = self._score_low(value, c["warn"], c["problem"])
                items.append(self._item("vibration", "Vibration", score, status, value, c["unit"], f"RMS={value:.3f}", c["suggestion"], found_topics=[t], required_topics=["sensor_combined|vehicle_acceleration|vehicle_imu"], sample_count=n))
            else:
                items.append(self._item("vibration", "Vibration", None, "unavailable", None, c["unit"], "No valid acceleration samples.", c["suggestion"], found_topics=[t], required_topics=["sensor_combined|vehicle_acceleration|vehicle_imu"], sample_count=n))
        # Attitude / Angular / Altitude
        for key, disp, sp_t, act_t, sp_candidates, act_candidates, unit_key in [
            ("attitude_tracking_error", "Attitude Tracking Error", "vehicle_attitude_setpoint", "vehicle_attitude", ["roll_sp_euler", "pitch_sp_euler", "yaw_sp_euler"], ["roll_euler", "pitch_euler", "yaw_euler"], "attitude_tracking_error"),
            ("angular_tracking_error", "Angular Tracking Error", "vehicle_rates_setpoint", "vehicle_angular_velocity", ["roll", "pitch", "yaw"], ["xyz[0]", "xyz[1]", "xyz[2]"], "angular_tracking_error"),
            ("altitude_tracking_error", "Altitude Tracking Error", "vehicle_local_position_setpoint", "vehicle_local_position", ["z"], ["z"], "altitude_tracking_error"),
        ]:
            cfg = self._metric_cfg(unit_key)
            st = self._find_topic(dataset, sp_t)
            at = self._find_topic(dataset, act_t)
            required_topics = [sp_t, act_t, "vehicle_status.nav_state"]
            if not st or not at:
                items.append(self._item(
                    key, disp, None, "unavailable", None, cfg["unit"], "Required topics not found.", cfg["suggestion"],
                    required_topics=required_topics,
                    found_topics=[x for x in [st, at] if x],
                    is_core=(key != "altitude_tracking_error"),
                    evaluation_strategy="mode_filtered_nav_state_segments",
                    not_evaluated_reason="setpoint 또는 actual 토픽이 없어 비행모드별 추종 평가를 할 수 없습니다.",
                ))
                continue
            sdf = self._with_timestamp_sec(dataset.topics[st].dataframe)
            adf = self._with_timestamp_sec(dataset.topics[at].dataframe)
            sp_cols = [pl.col("timestamp_sec").cast(pl.Float64)]
            act_cols = [pl.col("timestamp_sec").cast(pl.Float64)]
            used_signals = []
            axis_aliases = []
            for i in range(min(len(sp_candidates), len(act_candidates))):
                ss = self._find_signal(sdf, [sp_candidates[i]])
                aa = self._find_signal(adf, [act_candidates[i]])
                if not ss or not aa:
                    continue
                sp_alias = f"sp_{i}"
                act_alias = f"act_{i}"
                sp_cols.append(pl.col(ss).cast(pl.Float64).alias(sp_alias))
                act_cols.append(pl.col(aa).cast(pl.Float64).alias(act_alias))
                used_signals.extend([f"{st}.{ss}", f"{at}.{aa}"])
                axis_aliases.append((act_alias, sp_alias))
            if not axis_aliases:
                items.append(self._item(
                    key, disp, None, "unavailable", None, cfg["unit"], "No valid signal pairs.", cfg["suggestion"],
                    required_topics=required_topics,
                    found_topics=[st, at],
                    is_core=(key != "altitude_tracking_error"),
                    evaluation_strategy="mode_filtered_nav_state_segments",
                    not_evaluated_reason="setpoint와 actual 신호쌍을 찾지 못했습니다.",
                ))
                continue
            try:
                sp_df = sdf.select(sp_cols).sort("timestamp_sec")
                act_df = adf.select(act_cols).sort("timestamp_sec")
                mm = act_df.join_asof(sp_df, on="timestamp_sec", strategy="nearest")
            except Exception:
                mm = None
            if mm is None or mm.height <= 0:
                items.append(self._item(
                    key, disp, None, "unavailable", None, cfg["unit"], "No valid signal pairs.", cfg["suggestion"],
                    required_topics=required_topics,
                    found_topics=[st, at],
                    is_core=(key != "altitude_tracking_error"),
                    evaluation_strategy="mode_filtered_nav_state_segments",
                    not_evaluated_reason="actual과 setpoint 시계열을 시간축으로 정렬하지 못했습니다.",
                ))
                continue
            err_stack = []
            for act_alias, sp_alias in axis_aliases:
                err_stack.append(np.abs(np.asarray(mm[act_alias].to_numpy(), dtype=np.float64) - np.asarray(mm[sp_alias].to_numpy(), dtype=np.float64)))
            err_matrix = np.column_stack(err_stack) if err_stack else np.empty((mm.height, 0))
            valid_counts = np.sum(np.isfinite(err_matrix), axis=1) if err_matrix.size else np.zeros(mm.height, dtype=np.int64)
            # ``np.where`` evaluates both branches, so wrapping ``nanmean``
            # still emits "Mean of empty slice" for rows where every axis is
            # missing.  Compute only valid rows to keep normal GUI loads free
            # of misleading runtime warnings.
            row_err = np.full(mm.height, np.nan, dtype=np.float64)
            valid_rows = valid_counts > 0
            if np.any(valid_rows):
                row_err[valid_rows] = (
                    np.nansum(err_matrix[valid_rows], axis=1)
                    / valid_counts[valid_rows]
                )
            applicable_modes, excluded_modes = self._metric_mode_policy(unit_key)
            mode_ctx = self._build_mode_filtered_context(
                dataset,
                mm.select([pl.col("timestamp_sec").cast(pl.Float64)]),
                unit_key,
                valid_mask=(valid_counts > 0),
                metric_values=row_err,
                applicable_modes=applicable_modes,
                excluded_modes=excluded_modes,
                warn=cfg["warn"],
                problem=cfg["problem"],
                unit=cfg["unit"],
            )
            found_topics = [x for x in [st, at, mode_ctx.get("nav_state_topic")] if x]
            if mode_ctx.get("nav_state_topic") and mode_ctx.get("nav_state_signal"):
                used_signals.append(f"{mode_ctx['nav_state_topic']}.{mode_ctx['nav_state_signal']}")
            if mode_ctx.get("status_hint"):
                reason = mode_ctx.get("not_evaluated_reason") or "평가 가능한 비행모드 구간이 없습니다."
                items.append(self._item(
                    key, disp, None, mode_ctx["status_hint"], None, cfg["unit"], reason, cfg["suggestion"],
                    required_topics=required_topics,
                    found_topics=found_topics,
                    sample_count=mode_ctx.get("sample_count", 0),
                    is_core=(key != "altitude_tracking_error"),
                    used_signals=used_signals,
                    mode_gate="nav_state_applicable_modes",
                    mode_gate_reason="vehicle_status.nav_state 기준으로 적용 가능한 비행모드 구간만 점수화했습니다.",
                    fallback_used=False,
                    evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                    total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                    evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                    excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                    applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                    applicable_modes=mode_ctx.get("applicable_modes"),
                    excluded_modes=mode_ctx.get("excluded_modes"),
                    excluded_reasons=mode_ctx.get("excluded_reasons"),
                    evaluated_modes=mode_ctx.get("evaluated_modes"),
                    mode_results=mode_ctx.get("mode_results"),
                    not_evaluated_reason=mode_ctx.get("not_evaluated_reason"),
                ))
                continue
            eval_mask = mode_ctx.get("evaluated_mask", np.asarray([], dtype=bool))
            weights = mode_ctx.get("weights", np.asarray([], dtype=np.float64))
            value = self._weighted_mean(row_err[eval_mask], weights[eval_mask]) if int(np.sum(eval_mask)) > 0 else None
            if value is None:
                items.append(self._item(
                    key, disp, None, "unavailable", None, cfg["unit"], "No valid signal pairs.", cfg["suggestion"],
                    required_topics=required_topics,
                    found_topics=found_topics,
                    sample_count=mode_ctx.get("sample_count", 0),
                    is_core=(key != "altitude_tracking_error"),
                    used_signals=used_signals,
                    evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                    total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                    evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                    excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                    applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                    applicable_modes=mode_ctx.get("applicable_modes"),
                    excluded_modes=mode_ctx.get("excluded_modes"),
                    excluded_reasons=mode_ctx.get("excluded_reasons"),
                    evaluated_modes=mode_ctx.get("evaluated_modes"),
                    mode_results=mode_ctx.get("mode_results"),
                    not_evaluated_reason="적용 가능한 비행모드 구간에 유효한 오차 샘플이 없습니다.",
                ))
                continue
            score, status = self._score_low(value, cfg["warn"], cfg["problem"])
            applied_ratio = float(mode_ctx.get("applied_ratio_pct", 0.0))
            mode_factor = max(0.35, float(mode_ctx.get("mode_factor", 0.0))) if applied_ratio > 0.0 else 0.0
            conf, conf_breakdown = self._confidence_details(
                int(mode_ctx.get("sample_count", 0)),
                len(required_topics),
                len(found_topics),
                quality=1.0,
                mode_factor=mode_factor,
                reference_class=cfg.get("reference_class", "px4_explicit"),
                fallback_penalty=0.0,
            )
            reason_parts = [
                f"MAE={value:.3f}",
                f"evaluated={float(mode_ctx.get('evaluated_time_sec', 0.0)):.1f}s",
                f"coverage={applied_ratio:.1f}%",
            ]
            items.append(self._item(
                key, disp, score, status, value, cfg["unit"], " / ".join(reason_parts), cfg["suggestion"],
                required_topics=required_topics,
                found_topics=found_topics,
                sample_count=int(mode_ctx.get("sample_count", 0)),
                confidence=conf,
                confidence_breakdown=conf_breakdown,
                is_core=(key != "altitude_tracking_error"),
                used_signals=used_signals,
                reference_class=cfg.get("reference_class", "px4_explicit"),
                signal_semantics=cfg.get("signal_semantics", "px4_explicit"),
                threshold_source=cfg.get("threshold_source", ""),
                threshold_rationale=cfg.get("threshold_rationale", ""),
                reference_docs=cfg.get("reference_docs", []),
                mode_gate="nav_state_applicable_modes",
                mode_gate_reason="vehicle_status.nav_state 기준으로 적용 가능한 비행모드 구간만 점수화했습니다.",
                normalization_basis=cfg.get("normalization_basis", "시간 가중 평균 오차"),
                fallback_used=False,
                evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                applicable_modes=mode_ctx.get("applicable_modes"),
                excluded_modes=mode_ctx.get("excluded_modes"),
                excluded_reasons=mode_ctx.get("excluded_reasons"),
                evaluated_modes=mode_ctx.get("evaluated_modes"),
                mode_results=mode_ctx.get("mode_results"),
                evidence_values={"time_weighted_mae": round(float(value), 3)},
            ))
        # Innovation fail ratio
        cfg = self._metric_cfg("innovation_fail_ratio"); et = self._find_topic(dataset, ["estimator_status", "ekf2_innovations"])
        if not et:
            items.append(self._item("innovation_fail_ratio", "Innovation Fail Ratio", None, "unavailable", None, cfg["unit"], "Estimator topic not found.", cfg["suggestion"], required_topics=["estimator_status|ekf2_innovations"]))
        else:
            df = dataset.topics[et].dataframe; fc = self._find_signal(df, ["innovation_check_flags"]); n = 0
            if fc:
                x = self._to_float(df[fc].to_numpy()); n = int(x.size); ratio = float(np.mean(x > 0) * 100.0) if x.size else np.nan
            else:
                tr = [c for c in df.columns if "test_ratio" in c.lower()]
                vals = [np.abs(self._to_float(df[c].to_numpy())) for c in tr if self._to_float(df[c].to_numpy()).size > 0]
                if vals:
                    m = min(len(v) for v in vals); stacked = np.column_stack([v[:m] for v in vals]); n = int(m); ratio = float(np.mean(np.max(stacked, axis=1) > 1.0) * 100.0)
                else:
                    ratio = np.nan
            if np.isfinite(ratio):
                score, status = self._score_low(ratio, cfg["warn"], cfg["problem"])
                items.append(self._item("innovation_fail_ratio", "Innovation Fail Ratio", score, status, ratio, cfg["unit"], f"ratio={ratio:.2f}%", cfg["suggestion"], required_topics=["estimator_status|ekf2_innovations"], found_topics=[et], sample_count=n))
            else:
                items.append(self._item("innovation_fail_ratio", "Innovation Fail Ratio", None, "unavailable", None, cfg["unit"], "No innovation fields available.", cfg["suggestion"], required_topics=["estimator_status|ekf2_innovations"], found_topics=[et]))
        # GPS quality
        cfg = self._metric_cfg("gps_quality")
        gt = self._find_topic(dataset, ["vehicle_gps_position", "sensor_gps"])
        et = self._find_topic(dataset, "estimator_status")
        if not gt and not et:
            items.append(self._item(
                "gps_quality", "GPS Quality", None, "unavailable", None, "score",
                "GPS and estimator topics not found.", cfg["suggestion"],
                required_topics=["vehicle_gps_position|sensor_gps", "estimator_status"],
            ))
        else:
            score = 100.0
            reasons = []
            evidence_values = {}
            found_topics = [x for x in [gt, et] if x]
            used_signals = []
            n = 0
            if gt:
                df = dataset.topics[gt].dataframe
                fix = self._find_signal(df, ["fix_type", "gps_fix_type"])
                sat = self._find_signal(df, ["satellites_used", "satellites", "nsats"])
                if fix:
                    x = self._to_float(df[fix].to_numpy())
                    n = max(n, int(x.size))
                    if x.size:
                        v = float(np.median(x))
                        evidence_values["fix_median"] = round(v, 2)
                        used_signals.append(f"{gt}.{fix}")
                        reasons.append(f"fix={v:.1f}")
                        if v < 3.0:
                            score = min(score, 35.0)
                        elif v < 4.0:
                            score -= 15.0
                if sat:
                    x = self._to_float(df[sat].to_numpy())
                    n = max(n, int(x.size))
                    if x.size:
                        v = float(np.median(x))
                        evidence_values["satellites_median"] = round(v, 2)
                        used_signals.append(f"{gt}.{sat}")
                        reasons.append(f"sat={v:.1f}")
                        if v < 8.0:
                            score -= 25.0
                        elif v < 12.0:
                            score -= 12.0
            fail_labels = []
            if et:
                edf = dataset.topics[et].dataframe
                fail_sig = self._find_signal(edf, ["gps_check_fail_flags"])
                horiz_sig = self._find_signal(edf, ["pos_horiz_accuracy"])
                vert_sig = self._find_signal(edf, ["pos_vert_accuracy"])
                out_pos_sig = self._find_signal(edf, ["output_tracking_error[2]"])
                if fail_sig:
                    raw = self._to_float(edf[fail_sig].to_numpy())
                    n = max(n, int(raw.size))
                    if raw.size:
                        fail_ratio = float(np.mean(raw > 0.0) * 100.0)
                        evidence_values["gps_check_fail_ratio_pct"] = round(fail_ratio, 2)
                        used_signals.append(f"{et}.{fail_sig}")
                        bit_counts = {}
                        for bit, label in self.GPS_CHECK_FAIL_LABELS.items():
                            count = int(np.sum((raw.astype(np.int64) & (1 << bit)) > 0))
                            if count > 0:
                                bit_counts[label] = count
                        if bit_counts:
                            fail_labels = [f"{name}={count}" for name, count in sorted(bit_counts.items(), key=lambda x: (-x[1], x[0]))[:4]]
                        reasons.append(f"gps_fail={fail_ratio:.1f}%")
                        if fail_ratio > 3.0:
                            score -= 35.0
                        elif fail_ratio > 0.0:
                            score -= 15.0
                if horiz_sig:
                    raw = self._to_float(edf[horiz_sig].to_numpy())
                    n = max(n, int(raw.size))
                    if raw.size:
                        p95 = float(np.percentile(raw, 95))
                        evidence_values["pos_horiz_accuracy_p95_m"] = round(p95, 3)
                        used_signals.append(f"{et}.{horiz_sig}")
                        reasons.append(f"hacc_p95={p95:.2f}m")
                        if p95 > 2.0:
                            score -= 20.0
                        elif p95 > 1.0:
                            score -= 10.0
                if vert_sig:
                    raw = self._to_float(edf[vert_sig].to_numpy())
                    n = max(n, int(raw.size))
                    if raw.size:
                        p95 = float(np.percentile(raw, 95))
                        evidence_values["pos_vert_accuracy_p95_m"] = round(p95, 3)
                        used_signals.append(f"{et}.{vert_sig}")
                        reasons.append(f"vacc_p95={p95:.2f}m")
                        if p95 > 4.0:
                            score -= 15.0
                        elif p95 > 2.0:
                            score -= 8.0
                if out_pos_sig:
                    raw = self._to_float(edf[out_pos_sig].to_numpy())
                    n = max(n, int(raw.size))
                    if raw.size:
                        p95 = float(np.percentile(raw, 95))
                        evidence_values["estimator_output_tracking_pos_p95_m"] = round(p95, 3)
                        used_signals.append(f"{et}.{out_pos_sig}")
            score = max(5.0, min(100.0, score))
            status = self._status_from_score(score, float(cfg.get("warn_score_min", 80.0)), float(cfg.get("problem_score_min", 60.0)))
            quality = 0.95 if gt and et else 0.75
            conf, conf_breakdown = self._confidence_details(
                n,
                2,
                len(found_topics),
                quality=quality,
                mode_factor=1.0,
                reference_class="px4_explicit",
                fallback_penalty=0.10 if not (gt and et) else 0.0,
            )
            evidence_parts = []
            if fail_labels:
                evidence_parts.append("GPS fail bits: " + ", ".join(fail_labels))
            if "estimator_output_tracking_pos_p95_m" in evidence_values:
                evidence_parts.append(f"output_tracking_error[2] P95={evidence_values['estimator_output_tracking_pos_p95_m']:.2f} m")
            items.append(self._item(
                "gps_quality", "GPS Quality", score, status, score, "score",
                " / ".join(reasons) if reasons else "GPS/estimator metrics partially available",
                cfg["suggestion"],
                required_topics=["vehicle_gps_position|sensor_gps", "estimator_status"],
                found_topics=found_topics,
                sample_count=n,
                used_signals=used_signals,
                confidence=conf,
                confidence_breakdown=conf_breakdown,
                evidence=" | ".join(evidence_parts) if evidence_parts else "PX4 GPS + estimator health metrics used",
                reference_class="px4_explicit",
                signal_semantics="px4_explicit",
                threshold_source=str(cfg.get("threshold_source", "PX4 GPS checks + product accuracy targets")),
                threshold_rationale=str(cfg.get("threshold_rationale", "Combine fix quality, EKF GPS fail-bit ratio, and estimator position accuracy.")),
                reference_docs=cfg.get("reference_docs", []),
                mode_gate="all_logged_samples",
                mode_gate_reason="GPS and estimator health are meaningful across the logged flight window.",
                normalization_basis="score from fix/sat + gps_fail_ratio + position_accuracy",
                fallback_used=not (gt and et),
                fallback_source="GPS-only or estimator-only subset" if not (gt and et) else "",
                fallback_reason="One of GPS topic / estimator_status was missing." if not (gt and et) else "",
                evidence_values=evidence_values,
            ))
        # Communication quality
        cfg = self._metric_cfg("communication_quality")
        tt = self._find_topic(dataset, "telemetry_status")
        rt = self._find_topic(dataset, "radio_status")
        if not tt and not rt:
            items.append(self._item(
                "communication_quality", "Communication Quality", None, "unavailable", None, "score",
                "telemetry_status and radio_status not found.", cfg["suggestion"],
                required_topics=["telemetry_status", "radio_status"],
            ))
        else:
            score = 100.0
            reasons = []
            evidence_values = {}
            used_signals = []
            found_topics = [x for x in [tt, rt] if x]
            n = 0
            fallback_used = False
            fallback_source = ""
            if tt:
                tdf = dataset.topics[tt].dataframe
                lost_sig = self._find_signal(tdf, ["rx_message_lost_rate"])
                tx_rate_sig = self._find_signal(tdf, ["tx_rate_avg"])
                tx_err_sig = self._find_signal(tdf, ["tx_error_rate_avg"])
                tx_ovr_sig = self._find_signal(tdf, ["tx_buffer_overruns"])
                rx_ovr_sig = self._find_signal(tdf, ["rx_buffer_overruns"])
                rx_drop_sig = self._find_signal(tdf, ["rx_packet_drop_count"])
                rx_parse_sig = self._find_signal(tdf, ["rx_parse_errors"])
                if lost_sig:
                    raw = self._to_float(tdf[lost_sig].to_numpy())
                    n = max(n, int(raw.size))
                    if raw.size:
                        lost_med = float(np.median(raw))
                        lost_pct = lost_med * 100.0 if lost_med <= 1.5 else lost_med
                        evidence_values["telemetry_lost_rate_pct"] = round(lost_pct, 3)
                        used_signals.append(f"{tt}.{lost_sig}")
                        reasons.append(f"lost={lost_pct:.2f}%")
                        if lost_pct > 2.0:
                            score -= 35.0
                        elif lost_pct > 0.5:
                            score -= 15.0
                if tx_rate_sig and tx_err_sig:
                    tx_rate = self._to_float(tdf[tx_rate_sig].to_numpy())
                    tx_err = self._to_float(tdf[tx_err_sig].to_numpy())
                    if tx_rate.size and tx_err.size:
                        m = min(tx_rate.size, tx_err.size)
                        ratio = 100.0 * tx_err[:m] / np.maximum(tx_rate[:m], 1e-6)
                        ratio = ratio[np.isfinite(ratio)]
                        n = max(n, int(ratio.size))
                        if ratio.size:
                            med_ratio = float(np.median(ratio))
                            evidence_values["telemetry_tx_error_ratio_pct"] = round(med_ratio, 3)
                            used_signals.extend([f"{tt}.{tx_rate_sig}", f"{tt}.{tx_err_sig}"])
                            reasons.append(f"tx_err={med_ratio:.2f}%")
                            if med_ratio > 5.0:
                                score -= 15.0
                            elif med_ratio > 1.0:
                                score -= 8.0
                for sig_name, key_name in (
                    (tx_ovr_sig, "telemetry_tx_buffer_overruns_delta"),
                    (rx_ovr_sig, "telemetry_rx_buffer_overruns_delta"),
                    (rx_drop_sig, "telemetry_rx_packet_drop_delta"),
                    (rx_parse_sig, "telemetry_rx_parse_errors_delta"),
                ):
                    if not sig_name:
                        continue
                    raw = self._to_float(tdf[sig_name].to_numpy())
                    n = max(n, int(raw.size))
                    if raw.size:
                        delta = float(max(0.0, raw[-1] - raw[0]))
                        evidence_values[key_name] = round(delta, 3)
                        used_signals.append(f"{tt}.{sig_name}")
                        if "overruns" in key_name and delta > 0.0:
                            score -= 12.0 if delta <= 3.0 else 24.0
                        elif "drop" in key_name and delta > 0.0:
                            score -= 8.0 if delta <= 5.0 else 16.0
                        elif "parse" in key_name and delta > 0.0:
                            score -= 5.0 if delta <= 5.0 else 12.0
            if rt:
                rdf = dataset.topics[rt].dataframe
                txbuf_sig = self._find_signal(rdf, ["txbuf"])
                rxerr_sig = self._find_signal(rdf, ["rxerrors"])
                rssi_sig = self._find_signal(rdf, ["rssi"])
                noise_sig = self._find_signal(rdf, ["noise"])
                if txbuf_sig:
                    raw = self._to_float(rdf[txbuf_sig].to_numpy())
                    n = max(n, int(raw.size))
                    if raw.size:
                        p95 = float(np.percentile(raw, 95))
                        high_ratio = float(np.mean(raw > 90.0) * 100.0)
                        evidence_values["radio_txbuf_p95_pct"] = round(p95, 3)
                        evidence_values["radio_txbuf_high_ratio_pct"] = round(high_ratio, 3)
                        used_signals.append(f"{rt}.{txbuf_sig}")
                        if not tt:
                            reasons.append(f"txbuf_p95={p95:.1f}%")
                        if high_ratio > 5.0:
                            score -= 20.0
                        elif high_ratio > 1.0:
                            score -= 10.0
                if rxerr_sig:
                    raw = self._to_float(rdf[rxerr_sig].to_numpy())
                    n = max(n, int(raw.size))
                    if raw.size:
                        delta = float(max(0.0, raw[-1] - raw[0]))
                        evidence_values["radio_rxerrors_delta"] = round(delta, 3)
                        used_signals.append(f"{rt}.{rxerr_sig}")
                        if not tt:
                            reasons.append(f"rxerrors={delta:.0f}")
                        if delta > 50.0:
                            score -= 18.0
                        elif delta > 0.0:
                            score -= 8.0
                if rssi_sig and noise_sig:
                    rssi = self._to_float(rdf[rssi_sig].to_numpy())
                    noise = self._to_float(rdf[noise_sig].to_numpy())
                    if rssi.size and noise.size:
                        m = min(rssi.size, noise.size)
                        margin = rssi[:m] - noise[:m]
                        margin = margin[np.isfinite(margin)]
                        if margin.size:
                            evidence_values["radio_rssi_margin_median"] = round(float(np.median(margin)), 3)
                            used_signals.extend([f"{rt}.{rssi_sig}", f"{rt}.{noise_sig}"])
            if not tt and rt:
                fallback_used = True
                fallback_source = "radio_status only"
            score = max(5.0, min(100.0, score))
            status = self._status_from_score(score, float(cfg.get("warn_score_min", 80.0)), float(cfg.get("problem_score_min", 60.0)))
            quality = 0.95 if tt else 0.70
            conf, conf_breakdown = self._confidence_details(
                n,
                2,
                len(found_topics),
                quality=quality,
                mode_factor=1.0,
                reference_class="mixed_reference",
                fallback_penalty=0.20 if fallback_used else 0.0,
            )
            items.append(self._item(
                "communication_quality", "Communication Quality", score, status, score, "score",
                " / ".join(reasons) if reasons else "Telemetry/radio counters partially available",
                cfg["suggestion"],
                required_topics=["telemetry_status", "radio_status"],
                found_topics=found_topics,
                sample_count=n,
                used_signals=used_signals,
                confidence=conf,
                confidence_breakdown=conf_breakdown,
                evidence="PX4 transport counters and radio health fields combined for link-quality scoring.",
                reference_class="mixed_reference",
                signal_semantics="px4_explicit",
                threshold_source=str(cfg.get("threshold_source", "PX4 field semantics + product operating thresholds")),
                threshold_rationale=str(cfg.get("threshold_rationale", "Packet loss, buffer pressure, and receive error growth are more reliable than RSSI alone.")),
                reference_docs=cfg.get("reference_docs", []),
                mode_gate="all_logged_samples",
                mode_gate_reason="Link-quality counters are meaningful across the flight log regardless of control mode.",
                normalization_basis="score from telemetry loss/error counters, with radio fallback context",
                fallback_used=fallback_used,
                fallback_source=fallback_source,
                fallback_reason="telemetry_status unavailable; using radio_status-only context." if fallback_used else "",
                evidence_values=evidence_values,
            ))
        # Failsafe and mission stability
        cfg_fs = self._metric_cfg("failsafe_event_count"); cfg_ms = self._metric_cfg("mission_stability"); st = self._find_topic(dataset, "vehicle_status")
        if not st:
            items.append(self._item(
                "failsafe_event_count", "Failsafe Event Count", None, "unavailable", None, cfg_fs["unit"],
                "vehicle_status not found.", cfg_fs["suggestion"], required_topics=["vehicle_status"],
            ))
            items.append(self._item(
                "mission_stability", "Mission Stability", None, "unavailable", None, cfg_ms["unit"],
                "vehicle_status not found.", cfg_ms["suggestion"], required_topics=["vehicle_status"],
            ))
        else:
            df = self._with_timestamp_sec(dataset.topics[st].dataframe)
            fc = self._find_signal(df, ["failsafe", "failsafe_state"])
            nc = self._find_signal(df, ["nav_state", "main_state"])
            intent_sig = self._find_signal(df, ["nav_state_user_intention"])
            gcs_loss_sig = self._find_signal(df, ["gcs_connection_lost_counter"])
            f_events = 0; switches = 0; nav_div_ratio = None; gcs_loss_events = 0
            used_signals = []
            if fc:
                x = self._to_float(df[fc].to_numpy()); f_events = self._count_rising_edges(x); used_signals.append(f"{st}.{fc}")
            if nc:
                x = self._to_float(df[nc].to_numpy()); switches = int(np.sum(np.abs(np.diff(x)) > 0)) if x.size > 1 else 0; used_signals.append(f"{st}.{nc}")
            if intent_sig and nc:
                nav_arr = self._to_float(df[nc].to_numpy())
                intent_arr = self._to_float(df[intent_sig].to_numpy())
                if nav_arr.size and intent_arr.size:
                    m = min(nav_arr.size, intent_arr.size)
                    nav_div_ratio = float(np.mean(np.abs(nav_arr[:m] - intent_arr[:m]) > 0.0) * 100.0)
                    used_signals.append(f"{st}.{intent_sig}")
            if gcs_loss_sig:
                x = self._to_float(df[gcs_loss_sig].to_numpy())
                if x.size:
                    gcs_loss_events = int(max(0.0, x[-1] - x[0]))
                    used_signals.append(f"{st}.{gcs_loss_sig}")
            msg_cls = self._classify_logged_messages(getattr(self, "_eval_messages", []))
            msg_counts = msg_cls.get("counts", {}) if isinstance(msg_cls, dict) else {}
            msg_examples = msg_cls.get("examples", {}) if isinstance(msg_cls, dict) else {}
            msg_total = int(sum(int(v) for v in msg_counts.values())) if isinstance(msg_counts, dict) else 0
            recoverable_msg = int(
                msg_counts.get("rc_loss", 0)
                + msg_counts.get("data_link_loss", 0)
                + msg_counts.get("offboard_loss", 0)
                + msg_counts.get("battery", 0)
            )
            critical_events = int(
                msg_counts.get("termination", 0)
                + msg_counts.get("motor_failure", 0)
                + msg_counts.get("geofence", 0)
            )
            flight_sec = self._flight_time_sec(dataset)
            effective_event_count = max(int(f_events), int(recoverable_msg + critical_events + gcs_loss_events))
            effective_hours = max(flight_sec / 3600.0, 10.0 / 60.0) if effective_event_count > 0 else max(flight_sec / 3600.0, 1e-6)
            effective_rate_hr = float(effective_event_count / effective_hours) if effective_event_count > 0 else 0.0

            # 1) Explicit failsafe event count: absolute occurrence count is the
            # user-facing metric. Any occurrence is at least warning-level.
            if critical_events > 0:
                failsafe_score = 25.0
                failsafe_status = "problem"
            elif effective_event_count <= 0:
                failsafe_score = 100.0
                failsafe_status = "good"
            elif effective_event_count >= int(max(1.0, float(cfg_fs.get("problem", 3.0)))):
                failsafe_score = 35.0
                failsafe_status = "problem"
            else:
                failsafe_score = 72.0
                failsafe_status = "warning"
            failsafe_value = float(effective_event_count)
            common_evidence_values = {
                "failsafe_edges": int(f_events),
                "recoverable_message_events": int(recoverable_msg),
                "critical_message_events": int(critical_events),
                "gcs_connection_lost_events": int(gcs_loss_events),
                "mode_switch_count": int(switches),
                "flight_time_min": round(flight_sec / 60.0, 2),
                "effective_event_rate_per_hr": round(effective_rate_hr, 3),
            }
            if nav_div_ratio is not None:
                common_evidence_values["nav_state_divergence_ratio_pct"] = round(nav_div_ratio, 3)
            failsafe_reason_parts = [
                f"event_count={effective_event_count}",
                f"critical={critical_events}",
            ]
            if gcs_loss_events > 0:
                failsafe_reason_parts.append(f"gcs_loss={gcs_loss_events}")
            evidence_parts = []
            for label in ("rc_loss", "data_link_loss", "offboard_loss", "battery", "termination", "motor_failure", "geofence"):
                examples = msg_examples.get(label, []) if isinstance(msg_examples, dict) else []
                if examples:
                    evidence_parts.append(f"{label}: {examples[0]}")
            ref_class = "mixed_reference"
            conf_fs, conf_breakdown_fs = self._confidence_details(
                int(df.height),
                1,
                1,
                quality=0.90 if (fc or gcs_loss_sig or msg_total > 0) else 0.65,
                mode_factor=1.0,
                reference_class=ref_class,
                fallback_penalty=0.0 if (fc or msg_total > 0) else 0.20,
            )
            items.append(self._item(
                "failsafe_event_count", "Failsafe Event Count", failsafe_score, failsafe_status, failsafe_value, cfg_fs["unit"],
                " / ".join(failsafe_reason_parts), cfg_fs["suggestion"],
                required_topics=["vehicle_status"],
                found_topics=[st],
                sample_count=int(df.height),
                used_signals=used_signals,
                confidence=conf_fs,
                confidence_breakdown=conf_breakdown_fs,
                evidence=" | ".join(evidence_parts) if evidence_parts else "vehicle_status failsafe edge and logged_messages used for trigger classification",
                reference_class=ref_class,
                signal_semantics="px4_explicit",
                threshold_source=str(cfg_fs.get("threshold_source", "Product default: any explicit failsafe event => warning, repeated or critical events => problem")),
                threshold_rationale=str(cfg_fs.get("threshold_rationale", "Absolute failsafe occurrence count is the clearest operator-facing safety metric.")),
                reference_docs=cfg_fs.get("reference_docs", []),
                mode_gate="all_logged_samples",
                mode_gate_reason="Failsafe occurrence is evaluated over the full logged flight window.",
                normalization_basis="absolute explicit/recoverable failsafe event count with critical-event escalation",
                fallback_used=not bool(fc or msg_total > 0),
                fallback_source="mode-switch heuristic" if not bool(fc or msg_total > 0) else "",
                fallback_reason="No explicit failsafe edge or failsafe-classified messages; score is less direct." if not bool(fc or msg_total > 0) else "",
                evidence_values=common_evidence_values,
                evaluated_modes=self._nav_state_labels(self._to_float(df[nc].to_numpy())[:12] if nc else []),
            ))

            # 2) Mission stability: separate from failsafe occurrence. Prefer
            # nav_state vs user-intention divergence; fall back to mode-switch
            # rate only when intention telemetry is unavailable.
            mission_used_signals = list(used_signals)
            if fc:
                mission_used_signals = [x for x in mission_used_signals if x != f"{st}.{fc}"]
            mission_fallback_used = False
            mission_fallback_source = ""
            mission_fallback_reason = ""
            mission_evidence_values = dict(common_evidence_values)
            if nav_div_ratio is not None:
                mission_score, _ = self._score_low(float(nav_div_ratio), 2.0, 10.0)
                mission_status = self._status_from_score(
                    mission_score,
                    float(cfg_ms.get("warn_score_min", 80.0)),
                    float(cfg_ms.get("problem_score_min", 60.0)),
                )
                mission_value = float(mission_score)
                mission_reason = f"nav_div={nav_div_ratio:.2f}% / mode_switches={switches}"
                mission_normalization = "mission stability score from nav_state_user_intention divergence ratio"
                mission_threshold_source = str(cfg_ms.get("threshold_source", "Product default: nav_state divergence score, with mode-switch context"))
                mission_threshold_rationale = str(cfg_ms.get("threshold_rationale", "Mission stability is best represented by how often the actual mode diverges from intended mode."))
            else:
                switch_hours = max(flight_sec / 3600.0, 10.0 / 60.0) if switches > 0 else max(flight_sec / 3600.0, 1e-6)
                switch_rate_hr = float(switches / switch_hours) if switches > 0 else 0.0
                mission_score, _ = self._score_low(float(switch_rate_hr), 20.0, 60.0)
                mission_status = self._status_from_score(
                    mission_score,
                    float(cfg_ms.get("warn_score_min", 80.0)),
                    float(cfg_ms.get("problem_score_min", 60.0)),
                )
                mission_value = float(mission_score)
                mission_reason = f"mode_switch_rate={switch_rate_hr:.2f}/hr / switches={switches}"
                mission_normalization = "mission stability score from mode-switch rate fallback"
                mission_threshold_source = str(cfg_ms.get("threshold_source", "Fallback: mode-switch rate when nav_state_user_intention is unavailable"))
                mission_threshold_rationale = str(cfg_ms.get("threshold_rationale", "Frequent mode switching is a weaker but useful fallback indicator of unstable mission execution."))
                mission_fallback_used = True
                mission_fallback_source = "mode_switch_rate_per_hr"
                mission_fallback_reason = "nav_state_user_intention signal unavailable; used mode-switch frequency as a lower-confidence proxy."
                mission_evidence_values["mode_switch_rate_per_hr"] = round(switch_rate_hr, 3)
            conf_ms, conf_breakdown_ms = self._confidence_details(
                int(df.height),
                1,
                1,
                quality=0.92 if nav_div_ratio is not None else 0.68,
                mode_factor=1.0,
                reference_class=ref_class,
                fallback_penalty=0.22 if mission_fallback_used else 0.0,
            )
            items.append(self._item(
                "mission_stability", "Mission Stability", mission_score, mission_status, mission_value, cfg_ms["unit"],
                mission_reason, cfg_ms["suggestion"],
                required_topics=["vehicle_status"],
                found_topics=[st],
                sample_count=int(df.height),
                used_signals=mission_used_signals,
                confidence=conf_ms,
                confidence_breakdown=conf_breakdown_ms,
                evidence="nav_state_user_intention divergence and mode-switch pattern used" if nav_div_ratio is not None else "mode-switch pattern used as mission-stability fallback",
                reference_class=ref_class,
                signal_semantics="px4_explicit",
                threshold_source=mission_threshold_source,
                threshold_rationale=mission_threshold_rationale,
                reference_docs=cfg_ms.get("reference_docs", []),
                mode_gate="all_logged_samples",
                mode_gate_reason="Mission stability is evaluated over the full logged flight window.",
                normalization_basis=mission_normalization,
                fallback_used=mission_fallback_used,
                fallback_source=mission_fallback_source,
                fallback_reason=mission_fallback_reason,
                evidence_values=mission_evidence_values,
                evaluated_modes=self._nav_state_labels(self._to_float(df[nc].to_numpy())[:12] if nc else []),
            ))
        # Controller saturation.  Only actuator_motors has a documented physical
        # normalized bound.  Inferring a limit from PWM percentiles makes ordinary
        # dwell/clustering look like saturation, so raw actuator_outputs is not
        # scored without explicit output limits.
        cfg = self._metric_cfg("controller_saturation")
        motor_topic = self._find_topic(dataset, "actuator_motors")
        armed_topic = self._find_topic(dataset, "actuator_armed")
        landed_topic = self._find_topic(dataset, "vehicle_land_detected")
        pwm_topic = self._find_topic(dataset, "actuator_outputs")
        required_sat_topics = [
            "actuator_motors.control[]",
            "actuator_armed.armed",
            "vehicle_land_detected.landed",
        ]
        found_sat_topics = [x for x in [motor_topic, armed_topic, landed_topic, pwm_topic] if x]
        if not motor_topic or not armed_topic or not landed_topic:
            missing_reason = (
                "Controller saturation requires normalized actuator_motors plus armed and in-air evidence. "
                "PWM observed percentiles are not physical limits and are intentionally not scored."
            )
            items.append(self._item(
                "controller_saturation", "Controller Saturation", None, "unavailable", None, cfg["unit"],
                missing_reason, cfg["suggestion"], required_topics=required_sat_topics,
                found_topics=found_sat_topics, sample_count=0,
                reference_class="px4_explicit", signal_semantics="px4_explicit",
                threshold_source="PX4 actuator_motors normalized upper bound",
                threshold_rationale="Score only the documented [0,1] motor command ceiling while armed and airborne.",
                mode_gate="armed_and_airborne",
                mode_gate_reason="Disarmed and landed samples cannot demonstrate in-flight control-authority saturation.",
                normalization_basis="time-weighted fraction at control >= 0.98",
                not_evaluated_reason=missing_reason,
            ))
        else:
            try:
                motor_df = self._with_timestamp_sec(dataset.topics[motor_topic].dataframe)
                armed_df = self._with_timestamp_sec(dataset.topics[armed_topic].dataframe)
                landed_df = self._with_timestamp_sec(dataset.topics[landed_topic].dataframe)
                armed_sig = self._find_signal(armed_df, ["armed"])
                landed_sig = self._find_signal(landed_df, ["landed"])
                chans = [c for c in motor_df.columns if re.fullmatch(r"control\[\d+\]", str(c))]
                if not armed_sig or not landed_sig or not chans:
                    raise ValueError("missing normalized motor channels or flight-state signals")
                active = motor_df.select(
                    [pl.col("timestamp_sec").cast(pl.Float64)]
                    + [pl.col(c).cast(pl.Float64) for c in chans]
                ).sort("timestamp_sec")
                active = active.join_asof(
                    armed_df.select([
                        pl.col("timestamp_sec").cast(pl.Float64),
                        pl.col(armed_sig).cast(pl.Float64).alias("armed_gate"),
                    ]).drop_nulls().sort("timestamp_sec"),
                    on="timestamp_sec", strategy="nearest",
                ).join_asof(
                    landed_df.select([
                        pl.col("timestamp_sec").cast(pl.Float64),
                        pl.col(landed_sig).cast(pl.Float64).alias("landed_gate"),
                    ]).drop_nulls().sort("timestamp_sec"),
                    on="timestamp_sec", strategy="nearest",
                ).filter((pl.col("armed_gate") > 0.5) & (pl.col("landed_gate") <= 0.5))
                ts = np.asarray(active["timestamp_sec"].to_numpy(), dtype=np.float64)
                row_weights = self._time_weights_from_timestamps(ts)
                sats = []
                for cch in chans:
                    values = np.asarray(active[cch].to_numpy(), dtype=np.float64)
                    valid = np.isfinite(values) & np.isfinite(row_weights) & (row_weights > 0.0)
                    if int(np.sum(valid)) < 30:
                        continue
                    duration = float(np.sum(row_weights[valid]))
                    if duration <= 1e-9:
                        continue
                    upper_duration = float(np.sum(row_weights[valid & (values >= 0.98)]))
                    sats.append((cch, 100.0 * upper_duration / duration, int(np.sum(valid))))
                if not sats:
                    raise ValueError("insufficient armed-airborne normalized motor samples")
                worst, value, used_n = max(sats, key=lambda x: x[1])
                score, status = self._score_low(value, cfg["warn"], cfg["problem"])
                conf, conf_breakdown = self._confidence_details(
                    used_n, 3, 3, quality=1.0, mode_factor=1.0, reference_class="px4_explicit"
                )
                items.append(self._item(
                    "controller_saturation", "Controller Saturation", score, status, value, cfg["unit"],
                    f"Upper-limit duration={value:.2f}% (worst={worst}, physical limit=1.0)", cfg["suggestion"],
                    required_topics=required_sat_topics, found_topics=found_sat_topics,
                    sample_count=used_n,
                    used_signals=[
                        f"{motor_topic}.{worst}",
                        f"{armed_topic}.{armed_sig}",
                        f"{landed_topic}.{landed_sig}",
                    ],
                    confidence=conf, confidence_breakdown=conf_breakdown,
                    reference_class="px4_explicit", signal_semantics="px4_explicit",
                    threshold_source="PX4 actuator_motors normalized upper bound",
                    threshold_rationale="A motor command at or above 0.98 is within 2% of the documented 1.0 ceiling.",
                    mode_gate="armed_and_airborne",
                    mode_gate_reason="Only armed samples with vehicle_land_detected.landed=false are scored.",
                    normalization_basis="time-weighted fraction at control >= 0.98",
                    evidence_values={
                        "upper_limit": 1.0,
                        "upper_saturation_threshold": 0.98,
                        "upper_saturation_duration_pct": round(float(value), 3),
                        "samples_used": int(used_n),
                    },
                ))
            except Exception as exc:
                reason = (
                    "Normalized actuator motor saturation could not be evaluated with armed/in-air gating: "
                    f"{exc}. PWM observed percentiles were not used."
                )
                items.append(self._item(
                    "controller_saturation", "Controller Saturation", None, "unavailable", None, cfg["unit"],
                    reason, cfg["suggestion"], required_topics=required_sat_topics,
                    found_topics=found_sat_topics, sample_count=0,
                    reference_class="px4_explicit", signal_semantics="px4_explicit",
                    threshold_source="PX4 actuator_motors normalized upper bound",
                    mode_gate="armed_and_airborne",
                    normalization_basis="time-weighted fraction at control >= 0.98",
                    not_evaluated_reason=reason,
                ))
        return items
    def _eval_airframe(self, dataset, airframe: str, parameters=None) -> List[dict]:
        out = []
        tecs_parameter_context = self._build_tecs_parameter_context(parameters)
        hover_cfg = self._metric_cfg("hover_altitude_hold_error", airframe=airframe)
        hover_seg = self.cfg.get("segments", {}).get("hover", {})

        if airframe in ("MULTICOPTER", "VTOL", "HELICOPTER"):
            st = self._find_topic(dataset, "vehicle_local_position_setpoint")
            at = self._find_topic(dataset, "vehicle_local_position")
            if st and at:
                sdf = self._with_timestamp_sec(dataset.topics[st].dataframe)
                adf = self._with_timestamp_sec(dataset.topics[at].dataframe)
                sp = self._find_signal(sdf, ["z"])
                ac = self._find_signal(adf, ["z"])
                vx = self._find_signal(adf, ["vx", "vel_x"])
                vy = self._find_signal(adf, ["vy", "vel_y"])
                vz = self._find_signal(adf, ["vz", "vel_z"])
                if all([sp, ac, vx, vy, vz]):
                    m = self._join_ts(adf, ac, sdf, sp)
                    if m is not None:
                        aux = adf.select([
                            pl.col("timestamp_sec").cast(pl.Float64),
                            pl.col(vx).cast(pl.Float64).alias("vx"),
                            pl.col(vy).cast(pl.Float64).alias("vy"),
                            pl.col(vz).cast(pl.Float64).alias("vz"),
                        ]).drop_nulls().sort("timestamp_sec")
                        if aux.height > 0:
                            m2 = m.join_asof(aux, on="timestamp_sec", strategy="nearest").drop_nulls(subset=["actual", "setpoint", "vx", "vy", "vz"])
                            if m2.height > 0:
                                hspd = np.sqrt(m2["vx"].to_numpy() ** 2 + m2["vy"].to_numpy() ** 2)
                                mask = (hspd <= float(hover_seg.get("hspd_max", 2.0))) & (np.abs(m2["vz"].to_numpy()) <= float(hover_seg.get("vz_max", 0.5)))
                                need = int(hover_seg.get("min_samples", 80))
                                err = np.abs(np.asarray(m2["actual"].to_numpy(), dtype=np.float64) - np.asarray(m2["setpoint"].to_numpy(), dtype=np.float64))
                                valid_mask = mask & np.isfinite(err)
                                applicable_modes, excluded_modes = self._metric_mode_policy("hover_altitude_hold_error", airframe=airframe)
                                mode_ctx = self._build_mode_filtered_context(
                                    dataset,
                                    m2.select([pl.col("timestamp_sec").cast(pl.Float64)]),
                                    "hover_altitude_hold_error",
                                    valid_mask=valid_mask,
                                    metric_values=err,
                                    airframe=airframe,
                                    applicable_modes=applicable_modes,
                                    excluded_modes=excluded_modes,
                                    warn=hover_cfg["warn"],
                                    problem=hover_cfg["problem"],
                                    unit=hover_cfg["unit"],
                                )
                                found_topics = [x for x in [st, at, mode_ctx.get("nav_state_topic")] if x]
                                used_signals = [f"{st}.{sp}", f"{at}.{ac}", f"{at}.{vx}", f"{at}.{vy}", f"{at}.{vz}"]
                                if mode_ctx.get("nav_state_topic") and mode_ctx.get("nav_state_signal"):
                                    used_signals.append(f"{mode_ctx['nav_state_topic']}.{mode_ctx['nav_state_signal']}")
                                if int(np.sum(mode_ctx.get("applicable_mask", np.asarray([], dtype=bool)) & ~mask)) > 0:
                                    mode_ctx["excluded_reasons"] = list(mode_ctx.get("excluded_reasons", [])) + ["적용 가능한 비행모드 안에서도 hover 조건(수평속도/수직속도 기준)을 만족하지 않은 구간은 제외했습니다."]
                                if mode_ctx.get("status_hint"):
                                    out.append(self._item(
                                        "hover_altitude_hold_error", "Hover Altitude Hold Error", None, mode_ctx["status_hint"], None, hover_cfg["unit"],
                                        mode_ctx.get("not_evaluated_reason") or "평가 가능한 hover 구간이 없습니다.", hover_cfg["suggestion"],
                                        scope="airframe", segment="hover",
                                        required_topics=["vehicle_local_position_setpoint.z", "vehicle_local_position.z", "vehicle_local_position.vx/vy/vz", "vehicle_status.nav_state"],
                                        found_topics=found_topics, sample_count=mode_ctx.get("sample_count", 0),
                                        used_signals=used_signals,
                                        mode_gate="nav_state_applicable_modes",
                                        mode_gate_reason="vehicle_status.nav_state 기준 자동 비행모드 중 hover 조건을 만족한 구간만 점수화했습니다.",
                                        evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                                        total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                                        evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                                        excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                                        applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                                        applicable_modes=mode_ctx.get("applicable_modes"),
                                        excluded_modes=mode_ctx.get("excluded_modes"),
                                        excluded_reasons=mode_ctx.get("excluded_reasons"),
                                        evaluated_modes=mode_ctx.get("evaluated_modes"),
                                        mode_results=mode_ctx.get("mode_results"),
                                        not_evaluated_reason=mode_ctx.get("not_evaluated_reason"),
                                    ))
                                elif int(mode_ctx.get("sample_count", 0)) < need:
                                    out.append(self._item(
                                        "hover_altitude_hold_error", "Hover Altitude Hold Error", None, "unavailable", None, hover_cfg["unit"],
                                        f"Hover 조건을 만족한 유효 샘플이 부족합니다. need={need}, used={int(mode_ctx.get('sample_count', 0))}.", hover_cfg["suggestion"],
                                        scope="airframe", segment="hover",
                                        required_topics=["vehicle_local_position_setpoint.z", "vehicle_local_position.z", "vehicle_local_position.vx/vy/vz", "vehicle_status.nav_state"],
                                        found_topics=found_topics, sample_count=mode_ctx.get("sample_count", 0),
                                        used_signals=used_signals,
                                        mode_gate="nav_state_applicable_modes",
                                        mode_gate_reason="vehicle_status.nav_state 기준 자동 비행모드 중 hover 조건을 만족한 구간만 점수화했습니다.",
                                        evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                                        total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                                        evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                                        excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                                        applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                                        applicable_modes=mode_ctx.get("applicable_modes"),
                                        excluded_modes=mode_ctx.get("excluded_modes"),
                                        excluded_reasons=mode_ctx.get("excluded_reasons"),
                                        evaluated_modes=mode_ctx.get("evaluated_modes"),
                                        mode_results=mode_ctx.get("mode_results"),
                                        not_evaluated_reason="적용 가능한 자동 비행모드 안에 hover 조건을 만족한 유효 샘플이 충분하지 않습니다.",
                                    ))
                                else:
                                    eval_mask = mode_ctx.get("evaluated_mask", np.asarray([], dtype=bool))
                                    weights = mode_ctx.get("weights", np.asarray([], dtype=np.float64))
                                    v = self._weighted_mean(err[eval_mask], weights[eval_mask]) if int(np.sum(eval_mask)) > 0 else None
                                    if v is not None:
                                        s, stt = self._score_low(v, hover_cfg["warn"], hover_cfg["problem"])
                                        conf, conf_breakdown = self._confidence_details(
                                            int(mode_ctx.get("sample_count", 0)),
                                            4,
                                            len(found_topics),
                                            quality=min(1.0, float(mode_ctx.get("sample_count", 0)) / float(max(need, 1))),
                                            mode_factor=max(0.35, float(mode_ctx.get("mode_factor", 0.0))),
                                            reference_class=hover_cfg.get("reference_class", "px4_explicit"),
                                            fallback_penalty=0.0,
                                        )
                                        out.append(self._item(
                                            "hover_altitude_hold_error", "Hover Altitude Hold Error", s, stt, v, hover_cfg["unit"],
                                            f"Hover samples={int(mode_ctx.get('sample_count', 0))}, MAE={v:.2f} m", hover_cfg["suggestion"],
                                            scope="airframe", segment="hover",
                                            required_topics=["vehicle_local_position_setpoint.z", "vehicle_local_position.z", "vehicle_local_position.vx/vy/vz", "vehicle_status.nav_state"],
                                            found_topics=found_topics, sample_count=int(mode_ctx.get("sample_count", 0)),
                                            confidence=conf,
                                            confidence_breakdown=conf_breakdown,
                                            used_signals=used_signals,
                                            mode_gate="nav_state_applicable_modes",
                                            mode_gate_reason="vehicle_status.nav_state 기준 자동 비행모드 중 hover 조건을 만족한 구간만 점수화했습니다.",
                                            evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                                            total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                                            evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                                            excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                                            applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                                            applicable_modes=mode_ctx.get("applicable_modes"),
                                            excluded_modes=mode_ctx.get("excluded_modes"),
                                            excluded_reasons=mode_ctx.get("excluded_reasons"),
                                            evaluated_modes=mode_ctx.get("evaluated_modes"),
                                            mode_results=mode_ctx.get("mode_results"),
                                            evidence_values={"time_weighted_mae": round(float(v), 3)},
                                        ))

        if airframe in ("FIXED_WING", "VTOL"):
            cfg = self._metric_cfg("tecs_altitude_tracking_error", airframe=airframe)
            st = self._find_topic(dataset, "vehicle_local_position_setpoint")
            at = self._find_topic(dataset, "vehicle_local_position")
            if st and at:
                sdf = dataset.topics[st].dataframe
                adf = dataset.topics[at].dataframe
                sp = self._find_signal(sdf, ["z"])
                ac = self._find_signal(adf, ["z"])
                m = self._join_ts(adf, ac, sdf, sp) if (sp and ac) else None
                if m is not None:
                    e = np.abs(np.asarray(m["actual"].to_numpy(), dtype=np.float64) - np.asarray(m["setpoint"].to_numpy(), dtype=np.float64))
                    applicable_modes, excluded_modes = self._metric_mode_policy("tecs_altitude_tracking_error", airframe=airframe)
                    mode_ctx = self._build_mode_filtered_context(
                        dataset,
                        m.select([pl.col("timestamp_sec").cast(pl.Float64)]),
                        "tecs_altitude_tracking_error",
                        valid_mask=np.isfinite(e),
                        metric_values=e,
                        airframe=airframe,
                        applicable_modes=applicable_modes,
                        excluded_modes=excluded_modes,
                        warn=cfg["warn"],
                        problem=cfg["problem"],
                        unit=cfg["unit"],
                    )
                    found_topics = [x for x in [st, at, mode_ctx.get("nav_state_topic")] if x]
                    used_signals = [f"{st}.{sp}", f"{at}.{ac}"]
                    if mode_ctx.get("nav_state_topic") and mode_ctx.get("nav_state_signal"):
                        used_signals.append(f"{mode_ctx['nav_state_topic']}.{mode_ctx['nav_state_signal']}")
                    if mode_ctx.get("status_hint"):
                        out.append(self._item(
                            "tecs_altitude_tracking_error", "TECS Altitude Tracking Error", None, mode_ctx["status_hint"], None, cfg["unit"],
                            mode_ctx.get("not_evaluated_reason") or "평가 가능한 TECS 고도 추종 구간이 없습니다.", cfg["suggestion"],
                            scope="airframe", segment="cruise", required_topics=["vehicle_local_position_setpoint.z", "vehicle_local_position.z", "vehicle_status.nav_state"],
                            found_topics=found_topics, sample_count=mode_ctx.get("sample_count", 0),
                            used_signals=used_signals, parameter_context=tecs_parameter_context,
                            mode_gate="nav_state_applicable_modes",
                            mode_gate_reason="vehicle_status.nav_state 기준 TECS 고도 추종이 의미 있는 비행모드만 점수화했습니다.",
                            evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                            total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                            evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                            excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                            applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                            applicable_modes=mode_ctx.get("applicable_modes"),
                            excluded_modes=mode_ctx.get("excluded_modes"),
                            excluded_reasons=mode_ctx.get("excluded_reasons"),
                            evaluated_modes=mode_ctx.get("evaluated_modes"),
                            mode_results=mode_ctx.get("mode_results"),
                            not_evaluated_reason=mode_ctx.get("not_evaluated_reason"),
                        ))
                    else:
                        eval_mask = mode_ctx.get("evaluated_mask", np.asarray([], dtype=bool))
                        weights = mode_ctx.get("weights", np.asarray([], dtype=np.float64))
                        v = self._weighted_mean(e[eval_mask], weights[eval_mask]) if int(np.sum(eval_mask)) > 0 else None
                        if v is not None:
                            s, stt = self._score_low(v, cfg["warn"], cfg["problem"])
                            conf, conf_breakdown = self._confidence_details(
                                int(mode_ctx.get("sample_count", 0)),
                                3,
                                len(found_topics),
                                quality=1.0,
                                mode_factor=max(0.35, float(mode_ctx.get("mode_factor", 0.0))),
                                reference_class=cfg.get("reference_class", "px4_explicit"),
                                fallback_penalty=0.0,
                            )
                            out.append(self._item(
                                "tecs_altitude_tracking_error", "TECS Altitude Tracking Error", s, stt, v, cfg["unit"], f"MAE={v:.2f} m", cfg["suggestion"],
                                scope="airframe", segment="cruise", required_topics=["vehicle_local_position_setpoint.z", "vehicle_local_position.z", "vehicle_status.nav_state"],
                                found_topics=found_topics, sample_count=int(mode_ctx.get("sample_count", 0)),
                                used_signals=used_signals, parameter_context=tecs_parameter_context,
                                confidence=conf, confidence_breakdown=conf_breakdown,
                                mode_gate="nav_state_applicable_modes",
                                mode_gate_reason="vehicle_status.nav_state 기준 TECS 고도 추종이 의미 있는 비행모드만 점수화했습니다.",
                                evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                                total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                                evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                                excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                                applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                                applicable_modes=mode_ctx.get("applicable_modes"),
                                excluded_modes=mode_ctx.get("excluded_modes"),
                                excluded_reasons=mode_ctx.get("excluded_reasons"),
                                evaluated_modes=mode_ctx.get("evaluated_modes"),
                                mode_results=mode_ctx.get("mode_results"),
                                evidence_values={"time_weighted_mae": round(float(v), 3)},
                            ))

            cfg = self._metric_cfg("tecs_airspeed_tracking_error", airframe=airframe)
            tt = self._find_topic(dataset, "tecs_status")
            if tt:
                df = dataset.topics[tt].dataframe
                sp = self._find_signal(df, ["true_airspeed_sp"])
                ac = self._find_signal(df, ["true_airspeed_filtered", "true_airspeed"])
                m = self._join_ts(df, ac, df, sp) if (sp and ac) else None
                if m is not None:
                    e = np.abs(np.asarray(m["actual"].to_numpy(), dtype=np.float64) - np.asarray(m["setpoint"].to_numpy(), dtype=np.float64))
                    applicable_modes, excluded_modes = self._metric_mode_policy("tecs_airspeed_tracking_error", airframe=airframe)
                    mode_ctx = self._build_mode_filtered_context(
                        dataset,
                        m.select([pl.col("timestamp_sec").cast(pl.Float64)]),
                        "tecs_airspeed_tracking_error",
                        valid_mask=np.isfinite(e),
                        metric_values=e,
                        airframe=airframe,
                        applicable_modes=applicable_modes,
                        excluded_modes=excluded_modes,
                        warn=cfg["warn"],
                        problem=cfg["problem"],
                        unit=cfg["unit"],
                    )
                    found_topics = [x for x in [tt, mode_ctx.get("nav_state_topic")] if x]
                    used_signals = [f"{tt}.{sp}", f"{tt}.{ac}"]
                    if mode_ctx.get("nav_state_topic") and mode_ctx.get("nav_state_signal"):
                        used_signals.append(f"{mode_ctx['nav_state_topic']}.{mode_ctx['nav_state_signal']}")
                    if mode_ctx.get("status_hint"):
                        out.append(self._item(
                            "tecs_airspeed_tracking_error", "TECS Airspeed Tracking Error", None, mode_ctx["status_hint"], None, cfg["unit"],
                            mode_ctx.get("not_evaluated_reason") or "평가 가능한 TECS 속도 추종 구간이 없습니다.", cfg["suggestion"],
                            scope="airframe", segment="cruise", required_topics=["tecs_status.true_airspeed_sp", "tecs_status.true_airspeed_filtered", "vehicle_status.nav_state"],
                            found_topics=found_topics, sample_count=mode_ctx.get("sample_count", 0),
                            used_signals=used_signals, parameter_context=tecs_parameter_context,
                            mode_gate="nav_state_applicable_modes",
                            mode_gate_reason="vehicle_status.nav_state 기준 TECS 속도 추종이 의미 있는 비행모드만 점수화했습니다.",
                            evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                            total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                            evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                            excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                            applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                            applicable_modes=mode_ctx.get("applicable_modes"),
                            excluded_modes=mode_ctx.get("excluded_modes"),
                            excluded_reasons=mode_ctx.get("excluded_reasons"),
                            evaluated_modes=mode_ctx.get("evaluated_modes"),
                            mode_results=mode_ctx.get("mode_results"),
                            not_evaluated_reason=mode_ctx.get("not_evaluated_reason"),
                        ))
                    else:
                        eval_mask = mode_ctx.get("evaluated_mask", np.asarray([], dtype=bool))
                        weights = mode_ctx.get("weights", np.asarray([], dtype=np.float64))
                        v = self._weighted_mean(e[eval_mask], weights[eval_mask]) if int(np.sum(eval_mask)) > 0 else None
                        if v is not None:
                            s, stt = self._score_low(v, cfg["warn"], cfg["problem"])
                            conf, conf_breakdown = self._confidence_details(
                                int(mode_ctx.get("sample_count", 0)),
                                2,
                                len(found_topics),
                                quality=1.0,
                                mode_factor=max(0.35, float(mode_ctx.get("mode_factor", 0.0))),
                                reference_class=cfg.get("reference_class", "px4_explicit"),
                                fallback_penalty=0.0,
                            )
                            out.append(self._item(
                                "tecs_airspeed_tracking_error", "TECS Airspeed Tracking Error", s, stt, v, cfg["unit"], f"MAE={v:.2f} m/s", cfg["suggestion"],
                                scope="airframe", segment="cruise", required_topics=["tecs_status.true_airspeed_sp", "tecs_status.true_airspeed_filtered", "vehicle_status.nav_state"],
                                found_topics=found_topics, sample_count=int(mode_ctx.get("sample_count", 0)),
                                used_signals=used_signals, parameter_context=tecs_parameter_context,
                                confidence=conf, confidence_breakdown=conf_breakdown,
                                mode_gate="nav_state_applicable_modes",
                                mode_gate_reason="vehicle_status.nav_state 기준 TECS 속도 추종이 의미 있는 비행모드만 점수화했습니다.",
                                evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                                total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                                evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                                excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                                applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                                applicable_modes=mode_ctx.get("applicable_modes"),
                                excluded_modes=mode_ctx.get("excluded_modes"),
                                excluded_reasons=mode_ctx.get("excluded_reasons"),
                                evaluated_modes=mode_ctx.get("evaluated_modes"),
                                mode_results=mode_ctx.get("mode_results"),
                                evidence_values={"time_weighted_mae": round(float(v), 3)},
                            ))

            # Phase 2: TECS total energy balance error (height-equivalent)
            cfg = self._metric_cfg("tecs_energy_balance_error", airframe=airframe)
            z_actual, zt, _ = self._extract_series(dataset, "vehicle_local_position", ["z"], "z_actual")
            z_sp, zst, _ = self._extract_series(dataset, "vehicle_local_position_setpoint", ["z"], "z_sp")
            v_actual, vt, _ = self._extract_series(
                dataset,
                ["tecs_status", "airspeed", "airspeed_validated"],
                ["true_airspeed_filtered", "true_airspeed_m_s", "true_airspeed", "indicated_airspeed_m_s"],
                "v_actual",
            )
            v_sp, vst, _ = self._extract_series(dataset, "tecs_status", ["true_airspeed_sp"], "v_sp")
            if z_actual is not None and z_sp is not None and v_actual is not None and v_sp is not None:
                try:
                    em = z_actual.join_asof(z_sp, on="timestamp_sec", strategy="nearest")
                    em = em.join_asof(v_actual, on="timestamp_sec", strategy="nearest")
                    em = em.join_asof(v_sp, on="timestamp_sec", strategy="nearest")
                    em = em.drop_nulls(subset=["z_actual", "z_sp", "v_actual", "v_sp"])
                    if em.height > 0:
                        g = 9.80665
                        h_actual = (-em["z_actual"].to_numpy()) + (em["v_actual"].to_numpy() ** 2) / (2.0 * g)
                        h_sp = (-em["z_sp"].to_numpy()) + (em["v_sp"].to_numpy() ** 2) / (2.0 * g)
                        e = np.abs(h_actual - h_sp)
                        applicable_modes, excluded_modes = self._metric_mode_policy("tecs_energy_balance_error", airframe=airframe)
                        mode_ctx = self._build_mode_filtered_context(
                            dataset,
                            em.select([pl.col("timestamp_sec").cast(pl.Float64)]),
                            "tecs_energy_balance_error",
                            valid_mask=np.isfinite(e),
                            metric_values=e,
                            airframe=airframe,
                            applicable_modes=applicable_modes,
                            excluded_modes=excluded_modes,
                            warn=cfg["warn"],
                            problem=cfg["problem"],
                            unit=cfg["unit"],
                        )
                        found_topics = [x for x in [zt, zst, vt, vst, mode_ctx.get("nav_state_topic")] if x]
                        used_signals = [x for x in [f"{zt}.z" if zt else "", f"{zst}.z" if zst else "", f"{vt}.v_actual" if vt else "", f"{vst}.true_airspeed_sp" if vst else ""] if x]
                        if mode_ctx.get("nav_state_topic") and mode_ctx.get("nav_state_signal"):
                            used_signals.append(f"{mode_ctx['nav_state_topic']}.{mode_ctx['nav_state_signal']}")
                        if mode_ctx.get("status_hint"):
                            out.append(self._item(
                                "tecs_energy_balance_error", "TECS Energy Balance Error", None, mode_ctx["status_hint"], None, cfg["unit"],
                                mode_ctx.get("not_evaluated_reason") or "평가 가능한 TECS 에너지 구간이 없습니다.", cfg["suggestion"],
                                scope="airframe", segment="cruise",
                                required_topics=["vehicle_local_position.z", "vehicle_local_position_setpoint.z", "tecs_status.true_airspeed_sp", "tecs/airspeed actual", "vehicle_status.nav_state"],
                                found_topics=found_topics, sample_count=mode_ctx.get("sample_count", 0),
                                used_signals=used_signals, parameter_context=tecs_parameter_context,
                                mode_gate="nav_state_applicable_modes",
                                mode_gate_reason="vehicle_status.nav_state 기준 TECS 에너지 평가가 의미 있는 비행모드만 점수화했습니다.",
                                evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                                total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                                evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                                excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                                applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                                applicable_modes=mode_ctx.get("applicable_modes"),
                                excluded_modes=mode_ctx.get("excluded_modes"),
                                excluded_reasons=mode_ctx.get("excluded_reasons"),
                                evaluated_modes=mode_ctx.get("evaluated_modes"),
                                mode_results=mode_ctx.get("mode_results"),
                                not_evaluated_reason=mode_ctx.get("not_evaluated_reason"),
                            ))
                        else:
                            eval_mask = mode_ctx.get("evaluated_mask", np.asarray([], dtype=bool))
                            weights = mode_ctx.get("weights", np.asarray([], dtype=np.float64))
                            v = self._weighted_mean(e[eval_mask], weights[eval_mask]) if int(np.sum(eval_mask)) > 0 else None
                            if v is not None:
                                s, stt = self._score_low(v, cfg["warn"], cfg["problem"])
                                conf, conf_breakdown = self._confidence_details(
                                    int(mode_ctx.get("sample_count", 0)),
                                    5,
                                    len(found_topics),
                                    quality=1.0,
                                    mode_factor=max(0.35, float(mode_ctx.get("mode_factor", 0.0))),
                                    reference_class=cfg.get("reference_class", "px4_explicit"),
                                    fallback_penalty=0.0,
                                )
                                out.append(self._item(
                                    "tecs_energy_balance_error", "TECS Energy Balance Error", s, stt, v, cfg["unit"], f"Energy-height MAE={v:.2f} m", cfg["suggestion"],
                                    scope="airframe", segment="cruise",
                                    required_topics=["vehicle_local_position.z", "vehicle_local_position_setpoint.z", "tecs_status.true_airspeed_sp", "tecs/airspeed actual", "vehicle_status.nav_state"],
                                    found_topics=found_topics, sample_count=int(mode_ctx.get("sample_count", 0)),
                                    used_signals=used_signals, parameter_context=tecs_parameter_context,
                                    confidence=conf, confidence_breakdown=conf_breakdown,
                                    mode_gate="nav_state_applicable_modes",
                                    mode_gate_reason="vehicle_status.nav_state 기준 TECS 에너지 평가가 의미 있는 비행모드만 점수화했습니다.",
                                    evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                                    total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                                    evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                                    excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                                    applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                                    applicable_modes=mode_ctx.get("applicable_modes"),
                                    excluded_modes=mode_ctx.get("excluded_modes"),
                                    excluded_reasons=mode_ctx.get("excluded_reasons"),
                                    evaluated_modes=mode_ctx.get("evaluated_modes"),
                                    mode_results=mode_ctx.get("mode_results"),
                                    evidence_values={"time_weighted_mae": round(float(v), 3)},
                                ))
                except Exception:
                    pass
            elif airframe == "FIXED_WING":
                out.append(self._item(
                    "tecs_energy_balance_error", "TECS Energy Balance Error", None, "unavailable", None, cfg["unit"], "Missing altitude or airspeed pair for energy metric.", cfg["suggestion"],
                    scope="airframe", segment="cruise",
                    required_topics=["vehicle_local_position.z", "vehicle_local_position_setpoint.z", "tecs_status.true_airspeed_sp", "tecs/airspeed actual"],
                    found_topics=[x for x in [zt, zst, vt, vst] if x], sample_count=0,
                    parameter_context=tecs_parameter_context,
                ))

            # Phase 2: Airspeed consistency.  GPS speed is converted to an
            # air-relative reference with the logged wind vector before comparison.
            cfg = self._metric_cfg("airspeed_consistency", airframe=airframe)
            asp, aspt, asp_sig = self._extract_series(
                dataset,
                ["airspeed_validated", "tecs_status", "airspeed"],
                ["true_airspeed_m_s", "true_airspeed_filtered", "true_airspeed", "indicated_airspeed_m_s"],
                "airspeed",
            )
            gs_series, _gps_speed_topic, _wind_topic, gs_used_signals, air_ref_found_topics = self._extract_wind_corrected_speed_series(dataset)
            lpt = self._find_topic(dataset, "vehicle_local_position")
            airspeed_consistency_added = False
            if asp is not None and gs_series is not None:
                try:
                    mm = asp.join_asof(gs_series, on="timestamp_sec", strategy="nearest").drop_nulls(subset=["airspeed", "air_relative_speed"])
                    mm = mm.filter(
                        (pl.col("timestamp_sec") - pl.col("air_reference_timestamp_sec")).abs() <= pl.lit(1.0)
                    )
                    if mm.height > 0:
                        d = np.asarray(mm["airspeed"].to_numpy(), dtype=np.float64) - np.asarray(mm["air_relative_speed"].to_numpy(), dtype=np.float64)
                        valid_mask = np.isfinite(d)
                        applicable_modes, excluded_modes = self._metric_mode_policy(
                            "airspeed_consistency",
                            airframe=airframe,
                            default_applicable=self.FORWARD_FLIGHT_APPLICABLE_MODE_LABELS,
                            default_excluded=self.SETPOINT_EXCLUDED_MODE_LABELS,
                        )
                        mode_ctx = self._build_mode_filtered_context(
                            dataset,
                            mm,
                            "airspeed_consistency",
                            valid_mask=valid_mask,
                            metric_values=d,
                            airframe=airframe,
                            applicable_modes=applicable_modes,
                            excluded_modes=excluded_modes,
                            warn=cfg["warn"],
                            problem=cfg["problem"],
                            unit=cfg["unit"],
                        )
                        found_topics = list(dict.fromkeys([x for x in [aspt] + air_ref_found_topics + [mode_ctx.get("nav_state_topic")] if x]))
                        used_signals = [x for x in ([f"{aspt}.{asp_sig}" if aspt and asp_sig else ""] + gs_used_signals) if x]
                        if mode_ctx.get("nav_state_topic") and mode_ctx.get("nav_state_signal"):
                            used_signals.append(f"{mode_ctx['nav_state_topic']}.{mode_ctx['nav_state_signal']}")
                        if mode_ctx.get("status_hint"):
                            out.append(self._item(
                                "airspeed_consistency", "Airspeed Consistency", None, mode_ctx["status_hint"], None, cfg["unit"],
                                mode_ctx.get("not_evaluated_reason") or "평가 가능한 순항/전진비행 구간이 없습니다.", cfg["suggestion"],
                                scope="airframe", segment="cruise",
                                required_topics=["airspeed/validated TAS", "vehicle_gps_position velocity vector", "wind vector", "vehicle_status.nav_state"],
                                found_topics=found_topics, sample_count=mode_ctx.get("sample_count", 0),
                                used_signals=used_signals,
                                parameter_context=tecs_parameter_context,
                                mode_gate="nav_state_applicable_modes",
                                mode_gate_reason="vehicle_status.nav_state 기준 전진비행/순항에 해당하는 비행모드만 점수화했습니다.",
                                evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                                total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                                evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                                excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                                applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                                applicable_modes=mode_ctx.get("applicable_modes"),
                                excluded_modes=mode_ctx.get("excluded_modes"),
                                excluded_reasons=mode_ctx.get("excluded_reasons"),
                                evaluated_modes=mode_ctx.get("evaluated_modes"),
                                mode_results=mode_ctx.get("mode_results"),
                                not_evaluated_reason=mode_ctx.get("not_evaluated_reason"),
                            ))
                            airspeed_consistency_added = True
                        else:
                            eval_mask = mode_ctx.get("evaluated_mask", np.asarray([], dtype=bool))
                            weights = mode_ctx.get("weights", np.asarray([], dtype=np.float64))
                            d_use = d[eval_mask]
                            if int(np.sum(eval_mask)) > 0:
                                center = float(np.median(d_use[np.isfinite(d_use)]))
                                centered = d - center
                                use_vals = centered[eval_mask]
                                w_use = weights[eval_mask]
                                good = np.isfinite(use_vals) & np.isfinite(w_use) & (w_use > 0.0)
                                if int(np.sum(good)) > 0:
                                    v = float(np.sqrt(np.sum((use_vals[good] ** 2) * w_use[good]) / max(np.sum(w_use[good]), 1e-9)))
                                else:
                                    use_vals = use_vals[np.isfinite(use_vals)]
                                    v = float(np.sqrt(np.mean(use_vals * use_vals))) if use_vals.size > 0 else None
                            else:
                                v = None
                            if v is not None:
                                s, stt = self._score_low(v, cfg["warn"], cfg["problem"])
                                mode_labels = np.asarray([self._nav_state_label(x) for x in mode_ctx["joined_frame"]["nav_state"].to_numpy()], dtype=object)
                                mode_results = self._mode_results_for_rms_metric(
                                    mode_labels,
                                    eval_mask,
                                    centered,
                                    weights,
                                    warn=cfg["warn"],
                                    problem=cfg["problem"],
                                    unit=cfg["unit"],
                                )
                                try:
                                    rms_timestamps = np.asarray(mode_ctx["joined_frame"]["timestamp_sec"].to_numpy(), dtype=np.float64)
                                except Exception:
                                    rms_timestamps = None
                                self._inject_mode_time_intervals(mode_results, rms_timestamps, mode_labels, eval_mask)
                                conf, conf_breakdown = self._confidence_details(
                                    int(mode_ctx.get("sample_count", 0)),
                                    4,
                                    len(found_topics),
                                    quality=1.0,
                                    mode_factor=max(0.35, float(mode_ctx.get("mode_factor", 0.0))),
                                    reference_class=cfg.get("reference_class", "product_default_threshold"),
                                    fallback_penalty=0.0,
                                )
                                out.append(self._item(
                                    "airspeed_consistency", "Airspeed Consistency", s, stt, v, cfg["unit"], f"Wind-corrected residual RMS={v:.2f} m/s", cfg["suggestion"],
                                    scope="airframe", segment="cruise",
                                    required_topics=["airspeed/validated TAS", "vehicle_gps_position velocity vector", "wind vector", "vehicle_status.nav_state"],
                                    found_topics=found_topics, sample_count=int(mode_ctx.get("sample_count", 0)),
                                    used_signals=used_signals,
                                    parameter_context=tecs_parameter_context,
                                    confidence=conf,
                                    confidence_breakdown=conf_breakdown,
                                    mode_gate="nav_state_applicable_modes",
                                    mode_gate_reason="vehicle_status.nav_state 기준 전진비행/순항에 해당하는 비행모드만 점수화했습니다.",
                                    evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                                    total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                                    evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                                    excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                                    applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                                    applicable_modes=mode_ctx.get("applicable_modes"),
                                    excluded_modes=mode_ctx.get("excluded_modes"),
                                    excluded_reasons=mode_ctx.get("excluded_reasons"),
                                    evaluated_modes=mode_ctx.get("evaluated_modes"),
                                    mode_results=mode_results,
                                    reference_class="derived_from_px4",
                                    signal_semantics="px4_explicit",
                                    threshold_source=str(cfg.get("threshold_source", "TAS versus GPS-minus-wind residual RMS")),
                                    threshold_rationale="Ground velocity is converted to air-relative velocity before TAS comparison.",
                                    normalization_basis="time-weighted RMS after removing the median TAS-minus-air-reference bias",
                                    evidence_values={"time_weighted_rms": round(float(v), 3), "median_bias_removed_mps": round(float(center), 3)},
                                ))
                                airspeed_consistency_added = True
                except Exception:
                    pass

            if not airspeed_consistency_added:
                missing = []
                if asp is None:
                    missing.append("airspeed/TAS")
                if gs_series is None:
                    missing.append("GPS velocity plus wind vector")
                reason = (
                    f"Missing {', '.join(missing) if missing else 'sufficient wind-corrected comparison samples'}. "
                    "Raw TAS-minus-ground-speed is not scored because wind can dominate that difference."
                )
                out.append(self._item(
                    "airspeed_consistency", "Airspeed Consistency", None, "unavailable", None, cfg["unit"],
                    reason, cfg["suggestion"], scope="airframe", segment="cruise",
                    required_topics=["airspeed/validated TAS", "vehicle_gps_position velocity vector", "wind vector", "vehicle_status.nav_state"],
                    found_topics=list(dict.fromkeys([x for x in [aspt] + air_ref_found_topics if x])),
                    sample_count=0,
                    used_signals=[x for x in ([f"{aspt}.{asp_sig}" if aspt and asp_sig else ""] + gs_used_signals) if x],
                    reference_class="derived_from_px4", signal_semantics="px4_explicit",
                    threshold_source=str(cfg.get("threshold_source", "TAS versus GPS-minus-wind residual RMS")),
                    mode_gate="nav_state_applicable_modes_and_fixed_wing_phase",
                    mode_gate_reason="VTOL comparison additionally requires vehicle_type=fixed-wing and not in transition.",
                    normalization_basis="time-weighted RMS after removing median bias",
                    not_evaluated_reason=reason,
                ))

            # Airspeed scale appropriateness — primary metric:
            # |TAS_mean - |GPS velocity - wind|| in m/s.
            # AirspeedWind.tas_scale_validated/raw는 있으면 참고용 컨텍스트로만 추가.
            cfg = self._metric_cfg("airspeed_scale_appropriateness", airframe=airframe)
            cfg_scale, cfg_scale_name = self._parameter_float(parameters, ["ASPD_SCALE_1", "ASPD_SCALE_2", "ASPD_SCALE_3", "ASPD_SCALE"])
            scale_topic = self._find_topic(dataset, ["airspeed_wind", "airspeedwind", "airspeed_wind_estimate"])
            val_ref_supp = None
            raw_ref_supp = None
            sig_valid = None
            sig_raw = None
            if scale_topic:
                try:
                    sdf = self._with_timestamp_sec(dataset.topics[scale_topic].dataframe)
                    sig_valid = self._find_signal(sdf, ["tas_scale_validated", "tas_scale"])
                    sig_raw = self._find_signal(sdf, ["tas_scale_raw"])
                    if sig_valid:
                        v_arr = self._to_float(sdf[sig_valid].to_numpy())
                        if v_arr.size >= 10:
                            tail_n = max(10, min(int(v_arr.size * 0.2), 200))
                            val_ref_supp = float(np.median(v_arr[-tail_n:]))
                    if sig_raw:
                        r_arr = self._to_float(sdf[sig_raw].to_numpy())
                        if r_arr.size >= 10:
                            tail_n = max(10, min(int(r_arr.size * 0.2), 200))
                            raw_ref_supp = float(np.median(r_arr[-tail_n:]))
                except Exception:
                    val_ref_supp = None
                    raw_ref_supp = None

            scale_required_topics = ["airspeed/validated TAS", "vehicle_gps_position velocity vector", "wind vector"]
            scale_row_added = False
            vz = None
            try:
                if asp is not None and gs_series is not None:
                    mm = asp.join_asof(gs_series, on="timestamp_sec", strategy="nearest").drop_nulls(subset=["airspeed", "air_relative_speed"])
                    mm = mm.filter(
                        (pl.col("timestamp_sec") - pl.col("air_reference_timestamp_sec")).abs() <= pl.lit(1.0)
                    )
                    if lpt:
                        try:
                            ldf = self._with_timestamp_sec(dataset.topics[lpt].dataframe)
                            vz = self._find_signal(ldf, ["vz", "vel_d", "vel_z"])
                            if vz:
                                vz_join = ldf.select([
                                    pl.col("timestamp_sec").cast(pl.Float64),
                                    pl.col(vz).cast(pl.Float64).alias("vz_conditional"),
                                ]).drop_nulls().sort("timestamp_sec")
                                if vz_join.height > 0:
                                    mm = mm.join_asof(vz_join, on="timestamp_sec", strategy="nearest")
                        except Exception:
                            vz = None
                    att_topic = self._find_topic(dataset, "vehicle_attitude")
                    roll_sig = None
                    pitch_sig = None
                    if att_topic:
                        try:
                            adf = self._with_timestamp_sec(dataset.topics[att_topic].dataframe)
                            roll_sig = self._find_signal(adf, ["roll_euler", "roll"])
                            pitch_sig = self._find_signal(adf, ["pitch_euler", "pitch"])
                            if roll_sig and pitch_sig:
                                att_join = adf.select([
                                    pl.col("timestamp_sec").cast(pl.Float64),
                                    pl.col(roll_sig).cast(pl.Float64).alias("roll_conditional"),
                                    pl.col(pitch_sig).cast(pl.Float64).alias("pitch_conditional"),
                                ]).drop_nulls().sort("timestamp_sec")
                                if att_join.height > 0:
                                    mm = mm.join_asof(att_join, on="timestamp_sec", strategy="nearest")
                        except Exception:
                            roll_sig = None
                            pitch_sig = None
                    if mm.height > 0:
                        tas_arr = np.asarray(mm["airspeed"].to_numpy(), dtype=np.float64)
                        gps_arr = np.asarray(mm["air_relative_speed"].to_numpy(), dtype=np.float64)
                        pair_count = min(int(tas_arr.size), int(gps_arr.size))
                        if pair_count >= 10:
                            tas_arr = tas_arr[:pair_count]
                            gps_arr = gps_arr[:pair_count]
                            base_mask = np.isfinite(tas_arr) & np.isfinite(gps_arr) & (gps_arr > 3.0)
                            filter_notes = []
                            cruise_mask = base_mask.copy()
                            if "roll_conditional" in mm.columns and "pitch_conditional" in mm.columns:
                                roll_arr = np.asarray(mm["roll_conditional"].to_numpy()[:pair_count], dtype=np.float64)
                                pitch_arr = np.asarray(mm["pitch_conditional"].to_numpy()[:pair_count], dtype=np.float64)
                                cruise_mask &= np.isfinite(roll_arr) & np.isfinite(pitch_arr)
                                # MathEngine's explicit *_euler fields are degrees;
                                # raw roll/pitch fields, when present, are radians.
                                roll_limit = 20.0 if "euler" in str(roll_sig).lower() else math.radians(20.0)
                                pitch_limit = 12.0 if "euler" in str(pitch_sig).lower() else math.radians(12.0)
                                cruise_mask &= np.abs(roll_arr) <= roll_limit
                                cruise_mask &= np.abs(pitch_arr) <= pitch_limit
                                filter_notes.append("|roll|<=20deg")
                                filter_notes.append("|pitch|<=12deg")
                            if "vz_conditional" in mm.columns:
                                vz_arr = np.asarray(mm["vz_conditional"].to_numpy()[:pair_count], dtype=np.float64)
                                cruise_mask &= np.isfinite(vz_arr)
                                cruise_mask &= np.abs(vz_arr) <= 1.5
                                filter_notes.append("|vz|<=1.5m/s")
                            applicable_modes, excluded_modes = self._metric_mode_policy(
                                "airspeed_scale_appropriateness",
                                airframe=airframe,
                                default_applicable=self.FORWARD_FLIGHT_APPLICABLE_MODE_LABELS,
                                default_excluded=self.SETPOINT_EXCLUDED_MODE_LABELS,
                            )
                            cruise_mode_ctx = self._build_mode_filtered_context(
                                dataset,
                                mm.head(pair_count),
                                "airspeed_scale_appropriateness",
                                valid_mask=cruise_mask,
                                metric_values=(tas_arr - gps_arr),
                                airframe=airframe,
                                applicable_modes=applicable_modes,
                                excluded_modes=excluded_modes,
                                warn=cfg["warn"],
                                problem=cfg["problem"],
                                unit=cfg["unit"],
                            )
                            base_mode_ctx = self._build_mode_filtered_context(
                                dataset,
                                mm.head(pair_count),
                                "airspeed_scale_appropriateness",
                                valid_mask=base_mask,
                                metric_values=(tas_arr - gps_arr),
                                airframe=airframe,
                                applicable_modes=applicable_modes,
                                excluded_modes=excluded_modes,
                                warn=cfg["warn"],
                                problem=cfg["problem"],
                                unit=cfg["unit"],
                            )
                            used_conditional_cruise = int(cruise_mode_ctx.get("sample_count", 0)) >= 10
                            chosen_mode_ctx = cruise_mode_ctx if used_conditional_cruise else base_mode_ctx
                            eval_mask = chosen_mode_ctx.get("evaluated_mask", np.asarray([], dtype=bool))
                            used_count = int(np.sum(eval_mask))
                            if used_count >= 10:
                                tas_use = tas_arr[eval_mask]
                                gps_use = gps_arr[eval_mask]
                                tas_mean = float(np.mean(tas_use))
                                gps_mean = float(np.mean(gps_use))
                                gap_signed = tas_mean - gps_mean   # +면 TAS가 큼, -면 TAS가 작음
                                gap_abs = abs(gap_signed)
                                metric_value = gap_abs
                                score, status = self._score_low(metric_value, cfg["warn"], cfg["problem"])
                                implied_ratio = float(gps_mean / max(abs(tas_mean), 1e-6))
                                implied_pct = float((implied_ratio - 1.0) * 100.0)
                                step_limit_pct = 2.0 if status == "good" else (4.0 if status == "warning" else 6.0)
                                bounded_step_pct = float(max(-step_limit_pct, min(step_limit_pct, implied_pct)))
                                direction = "상향" if gap_signed < -0.05 else ("하향" if gap_signed > 0.05 else "유지")
                                warn_thr = float(cfg["warn"])
                                ground_ref_label = "wind-corrected air-relative speed"
                                scale_context = list(tecs_parameter_context)
                                if cfg_scale is not None:
                                    scale_context.insert(0, f"{cfg_scale_name}={cfg_scale:.3f} (설정된 airspeed scale)")
                                if aspt and asp_sig:
                                    scale_context.append(f"TAS source={aspt}.{asp_sig}")
                                if gs_used_signals:
                                    scale_context.append(f"Air-relative reference source={', '.join(gs_used_signals)}")
                                if val_ref_supp is not None:
                                    scale_context.append(f"{scale_topic}.{sig_valid}={val_ref_supp:.3f} (참고: validated scale 중앙값)")
                                if raw_ref_supp is not None:
                                    scale_context.append(f"{scale_topic}.{sig_raw}={raw_ref_supp:.3f} (참고: raw scale 중앙값)")
                                if used_conditional_cruise:
                                    scale_context.append(f"순항 판정 기준: {' / '.join(filter_notes) if filter_notes else '순항 평균'}")
                                else:
                                    scale_context.append("순항 판정 기준: 조건부 순항 샘플 부족 -> 적용 가능한 전진비행 구간 전체 사용")
                                scale_context.append(f"TAS 평균={tas_mean:.2f} m/s")
                                scale_context.append(f"{ground_ref_label} 평균={gps_mean:.2f} m/s")
                                scale_context.append(f"평균 차이(TAS-air reference)={gap_signed:+.2f} m/s")
                                if cfg_scale is not None:
                                    full_target_scale = float(cfg_scale * implied_ratio)
                                    first_step_scale = float(cfg_scale * (1.0 + bounded_step_pct / 100.0))
                                    scale_context.append(f"직접 비율 기준 scale 추정={full_target_scale:.3f}")
                                    scale_context.append(f"보수적 1차 조정안={first_step_scale:.3f} ({bounded_step_pct:+.1f}%)")
                                reason = (
                                    f"순항 평균 TAS={tas_mean:.2f} m/s, {ground_ref_label}={gps_mean:.2f} m/s, "
                                    f"차이={gap_abs:.2f} m/s (허용 ±{warn_thr:.1f} m/s)"
                                )
                                evidence = (
                                    f"{'순항 필터 사용' if used_conditional_cruise else '전구간 평균 사용'} | "
                                    f"samples={used_count}, filters={' / '.join(filter_notes) if filter_notes else 'none'}"
                                )
                                if status == "good":
                                    suggestion = (
                                        f"TAS와 {ground_ref_label} 순항 평균이 ±{warn_thr:.1f} m/s 이내로 일치하여 airspeed scale은 정상 범위로 보입니다. "
                                        "다음 비행에서도 동일한 경향인지 한 번 더 확인하세요."
                                    )
                                else:
                                    if direction == "유지":
                                        suggestion = (
                                            "평균 차이가 남아 있지만 방향성이 모호합니다. 동일 조건의 반복 비행으로 경향을 먼저 재확인하고, "
                                            "pitot 설치 상태와 속도 로그 품질을 함께 점검하세요."
                                        )
                                    elif cfg_scale is not None:
                                        full_target_scale = float(cfg_scale * implied_ratio)
                                        first_step_scale = float(cfg_scale * (1.0 + bounded_step_pct / 100.0))
                                        suggestion = (
                                            f"순항 평균에서 TAS가 {ground_ref_label}보다 {gap_abs:.2f} m/s "
                                            f"{'낮게' if direction == '상향' else '높게'} 측정되어 "
                                            f"{cfg_scale_name}을(를) {direction} 권장합니다. "
                                            f"보수적 1차 조정안: {cfg_scale:.3f} -> {first_step_scale:.3f} ({bounded_step_pct:+.1f}%) "
                                            f"(직접 비율 기준 추정값 {full_target_scale:.3f}). "
                                            "한 번에 full-ratio 보정보다 소폭 조정 후 동일 조건 비행으로 재검증을 권장합니다."
                                        )
                                    else:
                                        suggestion = (
                                            f"순항 평균에서 TAS가 {ground_ref_label}보다 {gap_abs:.2f} m/s "
                                            f"{'낮게' if direction == '상향' else '높게'} 측정됩니다. "
                                            f"ASPD_SCALE_n을(를) {direction} 방향으로 소폭 조정한 뒤 동일 조건 비행으로 재확인하세요."
                                        )
                                scale_conf, scale_conf_breakdown = self._confidence_details(
                                    used_count,
                                    4,
                                    len(set([x for x in [aspt] + air_ref_found_topics + [chosen_mode_ctx.get("nav_state_topic")] if x])),
                                    quality=0.85 if used_conditional_cruise else 0.65,
                                    mode_factor=max(0.35, float(chosen_mode_ctx.get("mode_factor", 0.0))),
                                    reference_class=cfg.get("reference_class", "product_default_threshold"),
                                    fallback_penalty=0.08 if not used_conditional_cruise else 0.0,
                                )
                                scale_likely = None
                                scale_priority = None
                                scale_actions = None
                                scale_checklist = None
                                if status in ("warning", "problem"):
                                    scale_likely = [
                                        "현재 ASPD_SCALE_n이 순항 평균 속도 관계와 맞지 않을 가능성",
                                        "풍속 추정 오차 또는 pitot scale 오차가 남아 있을 가능성",
                                    ]
                                    scale_priority = [
                                        "Airspeed vs GPS-minus-wind air-relative speed",
                                        "같은 고도/스로틀의 순항 평균 비교와 wind estimate 품질",
                                        "가능하면 여러 heading을 포함한 loiter 재비행",
                                    ]
                                    scale_actions = [suggestion]
                                    scale_checklist = [
                                        f"재비행 후 TAS-air-reference 차이가 ±{warn_thr:.1f} m/s 이내로 줄어드는지 확인",
                                        "stall/overspeed 경향이 악화되지 않는지 확인",
                                    ]
                                found_topics = list(dict.fromkeys([x for x in [aspt] + air_ref_found_topics + [lpt, att_topic, scale_topic, chosen_mode_ctx.get("nav_state_topic")] if x]))
                                used_signals = [x for x in (
                                    [f"{aspt}.{asp_sig}" if aspt and asp_sig else ""]
                                    + gs_used_signals
                                    + [f"{lpt}.{vz}" if lpt and vz else "", f"{att_topic}.{roll_sig}" if att_topic and roll_sig else "", f"{att_topic}.{pitch_sig}" if att_topic and pitch_sig else ""]
                                ) if x]
                                if chosen_mode_ctx.get("nav_state_topic") and chosen_mode_ctx.get("nav_state_signal"):
                                    used_signals.append(f"{chosen_mode_ctx['nav_state_topic']}.{chosen_mode_ctx['nav_state_signal']}")
                                mode_labels = np.asarray([self._nav_state_label(x) for x in chosen_mode_ctx["joined_frame"]["nav_state"].to_numpy()], dtype=object)
                                mode_results = self._mode_results_for_gap_metric(
                                    mode_labels,
                                    eval_mask,
                                    tas_arr,
                                    gps_arr,
                                    chosen_mode_ctx.get("weights", np.asarray([], dtype=np.float64)),
                                    warn=cfg["warn"],
                                    problem=cfg["problem"],
                                    unit=cfg["unit"],
                                )
                                try:
                                    gap_timestamps = np.asarray(chosen_mode_ctx["joined_frame"]["timestamp_sec"].to_numpy(), dtype=np.float64)
                                except Exception:
                                    gap_timestamps = None
                                self._inject_mode_time_intervals(mode_results, gap_timestamps, mode_labels, eval_mask)
                                out.append(self._item(
                                    "airspeed_scale_appropriateness", "Airspeed Scale Appropriateness",
                                    score, status, metric_value, cfg["unit"], reason, suggestion,
                                    scope="airframe", segment="cruise",
                                    required_topics=scale_required_topics + ["vehicle_status.nav_state"],
                                    found_topics=found_topics,
                                    sample_count=int(used_count),
                                    confidence=scale_conf,
                                    confidence_breakdown=scale_conf_breakdown,
                                    evidence=evidence,
                                    used_signals=used_signals,
                                    airframe=airframe,
                                    likely_causes=scale_likely,
                                    priority_checks=scale_priority,
                                    tuning_actions=scale_actions,
                                    verification_checklist=scale_checklist,
                                    parameter_context=scale_context,
                                    mode_gate="nav_state_applicable_modes",
                                    mode_gate_reason="vehicle_status.nav_state 기준 전진비행/순항에 해당하는 비행모드만 점수화했습니다.",
                                    evaluation_strategy=chosen_mode_ctx.get("evaluation_strategy"),
                                    total_log_time_sec=chosen_mode_ctx.get("total_log_time_sec"),
                                    evaluated_time_sec=chosen_mode_ctx.get("evaluated_time_sec"),
                                    excluded_time_sec=chosen_mode_ctx.get("excluded_time_sec"),
                                    applied_ratio_pct=chosen_mode_ctx.get("applied_ratio_pct"),
                                    applicable_modes=chosen_mode_ctx.get("applicable_modes"),
                                    excluded_modes=chosen_mode_ctx.get("excluded_modes"),
                                    excluded_reasons=chosen_mode_ctx.get("excluded_reasons"),
                                    evaluated_modes=chosen_mode_ctx.get("evaluated_modes"),
                                    fallback_used=not used_conditional_cruise,
                                    fallback_source="" if used_conditional_cruise else "applicable forward-flight samples without conditional cruise filters",
                                    fallback_reason="" if used_conditional_cruise else "조건부 순항 샘플이 부족하여 적용 가능한 전진비행 구간 전체로 계산했습니다.",
                                    mode_results=mode_results,
                                    reference_class="derived_from_px4",
                                    signal_semantics="px4_explicit",
                                    threshold_source=str(cfg.get("threshold_source", "TAS versus GPS-minus-wind cruise-mean gap")),
                                    threshold_rationale="Wind-vector correction is required before interpreting the mean difference as an airspeed-scale indicator.",
                                    normalization_basis="absolute difference between cruise-mean TAS and GPS-minus-wind velocity magnitude",
                                    evidence_values={
                                        "tas_mean_mps": round(float(tas_mean), 3),
                                        "air_reference_mean_mps": round(float(gps_mean), 3),
                                        "tas_air_reference_gap_mps": round(float(metric_value), 3),
                                    },
                                ))
                                scale_row_added = True
                            elif chosen_mode_ctx.get("status_hint"):
                                scale_context = list(tecs_parameter_context)
                                if cfg_scale is not None:
                                    scale_context.insert(0, f"{cfg_scale_name}={cfg_scale:.3f} (설정된 airspeed scale)")
                                out.append(self._item(
                                    "airspeed_scale_appropriateness", "Airspeed Scale Appropriateness", None, chosen_mode_ctx["status_hint"], None, cfg["unit"],
                                    chosen_mode_ctx.get("not_evaluated_reason") or "평가 가능한 전진비행 구간이 없습니다.", cfg["suggestion"],
                                    scope="airframe", segment="cruise",
                                    required_topics=scale_required_topics + ["vehicle_status.nav_state"],
                                    found_topics=list(dict.fromkeys([x for x in [aspt] + air_ref_found_topics + [lpt, att_topic, scale_topic, chosen_mode_ctx.get("nav_state_topic")] if x])),
                                    sample_count=int(chosen_mode_ctx.get("sample_count", 0)),
                                    parameter_context=scale_context,
                                    mode_gate="nav_state_applicable_modes",
                                    mode_gate_reason="vehicle_status.nav_state 기준 전진비행/순항에 해당하는 비행모드만 점수화했습니다.",
                                    evaluation_strategy=chosen_mode_ctx.get("evaluation_strategy"),
                                    total_log_time_sec=chosen_mode_ctx.get("total_log_time_sec"),
                                    evaluated_time_sec=chosen_mode_ctx.get("evaluated_time_sec"),
                                    excluded_time_sec=chosen_mode_ctx.get("excluded_time_sec"),
                                    applied_ratio_pct=chosen_mode_ctx.get("applied_ratio_pct"),
                                    applicable_modes=chosen_mode_ctx.get("applicable_modes"),
                                    excluded_modes=chosen_mode_ctx.get("excluded_modes"),
                                    excluded_reasons=chosen_mode_ctx.get("excluded_reasons"),
                                    evaluated_modes=chosen_mode_ctx.get("evaluated_modes"),
                                    mode_results=chosen_mode_ctx.get("mode_results"),
                                    not_evaluated_reason=chosen_mode_ctx.get("not_evaluated_reason"),
                                ))
                                scale_row_added = True
            except Exception:
                scale_row_added = False

            if not scale_row_added:
                scale_context = list(tecs_parameter_context)
                if cfg_scale is not None:
                    scale_context.insert(0, f"{cfg_scale_name}={cfg_scale:.3f} (설정된 airspeed scale)")
                if aspt and asp_sig:
                    scale_context.append(f"TAS source={aspt}.{asp_sig}")
                if gs_used_signals:
                    scale_context.append(f"Air-relative reference source={', '.join(gs_used_signals)}")
                if val_ref_supp is not None:
                    scale_context.append(f"{scale_topic}.{sig_valid}={val_ref_supp:.3f} (참고: validated scale 중앙값)")
                if raw_ref_supp is not None:
                    scale_context.append(f"{scale_topic}.{sig_raw}={raw_ref_supp:.3f} (참고: raw scale 중앙값)")
                missing_parts = []
                if asp is None:
                    missing_parts.append("airspeed/TAS 신호")
                if gs_series is None:
                    missing_parts.append("GPS velocity + wind vector 신호")
                if asp is not None and gs_series is not None:
                    missing_parts.append("충분한 TAS-air-reference 비교 샘플 (10개 미만)")
                missing_text = ", ".join(missing_parts) if missing_parts else "TAS-air-reference 비교 샘플"
                out.append(self._item(
                    "airspeed_scale_appropriateness", "Airspeed Scale Appropriateness", None, "unavailable", None, cfg["unit"],
                    f"Missing {missing_text}.", cfg["suggestion"],
                    scope="airframe", segment="cruise",
                    required_topics=scale_required_topics,
                    found_topics=list(dict.fromkeys([x for x in [aspt] + air_ref_found_topics + [lpt, scale_topic] if x])),
                    sample_count=0,
                    parameter_context=scale_context,
                    reference_class="derived_from_px4",
                    signal_semantics="px4_explicit",
                    threshold_source=str(cfg.get("threshold_source", "TAS versus GPS-minus-wind cruise-mean gap")),
                    not_evaluated_reason="Wind-corrected air-relative reference is required; raw TAS-minus-ground-speed is intentionally not scored.",
                ))

            # Navigation path tracking
            # The evaluator below deliberately does not use local-position to
            # waypoint-setpoint distance as cross-track error.  That legacy
            # proxy remains bypassed so older logs without an explicit PX4
            # cross-track field are reported unavailable rather than mis-scored.
            out.extend(self._eval_navigation_path_tracking(dataset, airframe, parameters=parameters))
            cfg = self._metric_cfg("navigation_path_tracking_error", airframe=airframe)
            st = None
            at = None
            nav_row_added = True
            if st and at:
                try:
                    sdf = self._with_timestamp_sec(dataset.topics[st].dataframe)
                    adf = self._with_timestamp_sec(dataset.topics[at].dataframe)
                    sx = self._find_signal(sdf, ["x"])
                    sy = self._find_signal(sdf, ["y"])
                    ax = self._find_signal(adf, ["x"])
                    ay = self._find_signal(adf, ["y"])
                    xy_valid_sig = self._find_signal(adf, ["xy_valid"])
                    vxy_valid_sig = self._find_signal(adf, ["v_xy_valid"])
                    if all([sx, sy, ax, ay]):
                        sp_xy = sdf.select([
                            pl.col("timestamp_sec").cast(pl.Float64),
                            pl.col(sx).cast(pl.Float64).alias("sp_x"),
                            pl.col(sy).cast(pl.Float64).alias("sp_y"),
                        ]).drop_nulls().sort("timestamp_sec")
                        act_cols = [
                            pl.col("timestamp_sec").cast(pl.Float64),
                            pl.col(ax).cast(pl.Float64).alias("act_x"),
                            pl.col(ay).cast(pl.Float64).alias("act_y"),
                        ]
                        if xy_valid_sig:
                            act_cols.append(pl.col(xy_valid_sig).cast(pl.Float64).alias("xy_valid"))
                        if vxy_valid_sig:
                            act_cols.append(pl.col(vxy_valid_sig).cast(pl.Float64).alias("v_xy_valid"))
                        act_xy = adf.select(act_cols).drop_nulls(subset=["act_x", "act_y"]).sort("timestamp_sec")
                        mm = act_xy.join_asof(sp_xy, on="timestamp_sec", strategy="nearest").drop_nulls()
                        if mm.height > 20:
                            err = np.sqrt((mm["act_x"].to_numpy() - mm["sp_x"].to_numpy()) ** 2 + (mm["act_y"].to_numpy() - mm["sp_y"].to_numpy()) ** 2)
                            valid_mask = np.isfinite(err)
                            applicable_modes, excluded_modes = self._metric_mode_policy(
                                "navigation_path_tracking_error",
                                airframe=airframe,
                                default_applicable=self.PATH_SETPOINT_APPLICABLE_MODE_LABELS,
                                default_excluded=self.SETPOINT_EXCLUDED_MODE_LABELS,
                            )
                            mode_ctx = self._build_mode_filtered_context(
                                dataset,
                                mm.select([pl.col("timestamp_sec").cast(pl.Float64)]),
                                "navigation_path_tracking_error",
                                valid_mask=valid_mask,
                                metric_values=err,
                                airframe=airframe,
                                applicable_modes=applicable_modes,
                                excluded_modes=excluded_modes,
                                warn=cfg["warn"],
                                problem=cfg["problem"],
                                unit=cfg["unit"],
                            )
                            found_topics = [x for x in [st, at, mode_ctx.get("nav_state_topic")] if x]
                            used_signals = [x for x in [
                                f"{st}.{sx}", f"{st}.{sy}", f"{at}.{ax}", f"{at}.{ay}",
                                f"{at}.{xy_valid_sig}" if at and xy_valid_sig else "",
                                f"{at}.{vxy_valid_sig}" if at and vxy_valid_sig else "",
                                f"{mode_ctx['nav_state_topic']}.{mode_ctx['nav_state_signal']}" if mode_ctx.get("nav_state_topic") and mode_ctx.get("nav_state_signal") else "",
                            ] if x]
                            if mode_ctx.get("status_hint"):
                                out.append(self._item(
                                    "navigation_path_tracking_error", "Navigation Path Tracking Error", None, mode_ctx["status_hint"], None, cfg["unit"],
                                    mode_ctx.get("not_evaluated_reason") or "경로 추종을 평가할 자동 비행모드 구간이 없습니다.", cfg["suggestion"],
                                    scope="airframe", segment="path",
                                    required_topics=[
                                        "vehicle_local_position_setpoint.x/y",
                                        "vehicle_local_position.x/y",
                                        "vehicle_status.nav_state",
                                    ],
                                    found_topics=found_topics, sample_count=mode_ctx.get("sample_count", 0),
                                    used_signals=used_signals,
                                    reference_class="px4_explicit",
                                    signal_semantics="px4_explicit",
                                    threshold_source=str(cfg.get("threshold_source", "PX4 setpoint/position semantics + product default path MAE thresholds")),
                                    threshold_rationale=str(cfg.get("threshold_rationale", "Use only setpoint-to-actual XY path error for the main path-tracking score.")),
                                    reference_docs=cfg.get("reference_docs", []),
                                    mode_gate="nav_state_applicable_modes",
                                    mode_gate_reason="vehicle_status.nav_state 기준 경로 추종이 의미 있는 비행모드만 점수화했습니다.",
                                    normalization_basis="시간 가중 평균 경로 오차(MAE)",
                                    fallback_used=False,
                                    evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                                    total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                                    evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                                    excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                                    applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                                    applicable_modes=mode_ctx.get("applicable_modes"),
                                    excluded_modes=mode_ctx.get("excluded_modes"),
                                    excluded_reasons=mode_ctx.get("excluded_reasons"),
                                    evaluated_modes=mode_ctx.get("evaluated_modes"),
                                    mode_results=mode_ctx.get("mode_results"),
                                    not_evaluated_reason=mode_ctx.get("not_evaluated_reason"),
                                ))
                                reliability_cfg = self._metric_cfg("navigation_path_tracking_reliability", airframe=airframe)
                                out.append(self._item(
                                    "navigation_path_tracking_reliability", "Navigation Path Tracking Reliability", None, mode_ctx["status_hint"], None, str(reliability_cfg.get("unit", "score")),
                                    mode_ctx.get("not_evaluated_reason") or "경로 추종 신뢰도를 평가할 자동 비행모드 구간이 없습니다.", str(reliability_cfg.get("suggestion", "Review NAV_ACC_RAD and local-position validity together with path-tail error.")),
                                    scope="airframe", segment="path",
                                    required_topics=[
                                        "vehicle_local_position_setpoint.x/y",
                                        "vehicle_local_position.x/y",
                                        "vehicle_status.nav_state",
                                        "vehicle_local_position.xy_valid|vehicle_local_position.v_xy_valid",
                                    ],
                                    found_topics=found_topics, sample_count=mode_ctx.get("sample_count", 0),
                                    used_signals=used_signals,
                                    reference_class="mixed_reference",
                                    signal_semantics="px4_explicit",
                                    threshold_source=str(reliability_cfg.get("threshold_source", "NAV_ACC_RAD + vehicle_local_position xy_valid/v_xy_valid cross-check")),
                                    threshold_rationale=str(reliability_cfg.get("threshold_rationale", "Separate path-tracking confidence from pure path error by cross-checking tail error against NAV_ACC_RAD and local-position validity ratio.")),
                                    mode_gate="nav_state_applicable_modes",
                                    mode_gate_reason="vehicle_status.nav_state 기준 경로 추종이 의미 있는 비행모드만 점수화했습니다.",
                                    normalization_basis="P95/reference ratio and xy_valid ratio cross-check",
                                    fallback_used=False,
                                    evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                                    total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                                    evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                                    excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                                    applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                                    applicable_modes=mode_ctx.get("applicable_modes"),
                                    excluded_modes=mode_ctx.get("excluded_modes"),
                                    excluded_reasons=mode_ctx.get("excluded_reasons"),
                                    evaluated_modes=mode_ctx.get("evaluated_modes"),
                                    not_evaluated_reason=mode_ctx.get("not_evaluated_reason"),
                                ))
                                nav_row_added = True
                            else:
                                eval_mask = mode_ctx.get("evaluated_mask", np.asarray([], dtype=bool))
                                weights = mode_ctx.get("weights", np.asarray([], dtype=np.float64))
                                err_use = err[eval_mask]
                                weight_use = weights[eval_mask]
                                used_count = int(np.sum(eval_mask))
                                mae = self._weighted_mean(err_use, weight_use)
                                p95 = float(np.percentile(err_use[np.isfinite(err_use)], 95)) if used_count else np.nan
                                p99 = float(np.percentile(err_use[np.isfinite(err_use)], 99)) if used_count else np.nan
                                ref_radius, ref_name = self._parameter_float(parameters, ["NAV_ACC_RAD"])
                                ref_ratio = None if ref_radius is None or ref_radius <= 1e-6 else float(p95 / ref_radius)
                                mae_score, _ = self._score_low(mae, cfg["warn"], cfg["problem"])
                                threshold_notes = [f"MAE warn/problem={float(cfg['warn']):.1f}/{float(cfg['problem']):.1f} m"]
                                if ref_ratio is not None:
                                    threshold_notes.append(f"NAV_ACC_RAD={float(ref_radius):.2f} m")
                                valid_ratio = None
                                if "xy_valid" in mm.columns or "v_xy_valid" in mm.columns:
                                    nav_valid_mask = np.ones(len(mm), dtype=bool)
                                    if "xy_valid" in mm.columns:
                                        xy_valid_arr = np.asarray(mm["xy_valid"].to_numpy(), dtype=np.float64)
                                        nav_valid_mask &= np.isfinite(xy_valid_arr) & (xy_valid_arr > 0.5)
                                    if "v_xy_valid" in mm.columns:
                                        vxy_valid_arr = np.asarray(mm["v_xy_valid"].to_numpy(), dtype=np.float64)
                                        nav_valid_mask &= np.isfinite(vxy_valid_arr) & (vxy_valid_arr > 0.5)
                                    valid_ratio = float(np.mean(nav_valid_mask[eval_mask]) * 100.0) if int(np.sum(eval_mask)) > 0 else None
                                final_score = float(mae_score) if mae_score is not None else None
                                stt = self._status_from_score(final_score)
                                reason_parts = [f"MAE={mae:.2f}m", f"P95={p95:.2f}m", f"P99={p99:.2f}m"]
                                evidence_values = {
                                    "path_mae_m": round(mae, 3),
                                    "path_p95_m": round(p95, 3),
                                    "path_p99_m": round(p99, 3),
                                    "samples_used": int(used_count),
                                }
                                conf, conf_breakdown = self._confidence_details(
                                    used_count,
                                    3,
                                    len([x for x in [st, at, mode_ctx.get("nav_state_topic")] if x]),
                                    quality=0.95,
                                    mode_factor=max(0.35, float(mode_ctx.get("mode_factor", 0.0))),
                                    reference_class="px4_explicit",
                                    fallback_penalty=0.0,
                                )
                                out.append(self._item(
                                    "navigation_path_tracking_error", "Navigation Path Tracking Error", final_score, stt, mae, cfg["unit"], " / ".join(reason_parts), cfg["suggestion"],
                                    scope="airframe", segment="path",
                                    required_topics=[
                                        "vehicle_local_position_setpoint.x/y",
                                        "vehicle_local_position.x/y",
                                        "vehicle_status.nav_state",
                                        "vehicle_local_position.xy_valid|vehicle_local_position.v_xy_valid",
                                    ],
                                    found_topics=found_topics, sample_count=used_count,
                                    used_signals=used_signals,
                                    confidence=conf,
                                    confidence_breakdown=conf_breakdown,
                                    evidence=" | ".join(threshold_notes),
                                    reference_class="px4_explicit",
                                    signal_semantics="px4_explicit",
                                    threshold_source=str(cfg.get("threshold_source", "PX4 setpoint/position semantics + product default path MAE thresholds")),
                                    threshold_rationale=str(cfg.get("threshold_rationale", "Use only setpoint-to-actual XY path error for the main path-tracking score.")),
                                    reference_docs=cfg.get("reference_docs", []),
                                    mode_gate="nav_state_applicable_modes",
                                    mode_gate_reason="vehicle_status.nav_state 기준 경로 추종이 의미 있는 비행모드만 점수화했습니다.",
                                    normalization_basis="시간 가중 평균 경로 오차(MAE)",
                                    fallback_used=False,
                                    evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                                    total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                                    evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                                    excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                                    applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                                    applicable_modes=mode_ctx.get("applicable_modes"),
                                    excluded_modes=mode_ctx.get("excluded_modes"),
                                    excluded_reasons=mode_ctx.get("excluded_reasons"),
                                    evidence_values=evidence_values,
                                    evaluated_modes=mode_ctx.get("evaluated_modes"),
                                    mode_results=mode_ctx.get("mode_results"),
                                ))
                                reliability_cfg = self._metric_cfg("navigation_path_tracking_reliability", airframe=airframe)
                                reliability_scores = []
                                reliability_reason_parts = []
                                reliability_threshold_notes = []
                                reliability_evidence_values = {
                                    "path_p95_m": round(p95, 3),
                                    "path_p99_m": round(p99, 3),
                                    "samples_used": int(used_count),
                                }
                                reliability_param_context = []
                                if ref_ratio is not None:
                                    ratio_warn = float(reliability_cfg.get("warn_p95_ratio", 0.8))
                                    ratio_problem = float(reliability_cfg.get("problem_p95_ratio", 1.2))
                                    ratio_score, _ = self._score_low(ref_ratio, ratio_warn, ratio_problem)
                                    reliability_scores.append(float(ratio_score))
                                    reliability_reason_parts.append(f"P95/ref={ref_ratio:.2f}")
                                    reliability_threshold_notes.append(f"P95/ref warn/problem={ratio_warn:.2f}/{ratio_problem:.2f}")
                                    reliability_evidence_values["reference_radius_m"] = round(float(ref_radius), 3)
                                    reliability_evidence_values["path_ref_ratio"] = round(ref_ratio, 3)
                                    reliability_param_context.append(f"{ref_name or 'NAV_ACC_RAD'}={float(ref_radius):.3f} m (경로 수용 반경 참고 기준)")
                                if valid_ratio is not None:
                                    valid_warn = float(reliability_cfg.get("xy_valid_warn", 98.0))
                                    valid_problem = float(reliability_cfg.get("xy_valid_problem", 95.0))
                                    valid_score, _ = self._score_high(valid_ratio, valid_warn, valid_problem)
                                    reliability_scores.append(float(valid_score))
                                    reliability_reason_parts.append(f"xy_valid={valid_ratio:.1f}%")
                                    reliability_threshold_notes.append(f"xy_valid warn/problem={valid_warn:.1f}/{valid_problem:.1f}%")
                                    reliability_evidence_values["xy_valid_ratio_pct"] = round(valid_ratio, 3)
                                if reliability_scores:
                                    reliability_score = min(reliability_scores)
                                    reliability_status = self._status_from_score(
                                        reliability_score,
                                        good_min=float(reliability_cfg.get("warn_score_min", 80.0)),
                                        warn_min=float(reliability_cfg.get("problem_score_min", 60.0)),
                                    )
                                    reliability_quality = 0.95 if len(reliability_scores) >= 2 else 0.85
                                    reliability_conf, reliability_conf_breakdown = self._confidence_details(
                                        used_count,
                                        4,
                                        len([x for x in [st, at, mode_ctx.get("nav_state_topic"), at if (xy_valid_sig or vxy_valid_sig) else None] if x]),
                                        quality=reliability_quality,
                                        mode_factor=max(0.35, float(mode_ctx.get("mode_factor", 0.0))),
                                        reference_class="mixed_reference",
                                        fallback_penalty=0.0 if len(reliability_scores) >= 2 else 0.15,
                                    )
                                    out.append(self._item(
                                        "navigation_path_tracking_reliability",
                                        "Navigation Path Tracking Reliability",
                                        reliability_score,
                                        reliability_status,
                                        reliability_score,
                                        str(reliability_cfg.get("unit", "score")),
                                        " / ".join(reliability_reason_parts),
                                        str(reliability_cfg.get("suggestion", "Review NAV_ACC_RAD and local-position validity together with path-tail error.")),
                                        scope="airframe",
                                        segment="path",
                                        required_topics=[
                                            "vehicle_local_position_setpoint.x/y",
                                            "vehicle_local_position.x/y",
                                            "vehicle_status.nav_state",
                                            "vehicle_local_position.xy_valid|vehicle_local_position.v_xy_valid",
                                        ],
                                        found_topics=found_topics,
                                        sample_count=used_count,
                                        used_signals=used_signals,
                                        confidence=reliability_conf,
                                        confidence_breakdown=reliability_conf_breakdown,
                                        evidence=" | ".join(reliability_threshold_notes),
                                        parameter_context=reliability_param_context,
                                        reference_class="mixed_reference",
                                        signal_semantics="px4_explicit",
                                        threshold_source=str(reliability_cfg.get("threshold_source", "NAV_ACC_RAD + vehicle_local_position xy_valid/v_xy_valid cross-check")),
                                        threshold_rationale=str(reliability_cfg.get("threshold_rationale", "Separate path-tracking confidence from pure path error by cross-checking tail error against NAV_ACC_RAD and local-position validity ratio.")),
                                        reference_docs=reliability_cfg.get("reference_docs", []),
                                        mode_gate="nav_state_applicable_modes",
                                        mode_gate_reason="vehicle_status.nav_state 기준 경로 추종이 의미 있는 비행모드만 점수화했습니다.",
                                        normalization_basis="P95/reference ratio and xy_valid ratio cross-check",
                                        fallback_used=bool(len(reliability_scores) < 2),
                                        fallback_source="single cross-check component" if len(reliability_scores) < 2 else "",
                                        fallback_reason="NAV_ACC_RAD 또는 xy_valid 중 한쪽만 있어 단일 근거로 신뢰도를 계산했습니다." if len(reliability_scores) < 2 else "",
                                        evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                                        total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                                        evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                                        excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                                        applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                                        applicable_modes=mode_ctx.get("applicable_modes"),
                                        excluded_modes=mode_ctx.get("excluded_modes"),
                                        excluded_reasons=mode_ctx.get("excluded_reasons"),
                                        evidence_values=reliability_evidence_values,
                                        evaluated_modes=mode_ctx.get("evaluated_modes"),
                                    ))
                                else:
                                    out.append(self._item(
                                        "navigation_path_tracking_reliability",
                                        "Navigation Path Tracking Reliability",
                                        None,
                                        "unavailable",
                                        None,
                                        str(reliability_cfg.get("unit", "score")),
                                        "Missing NAV_ACC_RAD and vehicle_local_position.xy_valid/v_xy_valid cross-check inputs.",
                                        str(reliability_cfg.get("suggestion", "Review NAV_ACC_RAD and local-position validity together with path-tail error.")),
                                        scope="airframe",
                                        segment="path",
                                        required_topics=[
                                            "vehicle_local_position_setpoint.x/y",
                                            "vehicle_local_position.x/y",
                                            "vehicle_status.nav_state",
                                            "vehicle_local_position.xy_valid|vehicle_local_position.v_xy_valid",
                                        ],
                                        found_topics=found_topics,
                                        sample_count=used_count,
                                        used_signals=used_signals,
                                        reference_class="mixed_reference",
                                        signal_semantics="px4_explicit",
                                        threshold_source=str(reliability_cfg.get("threshold_source", "NAV_ACC_RAD + vehicle_local_position xy_valid/v_xy_valid cross-check")),
                                        threshold_rationale=str(reliability_cfg.get("threshold_rationale", "Separate path-tracking confidence from pure path error by cross-checking tail error against NAV_ACC_RAD and local-position validity ratio.")),
                                        mode_gate="nav_state_applicable_modes",
                                        mode_gate_reason="vehicle_status.nav_state 기준 경로 추종이 의미 있는 비행모드만 점수화했습니다.",
                                        normalization_basis="P95/reference ratio and xy_valid ratio cross-check",
                                        evaluation_strategy=mode_ctx.get("evaluation_strategy"),
                                        total_log_time_sec=mode_ctx.get("total_log_time_sec"),
                                        evaluated_time_sec=mode_ctx.get("evaluated_time_sec"),
                                        excluded_time_sec=mode_ctx.get("excluded_time_sec"),
                                        applied_ratio_pct=mode_ctx.get("applied_ratio_pct"),
                                        applicable_modes=mode_ctx.get("applicable_modes"),
                                        excluded_modes=mode_ctx.get("excluded_modes"),
                                        excluded_reasons=mode_ctx.get("excluded_reasons"),
                                        evaluated_modes=mode_ctx.get("evaluated_modes"),
                                        not_evaluated_reason="NAV_ACC_RAD 및 xy_valid/v_xy_valid 근거가 없어 경로 추종 신뢰도 교차검증을 계산할 수 없습니다.",
                                    ))
                                nav_row_added = True
                except Exception:
                    pass
            if not nav_row_added:
                out.append(self._item(
                    "navigation_path_tracking_error", "Navigation Path Tracking Error", None, "unavailable", None, cfg["unit"],
                    "Missing vehicle_local_position_setpoint.x/y or vehicle_local_position.x/y.", cfg["suggestion"],
                    scope="airframe", segment="path",
                    required_topics=["vehicle_local_position_setpoint.x/y", "vehicle_local_position.x/y"],
                    found_topics=[x for x in [st, at] if x], sample_count=0,
                ))
                reliability_cfg = self._metric_cfg("navigation_path_tracking_reliability", airframe=airframe)
                out.append(self._item(
                    "navigation_path_tracking_reliability", "Navigation Path Tracking Reliability", None, "unavailable", None, str(reliability_cfg.get("unit", "score")),
                    "Missing vehicle_local_position_setpoint.x/y or vehicle_local_position.x/y.", str(reliability_cfg.get("suggestion", "Review NAV_ACC_RAD and local-position validity together with path-tail error.")),
                    scope="airframe", segment="path",
                    required_topics=["vehicle_local_position_setpoint.x/y", "vehicle_local_position.x/y"],
                    found_topics=[x for x in [st, at] if x], sample_count=0,
                ))

            # Phase 2: Throttle-pitch coupling index
            cfg = self._metric_cfg("throttle_pitch_coupling", airframe=airframe)
            thr, tht, _ = self._extract_series(
                dataset,
                ["tecs_status", "actuator_controls", "vehicle_thrust_setpoint"],
                ["throttle_sp", "throttle_setpoint", "throttle_trim", "control[3]", "xyz[2]"],
                "throttle",
            )
            pitch, pit_t, _ = self._extract_series(dataset, "vehicle_attitude", ["pitch_euler", "pitch"], "pitch")
            if thr is not None and pitch is not None:
                try:
                    mm = thr.join_asof(pitch, on="timestamp_sec", strategy="nearest").drop_nulls(subset=["throttle", "pitch"])
                    if mm.height > 40:
                        d_thr = np.diff(mm["throttle"].to_numpy())
                        d_pitch = np.diff(mm["pitch"].to_numpy())
                        m = np.isfinite(d_thr) & np.isfinite(d_pitch)
                        d_thr = d_thr[m]
                        d_pitch = d_pitch[m]
                        if d_thr.size > 30 and np.std(d_thr) > 1e-8 and np.std(d_pitch) > 1e-8:
                            corr = float(np.corrcoef(d_thr, d_pitch)[0, 1])
                            corr = 0.0 if not np.isfinite(corr) else abs(corr)
                            v = float(corr * 100.0)
                            s, stt = self._score_low(v, cfg["warn"], cfg["problem"])
                            out.append(self._item(
                                "throttle_pitch_coupling", "Throttle-Pitch Coupling Index", s, stt, v, cfg["unit"], f"|corr(dThrottle,dPitch)|={v:.1f}%", cfg["suggestion"],
                                scope="airframe", segment="cruise",
                                required_topics=["throttle source", "vehicle_attitude.pitch_euler"],
                                found_topics=[x for x in [tht, pit_t] if x], sample_count=int(d_thr.size),
                            ))
                except Exception:
                    pass

        # Phase 2: RPM stability (rotorcraft-centric)
        if airframe in ("HELICOPTER", "VTOL"):
            cfg = self._metric_cfg("rpm_stability", airframe=airframe)
            rpm, rtopic, rsig = self._find_rpm_series(dataset)
            if rpm is not None:
                rv = self._to_float(rpm["rpm"].to_numpy())
                if rv.size > 40:
                    mean_abs = float(max(abs(np.mean(rv)), 1e-9))
                    cv = float(np.std(rv) / mean_abs * 100.0)
                    s, stt = self._score_low(cv, cfg["warn"], cfg["problem"])
                    out.append(self._item(
                        "rpm_stability", "RPM Stability", s, stt, cv, cfg["unit"], f"CV={cv:.2f}% ({rsig})", cfg["suggestion"],
                        scope="airframe", segment="all", required_topics=["any *_rpm field"], found_topics=[x for x in [rtopic] if x], sample_count=int(rv.size),
                    ))
            elif airframe == "HELICOPTER":
                out.append(self._item(
                    "rpm_stability", "RPM Stability", None, "unavailable", None, cfg["unit"], "RPM signal not found.", cfg["suggestion"],
                    scope="airframe", segment="all", required_topics=["any *_rpm field"], found_topics=[],
                ))

        # Phase 2: Rotor harmonic vibration severity
        if airframe in ("MULTICOPTER", "VTOL", "HELICOPTER"):
            cfg = self._metric_cfg("rotor_harmonic_vibration", airframe=airframe)
            t = self._find_topic(dataset, ["sensor_combined", "vehicle_acceleration", "vehicle_imu"])
            if t:
                try:
                    df = self._with_timestamp_sec(dataset.topics[t].dataframe)
                    ax = self._find_signal(df, ["accelerometer_m_s2[0]", "accel_m_s2[0]", "xyz[0]", "accel[0]"])
                    ay = self._find_signal(df, ["accelerometer_m_s2[1]", "accel_m_s2[1]", "xyz[1]", "accel[1]"])
                    az = self._find_signal(df, ["accelerometer_m_s2[2]", "accel_m_s2[2]", "xyz[2]", "accel[2]"])
                    if ax and ay and az and "timestamp_sec" in df.columns:
                        md = df.select([
                            pl.col("timestamp_sec").cast(pl.Float64).alias("t"),
                            pl.col(ax).cast(pl.Float64).alias("ax"),
                            pl.col(ay).cast(pl.Float64).alias("ay"),
                            pl.col(az).cast(pl.Float64).alias("az"),
                        ]).drop_nulls().sort("t")
                        if md.height > 300:
                            ts = md["t"].to_numpy()
                            dt = np.diff(ts)
                            dt = dt[np.isfinite(dt) & (dt > 1e-5)]
                            if dt.size > 0:
                                d = float(np.median(dt))
                                fs = 1.0 / max(d, 1e-6)
                                mag = np.sqrt(md["ax"].to_numpy() ** 2 + md["ay"].to_numpy() ** 2 + md["az"].to_numpy() ** 2)
                                mag = mag[np.isfinite(mag)]
                                if mag.size >= 256 and fs > 5.0:
                                    n = int(min(mag.size, 8192))
                                    sig = mag[-n:] - float(np.mean(mag[-n:]))
                                    win = np.hanning(n)
                                    sp = np.abs(np.fft.rfft(sig * win)) ** 2
                                    fr = np.fft.rfftfreq(n, d=d)
                                    valid_mask = (fr >= 5.0) & (fr <= min(300.0, fs * 0.45))
                                    band = sp[valid_mask]
                                    fr_band = fr[valid_mask]
                                    if band.size > 20:
                                        peak = float(np.max(band))
                                        med = float(np.median(band) + 1e-12)
                                        db = float(10.0 * np.log10((peak + 1e-12) / med))
                                        dom_freq = float(fr_band[int(np.argmax(band))]) if fr_band.size else 0.0
                                        ranges = {
                                            "low": (5.0, 30.0),
                                            "mid": (30.0, 90.0),
                                            "high": (90.0, min(300.0, fs * 0.45)),
                                        }
                                        band_stats = []
                                        for bname, (f0, f1) in ranges.items():
                                            if f1 <= f0:
                                                continue
                                            bm = (fr >= f0) & (fr < f1)
                                            bp = sp[bm]
                                            if bp.size > 5:
                                                bp_peak = float(np.max(bp))
                                                bp_med = float(np.median(bp) + 1e-12)
                                                bp_db = float(10.0 * np.log10((bp_peak + 1e-12) / bp_med))
                                                band_stats.append((bname, bp_db))
                                        band_stats.sort(key=lambda x: x[1], reverse=True)
                                        top_band = band_stats[0][0] if band_stats else "n/a"
                                        bands_text = ", ".join(f"{bn}:{bv:.1f}dB" for bn, bv in band_stats[:3])
                                        s, stt = self._score_low(db, cfg["warn"], cfg["problem"])
                                        out.append(self._item(
                                            "rotor_harmonic_vibration",
                                            "Rotor Harmonic Vibration",
                                            s,
                                            stt,
                                            db,
                                            cfg["unit"],
                                            f"Peak/median={db:.2f} dB, dom={dom_freq:.1f} Hz, band={top_band}",
                                            cfg["suggestion"],
                                            scope="airframe",
                                            segment="all",
                                            required_topics=["sensor_combined|vehicle_acceleration|vehicle_imu"],
                                            found_topics=[t],
                                            sample_count=n,
                                            evidence=f"Dominant {dom_freq:.1f} Hz, band levels [{bands_text}]",
                                            used_signals=[f"{t}.{ax}", f"{t}.{ay}", f"{t}.{az}"],
                                            airframe=airframe,
                                        ))
                except Exception:
                    pass

        # Phase 2: Control coupling index (cross-axis rate coupling)
        if airframe in ("MULTICOPTER", "FIXED_WING", "VTOL", "HELICOPTER"):
            cfg = self._metric_cfg("control_coupling_index", airframe=airframe)
            rt = self._find_topic(dataset, "vehicle_angular_velocity")
            if rt:
                try:
                    rdf = self._with_timestamp_sec(dataset.topics[rt].dataframe)
                    r0 = self._find_signal(rdf, ["xyz[0]", "rollspeed", "roll_rate"])
                    r1 = self._find_signal(rdf, ["xyz[1]", "pitchspeed", "pitch_rate"])
                    r2 = self._find_signal(rdf, ["xyz[2]", "yawspeed", "yaw_rate"])
                    if r0 and r1 and r2:
                        rv = rdf.select([
                            pl.col(r0).cast(pl.Float64).alias("r0"),
                            pl.col(r1).cast(pl.Float64).alias("r1"),
                            pl.col(r2).cast(pl.Float64).alias("r2"),
                        ]).drop_nulls()
                        if rv.height > 60:
                            arr = rv.to_numpy()
                            if np.isfinite(arr).all():
                                cc = np.corrcoef(arr, rowvar=False)
                                off = [abs(float(cc[0, 1])), abs(float(cc[0, 2])), abs(float(cc[1, 2]))]
                                v = float(max(off) * 100.0)
                                s, stt = self._score_low(v, cfg["warn"], cfg["problem"])
                                out.append(self._item(
                                    "control_coupling_index", "Control Coupling Index", s, stt, v, cfg["unit"], f"max|corr|={v:.1f}%", cfg["suggestion"],
                                    scope="airframe", segment="all", required_topics=["vehicle_angular_velocity.xyz[0..2]"],
                                    found_topics=[rt], sample_count=int(rv.height),
                                ))
                except Exception:
                    pass

        if airframe == "VTOL":
            windows = self._extract_transition_windows(dataset)
            cfg_d = self._metric_cfg("transition_duration", airframe=airframe)
            cfg_l = self._metric_cfg("transition_altitude_loss", airframe=airframe)
            if windows:
                transition_thresholds = self._resolve_vtol_transition_thresholds(
                    parameters,
                    cfg_d["warn"],
                    cfg_d["problem"],
                )
                transition_warn = float(transition_thresholds["warn"])
                transition_problem = float(transition_thresholds["problem"])
                total_log_time_sec = float(self._flight_time_sec(dataset))
                evaluated_time_sec = float(sum(max(0.0, w[2]) for w in windows))
                excluded_time_sec = max(0.0, total_log_time_sec - evaluated_time_sec)
                applied_ratio_pct = (100.0 * evaluated_time_sec / total_log_time_sec) if total_log_time_sec > 1e-9 else 0.0
                duration_mode_results = []
                for idx, (_, _, dur, samples) in enumerate(windows, start=1):
                    score_i, status_i = self._score_low(float(dur), transition_warn, transition_problem)
                    duration_mode_results.append({
                        "mode": f"Transition {idx}",
                        "status": status_i,
                        "status_label": self.STATUS_LABEL.get(status_i, status_i),
                        "score": round(float(score_i), 1),
                        "value": float(dur),
                        "unit": cfg_d["unit"],
                        "time_sec": round(float(dur), 3),
                        "evaluated_ratio_pct": round((100.0 * float(dur) / evaluated_time_sec), 3) if evaluated_time_sec > 1e-9 else 0.0,
                        "sample_count": int(samples),
                    })
                maxw = max(windows, key=lambda x: x[2])
                v = float(maxw[2])
                s, stt = self._score_low(v, transition_warn, transition_problem)
                out.append(self._item(
                    "transition_duration", "Transition Duration", s, stt, v, cfg_d["unit"],
                    (
                        f"transition_count={len(windows)}, max_duration={v:.2f}s, "
                        f"effective_warn={transition_warn:.2f}s, effective_problem={transition_problem:.2f}s"
                    ),
                    cfg_d["suggestion"],
                    scope="airframe", segment="transition", required_topics=["vehicle_status.in_transition_mode"],
                    found_topics=["vehicle_status"], sample_count=int(sum(w[3] for w in windows)),
                    parameter_context=transition_thresholds["parameter_context"],
                    threshold_source=str(transition_thresholds["threshold_source"]),
                    threshold_rationale=str(transition_thresholds["threshold_rationale"]),
                    normalization_basis=str(transition_thresholds["normalization_basis"]),
                    evaluation_strategy="transition_event_segments",
                    total_log_time_sec=total_log_time_sec,
                    evaluated_time_sec=evaluated_time_sec,
                    excluded_time_sec=excluded_time_sec,
                    applied_ratio_pct=applied_ratio_pct,
                    applicable_modes=["Transition"],
                    excluded_reasons=["전환 모드가 아닌 구간은 transition 평가 점수에서 제외했습니다."],
                    evaluated_modes=[x["mode"] for x in duration_mode_results],
                    mode_results=duration_mode_results,
                    evidence_values=dict(transition_thresholds.get("evidence_values", {}) or {}),
                ))
                lt = self._find_topic(dataset, "vehicle_local_position")
                if lt:
                    df = self._with_timestamp_sec(dataset.topics[lt].dataframe)
                    zs = self._find_signal(df, ["z", "alt_up"])
                    if zs:
                        z_df = df.select([pl.col("timestamp_sec").cast(pl.Float64), pl.col(zs).cast(pl.Float64).alias("z")]).drop_nulls().sort("timestamp_sec")
                        loss_results = []
                        for idx, (start_t, end_t, dur, _) in enumerate(windows, start=1):
                            seg = z_df.filter((pl.col("timestamp_sec") >= start_t) & (pl.col("timestamp_sec") <= end_t))
                            if seg.height < 2:
                                continue
                            z = self._to_float(seg["z"].to_numpy())
                            if z.size < 2:
                                continue
                            start_z = float(z[0])
                            if str(zs).lower() == "alt_up":
                                # Up-positive altitude: a loss is a decrease only.
                                loss_i = float(max(start_z - float(np.min(z)), 0.0))
                                altitude_convention = "up-positive"
                            else:
                                # PX4 local-position z is NED/down-positive: a loss
                                # is an increase in z.  An altitude gain must not be
                                # folded into the loss magnitude.
                                loss_i = float(max(float(np.max(z)) - start_z, 0.0))
                                altitude_convention = "NED down-positive"
                            score_i, status_i = self._score_low(loss_i, cfg_l["warn"], cfg_l["problem"])
                            loss_results.append({
                                "mode": f"Transition {idx}",
                                "status": status_i,
                                "status_label": self.STATUS_LABEL.get(status_i, status_i),
                                "score": round(float(score_i), 1),
                                "value": float(loss_i),
                                "unit": cfg_l["unit"],
                                "time_sec": round(float(dur), 3),
                                "evaluated_ratio_pct": round((100.0 * float(dur) / evaluated_time_sec), 3) if evaluated_time_sec > 1e-9 else 0.0,
                                "sample_count": int(z.size),
                                "altitude_convention": altitude_convention,
                            })
                        if loss_results:
                            worst_loss = max(loss_results, key=lambda item: float(item.get("value", 0.0)))
                            loss = float(worst_loss["value"])
                            s2, st2 = self._score_low(loss, cfg_l["warn"], cfg_l["problem"])
                            out.append(self._item(
                                "transition_altitude_loss", "Transition Altitude Loss", s2, st2, loss, cfg_l["unit"],
                                f"worst_transition={worst_loss['mode']}, downward_excursion={loss:.2f}m", cfg_l["suggestion"],
                                scope="airframe", segment="transition", required_topics=["vehicle_status.in_transition_mode", "vehicle_local_position.z"],
                                found_topics=["vehicle_status", lt], sample_count=int(sum(int(item.get("sample_count", 0)) for item in loss_results)),
                                evaluation_strategy="transition_event_segments",
                                total_log_time_sec=total_log_time_sec,
                                evaluated_time_sec=evaluated_time_sec,
                                excluded_time_sec=excluded_time_sec,
                                applied_ratio_pct=applied_ratio_pct,
                                applicable_modes=["Transition"],
                                excluded_reasons=["전환 모드가 아닌 구간은 transition 평가 점수에서 제외했습니다."],
                                evaluated_modes=[x["mode"] for x in loss_results],
                                mode_results=loss_results,
                                reference_class="derived_from_px4",
                                signal_semantics="px4_explicit",
                                threshold_rationale="Only downward altitude excursion from transition start is counted; upward excursion is not a loss.",
                                normalization_basis="maximum downward excursion from each transition start altitude",
                                evidence_values={
                                    "worst_downward_excursion_m": round(float(loss), 3),
                                    "altitude_convention": str(worst_loss.get("altitude_convention", "")),
                                },
                            ))
        return out

    def _extract_transition_windows(self, dataset) -> List[Tuple[float, float, float, int]]:
        st = self._find_topic(dataset, "vehicle_status")
        if not st:
            return []
        df = self._with_timestamp_sec(dataset.topics[st].dataframe)
        fs = self._find_signal(df, ["in_transition_mode"])
        if not fs or "timestamp_sec" not in df.columns:
            return []
        t = self._to_float(df["timestamp_sec"].to_numpy())
        f = self._to_float(df[fs].to_numpy())
        n = min(t.size, f.size)
        if n <= 1:
            return []
        t = t[:n]; b = f[:n] > 0.5
        out = []; i = 0
        while i < n:
            if not b[i]:
                i += 1
                continue
            s = i
            while i + 1 < n and b[i + 1]:
                i += 1
            e = i
            out.append((float(t[s]), float(t[e]), float(max(0.0, t[e] - t[s])), int(e - s + 1)))
            i += 1
        return out

    def _safe_parameter_recommendations(self, issue: dict, current_params: dict, airframe: str) -> List[dict]:
        if self.parameter_recommender is None:
            return []
        try:
            return self.parameter_recommender.recommend(issue, current_params, airframe)
        except Exception:
            return []

    def _build_tuning_issues(self, dataset, airframe: str, items: List[dict]) -> Tuple[List[dict], dict]:
        if airframe not in ("MULTICOPTER", "FIXED_WING", "VTOL"):
            return [], {}
        if self.feature_extractor is None or self.tuning_inference is None:
            return [], {}

        features = self.feature_extractor.extract_rate_loop_features(dataset, axes=("roll", "pitch", "yaw"))

        by_key = {}
        for it in items:
            if isinstance(it, dict):
                by_key[str(it.get("key", "")).strip()] = it

        # Inject common confounder values from evaluated metrics when available.
        sat_val = by_key.get("controller_saturation", {}).get("value")
        vib_val = by_key.get("vibration", {}).get("value")
        inn_val = by_key.get("innovation_fail_ratio", {}).get("value")
        for ax in list(features.keys()):
            if sat_val is not None:
                features[ax]["saturation_ratio"] = float(sat_val)
            if vib_val is not None:
                features[ax]["vibration_severity"] = float(vib_val)
            if inn_val is not None:
                features[ax]["innovation_ratio"] = float(inn_val)

        current_params_view = self._eval_parameters if isinstance(self._eval_parameters, dict) else {}

        issues = []
        axis_advisors = []
        for axis in ("roll", "pitch", "yaw"):
            if axis not in features:
                continue
            try:
                issue, advisor = self.tuning_inference.infer_rate_loop_issue(axis, features[axis], airframe=airframe)
                issue = self._apply_registry_to_issue(issue, airframe)
                advisor = self._apply_registry_to_axis_advisor(advisor, issue, airframe)
                issue["parameter_recommendations"] = self._safe_parameter_recommendations(issue, current_params_view, airframe)
                issues.append(issue)
                axis_advisors.append(advisor)
            except Exception:
                continue

        # --- Attitude loop applies to all three airframes (MC/VTOL use P-gain semantics; FW uses TC). ---
        try:
            attitude_features = self.feature_extractor.extract_attitude_loop_features(
                dataset, axes=("roll", "pitch", "yaw")
            )
        except Exception:
            attitude_features = {}
        for ax in ("roll", "pitch", "yaw"):
            if ax not in attitude_features:
                continue
            # Inject confounders into attitude features for confidence calc.
            f = attitude_features[ax]
            if sat_val is not None:
                f["saturation_ratio"] = float(sat_val)
            if vib_val is not None:
                f["vibration_severity"] = float(vib_val)
            if inn_val is not None:
                f["innovation_ratio"] = float(inn_val)
            try:
                issue, advisor = self.tuning_inference.infer_attitude_loop_issue(ax, f, airframe=airframe)
                issue = self._apply_registry_to_issue(issue, airframe)
                advisor = self._apply_registry_to_axis_advisor(advisor, issue, airframe)
                issue["parameter_recommendations"] = self._safe_parameter_recommendations(issue, current_params_view, airframe)
                issues.append(issue)
                axis_advisors.append(advisor)
            except Exception:
                continue

        # --- MC-specific limits: rate-MAX, acc, tilt (position controller / MPC_* params). ---
        if airframe in ("MULTICOPTER", "VTOL"):
            try:
                rate_limit_features = self.feature_extractor.extract_rate_limit_features(
                    dataset, current_params_view, axes=("roll", "pitch", "yaw")
                )
            except Exception:
                rate_limit_features = {}
            for ax in ("roll", "pitch", "yaw"):
                if ax not in rate_limit_features:
                    continue
                try:
                    issue, advisor = self.tuning_inference.infer_rate_limit_issue(ax, rate_limit_features[ax], airframe=airframe)
                    issue = self._apply_registry_to_issue(issue, airframe)
                    advisor = self._apply_registry_to_axis_advisor(advisor, issue, airframe)
                    issue["parameter_recommendations"] = self._safe_parameter_recommendations(issue, current_params_view, airframe)
                    issues.append(issue)
                    axis_advisors.append(advisor)
                except Exception:
                    continue

            try:
                acc_metrics = self.feature_extractor.extract_acc_limit_features(dataset, current_params_view)
            except Exception:
                acc_metrics = {}
            if acc_metrics:
                try:
                    issue, advisor = self.tuning_inference.infer_acc_limit_issue(acc_metrics, airframe=airframe)
                    issue = self._apply_registry_to_issue(issue, airframe)
                    advisor = self._apply_registry_to_axis_advisor(advisor, issue, airframe)
                    issue["parameter_recommendations"] = self._safe_parameter_recommendations(issue, current_params_view, airframe)
                    issues.append(issue)
                    axis_advisors.append(advisor)
                except Exception:
                    pass

            try:
                tilt_metrics = self.feature_extractor.extract_tilt_limit_features(dataset, current_params_view)
            except Exception:
                tilt_metrics = {}
            if tilt_metrics:
                try:
                    issue, advisor = self.tuning_inference.infer_tilt_limit_issue(tilt_metrics, airframe=airframe)
                    issue = self._apply_registry_to_issue(issue, airframe)
                    advisor = self._apply_registry_to_axis_advisor(advisor, issue, airframe)
                    issue["parameter_recommendations"] = self._safe_parameter_recommendations(issue, current_params_view, airframe)
                    issues.append(issue)
                    axis_advisors.append(advisor)
                except Exception:
                    pass

        # --- FW (and VTOL forward-flight) TECS altitude / airspeed response ---
        if airframe in ("FIXED_WING", "VTOL"):
            try:
                tecs_alt_feats = self.feature_extractor.extract_tecs_altitude_features(dataset)
            except Exception:
                tecs_alt_feats = {}
            if tecs_alt_feats:
                try:
                    issue, advisor = self.tuning_inference.infer_tecs_altitude_response_issue(
                        tecs_alt_feats, airframe=airframe
                    )
                    issue = self._apply_registry_to_issue(issue, airframe)
                    advisor = self._apply_registry_to_axis_advisor(advisor, issue, airframe)
                    issue["parameter_recommendations"] = self._safe_parameter_recommendations(issue, current_params_view, airframe)
                    issues.append(issue)
                    axis_advisors.append(advisor)
                except Exception:
                    pass

            try:
                tecs_air_feats = self.feature_extractor.extract_tecs_airspeed_features(dataset)
            except Exception:
                tecs_air_feats = {}
            if tecs_air_feats:
                try:
                    issue, advisor = self.tuning_inference.infer_tecs_airspeed_response_issue(
                        tecs_air_feats, airframe=airframe
                    )
                    issue = self._apply_registry_to_issue(issue, airframe)
                    advisor = self._apply_registry_to_axis_advisor(advisor, issue, airframe)
                    issue["parameter_recommendations"] = self._safe_parameter_recommendations(issue, current_params_view, airframe)
                    issues.append(issue)
                    axis_advisors.append(advisor)
                except Exception:
                    pass

        if airframe == "VTOL":
            td = by_key.get("transition_duration", {}) if isinstance(by_key.get("transition_duration", {}), dict) else {}
            tl = by_key.get("transition_altitude_loss", {}) if isinstance(by_key.get("transition_altitude_loss", {}), dict) else {}
            if td or tl:
                try:
                    td_mode_results = td.get("mode_results", []) if isinstance(td.get("mode_results", []), list) else []
                    trans_metrics = {
                        "duration": td.get("value", 0.0),
                        "altitude_loss": tl.get("value", 0.0),
                        "innovation_ratio": by_key.get("innovation_fail_ratio", {}).get("value", 0.0),
                        "saturation_ratio": sat_val if sat_val is not None else 0.0,
                        "vibration_severity": vib_val if vib_val is not None else 0.0,
                        # Real observation counts replace the prior hardcoded values
                        # (sample_count=400, maneuver_count=2.0) inside the inference engine.
                        "sample_count": int(td.get("sample_count", 0) or 0),
                        "transition_count": float(len(td_mode_results)),
                    }
                    tr_issue, tr_advisor = self.tuning_inference.infer_vtol_transition_issue(trans_metrics, airframe=airframe)
                    tr_issue = self._apply_registry_to_issue(tr_issue, airframe)
                    tr_advisor = self._apply_registry_to_axis_advisor(tr_advisor, tr_issue, airframe)
                    tr_issue["parameter_recommendations"] = self._safe_parameter_recommendations(tr_issue, current_params_view, airframe)
                    issues.append(tr_issue)
                    axis_advisors.append(tr_advisor)
                except Exception:
                    pass

        issues.sort(
            key=lambda x: (
                3 if str(x.get("severity", "")).lower() == "problem" else 2 if str(x.get("severity", "")).lower() == "warning" else 1,
                float(x.get("confidence", 0.0)),
            ),
            reverse=True,
        )
        advisor = {
            "profile": str(self.eval_profile.get("profile_id", "general_controller_tuning")) if isinstance(self.eval_profile, dict) else "general_controller_tuning",
            "scope": {"airframe": airframe, "axes": ["roll", "pitch", "yaw"], "loop": "rate"},
            "axes": axis_advisors,
            "registry_info": self.advisor_registry_info if isinstance(self.advisor_registry_info, dict) else {},
            "notes": [
                "Candidate-based recommendation only; not deterministic diagnosis.",
                "If confidence is low, prioritize additional logging/retest before parameter changes.",
            ],
        }
        if not features:
            advisor["note"] = "No usable roll/pitch/yaw rate-loop samples for tuning inference."
        return issues, advisor

    def _weighted_score(self, items: List[dict], only_core: bool = True) -> Optional[float]:
        ws, wt = 0.0, 0.0
        for it in items:
            if only_core and not bool(it.get("is_core", True)):
                continue
            sc = it.get("score")
            if sc is None:
                continue
            w = float(it.get("weight", 1.0)); ws += float(sc) * w; wt += w
        return None if wt <= 1e-9 else float(ws / wt)

    @staticmethod
    def _safe_float(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return default

    def evaluate(
        self,
        dataset,
        file_name: str = "",
        aircraft_type: str = "Unknown",
        parameters: Optional[dict] = None,
        firmware: Optional[dict] = None,
        messages: Optional[list] = None,
    ) -> dict:
        airframe = self._normalize_airframe(aircraft_type)
        self._eval_airframe_ctx = airframe
        self._eval_parameters = parameters if isinstance(parameters, dict) else {}
        self._eval_messages = list(messages or [])
        self._eval_firmware = firmware if isinstance(firmware, dict) else {}
        px4_version_info = self._extract_px4_version(self._eval_firmware)
        self._eval_px4_version_details = px4_version_info
        self._eval_px4_version_summary = str(px4_version_info.get("summary", "Unknown"))
        items = self._eval_common(dataset) + self._eval_airframe(dataset, airframe, parameters=parameters)
        overall_score = self._weighted_score(items, only_core=True)
        oc = self.cfg.get("overall", {})
        overall_status = self._status_from_score(overall_score, self._safe_float(oc.get("good_score_min", 80.0), 80.0), self._safe_float(oc.get("warning_score_min", 60.0), 60.0))
        common_score = self._weighted_score([x for x in items if x.get("scope") == "common"], only_core=True)
        airframe_score = self._weighted_score([x for x in items if x.get("scope") == "airframe"], only_core=True)
        segment_scores: Dict[str, Optional[float]] = {}
        for seg in sorted({str(x.get("segment", "all")) for x in items}):
            segment_scores[seg] = self._weighted_score([x for x in items if str(x.get("segment", "all")) == seg], only_core=True)
        counts = {"good": 0, "warning": 0, "problem": 0, "unavailable": 0, "excluded": 0}
        for it in items:
            s = str(it.get("status", "unavailable")); counts[s] = counts.get(s, 0) + 1
        issue_items = [x for x in items if x.get("score") is not None and x.get("status") in ("problem", "warning")]
        for x in issue_items:
            conf = self._safe_float(x.get("confidence"), 50.0) / 100.0
            score = self._safe_float(x.get("score"), 100.0)
            sev = 2.0 if x.get("status") == "problem" else 1.0
            x["_issue_rank"] = sev * (100.0 - score) * max(0.3, conf)
        issue_items.sort(key=lambda x: x.get("_issue_rank", 0.0), reverse=True)
        metric_top_issues = []
        for x in issue_items[:5]:
            cause = (x.get("likely_causes") or [])
            cause_txt = cause[0] if cause else x.get("reason", "")
            metric_top_issues.append(f"{x.get('name', x.get('key', 'metric'))}: {cause_txt} (confidence {self._safe_float(x.get('confidence'), 0):.1f}%)")
        metric_recs = []
        for x in issue_items:
            for a in (x.get("tuning_actions") or []):
                if a and a not in metric_recs:
                    metric_recs.append(a)
                if len(metric_recs) >= 8:
                    break
            r = str(x.get("recommendation", "")).strip()
            if r and r not in metric_recs:
                metric_recs.append(r)
            if len(metric_recs) >= 8:
                break

        tuning_issues, tuning_advisor = self._build_tuning_issues(dataset, airframe, items)
        recommendation_view = {}
        if self.recommendation_engine is not None and tuning_issues:
            recommendation_view = self.recommendation_engine.build(
                tuning_issues,
                overall_score=None if overall_score is None else round(overall_score, 1),
                overall_status=overall_status,
            )

        top_issues = []
        if recommendation_view:
            for ti in recommendation_view.get("top_issues", []):
                if not isinstance(ti, dict):
                    continue
                title = str(ti.get("title", "Issue")).strip()
                sev = str(ti.get("severity", "normal")).strip()
                cause = str(ti.get("short_cause", "")).strip()
                conf = self._safe_float(ti.get("confidence"), np.nan)
                if np.isfinite(conf):
                    top_issues.append(f"{title} [{sev}]: {cause} (confidence {conf:.1f}%)")
                else:
                    top_issues.append(f"{title} [{sev}]: {cause}")
        for txt in metric_top_issues:
            if txt not in top_issues:
                top_issues.append(txt)
            if len(top_issues) >= 8:
                break

        recs = []
        for x in recommendation_view.get("recommended_software_actions", []) if isinstance(recommendation_view, dict) else []:
            if x not in recs:
                recs.append(x)
        for x in recommendation_view.get("recommended_hardware_actions", []) if isinstance(recommendation_view, dict) else []:
            if x not in recs:
                recs.append(x)
        for x in metric_recs:
            if x not in recs:
                recs.append(x)
            if len(recs) >= 12:
                break

        confs = [self._safe_float(x.get("confidence"), np.nan) for x in items if x.get("status") != "unavailable"]
        tuning_confs = [self._safe_float(x.get("confidence"), np.nan) for x in tuning_issues if isinstance(x, dict)]
        conf_avg = float(np.nanmean(confs)) if len(confs) > 0 else None
        tuning_conf_avg = float(np.nanmean(tuning_confs)) if len(tuning_confs) > 0 else None
        low_conf = [str(x.get("name", x.get("key", "metric"))) for x in items if self._safe_float(x.get("confidence"), 0.0) < 50.0]
        return {
            "schema_version": "1.4",
            "traceability_schema_version": "1.0",
            "aircraft_type": airframe or "Unknown",
            "file_name": file_name or "",
            "px4_version": self._eval_px4_version_summary,
            "px4_version_details": copy.deepcopy(px4_version_info),
            "overall_score": None if overall_score is None else round(overall_score, 1),
            "overall_status": overall_status,
            "overall_status_label": self.STATUS_LABEL.get(overall_status, overall_status),
            "common_score": None if common_score is None else round(common_score, 1),
            "airframe_score": None if airframe_score is None else round(airframe_score, 1),
            "segment_scores": {k: (None if v is None else round(v, 1)) for k, v in segment_scores.items()},
            "items": items,
            "summary": {"good": int(counts.get("good", 0)), "warning": int(counts.get("warning", 0)), "problem": int(counts.get("problem", 0)), "unavailable": int(counts.get("unavailable", 0))},
            "top_issues": top_issues,
            "recommended_actions": recs,
            "confidence_summary": {
                "average_confidence": None if conf_avg is None else round(conf_avg, 1),
                "tuning_average_confidence": None if tuning_conf_avg is None else round(tuning_conf_avg, 1),
                "low_confidence_metrics": low_conf[:10],
            },
            "issues": tuning_issues,
            "control_tuning_advisor": tuning_advisor,
            "recommendation_view": recommendation_view,
            "advisor_registry_info": self.advisor_registry_info if isinstance(self.advisor_registry_info, dict) else {},
        }
