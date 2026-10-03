from __future__ import annotations

from typing import Dict, List, Tuple

from .confidence import compute_tuning_confidence


class TuningInferenceEngine:
    """
    Rule-based tuning candidate inference.
    Output is possibility-based recommendation, not deterministic diagnosis.
    """

    def __init__(self, rules: dict | None = None):
        self.rules = rules or {}

    @staticmethod
    def _severity_label(v: float, warn: float, problem: float) -> str:
        if v >= problem:
            return "problem"
        if v >= warn:
            return "warning"
        return "normal"

    @staticmethod
    def _dedup(xs: List[str]) -> List[str]:
        out = []
        for x in xs:
            t = str(x).strip()
            if t and t not in out:
                out.append(t)
        return out

    def infer_rate_loop_issue(self, axis: str, feats: Dict[str, float], airframe: str = "UNKNOWN") -> Tuple[dict, dict]:
        rcfg = (self.rules.get("rate_loop", {}) if isinstance(self.rules, dict) else {}) or {}
        tw = {
            "overshoot": float(rcfg.get("overshoot_warn", 18.0)),
            "overshoot_problem": float(rcfg.get("overshoot_problem", 35.0)),
            "rise_time": float(rcfg.get("rise_time_warn", 0.55)),
            "rise_time_problem": float(rcfg.get("rise_time_problem", 1.2)),
            "settling_time": float(rcfg.get("settling_time_warn", 1.2)),
            "settling_time_problem": float(rcfg.get("settling_time_problem", 2.5)),
            "steady_state_error": float(rcfg.get("steady_state_error_warn", 0.12)),
            "steady_state_error_problem": float(rcfg.get("steady_state_error_problem", 0.30)),
            "oscillation_index": float(rcfg.get("oscillation_warn", 14.0)),
            "oscillation_problem": float(rcfg.get("oscillation_problem", 30.0)),
            "dominant_frequency": float(rcfg.get("dominant_freq_warn", 10.0)),
            "dominant_frequency_problem": float(rcfg.get("dominant_freq_problem", 18.0)),
            "saturation_ratio": float(rcfg.get("saturation_warn", 8.0)),
            "saturation_problem": float(rcfg.get("saturation_problem", 20.0)),
            "vibration_severity": float(rcfg.get("vibration_warn", 3.5)),
            "vibration_problem": float(rcfg.get("vibration_problem", 6.0)),
            "control_effort_variation": float(rcfg.get("effort_var_warn", 0.12)),
            "control_effort_variation_problem": float(rcfg.get("effort_var_problem", 0.25)),
            "innovation_ratio": float(rcfg.get("innovation_warn", 5.0)),
            "innovation_problem": float(rcfg.get("innovation_problem", 10.0)),
        }

        values = {
            "overshoot": float(feats.get("overshoot", 0.0) or 0.0),
            "rise_time": float(feats.get("rise_time", 0.0) or 0.0),
            "settling_time": float(feats.get("settling_time", 0.0) or 0.0),
            "steady_state_error": float(feats.get("steady_state_error", 0.0) or 0.0),
            "oscillation_index": float(feats.get("oscillation_index", 0.0) or 0.0),
            "dominant_frequency": float(feats.get("dominant_frequency", 0.0) or 0.0),
            "saturation_ratio": float(feats.get("saturation_ratio", 0.0) or 0.0),
            "vibration_severity": float(feats.get("vibration_severity", 0.0) or 0.0),
            "control_effort_variation": float(feats.get("control_effort_variation", 0.0) or 0.0),
            "innovation_ratio": float(feats.get("innovation_ratio", 0.0) or 0.0),
        }

        problem_flags, warning_flags = [], []
        for k, v in values.items():
            warn_k = k
            prob_k = f"{k}_problem"
            if prob_k not in tw or warn_k not in tw:
                continue
            sev = self._severity_label(v, tw[warn_k], tw[prob_k])
            if sev == "problem":
                problem_flags.append(k)
            elif sev == "warning":
                warning_flags.append(k)

        if problem_flags:
            severity = "problem"
        elif warning_flags:
            severity = "warning"
        else:
            severity = "normal"

        likely_causes, software_actions, hardware_actions, checks, tuning_candidates = [], [], [], [], []

        overshoot_hi = "overshoot" in warning_flags or "overshoot" in problem_flags
        oscillation_hi = "oscillation_index" in warning_flags or "oscillation_index" in problem_flags
        rise_slow = "rise_time" in warning_flags or "rise_time" in problem_flags
        sse_hi = "steady_state_error" in warning_flags or "steady_state_error" in problem_flags
        sat_hi = "saturation_ratio" in warning_flags or "saturation_ratio" in problem_flags
        vib_hi = "vibration_severity" in warning_flags or "vibration_severity" in problem_flags
        innov_hi = "innovation_ratio" in warning_flags or "innovation_ratio" in problem_flags

        if sat_hi:
            likely_causes += ["Actuator saturation / authority margin 부족 가능성"]
            tuning_candidates += ["actuator authority check first"]
            software_actions += ["Gain 재튜닝보다 mixer/allocation 제한 구간 우선 분석 권장"]
            hardware_actions += ["추력/서보 여유도(프로펠러/모터/서보 sizing) 점검 권장"]
            checks += ["Actuator output saturation ratio 및 지속 구간 확인"]

        if vib_hi:
            likely_causes += ["Vibration 영향으로 제어 응답 왜곡 가능성", "D-term noise amplification 가능성"]
            tuning_candidates += ["filter tuning first", "hardware vibration mitigation first"]
            software_actions += ["Notch/LPF 재설정 검토"]
            hardware_actions += ["IMU 장착/댐퍼/체결/구조 강성 점검 권장"]
            checks += ["Vibration spectrum 및 dominant peak 변화 확인"]

        if innov_hi:
            likely_causes += ["Estimator innovation 증가로 tracking error 유발 가능성"]
            tuning_candidates += ["estimator / sensor check first"]
            software_actions += ["EKF 파라미터/innovation gate 설정 점검 권장"]
            hardware_actions += ["센서 오프셋/배선/노이즈 경로 점검 권장"]
            checks += ["Innovation spike와 tracking error 동시 발생 여부 검증"]

        if overshoot_hi and not sat_hi:
            likely_causes += ["P 과대 또는 D damping 부족 가능성"]
            tuning_candidates += ["P decrease candidate", "D increase candidate"]
            software_actions += ["Rate P 감소 검토", "Rate D 증가 검토(노이즈 조건 확인 전제)"]
            checks += ["Step 응답 overshoot 및 settling time 재시험"]

        if rise_slow and not sat_hi:
            likely_causes += ["P 부족 또는 feedforward mismatch 가능성"]
            tuning_candidates += ["P increase candidate"]
            software_actions += ["Rate P 소폭 증가 검토", "Feedforward 보정 검토"]
            checks += ["Rise time 개선 여부 및 overshoot 부작용 확인"]
            # Lag without overshoot strongly suggests FF is too low.
            if not overshoot_hi:
                tuning_candidates += ["FF increase candidate"]
                software_actions += ["Feedforward gain 증가 검토 (rate lag 해소)"]

        if sse_hi and not sat_hi:
            likely_causes += ["I 부족 또는 trim/bias 영향 가능성"]
            tuning_candidates += ["I increase candidate"]
            software_actions += ["I gain 증가 검토", "trim/bias 보정 검토"]
            checks += ["steady-state error 재현성 및 외란 분리 검증"]

        if oscillation_hi and not vib_hi:
            tuning_candidates += ["P decrease candidate", "D increase candidate"]
            software_actions += ["Oscillation 구간에서 P 감소 또는 D 보강 검토"]

        if not likely_causes and severity == "normal":
            likely_causes = ["해당 축 응답 특성은 기준 범위 내"]
            checks = ["동일 maneuver 재시험 시 일관성 확인"]

        confidence = compute_tuning_confidence(
            sample_count=int(feats.get("sample_count", 0) or 0),
            maneuver_count=float(feats.get("maneuver_count", 0.0) or 0.0),
            saturation_ratio=float(values.get("saturation_ratio", 0.0) or 0.0),
            vibration_severity=float(values.get("vibration_severity", 0.0) or 0.0),
            innovation_ratio=float(values.get("innovation_ratio", 0.0) or 0.0),
            rules=self.rules,
        )

        # If maneuver basis is weak, force conservative guidance.
        if float(feats.get("maneuver_count", 0.0) or 0.0) < 1.0:
            confidence = min(confidence, 45.0)
            checks = self._dedup(["튜닝 분석 근거 부족: step-like maneuver 재비행 권장"] + checks)
            software_actions = self._dedup(["확정형 파라미터 변경 전 추가 계측/재비행 우선"] + software_actions)

        issue = {
            "issue_id": f"rate_loop_{axis}",
            "title": f"{axis.title()} Rate Loop Response",
            "severity": severity,
            "metric_value": {
                "overshoot_percent": round(values["overshoot"], 3),
                "rise_time_s": round(values["rise_time"], 3),
                "settling_time_s": round(values["settling_time"], 3),
                "steady_state_error": round(values["steady_state_error"], 5),
                "oscillation_index": round(values["oscillation_index"], 3),
                "dominant_frequency_hz": round(values["dominant_frequency"], 3),
                "saturation_ratio_percent": round(values["saturation_ratio"], 3),
                "vibration_rms": round(values["vibration_severity"], 3),
                "control_effort_variation": round(values["control_effort_variation"], 6),
            },
            "threshold_reference": {
                "warn": {
                    "overshoot_percent": tw["overshoot"],
                    "rise_time_s": tw["rise_time"],
                    "settling_time_s": tw["settling_time"],
                    "steady_state_error": tw["steady_state_error"],
                    "oscillation_index": tw["oscillation_index"],
                    "saturation_ratio_percent": tw["saturation_ratio"],
                    "vibration_rms": tw["vibration_severity"],
                    "innovation_ratio_percent": tw["innovation_ratio"],
                },
                "problem": {
                    "overshoot_percent": tw["overshoot_problem"],
                    "rise_time_s": tw["rise_time_problem"],
                    "settling_time_s": tw["settling_time_problem"],
                    "steady_state_error": tw["steady_state_error_problem"],
                    "oscillation_index": tw["oscillation_problem"],
                    "saturation_ratio_percent": tw["saturation_problem"],
                    "vibration_rms": tw["vibration_problem"],
                    "innovation_ratio_percent": tw["innovation_problem"],
                },
            },
            "segment": str(feats.get("segment", "step_like") or "step_like"),
            "evidence": (
                f"overshoot={values['overshoot']:.2f}%, rise={values['rise_time']:.2f}s, "
                f"settling={values['settling_time']:.2f}s, sse={values['steady_state_error']:.4f}, "
                f"osc={values['oscillation_index']:.2f}, dom={values['dominant_frequency']:.2f}Hz, "
                f"sat={values['saturation_ratio']:.2f}%, vib={values['vibration_severity']:.2f}, "
                f"innov={values['innovation_ratio']:.2f}%"
            ),
            "likely_causes": self._dedup(likely_causes),
            "software_actions": self._dedup(software_actions),
            "hardware_actions": self._dedup(hardware_actions),
            "verification_checklist": self._dedup(checks),
            "confidence": float(confidence),
            "tuning_candidates": self._dedup(tuning_candidates),
            "airframe": airframe,
        }

        advisor = {
            "axis": axis,
            "severity": severity,
            "confidence": float(confidence),
            "tuning_candidates": self._dedup(tuning_candidates),
            "short_cause": issue["likely_causes"][0] if issue["likely_causes"] else "",
            "short_action": (
                issue["software_actions"][0]
                if issue["software_actions"]
                else (issue["hardware_actions"][0] if issue["hardware_actions"] else "")
            ),
        }
        return issue, advisor

    def infer_attitude_loop_issue(self, axis: str, feats: Dict[str, float], airframe: str = "MULTICOPTER") -> Tuple[dict, dict]:
        """Attitude (angle) loop response inference.

        Mirrors infer_rate_loop_issue structure but uses attitude_loop thresholds
        and proposes MC_ROLL_P / MC_PITCH_P / MC_YAW_P (attitude P gains).
        """
        cfg = (self.rules.get("attitude_loop", {}) if isinstance(self.rules, dict) else {}) or {}
        tw = {
            "overshoot": float(cfg.get("overshoot_warn", 15.0)),
            "overshoot_problem": float(cfg.get("overshoot_problem", 30.0)),
            "rise_time": float(cfg.get("rise_time_warn", 0.8)),
            "rise_time_problem": float(cfg.get("rise_time_problem", 1.5)),
            "settling_time": float(cfg.get("settling_time_warn", 1.5)),
            "settling_time_problem": float(cfg.get("settling_time_problem", 3.0)),
            "attitude_mae_deg": float(cfg.get("attitude_mae_warn", 3.0)),
            "attitude_mae_deg_problem": float(cfg.get("attitude_mae_problem", 8.0)),
        }
        values = {
            "overshoot": float(feats.get("overshoot", 0.0) or 0.0),
            "rise_time": float(feats.get("rise_time", 0.0) or 0.0),
            "settling_time": float(feats.get("settling_time", 0.0) or 0.0),
            "attitude_mae_deg": float(feats.get("attitude_mae_deg", 0.0) or 0.0),
        }

        problem_flags, warning_flags = [], []
        for k, v in values.items():
            warn_k = k
            prob_k = f"{k}_problem"
            if prob_k not in tw or warn_k not in tw:
                continue
            sev = self._severity_label(v, tw[warn_k], tw[prob_k])
            if sev == "problem":
                problem_flags.append(k)
            elif sev == "warning":
                warning_flags.append(k)

        if problem_flags:
            severity = "problem"
        elif warning_flags:
            severity = "warning"
        else:
            severity = "normal"

        likely_causes, software_actions, hardware_actions, checks, tuning_candidates = [], [], [], [], []

        overshoot_hi = "overshoot" in warning_flags or "overshoot" in problem_flags
        rise_slow = "rise_time" in warning_flags or "rise_time" in problem_flags
        mae_hi = "attitude_mae_deg" in warning_flags or "attitude_mae_deg" in problem_flags

        sat_hi = float(feats.get("saturation_ratio", 0.0) or 0.0) >= 8.0
        vib_hi = float(feats.get("vibration_severity", 0.0) or 0.0) >= 3.5

        if sat_hi:
            likely_causes += ["Actuator/inner-loop saturation으로 attitude 응답 저하 가능성"]
            tuning_candidates += ["actuator authority check first"]
            software_actions += ["Rate-loop saturation 해소 후 attitude 게인 검토"]
        if vib_hi:
            likely_causes += ["Vibration이 attitude 측정/추정 품질에 영향 가능성"]
            tuning_candidates += ["hardware vibration mitigation first"]

        # Airframe-aware candidate texts:
        #   MC: higher P = faster response (P decrease => slower / overshoot 완화)
        #   FW: lower TC = faster response (TC increase => slower / overshoot 완화)
        is_fw = str(airframe or "").upper() == "FIXED_WING"
        slower_candidate = "attitude TC increase candidate" if is_fw else "attitude P decrease candidate"
        faster_candidate = "attitude TC decrease candidate" if is_fw else "attitude P increase candidate"
        slower_action = "Attitude TC 증가 검토 (응답 완화)" if is_fw else "Attitude P 감소 검토"
        faster_action = "Attitude TC 감소 검토 (응답 가속)" if is_fw else "Attitude P 소폭 증가 검토"

        if overshoot_hi and not sat_hi:
            likely_causes += ["Attitude 응답 과대 (gain 과대 또는 TC 과소 가능성)"]
            tuning_candidates += [slower_candidate]
            software_actions += [slower_action]
            checks += ["Attitude step 응답 overshoot 재시험"]
        if rise_slow and not sat_hi:
            likely_causes += ["Attitude 응답 느림 (gain 부족 또는 TC 과대 가능성)"]
            tuning_candidates += [faster_candidate]
            software_actions += [faster_action]
            checks += ["Attitude rise time 개선 여부 확인"]
        if mae_hi and not sat_hi and not overshoot_hi:
            likely_causes += ["전체 attitude tracking error 상승 — gain 부족/TC 과대 또는 외란 가능성"]
            tuning_candidates += [faster_candidate]
            checks += ["MAE 추세와 외란 구간 분리 검증"]

        if not likely_causes and severity == "normal":
            likely_causes = ["Attitude loop 응답 특성은 기준 범위 내"]
            checks = ["동일 maneuver 재시험 시 일관성 확인"]

        confidence = compute_tuning_confidence(
            sample_count=int(feats.get("sample_count", 0) or 0),
            maneuver_count=float(feats.get("maneuver_count", 0.0) or 0.0),
            saturation_ratio=float(feats.get("saturation_ratio", 0.0) or 0.0),
            vibration_severity=float(feats.get("vibration_severity", 0.0) or 0.0),
            innovation_ratio=float(feats.get("innovation_ratio", 0.0) or 0.0),
            rules=self.rules,
        )
        if float(feats.get("maneuver_count", 0.0) or 0.0) < 1.0:
            # No step-like maneuver: lean on MAE only, cap confidence.
            confidence = min(confidence, 45.0)
            checks = self._dedup(["Attitude step maneuver 부족: 재비행 권장"] + checks)

        issue = {
            "issue_id": f"attitude_loop_{axis}",
            "title": f"{axis.title()} Attitude Loop Response",
            "severity": severity,
            "metric_value": {
                "overshoot_percent": round(values["overshoot"], 3),
                "rise_time_s": round(values["rise_time"], 3),
                "settling_time_s": round(values["settling_time"], 3),
                "attitude_mae_deg": round(values["attitude_mae_deg"], 3),
            },
            "threshold_reference": {
                "warn": {
                    "overshoot_percent": tw["overshoot"],
                    "rise_time_s": tw["rise_time"],
                    "settling_time_s": tw["settling_time"],
                    "attitude_mae_deg": tw["attitude_mae_deg"],
                },
                "problem": {
                    "overshoot_percent": tw["overshoot_problem"],
                    "rise_time_s": tw["rise_time_problem"],
                    "settling_time_s": tw["settling_time_problem"],
                    "attitude_mae_deg": tw["attitude_mae_deg_problem"],
                },
            },
            "segment": str(feats.get("segment", "attitude_step") or "attitude_step"),
            "evidence": (
                f"overshoot={values['overshoot']:.2f}%, rise={values['rise_time']:.2f}s, "
                f"settling={values['settling_time']:.2f}s, mae={values['attitude_mae_deg']:.2f}deg"
            ),
            "likely_causes": self._dedup(likely_causes),
            "software_actions": self._dedup(software_actions),
            "hardware_actions": self._dedup(hardware_actions),
            "verification_checklist": self._dedup(checks),
            "confidence": float(confidence),
            "tuning_candidates": self._dedup(tuning_candidates),
            "airframe": airframe,
        }
        advisor = {
            "axis": axis,
            "severity": severity,
            "confidence": float(confidence),
            "tuning_candidates": self._dedup(tuning_candidates),
            "short_cause": issue["likely_causes"][0] if issue["likely_causes"] else "",
            "short_action": issue["software_actions"][0] if issue["software_actions"] else "",
        }
        return issue, advisor

    def infer_rate_limit_issue(self, axis: str, feats: Dict[str, float], airframe: str = "MULTICOPTER") -> Tuple[dict, dict]:
        cfg = (self.rules.get("rate_limit", {}) if isinstance(self.rules, dict) else {}) or {}
        warn = float(cfg.get("saturation_warn", 5.0))
        problem = float(cfg.get("saturation_problem", 15.0))
        sat_pct = feats.get("saturation_ratio_percent")
        param_present = bool(feats.get("param_present", False))

        if not param_present or sat_pct is None:
            severity = "unavailable"
        else:
            sat_pct = float(sat_pct)
            severity = self._severity_label(sat_pct, warn, problem)

        tuning_candidates, likely_causes, sw_actions, checks = [], [], [], []
        if severity in ("warning", "problem"):
            likely_causes += [
                f"Rate setpoint이 {feats.get('rate_max_param', 'rate MAX')}에 자주 도달 — "
                "공격적 maneuver 또는 attitude P 과대 가능성"
            ]
            tuning_candidates += ["rate MAX increase candidate", "attitude P decrease candidate"]
            sw_actions += [
                f"{feats.get('rate_max_param', 'rate MAX')} 상향 검토 (기체 인가 여유 확인 전제)",
                "Attitude P 감소 검토 (요구 rate를 줄이는 방향)",
            ]
            checks += ["동일 maneuver에서 rate saturation ratio 재측정", "actuator saturation 동반 여부 확인"]

        if severity == "normal":
            likely_causes = ["Rate setpoint이 MAX에 거의 도달하지 않음"]
            checks = ["mission 의도에 비해 MAX가 너무 크지는 않은지 별도 확인"]
        if severity == "unavailable":
            likely_causes = [
                f"{feats.get('rate_max_param', 'MC_*RATE_MAX')} 파라미터가 로그에 없어 판단 불가" if not param_present else "rate setpoint sample 부족"
            ]

        confidence = 75.0 if severity != "unavailable" else 0.0
        issue = {
            "issue_id": f"rate_limit_saturation_{axis}",
            "title": f"{axis.title()} Rate Setpoint Saturation",
            "severity": severity,
            "metric_value": {
                "saturation_ratio_percent": sat_pct if param_present else None,
                "rate_max_param": feats.get("rate_max_param"),
                "rate_max_value": feats.get("rate_max_value"),
                "max_abs_rate_setpoint": feats.get("max_abs_rate_setpoint"),
            },
            "threshold_reference": {"warn": {"saturation_ratio_percent": warn},
                                    "problem": {"saturation_ratio_percent": problem}},
            "segment": "rate_limit",
            "evidence": (
                f"sat={sat_pct:.2f}%, max_sp={feats.get('max_abs_rate_setpoint', 'N/A')}, "
                f"{feats.get('rate_max_param', 'N/A')}={feats.get('rate_max_value', 'N/A')}"
                if param_present else f"{feats.get('rate_max_param', 'N/A')} 미존재"
            ),
            "likely_causes": self._dedup(likely_causes),
            "software_actions": self._dedup(sw_actions),
            "hardware_actions": [],
            "verification_checklist": self._dedup(checks),
            "confidence": float(confidence),
            "tuning_candidates": self._dedup(tuning_candidates),
            "airframe": airframe,
        }
        advisor = {
            "axis": axis,
            "severity": severity,
            "confidence": float(confidence),
            "tuning_candidates": self._dedup(tuning_candidates),
            "short_cause": issue["likely_causes"][0] if issue["likely_causes"] else "",
            "short_action": issue["software_actions"][0] if issue["software_actions"] else "",
        }
        return issue, advisor

    def infer_acc_limit_issue(self, acc_metrics: Dict[str, dict], airframe: str = "MULTICOPTER") -> Tuple[dict, dict]:
        cfg = (self.rules.get("acc_limit", {}) if isinstance(self.rules, dict) else {}) or {}
        warn = float(cfg.get("saturation_warn", 5.0))
        problem = float(cfg.get("saturation_problem", 15.0))

        axes = ["horizontal", "up", "down"]
        per_axis = {}
        worst_sev = "unavailable"
        sev_rank = {"unavailable": 0, "normal": 1, "warning": 2, "problem": 3}
        for k in axes:
            entry = acc_metrics.get(k, {}) if isinstance(acc_metrics, dict) else {}
            present = bool(entry.get("param_present", False))
            sat = entry.get("saturation_ratio_percent")
            if not present or sat is None:
                sev = "unavailable"
            else:
                sev = self._severity_label(float(sat), warn, problem)
            per_axis[k] = {"severity": sev, "saturation_ratio_percent": sat, "param": entry.get("param"), "limit": entry.get("limit"), "max_cmd": entry.get("max_cmd")}
            if sev_rank[sev] > sev_rank[worst_sev]:
                worst_sev = sev

        severity = worst_sev
        tuning_candidates, likely_causes, sw_actions, checks = [], [], [], []
        if severity in ("warning", "problem"):
            for k, info in per_axis.items():
                if info["severity"] in ("warning", "problem") and info["param"]:
                    tuning_candidates.append(f"{info['param']} increase candidate")
                    sw_actions.append(f"{info['param']} 상향 검토 (mission profile 대비 추력/구조 여유 확인)")
            likely_causes += ["Position controller가 acceleration limit에 자주 도달 — mission profile이 limit보다 공격적"]
            checks += ["같은 mission에서 limit 상향 후 attitude/altitude tracking 영향 동시 관찰"]
        if severity == "normal":
            likely_causes = ["Acceleration command가 limit 대비 여유 있음"]
        if severity == "unavailable":
            likely_causes = ["acceleration setpoint 또는 MPC_ACC_* 파라미터 부재로 판단 불가"]

        confidence = 75.0 if severity in ("normal", "warning", "problem") else 0.0
        evidence_parts = []
        for k, v in per_axis.items():
            sat_val = v.get("saturation_ratio_percent")
            sat_txt = "N/A" if sat_val is None else f"{float(sat_val):.2f}%"
            evidence_parts.append(f"{k}={sat_txt} ({v.get('severity', 'unavailable')})")
        issue = {
            "issue_id": "acceleration_limit_saturation",
            "title": "Acceleration Limit Saturation",
            "severity": severity,
            "metric_value": per_axis,
            "threshold_reference": {"warn": {"saturation_ratio_percent": warn},
                                    "problem": {"saturation_ratio_percent": problem}},
            "segment": "acc_limit",
            "evidence": "; ".join(evidence_parts),
            "likely_causes": self._dedup(likely_causes),
            "software_actions": self._dedup(sw_actions),
            "hardware_actions": [],
            "verification_checklist": self._dedup(checks),
            "confidence": float(confidence),
            "tuning_candidates": self._dedup(tuning_candidates),
            "airframe": airframe,
        }
        advisor = {
            "axis": "acceleration",
            "severity": severity,
            "confidence": float(confidence),
            "tuning_candidates": self._dedup(tuning_candidates),
            "short_cause": issue["likely_causes"][0] if issue["likely_causes"] else "",
            "short_action": issue["software_actions"][0] if issue["software_actions"] else "",
        }
        return issue, advisor

    def infer_tilt_limit_issue(self, tilt_metrics: Dict[str, float], airframe: str = "MULTICOPTER") -> Tuple[dict, dict]:
        cfg = (self.rules.get("tilt_limit", {}) if isinstance(self.rules, dict) else {}) or {}
        warn = float(cfg.get("saturation_warn", 5.0))
        problem = float(cfg.get("saturation_problem", 15.0))
        present = bool(tilt_metrics.get("param_present", False))
        sat = tilt_metrics.get("saturation_ratio_percent")

        if not present or sat is None:
            severity = "unavailable"
        else:
            severity = self._severity_label(float(sat), warn, problem)

        tuning_candidates, likely_causes, sw_actions, checks = [], [], [], []
        if severity in ("warning", "problem"):
            likely_causes += ["Setpoint tilt이 MPC_TILTMAX_AIR에 자주 도달 — mission profile 과격 또는 limit 보수적"]
            tuning_candidates += ["MPC_TILTMAX_AIR increase candidate"]
            sw_actions += ["MPC_TILTMAX_AIR 상향 검토 (추력 여유와 안정성 동시 고려)"]
            checks += ["limit 상향 시 thrust margin과 attitude 응답 동시 관찰"]
        if severity == "normal":
            likely_causes = ["Setpoint tilt이 limit 대비 여유 있음"]
        if severity == "unavailable":
            likely_causes = ["MPC_TILTMAX_AIR 파라미터 부재 또는 attitude_setpoint 부재로 판단 불가"]

        confidence = 75.0 if severity != "unavailable" else 0.0
        issue = {
            "issue_id": "tilt_limit_saturation",
            "title": "Tilt Limit Saturation",
            "severity": severity,
            "metric_value": {
                "saturation_ratio_percent": sat if present else None,
                "limit_deg": tilt_metrics.get("limit_deg"),
                "max_tilt_deg": tilt_metrics.get("max_tilt_deg"),
            },
            "threshold_reference": {"warn": {"saturation_ratio_percent": warn},
                                    "problem": {"saturation_ratio_percent": problem}},
            "segment": "tilt_limit",
            "evidence": (
                f"sat={sat:.2f}%, max_tilt={tilt_metrics.get('max_tilt_deg', 'N/A')}deg, "
                f"limit={tilt_metrics.get('limit_deg', 'N/A')}deg"
                if present else "MPC_TILTMAX_AIR 또는 attitude_setpoint 부재"
            ),
            "likely_causes": self._dedup(likely_causes),
            "software_actions": self._dedup(sw_actions),
            "hardware_actions": [],
            "verification_checklist": self._dedup(checks),
            "confidence": float(confidence),
            "tuning_candidates": self._dedup(tuning_candidates),
            "airframe": airframe,
        }
        advisor = {
            "axis": "tilt",
            "severity": severity,
            "confidence": float(confidence),
            "tuning_candidates": self._dedup(tuning_candidates),
            "short_cause": issue["likely_causes"][0] if issue["likely_causes"] else "",
            "short_action": issue["software_actions"][0] if issue["software_actions"] else "",
        }
        return issue, advisor

    def infer_tecs_altitude_response_issue(self, feats: Dict[str, float], airframe: str = "FIXED_WING") -> Tuple[dict, dict]:
        cfg = (self.rules.get("tecs_altitude", {}) if isinstance(self.rules, dict) else {}) or {}
        warn = float(cfg.get("altitude_mae_warn", 4.0))
        problem = float(cfg.get("altitude_mae_problem", 10.0))

        mae = feats.get("altitude_mae_m") if isinstance(feats, dict) else None
        max_err = feats.get("altitude_max_abs_error_m") if isinstance(feats, dict) else None
        sample_count = int(feats.get("sample_count", 0) or 0) if isinstance(feats, dict) else 0

        if mae is None or sample_count < 30:
            severity = "unavailable"
        else:
            severity = self._severity_label(float(mae), warn, problem)

        tuning_candidates, likely_causes, sw_actions, checks = [], [], [], []
        if severity in ("warning", "problem"):
            likely_causes += [
                "TECS altitude tracking error 상승 — altitude time-constant 과대 또는 climb/sink limit 보수적 가능성"
            ]
            tuning_candidates += [
                "FW_T_ALT_TC decrease candidate",
                "FW_T_CLMB_MAX increase candidate",
                "FW_T_SINK_MIN increase candidate",
            ]
            sw_actions += [
                "FW_T_ALT_TC 감소 검토 (altitude loop 응답 가속)",
                "FW_T_CLMB_MAX/FW_T_SINK_MIN 상향 검토 (climb/descent 권한)",
            ]
            checks += [
                "climb segment에서 altitude error 변화 재측정",
                "throttle/pitch saturation 동시 발생 여부 확인",
            ]
        elif severity == "normal":
            likely_causes = ["TECS altitude tracking은 기준 범위 내"]
            checks = ["climb/descent 전환 구간에서 추세만 확인"]
        else:
            likely_causes = ["TECS altitude 데이터 부족 — forward flight 구간 부재 또는 sample 부족"]

        confidence = 75.0 if severity in ("normal", "warning", "problem") else 0.0
        issue = {
            "issue_id": "tecs_altitude_response",
            "title": "TECS Altitude Response",
            "severity": severity,
            "metric_value": {
                "altitude_mae_m": mae,
                "altitude_max_abs_error_m": max_err,
                "sample_count": sample_count,
            },
            "threshold_reference": {
                "warn": {"altitude_mae_m": warn},
                "problem": {"altitude_mae_m": problem},
            },
            "segment": "forward_flight",
            "evidence": (
                f"alt_mae={mae:.2f}m, max_abs_err={max_err:.2f}m, n={sample_count}"
                if mae is not None else "altitude tracking 데이터 부재"
            ),
            "likely_causes": self._dedup(likely_causes),
            "software_actions": self._dedup(sw_actions),
            "hardware_actions": [],
            "verification_checklist": self._dedup(checks),
            "confidence": float(confidence),
            "tuning_candidates": self._dedup(tuning_candidates),
            "airframe": airframe,
        }
        advisor = {
            "axis": "altitude",
            "severity": severity,
            "confidence": float(confidence),
            "tuning_candidates": self._dedup(tuning_candidates),
            "short_cause": issue["likely_causes"][0] if issue["likely_causes"] else "",
            "short_action": issue["software_actions"][0] if issue["software_actions"] else "",
        }
        return issue, advisor

    def infer_tecs_airspeed_response_issue(self, feats: Dict[str, float], airframe: str = "FIXED_WING") -> Tuple[dict, dict]:
        cfg = (self.rules.get("tecs_airspeed", {}) if isinstance(self.rules, dict) else {}) or {}
        mae_warn = float(cfg.get("airspeed_mae_warn", 2.0))
        mae_problem = float(cfg.get("airspeed_mae_problem", 5.0))
        res_warn = float(cfg.get("residual_rms_warn", 1.5))
        res_problem = float(cfg.get("residual_rms_problem", 3.5))

        mae = feats.get("airspeed_mae_mps") if isinstance(feats, dict) else None
        residual = feats.get("airspeed_residual_rms_mps") if isinstance(feats, dict) else None
        max_err = feats.get("airspeed_max_abs_error_mps") if isinstance(feats, dict) else None
        sample_count = int(feats.get("sample_count", 0) or 0) if isinstance(feats, dict) else 0
        source = str(feats.get("source", "") or "") if isinstance(feats, dict) else ""

        # Determine severity: prefer MAE if available, else residual RMS.
        if mae is not None and sample_count >= 30:
            severity = self._severity_label(float(mae), mae_warn, mae_problem)
        elif residual is not None and sample_count >= 30:
            severity = self._severity_label(float(residual), res_warn, res_problem)
        else:
            severity = "unavailable"

        tuning_candidates, likely_causes, sw_actions, checks = [], [], [], []
        if severity in ("warning", "problem"):
            likely_causes += [
                "TECS airspeed tracking error 상승 — speed time-constant 과대 또는 throttle damping 과대 가능성"
            ]
            tuning_candidates += [
                "FW_T_TAS_TC decrease candidate",
                "FW_T_SPDWEIGHT increase candidate",
                "FW_T_THR_DAMP decrease candidate",
            ]
            sw_actions += [
                "FW_T_TAS_TC 감소 검토 (airspeed loop 응답 가속)",
                "FW_T_SPDWEIGHT 상향 검토 (speed 우선도 강화)",
            ]
            checks += [
                "cruise segment에서 airspeed error 재측정",
                "pitch 명령 안정성과 throttle 사용량 동시 관찰",
                "pitot 보정/wind estimation 정합성 점검",
            ]
        elif severity == "normal":
            likely_causes = ["TECS airspeed tracking은 기준 범위 내"]
            checks = ["바람/외란 구간에서 추세만 확인"]
        else:
            likely_causes = ["TECS airspeed 데이터 부족 — tecs_status/airspeed 부재 또는 sample 부족"]

        confidence = 75.0 if severity in ("normal", "warning", "problem") else 0.0
        # Residual-RMS-only signal (fallback) is less reliable than direct MAE.
        if mae is None and residual is not None and confidence > 0:
            confidence = 55.0

        issue = {
            "issue_id": "tecs_airspeed_response",
            "title": "TECS Airspeed Response",
            "severity": severity,
            "metric_value": {
                "airspeed_mae_mps": mae,
                "airspeed_residual_rms_mps": residual,
                "airspeed_max_abs_error_mps": max_err,
                "sample_count": sample_count,
                "source": source,
            },
            "threshold_reference": {
                "warn": {"airspeed_mae_mps": mae_warn, "airspeed_residual_rms_mps": res_warn},
                "problem": {"airspeed_mae_mps": mae_problem, "airspeed_residual_rms_mps": res_problem},
            },
            "segment": "forward_flight",
            "evidence": (
                (f"mae={mae:.2f}m/s" if mae is not None else "")
                + (f", residual_rms={residual:.2f}m/s" if residual is not None else "")
                + (f", source={source}" if source else "")
                + (f", n={sample_count}" if sample_count else "")
            ).lstrip(", "),
            "likely_causes": self._dedup(likely_causes),
            "software_actions": self._dedup(sw_actions),
            "hardware_actions": [],
            "verification_checklist": self._dedup(checks),
            "confidence": float(confidence),
            "tuning_candidates": self._dedup(tuning_candidates),
            "airframe": airframe,
        }
        advisor = {
            "axis": "airspeed",
            "severity": severity,
            "confidence": float(confidence),
            "tuning_candidates": self._dedup(tuning_candidates),
            "short_cause": issue["likely_causes"][0] if issue["likely_causes"] else "",
            "short_action": issue["software_actions"][0] if issue["software_actions"] else "",
        }
        return issue, advisor

    def infer_vtol_transition_issue(self, transition_metrics: Dict[str, float], airframe: str = "VTOL") -> Tuple[dict, dict]:
        cfg = (self.rules.get("vtol_transition", {}) if isinstance(self.rules, dict) else {}) or {}
        dur_warn = float(cfg.get("duration_warn", 16.0))
        dur_problem = float(cfg.get("duration_problem", 24.0))
        alt_warn = float(cfg.get("alt_loss_warn", 8.0))
        alt_problem = float(cfg.get("alt_loss_problem", 18.0))
        inn_warn = float(cfg.get("innovation_warn", 5.0))
        inn_problem = float(cfg.get("innovation_problem", 10.0))

        duration = float(transition_metrics.get("duration", 0.0) or 0.0)
        alt_loss = float(transition_metrics.get("altitude_loss", 0.0) or 0.0)
        innov = float(transition_metrics.get("innovation_ratio", 0.0) or 0.0)
        duration_sev = self._severity_label(duration, dur_warn, dur_problem)
        loss_sev = self._severity_label(alt_loss, alt_warn, alt_problem)
        innov_sev = self._severity_label(innov, inn_warn, inn_problem)
        sev_rank = {"normal": 1, "warning": 2, "problem": 3}
        severity = max([duration_sev, loss_sev, innov_sev], key=lambda s: sev_rank.get(s, 0))

        likely_causes, sw, hw, checks, cands = [], [], [], [], []
        if sev_rank.get(duration_sev, 0) >= 2:
            likely_causes += ["Mode transition logic/schedule 영향 가능성", "Transition thrust schedule 보수/과소 가능성"]
            sw += ["Transition throttle schedule 재조정 검토", "Mode switch trigger 조건 재검토"]
            # Parameter-oriented candidates for the recommender.
            cands += [
                "VT_F_TRANS_THR increase candidate",
                "VT_F_TRANS_DUR increase candidate",
            ]
            checks += ["Transition duration 분포(최대/평균) 재측정"]
        if sev_rank.get(loss_sev, 0) >= 2:
            likely_causes += ["Transition pitch/thrust blending 불균형 가능성"]
            sw += ["Transition pitch target profile 완화 검토", "Transition thrust support 증대 검토"]
            hw += ["추력 여유도/기체 중량 밸런스 점검 권장"]
            cands += [
                "VT_F_TRANS_THR increase candidate",
            ]
            checks += ["Transition 중 altitude dip와 attitude overshoot 동시 관찰"]
        if sev_rank.get(innov_sev, 0) >= 2:
            likely_causes += ["Transition 구간 estimator innovation 스파이크 가능성"]
            sw += ["Transition 구간 EKF 파라미터/게이트 점검 권장"]
            hw += ["센서 진동/배선 노이즈 경로 점검 권장"]
            cands += ["estimator / sensor check first"]
            checks += ["Transition window에서 innovation spike 재검증"]

        # Use actual transition observation count and sample count when caller
        # provides them. Earlier this was hardcoded (sample_count=400,
        # maneuver_count=2.0) which inflated confidence on under-observed logs.
        sample_count_in = transition_metrics.get("sample_count")
        try:
            sample_count = int(sample_count_in) if sample_count_in is not None else 0
        except (TypeError, ValueError):
            sample_count = 0
        transition_count_in = transition_metrics.get("transition_count")
        try:
            transition_count = float(transition_count_in) if transition_count_in is not None else 0.0
        except (TypeError, ValueError):
            transition_count = 0.0

        confidence = compute_tuning_confidence(
            sample_count=sample_count,
            maneuver_count=transition_count,
            saturation_ratio=float(transition_metrics.get("saturation_ratio", 0.0) or 0.0),
            vibration_severity=float(transition_metrics.get("vibration_severity", 0.0) or 0.0),
            innovation_ratio=innov,
            rules=self.rules,
        )
        if transition_count < 1.0:
            confidence = min(confidence, 40.0)

        issue = {
            "issue_id": "vtol_transition_tuning",
            "title": "VTOL Transition Tuning",
            "severity": severity,
            "metric_value": {
                "transition_duration_s": round(duration, 3),
                "transition_altitude_loss_m": round(alt_loss, 3),
                "innovation_ratio_percent": round(innov, 3),
            },
            "threshold_reference": {
                "warn": {
                    "transition_duration_s": dur_warn,
                    "transition_altitude_loss_m": alt_warn,
                    "innovation_ratio_percent": inn_warn,
                },
                "problem": {
                    "transition_duration_s": dur_problem,
                    "transition_altitude_loss_m": alt_problem,
                    "innovation_ratio_percent": inn_problem,
                },
            },
            "segment": "transition",
            "evidence": f"duration={duration:.2f}s, altitude_loss={alt_loss:.2f}m, innovation={innov:.2f}%",
            "likely_causes": self._dedup(likely_causes),
            "software_actions": self._dedup(sw),
            "hardware_actions": self._dedup(hw),
            "verification_checklist": self._dedup(checks),
            "confidence": float(confidence),
            "tuning_candidates": self._dedup(cands),
            "airframe": airframe,
        }
        advisor = {
            "axis": "transition",
            "severity": severity,
            "confidence": float(confidence),
            "tuning_candidates": self._dedup(cands),
            "short_cause": issue["likely_causes"][0] if issue["likely_causes"] else "",
            "short_action": issue["software_actions"][0] if issue["software_actions"] else (issue["hardware_actions"][0] if issue["hardware_actions"] else ""),
        }
        return issue, advisor
