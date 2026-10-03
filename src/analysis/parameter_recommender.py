from __future__ import annotations

import copy
from typing import Dict, List, Optional


DEFAULT_SAFETY = {
    "max_delta_percent_per_step": 10.0,
    "min_confidence_to_recommend": 70.0,
    "block_when_candidates_contain": [
        "actuator authority check first",
        "hardware vibration mitigation first",
        "estimator / sensor check first",
        "filter tuning first",
    ],
    "risk_tier_multipliers": {
        "low": 1.0,
        "medium": 0.7,
        "high": 0.5,
    },
    "staged_apply": {
        "enabled": True,
        "first_step_fraction": 0.5,
        "note": (
            "[단계 적용 권장] 1단계: 권장 Δ의 50%만 반영하고 동일 maneuver를 재비행해 "
            "verification 항목 모두 정상 범위인지 확인 → 2단계: 나머지 50%를 반영하고 "
            "재검증. 단계 사이에 진동/EKF innovation 악화 시 즉시 롤백."
        ),
    },
    "vtol_fallback_note": (
        "[VTOL FALLBACK] MULTICOPTER 매핑 기반 추정값입니다. VTOL hover 동역학(긴 "
        "모멘트 암, 큰 관성, FW 표면 항력)이 다르므로 MC 권장값을 그대로 적용하지 "
        "말고 단계적 검증을 권장합니다."
    ),
}


class ParameterRecommender:
    """Map textual tuning candidates to conservative PX4 parameter suggestions.

    Output rows always contain text-level guidance (rationale, verification,
    safety_notes); concrete `suggested` values are emitted ONLY when:
        - confidence >= safety.min_confidence_to_recommend, AND
        - severity is "warning" or "problem", AND
        - no confounder-first candidate is present, AND
        - the matched action is NOT marked `advisory_only`, AND
        - the parameter is present in the log's current_params, AND
        - any per-action gate (require_metric_below) passes.

    `advisory_only` rows still appear in the result list with `suggested=None`
    so the pilot sees the rationale/verification but no number to copy-paste.

    Safety chain applied to numeric suggestions (in order):
        1. confidence floor
        2. confounder-first candidate block
        3. per-action gate
        4. severity_delta lookup
        5. risk_tier multiplier (shrinks |Δ| toward 1.0 for medium/high)
        6. ±max_delta_percent_per_step cap
        7. PX4 range clamp
        8. near-no-op drop (|Δ%|<0.5% AND not clamped)

    Airframe fallback: VTOL hover-mode issues fall back to the MULTICOPTER
    mapping for the same issue_id when no VTOL-specific row exists. Every
    fallback recommendation has the safety.vtol_fallback_note prepended to
    its `safety_notes` so the pilot can never miss the distinction.
    """

    AIRFRAME_FALLBACK = {"VTOL": "MULTICOPTER"}
    VALID_TIERS = ("low", "medium", "high")

    def __init__(self, config: Optional[dict] = None):
        cfg = config if isinstance(config, dict) else {}
        safety_in = cfg.get("safety", {}) if isinstance(cfg.get("safety", {}), dict) else {}
        self.safety = copy.deepcopy(DEFAULT_SAFETY)
        for key in ("max_delta_percent_per_step", "min_confidence_to_recommend"):
            if key in safety_in:
                try:
                    self.safety[key] = float(safety_in[key])
                except Exception:
                    pass
        if isinstance(safety_in.get("block_when_candidates_contain"), list):
            self.safety["block_when_candidates_contain"] = [
                str(x).strip().lower()
                for x in safety_in["block_when_candidates_contain"]
                if str(x).strip()
            ]
        tier_in = safety_in.get("risk_tier_multipliers")
        if isinstance(tier_in, dict):
            merged = dict(self.safety["risk_tier_multipliers"])
            for tier in self.VALID_TIERS:
                if tier in tier_in:
                    try:
                        merged[tier] = float(tier_in[tier])
                    except Exception:
                        pass
            self.safety["risk_tier_multipliers"] = merged
        staged_in = safety_in.get("staged_apply")
        if isinstance(staged_in, dict):
            merged_staged = dict(self.safety["staged_apply"])
            if "enabled" in staged_in:
                merged_staged["enabled"] = bool(staged_in["enabled"])
            if "first_step_fraction" in staged_in:
                try:
                    merged_staged["first_step_fraction"] = float(staged_in["first_step_fraction"])
                except Exception:
                    pass
            if "note" in staged_in and str(staged_in["note"]).strip():
                merged_staged["note"] = str(staged_in["note"]).strip()
            self.safety["staged_apply"] = merged_staged
        if "vtol_fallback_note" in safety_in and str(safety_in["vtol_fallback_note"]).strip():
            self.safety["vtol_fallback_note"] = str(safety_in["vtol_fallback_note"]).strip()

        self._index: Dict[tuple, Dict[str, List[dict]]] = {}
        for entry in cfg.get("mappings", []) if isinstance(cfg.get("mappings", []), list) else []:
            if not isinstance(entry, dict):
                continue
            airframe = str(entry.get("airframe", "")).strip().upper()
            issue_id = str(entry.get("issue_id", "")).strip()
            if not airframe or not issue_id:
                continue
            cands_in = entry.get("candidates", {})
            if not isinstance(cands_in, dict):
                continue
            cands_norm: Dict[str, List[dict]] = {}
            for cand_text, actions in cands_in.items():
                ck = str(cand_text).strip()
                if not ck:
                    continue
                if not isinstance(actions, list):
                    continue
                normed = []
                for action in actions:
                    if not isinstance(action, dict):
                        continue
                    param = str(action.get("param", "")).strip()
                    if not param:
                        continue
                    sev_delta = action.get("severity_delta", {}) if isinstance(action.get("severity_delta", {}), dict) else {}
                    rng = action.get("range", []) if isinstance(action.get("range", []), list) else []
                    rng_pair = None
                    if len(rng) == 2:
                        try:
                            rng_pair = (float(rng[0]), float(rng[1]))
                        except Exception:
                            rng_pair = None
                    gate = action.get("gate", {}) if isinstance(action.get("gate", {}), dict) else {}
                    require_below = gate.get("require_metric_below", {}) if isinstance(gate.get("require_metric_below", {}), dict) else {}
                    risk_tier = str(action.get("risk_tier", "medium")).strip().lower()
                    if risk_tier not in self.VALID_TIERS:
                        risk_tier = "medium"
                    advisory_only = bool(action.get("advisory_only", False))
                    normed.append(
                        {
                            "param": param,
                            "severity_delta": {
                                "warning": float(sev_delta.get("warning", 1.0)),
                                "problem": float(sev_delta.get("problem", 1.0)),
                            },
                            "range": rng_pair,
                            "rationale": str(action.get("rationale", "")).strip(),
                            "verification": str(action.get("verification", "")).strip(),
                            "require_metric_below": {str(k): float(v) for k, v in require_below.items()},
                            "risk_tier": risk_tier,
                            "advisory_only": advisory_only,
                        }
                    )
                if normed:
                    cands_norm[ck] = normed
            if cands_norm:
                self._index[(airframe, issue_id)] = cands_norm

    @staticmethod
    def _safe_float(v, default=None):
        try:
            return float(v)
        except Exception:
            return default

    def _candidates_blocked(self, candidates_lower: List[str]) -> Optional[str]:
        blockers = self.safety.get("block_when_candidates_contain", []) or []
        for blk in blockers:
            if blk in candidates_lower:
                return blk
        return None

    def _gate_blocked(self, action: dict, issue_metrics: dict) -> Optional[str]:
        for metric_key, threshold in action.get("require_metric_below", {}).items():
            actual = self._safe_float(issue_metrics.get(metric_key))
            if actual is None:
                return f"{metric_key} 측정값 부재로 게이트 검증 불가"
            if actual >= float(threshold):
                return f"{metric_key} {actual:.2f} ≥ {threshold:.2f} (게이트 미통과)"
        return None

    def _staged_apply_text(self, current: float, suggested: float) -> Optional[str]:
        staged = self.safety.get("staged_apply", {}) or {}
        if not staged.get("enabled", False):
            return None
        try:
            frac = float(staged.get("first_step_fraction", 0.5))
        except Exception:
            frac = 0.5
        frac = max(0.0, min(1.0, frac))
        first_step = current + (suggested - current) * frac
        note = str(staged.get("note", "")).strip()
        if note:
            return f"{note} (1단계 권장값 ≈ {first_step:g})"
        return f"단계 적용: 1단계 ≈ {first_step:g} → 검증 → 2단계 ≈ {suggested:g}"

    def recommend(
        self,
        issue: dict,
        current_params: Optional[dict],
        airframe: str,
    ) -> List[dict]:
        if not isinstance(issue, dict):
            return []
        issue_id = str(issue.get("issue_id", "")).strip()
        if not issue_id:
            return []
        airframe_key = str(airframe or issue.get("airframe", "")).strip().upper()
        if not airframe_key:
            return []

        cand_map = self._index.get((airframe_key, issue_id))
        used_fallback = False
        if not cand_map:
            fallback_af = self.AIRFRAME_FALLBACK.get(airframe_key)
            if fallback_af:
                cand_map = self._index.get((fallback_af, issue_id))
                if cand_map:
                    used_fallback = True
        if not cand_map:
            return []

        severity = str(issue.get("severity", "normal")).strip().lower()
        if severity not in ("warning", "problem"):
            return []

        confidence = self._safe_float(issue.get("confidence"), 0.0) or 0.0
        confidence_ok = confidence >= float(self.safety["min_confidence_to_recommend"])

        tuning_candidates_raw = issue.get("tuning_candidates", []) if isinstance(issue.get("tuning_candidates", []), list) else []
        candidates_lower = [str(x).strip().lower() for x in tuning_candidates_raw]
        blocked_by = self._candidates_blocked(candidates_lower)
        if blocked_by:
            return []

        params = current_params if isinstance(current_params, dict) else {}
        metric_value = issue.get("metric_value", {}) if isinstance(issue.get("metric_value", {}), dict) else {}

        max_delta_pct = float(self.safety.get("max_delta_percent_per_step", 10.0))
        cap_ratio_hi = 1.0 + (max_delta_pct / 100.0)
        cap_ratio_lo = 1.0 - (max_delta_pct / 100.0)
        tier_mults = self.safety.get("risk_tier_multipliers", {}) or {}
        vtol_note = str(self.safety.get("vtol_fallback_note", "")).strip()

        results: List[dict] = []
        seen_params = set()

        for cand_text in tuning_candidates_raw:
            cand_key = str(cand_text).strip()
            if not cand_key:
                continue
            actions = cand_map.get(cand_key)
            if not actions:
                continue
            for action in actions:
                param = action["param"]
                if param in seen_params:
                    continue

                safety_notes: List[str] = []
                if used_fallback and vtol_note:
                    safety_notes.append(vtol_note)

                advisory_only = bool(action.get("advisory_only", False))
                risk_tier = str(action.get("risk_tier", "medium")).lower()
                tier_mult = float(tier_mults.get(risk_tier, 1.0))

                gate_reason = self._gate_blocked(action, metric_value)

                # Advisory-only path: emit guidance row, no number.
                if advisory_only:
                    note_lines = [f"[ADVISORY ONLY] risk_tier={risk_tier} — 단일 로그 분석으로는 안전 적정값 자동 산출 불가."]
                    if gate_reason:
                        note_lines.append(f"게이트: {gate_reason}")
                    safety_notes.extend(note_lines)
                    results.append(
                        {
                            "param": param,
                            "current": self._safe_float(params.get(param)),
                            "suggested": None,
                            "delta_percent": None,
                            "rationale": action.get("rationale", ""),
                            "verification": action.get("verification", ""),
                            "candidate": cand_key,
                            "severity": severity,
                            "confidence": float(confidence),
                            "safety_notes": safety_notes,
                            "clamped": False,
                            "risk_tier": risk_tier,
                            "advisory_only": True,
                            "staged_apply": None,
                        }
                    )
                    seen_params.add(param)
                    continue

                # Numeric path requires param presence + confidence + gate pass.
                if param not in params:
                    continue
                current = self._safe_float(params.get(param))
                if current is None:
                    continue
                if not confidence_ok:
                    safety_notes.append(
                        f"신뢰도 {confidence:.1f}% < 최소 권고 임계 {float(self.safety['min_confidence_to_recommend']):.0f}% — 수치 권장 보류"
                    )
                    results.append(
                        {
                            "param": param,
                            "current": float(current),
                            "suggested": None,
                            "delta_percent": None,
                            "rationale": action.get("rationale", ""),
                            "verification": action.get("verification", ""),
                            "candidate": cand_key,
                            "severity": severity,
                            "confidence": float(confidence),
                            "safety_notes": safety_notes,
                            "clamped": False,
                            "risk_tier": risk_tier,
                            "advisory_only": False,
                            "staged_apply": None,
                        }
                    )
                    seen_params.add(param)
                    continue
                if gate_reason:
                    safety_notes.append(f"게이트 미통과: {gate_reason} — 수치 권장 보류")
                    results.append(
                        {
                            "param": param,
                            "current": float(current),
                            "suggested": None,
                            "delta_percent": None,
                            "rationale": action.get("rationale", ""),
                            "verification": action.get("verification", ""),
                            "candidate": cand_key,
                            "severity": severity,
                            "confidence": float(confidence),
                            "safety_notes": safety_notes,
                            "clamped": False,
                            "risk_tier": risk_tier,
                            "advisory_only": False,
                            "staged_apply": None,
                        }
                    )
                    seen_params.add(param)
                    continue

                sev_delta = action.get("severity_delta", {})
                raw_ratio = float(sev_delta.get(severity, sev_delta.get("warning", 1.0)))

                # risk_tier multiplier shrinks |Δ| toward 1.0.
                tiered_ratio = 1.0 + (raw_ratio - 1.0) * tier_mult
                if tier_mult < 1.0 and abs(raw_ratio - 1.0) > 1e-9:
                    safety_notes.append(
                        f"risk_tier={risk_tier} ({tier_mult:g}×) 적용: 룰 Δ {(raw_ratio - 1.0) * 100:+.1f}% → {(tiered_ratio - 1.0) * 100:+.1f}%"
                    )
                ratio = tiered_ratio

                # ±cap.
                if ratio > cap_ratio_hi:
                    safety_notes.append(
                        f"±{max_delta_pct:.0f}% 캡 적용 ({(ratio - 1.0) * 100:+.1f}% → +{max_delta_pct:.1f}%)"
                    )
                    ratio = cap_ratio_hi
                elif ratio < cap_ratio_lo:
                    safety_notes.append(
                        f"±{max_delta_pct:.0f}% 캡 적용 ({(ratio - 1.0) * 100:+.1f}% → -{max_delta_pct:.1f}%)"
                    )
                    ratio = cap_ratio_lo

                suggested = current * ratio

                # PX4 range clamp.
                clamped = False
                rng = action.get("range")
                if rng is not None:
                    lo, hi = rng
                    if suggested < lo:
                        safety_notes.append(f"PX4 range [{lo:g}, {hi:g}] 하한 clamp")
                        suggested = float(lo)
                        clamped = True
                    elif suggested > hi:
                        safety_notes.append(f"PX4 range [{lo:g}, {hi:g}] 상한 clamp")
                        suggested = float(hi)
                        clamped = True

                if current == 0.0:
                    delta_pct = 0.0 if suggested == 0.0 else float("inf")
                else:
                    delta_pct = (suggested - current) / current * 100.0

                if abs(delta_pct) < 0.5 and not clamped:
                    continue

                staged = self._staged_apply_text(float(current), float(suggested))
                if staged:
                    safety_notes.append(staged)

                results.append(
                    {
                        "param": param,
                        "current": float(current),
                        "suggested": float(suggested),
                        "delta_percent": round(float(delta_pct), 2),
                        "rationale": action.get("rationale", ""),
                        "verification": action.get("verification", ""),
                        "candidate": cand_key,
                        "severity": severity,
                        "confidence": float(confidence),
                        "safety_notes": safety_notes,
                        "clamped": bool(clamped),
                        "risk_tier": risk_tier,
                        "advisory_only": False,
                        "staged_apply": staged,
                    }
                )
                seen_params.add(param)

        return results
