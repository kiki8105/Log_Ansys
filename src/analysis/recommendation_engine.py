from __future__ import annotations


class RecommendationEngine:
    @staticmethod
    def _dedup(items):
        out = []
        for x in items:
            t = str(x).strip()
            if t and t not in out:
                out.append(t)
        return out

    @staticmethod
    def _severity_rank(sev: str) -> int:
        s = str(sev or "").lower()
        if s == "problem":
            return 3
        if s == "warning":
            return 2
        if s == "normal":
            return 1
        return 0

    def build(self, issues: list, overall_score=None, overall_status="unavailable"):
        issues = [x for x in (issues or []) if isinstance(x, dict)]
        issues_sorted = sorted(
            issues,
            key=lambda x: (self._severity_rank(x.get("severity")), float(x.get("confidence", 0.0))),
            reverse=True,
        )

        top_issues = []
        all_causes, sw_actions, hw_actions, checks, confs = [], [], [], [], []
        for it in issues_sorted:
            sev = str(it.get("severity", "normal"))
            cause = (it.get("likely_causes") or [])
            sw = (it.get("software_actions") or [])
            hw = (it.get("hardware_actions") or [])
            vc = (it.get("verification_checklist") or [])
            conf = it.get("confidence")
            try:
                confs.append(float(conf))
            except Exception:
                pass

            all_causes.extend(cause)
            sw_actions.extend(sw)
            hw_actions.extend(hw)
            checks.extend(vc)
            if len(top_issues) < 6:
                top_issues.append(
                    {
                        "issue_id": it.get("issue_id", ""),
                        "title": it.get("title", "Issue"),
                        "severity": sev,
                        "short_cause": cause[0] if cause else "",
                        "short_recommended_action": sw[0] if sw else (hw[0] if hw else ""),
                        "confidence": float(conf) if conf is not None else None,
                    }
                )

        conf_avg = round(sum(confs) / len(confs), 1) if confs else None

        return {
            "top_issues": top_issues,
            "likely_causes": self._dedup(all_causes)[:20],
            "recommended_software_actions": self._dedup(sw_actions)[:20],
            "recommended_hardware_actions": self._dedup(hw_actions)[:20],
            "verification_checklist": self._dedup(checks)[:20],
            "confidence_summary": {"average_confidence": conf_avg, "count": len(confs)},
            "score_summary": {"overall_score": overall_score, "overall_status": overall_status},
        }

