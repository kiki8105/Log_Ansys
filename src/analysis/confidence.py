import math


def clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, float(v)))


def compute_tuning_confidence(
    *,
    sample_count: int,
    maneuver_count: float,
    saturation_ratio: float,
    vibration_severity: float,
    innovation_ratio: float,
    rules: dict | None = None,
) -> float:
    """
    Confidence for tuning recommendations.
    Penalize under-observed maneuvers and confounding factors.
    """
    cfg = (rules or {}).get("confidence", {}) if isinstance(rules, dict) else {}
    base = float(cfg.get("base", 90.0))
    min_maneuvers = float(cfg.get("min_maneuvers", 2.0))
    min_samples = int(cfg.get("min_samples", 240))
    sat_warn = float(cfg.get("sat_warn", 8.0))
    sat_problem = float(cfg.get("sat_problem", 20.0))
    vib_warn = float(cfg.get("vib_warn", 3.5))
    vib_problem = float(cfg.get("vib_problem", 6.0))
    innov_warn = float(cfg.get("innov_warn", 5.0))
    innov_problem = float(cfg.get("innov_problem", 10.0))

    # Data sufficiency
    if sample_count < min_samples:
        base -= 18.0
    if maneuver_count < min_maneuvers:
        base -= 24.0
    elif maneuver_count < (min_maneuvers + 1.0):
        base -= 10.0

    # Confounders
    if math.isfinite(saturation_ratio):
        if saturation_ratio >= sat_problem:
            base -= 18.0
        elif saturation_ratio >= sat_warn:
            base -= 9.0

    if math.isfinite(vibration_severity):
        if vibration_severity >= vib_problem:
            base -= 14.0
        elif vibration_severity >= vib_warn:
            base -= 7.0

    if math.isfinite(innovation_ratio):
        if innovation_ratio >= innov_problem:
            base -= 14.0
        elif innovation_ratio >= innov_warn:
            base -= 7.0

    return round(clamp(base, 5.0, 99.0), 1)

