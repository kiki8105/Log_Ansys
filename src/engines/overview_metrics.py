# src/engines/overview_metrics.py
# (2026-05-28) Overview 탭용 비행 요약 metric 계산.
# 비교 다이얼로그 좌측 첫 탭에 PX4 Flight Review 상단 정보(Airframe / HW / SW /
# 비행 거리·속도·고도·tilt 등)를 한 화면으로 보여주기 위해 신설.
#
# 정책:
#  - 각 metric 은 폭넓은 try/except + None fallback → ulg 가 일부 토픽이 없어도
#    Overview 탭 전체가 깨지지 않게 함.
#  - 단위: 거리 m / 속도 km/h / 각도 deg — Flight Review 표기 동일.
#  - dict 반환 순서가 곧 표시 순서. (Python 3.7+ dict insertion order 보존)

import math
import datetime

import numpy as np


# PX4 sys_autostart 번호 → 사용자 친화적 이름. 전체 매핑은 아니고 대표 항목만.
# 미매핑은 "Autostart #{num}" 로 fallback.
AIRFRAME_NAME_MAP = {
    "1100": "Generic Plane",
    "2100": "Standard Plane",
    "4001": "Generic Quadcopter",
    "4011": "DJI F450 Quad",
    "4012": "HolyBro S500",
    "8001": "Generic Hex",
    "12001": "Generic Octo",
    "13000": "Generic Standard VTOL",
    "13001": "Caipirinha QuadPlane",
    "13002": "Convergence VTOL",
    "13013": "Deltaquad",
    "13030": "Standard VTOL",
    "13050": "Tiltrotor VTOL",
}


# ---- 안전 추출 / 포맷 헬퍼 -----------------------------------------------

def _safe_col(df, col):
    """polars DataFrame 컬럼 → numpy. 없거나 비면 None."""
    try:
        if df is None or col not in df.columns:
            return None
        arr = df[col].to_numpy()
        if arr is None or len(arr) == 0:
            return None
        return arr
    except Exception:
        return None


def _format_duration(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "N/A"
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"


def _format_life_time(seconds):
    # Flight Review 와 동일한 "1 hours 17 seconds" 표기.
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "N/A"
    seconds = int(round(seconds))
    h = seconds // 3600
    rem = seconds - h * 3600
    m = rem // 60
    s = rem % 60
    if h > 0 and m == 0:
        return f"{h} hours {s} seconds"
    if h > 0:
        return f"{h} hours {m} minutes {s} seconds"
    if m > 0:
        return f"{m} minutes {s} seconds"
    return f"{s} seconds"


def _format_datetime_from_utc_us(utc_us, offset_min=0):
    # (2026-05-28) offset_min: PX4 SDLOG_UTC_OFFSET (분). 한국=+540 → 9시간 더함.
    # Flight Review 와 동일하게 로컬 timezone 으로 변환해 표시.
    if utc_us is None or utc_us <= 0:
        return "N/A"
    try:
        # ``utcfromtimestamp`` is deprecated because it returns a naive UTC
        # value.  Keep the existing display semantics while making UTC
        # explicit; the configured PX4 offset is applied immediately below.
        dt = datetime.datetime.fromtimestamp(utc_us / 1e6, tz=datetime.timezone.utc)
        if offset_min:
            dt = dt + datetime.timedelta(minutes=int(offset_min))
        return dt.strftime("%d-%m-%Y %H:%M")
    except Exception:
        return "N/A"


def _decode_px4_version(rel_text):
    """PX4 ver_sw_release / sys_os_ver_release (0xAABBCCDD) → 'vA.B.C'."""
    rel = str(rel_text or "").strip()
    if not rel:
        return ""
    try:
        rel_int = int(rel, 0) if rel.lower().startswith("0x") else int(rel)
        major = (rel_int >> 24) & 0xFF
        minor = (rel_int >> 16) & 0xFF
        patch = (rel_int >> 8) & 0xFF
        return f"v{major}.{minor}.{patch}"
    except Exception:
        return rel


def _fmt_m(v):
    if v is None or not math.isfinite(v):
        return "N/A"
    return f"{v:.1f} m" if abs(v) < 100 else f"{v:.0f} m"


def _fmt_mps(mps):
    # (2026-05-28) PX4 / 사용자 요청: 속도 단위를 m/s 로 표기. Flight Review 기본은 km/h.
    if mps is None or not math.isfinite(mps):
        return "N/A"
    return f"{mps:.1f} m/s"


def _fmt_deg(v):
    if v is None or not math.isfinite(v):
        return "N/A"
    return f"{v:.1f} deg"


# ---- 메타데이터 라벨 -----------------------------------------------------

def _airframe_label(firmware, parameters):
    # (2026-05-28 fix) SYS_AUTOSTART 는 ULog msg_info_dict 가 아니라 initial_parameters
    # 에 들어있는 PX4 파라미터. firmware dict 에서 찾으면 항상 비어있어서 "Unknown" 으로 떨어짐.
    raw = str(parameters.get("SYS_AUTOSTART", "")).strip()
    if not raw:
        # 일부 ulg 는 msg_info 에도 sys_autostart 가 저장됨 — 폴백.
        raw = str(firmware.get("sys_autostart", "")).strip()
    if not raw:
        return "Unknown"
    # parameters 값은 "13000" 또는 "13000.0" 형태일 수 있어 정수 정규화.
    try:
        autostart = str(int(float(raw)))
    except Exception:
        autostart = raw
    name = AIRFRAME_NAME_MAP.get(autostart)
    if name is None:
        return f"Autostart #{autostart}"
    return f"{name} ({autostart})"


def _hardware_label(firmware):
    hw = str(firmware.get("ver_hw", "")).strip()
    sub = str(firmware.get("ver_hw_subtype", "")).strip()
    if hw and sub:
        return f"{hw} ({sub})"
    return hw or sub or "N/A"


def _software_label(firmware):
    rel = _decode_px4_version(firmware.get("ver_sw_release", ""))
    sw = str(firmware.get("ver_sw", "")).strip()
    label = rel or "N/A"
    if sw:
        label = f"{label} ({sw[:8]})" if rel else f"({sw[:8]})"
    return label


def _os_label(firmware):
    name = str(firmware.get("sys_os_name", "")).strip()
    ver = _decode_px4_version(
        firmware.get("sys_os_ver_release", "") or firmware.get("sys_os_ver", "")
    )
    if name and ver:
        return f"{name}, {ver}"
    return name or "N/A"


def _estimator_label(parameters, dataset):
    grp = str(parameters.get("SYS_MC_EST_GROUP", "")).strip()
    if grp == "2":
        return "EKF2"
    if grp == "1":
        return "LPE"
    # 토픽 추정. PX4 1.13+ 는 기본 EKF2.
    if dataset is not None and "estimator_status_0" in getattr(dataset, "topics", {}):
        return "EKF2"
    return "EKF2"


def _vehicle_uuid(firmware):
    return str(firmware.get("sys_uuid", "")).strip() or "N/A"


# ---- 시간 / dropout ------------------------------------------------------

def _logging_start(ulog, dataset, parameters):
    # 절대 UTC: 첫 vehicle_gps_position.time_utc_usec → SDLOG_UTC_OFFSET 적용 후 로컬 시간 표시.
    if dataset is None:
        return "N/A"
    offset_min = 0
    try:
        raw = str(parameters.get("SDLOG_UTC_OFFSET", "")).strip()
        if raw:
            offset_min = int(float(raw))
    except Exception:
        offset_min = 0
    for name in ("vehicle_gps_position_0", "vehicle_gps_position_1"):
        topic = dataset.topics.get(name) if hasattr(dataset, "topics") else None
        if topic is None:
            continue
        arr = _safe_col(topic.dataframe, "time_utc_usec")
        if arr is None:
            continue
        valid = arr[arr > 0]
        if len(valid) > 0:
            return _format_datetime_from_utc_us(int(valid[0]), offset_min)
    return "N/A"


def _logging_duration(ulog):
    try:
        return _format_duration((ulog.last_timestamp - ulog.start_timestamp) / 1e6)
    except Exception:
        return "N/A"


def _dropouts_label(ulog):
    try:
        drops = getattr(ulog, "dropouts", None) or []
        if not drops:
            return "0"
        total_ms = sum(float(getattr(d, "duration", 0)) for d in drops)
        return f"{len(drops)} ({total_ms / 1000:.2f} s)"
    except Exception:
        return "N/A"


def _armed_seconds(dataset):
    # 이번 비행의 armed 구간 시간 합산.
    if dataset is None:
        return None
    for name in ("actuator_armed_0", "actuator_armed_1"):
        topic = dataset.topics.get(name) if hasattr(dataset, "topics") else None
        if topic is None:
            continue
        df = topic.dataframe
        ts = _safe_col(df, "timestamp")
        armed = _safe_col(df, "armed")
        if ts is None or armed is None or len(ts) < 2:
            continue
        ts_sec = ts.astype("float64") / 1e6
        armed_bool = armed.astype("bool")
        dt = np.diff(ts_sec)
        return float(np.sum(dt[armed_bool[:-1]]))
    return None


def _vehicle_life_seconds(parameters, current_armed_sec):
    # (2026-05-28) PX4 가 LND_FLIGHT_T_HI/LO 파라미터에 부팅 시점까지의 누적 비행 시간을
    # µs 단위 64비트로 저장 (land_detector 가 land 이벤트마다 갱신). ulg initial_parameters
    # 는 부팅 직후 값이므로 거기에 이번 비행 armed 시간을 더하면 비행 종료 시점의 누적값.
    # Flight Review 의 'Vehicle Life Flight Time' 과 동일 산식.
    try:
        hi_raw = str(parameters.get("LND_FLIGHT_T_HI", "")).strip()
        lo_raw = str(parameters.get("LND_FLIGHT_T_LO", "")).strip()
        hi = int(float(hi_raw)) if hi_raw else 0
        lo = int(float(lo_raw)) if lo_raw else 0
        # 32비트 분할 저장 → 64비트 결합. lo 가 음수로 해석될 수 있어 mask.
        boot_us = (hi << 32) | (lo & 0xFFFFFFFF)
        boot_sec = boot_us / 1e6 if boot_us > 0 else 0.0
    except Exception:
        boot_sec = 0.0
    cur = current_armed_sec if current_armed_sec else 0.0
    total = boot_sec + cur
    return total if total > 0 else None


# ---- 거리 / 고도 / 속도 / tilt -------------------------------------------

def _distance_and_altdiff(dataset):
    topic = dataset.topics.get("vehicle_local_position_0") if dataset else None
    if topic is None:
        return None, None
    df = topic.dataframe
    x = _safe_col(df, "x")
    y = _safe_col(df, "y")
    z = _safe_col(df, "z")
    distance = None
    alt_diff = None
    if x is not None and y is not None and len(x) > 1:
        dx = np.diff(x.astype("float64"))
        dy = np.diff(y.astype("float64"))
        seg = np.sqrt(dx * dx + dy * dy)
        seg = seg[np.isfinite(seg)]
        if len(seg) > 0:
            distance = float(np.sum(seg))
    if z is not None and len(z) > 0:
        zf = z.astype("float64")
        finite = zf[np.isfinite(zf)]
        if len(finite) > 0:
            alt_diff = float(np.max(finite) - np.min(finite))
    return distance, alt_diff


def _speed_metrics(dataset):
    # 반환: (avg_mc, max_3d, max_horiz, max_up, max_down) — 모두 m/s.
    topic = dataset.topics.get("vehicle_local_position_0") if dataset else None
    if topic is None:
        return (None, None, None, None, None)
    df = topic.dataframe
    vx = _safe_col(df, "vx")
    vy = _safe_col(df, "vy")
    vz = _safe_col(df, "vz")
    if vx is None or vy is None:
        return (None, None, None, None, None)
    vx = vx.astype("float64")
    vy = vy.astype("float64")
    horiz = np.sqrt(vx * vx + vy * vy)
    horiz_finite = horiz[np.isfinite(horiz)]
    max_horiz = float(np.max(horiz_finite)) if len(horiz_finite) else None
    avg_mc = float(np.mean(horiz_finite)) if len(horiz_finite) else None
    if vz is not None:
        vz = vz.astype("float64")
        v3 = np.sqrt(vx * vx + vy * vy + vz * vz)
        v3f = v3[np.isfinite(v3)]
        max_3d = float(np.max(v3f)) if len(v3f) else max_horiz
        vzf = vz[np.isfinite(vz)]
        # NED 기준: -vz = up, +vz = down.
        max_up = float(np.max(-vzf)) if len(vzf) else None
        max_down = float(np.max(vzf)) if len(vzf) else None
        if max_up is not None and max_up < 0:
            max_up = 0.0
        if max_down is not None and max_down < 0:
            max_down = 0.0
    else:
        max_3d = max_horiz
        max_up = None
        max_down = None
    return avg_mc, max_3d, max_horiz, max_up, max_down


def _max_tilt_deg(dataset):
    # tilt = arccos(R[2,2]) = arccos(1 - 2*(q1^2 + q2^2)).
    topic = dataset.topics.get("vehicle_attitude_0") if dataset else None
    if topic is None:
        return None
    df = topic.dataframe
    q1 = _safe_col(df, "q[1]")
    q2 = _safe_col(df, "q[2]")
    if q1 is None or q2 is None:
        return None
    q1 = q1.astype("float64")
    q2 = q2.astype("float64")
    cos_tilt = np.clip(1.0 - 2.0 * (q1 * q1 + q2 * q2), -1.0, 1.0)
    tilt = np.degrees(np.arccos(cos_tilt))
    finite = tilt[np.isfinite(tilt)]
    if len(finite) == 0:
        return None
    return float(np.max(finite))


# ---- 진입점 --------------------------------------------------------------

# Overview 탭에 표시할 행 순서 (외부에서 import 해 _populate_overview_table 이 사용).
OVERVIEW_ROW_ORDER = [
    "Airframe",
    "Hardware",
    "Software Version",
    "OS Version",
    "Estimator",
    "Logging Start",
    "Logging Duration",
    "Dropouts",
    "Vehicle Life Flight Time",
    "Vehicle UUID",
    "Distance",
    "Max Altitude Difference",
    "Average Speed MC",
    "Max Speed",
    "Max Speed Horizontal",
    "Max Speed Up",
    "Max Speed Down",
    "Max Tilt Angle",
]


def build_overview(ulog, dataset, firmware, parameters):
    """비교 다이얼로그 Overview 탭에 표시할 (key→display text) dict 반환.

    - ulog: pyulog.ULog 객체
    - dataset: core.log_model.LogDataset
    - firmware: _extract_log_metadata_for_dataset 가 만든 firmware dict
    - parameters: 같은 함수가 만든 parameters dict
    """
    firmware = firmware or {}
    parameters = parameters or {}

    try:
        distance, alt_diff = _distance_and_altdiff(dataset)
    except Exception:
        distance, alt_diff = None, None
    try:
        avg_mc, max_3d, max_horiz, max_up, max_down = _speed_metrics(dataset)
    except Exception:
        avg_mc = max_3d = max_horiz = max_up = max_down = None
    try:
        max_tilt = _max_tilt_deg(dataset)
    except Exception:
        max_tilt = None
    try:
        armed_sec = _armed_seconds(dataset)
    except Exception:
        armed_sec = None
    try:
        life_sec = _vehicle_life_seconds(parameters, armed_sec)
    except Exception:
        life_sec = armed_sec

    return {
        "Airframe":                 _airframe_label(firmware, parameters),
        "Hardware":                 _hardware_label(firmware),
        "Software Version":         _software_label(firmware),
        "OS Version":               _os_label(firmware),
        "Estimator":                _estimator_label(parameters, dataset),
        "Logging Start":            _logging_start(ulog, dataset, parameters),
        "Logging Duration":         _logging_duration(ulog),
        "Dropouts":                 _dropouts_label(ulog),
        "Vehicle Life Flight Time": _format_life_time(life_sec),
        "Vehicle UUID":             _vehicle_uuid(firmware),
        "Distance":                 _fmt_m(distance),
        "Max Altitude Difference":  _fmt_m(alt_diff),
        "Average Speed MC":         _fmt_mps(avg_mc),
        "Max Speed":                _fmt_mps(max_3d),
        "Max Speed Horizontal":     _fmt_mps(max_horiz),
        "Max Speed Up":             _fmt_mps(max_up),
        "Max Speed Down":           _fmt_mps(max_down),
        "Max Tilt Angle":           _fmt_deg(max_tilt),
    }
