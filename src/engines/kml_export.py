# src/engines/kml_export.py
# (2026-05-29) 비행 데이터 → KMZ export → Google Earth Pro에서 열기.
#
# KMZ 안 구성:
#  - doc.kml (XML)
#    - LineString: 비행 경로 (lon, lat, alt) 시계열
#      * alt는 이륙고도 기준 AGL(relative)로 변환
#      * KML altitudeMode는 relativeToGround 사용
#      * vehicle_status.nav_state 기준으로 경로를 구간 분할하고 모드별 색상 적용
#    - Placemark: Takeoff (녹색 핀), Landing (적색 핀)
#    - Document description: 비행 요약
#  - 핀 아이콘은 Google hosted (http://maps.google.com/...) — KMZ 안에 이미지 없음.
#
# 좌표 sampling: 너무 촘촘하면 Google Earth가 느려지므로 ~2000 points로 thin.

import math
import zipfile
from xml.sax.saxutils import escape as xml_escape

import numpy as np


# Google Earth Pro 호환 KML namespace.
KML_NS = "http://www.opengis.net/kml/2.2"

# 색상: KML은 ABGR hex (Alpha, Blue, Green, Red). 일반 RGB와 순서가 다름.
LINE_COLOR_ABGR = "ff0080ff"  # fallback 주황
TAKEOFF_ICON = "http://maps.google.com/mapfiles/kml/paddle/grn-blank.png"
LANDING_ICON = "http://maps.google.com/mapfiles/kml/paddle/red-blank.png"

# 경로 sampling 한계 — Google Earth가 50k+ 점에서 무거워짐.
MAX_PATH_POINTS = 2000

# Google Earth 표시 방식.
# relativeToGround: KML 고도값을 Google Earth 지형면 기준 상대고도(AGL)로 해석.
KML_ALTITUDE_MODE = "relativeToGround"

# PX4 vehicle_status.nav_state 값 표시명.
# 버전/브랜치에 따라 일부 enum은 다를 수 있으므로 Unknown fallback을 유지한다.
NAV_STATE_NAMES = {
    0: "Manual",
    1: "Altitude",
    2: "Position",
    3: "Mission",
    4: "Loiter",
    5: "RTL",
    6: "Acro",
    7: "Offboard",
    8: "Stabilized",
    9: "Rattitude",
    10: "Takeoff",
    11: "Land",
    12: "Follow Target",
    13: "Precision Land",
    14: "Orbit",
    15: "VTOL Takeoff",
    16: "Descend",
    17: "Terminate",
    18: "External 1",
    19: "External 2",
    20: "External 3",
    21: "External 4",
    22: "External 5",
    23: "External 6",
    24: "External 7",
    25: "External 8",
}

# 모드별 경로 색상. 값이 없으면 MODE_COLOR_DEFAULT 사용.
# 사용자는 일반 웹 색상 #RRGGBB 형식으로 커스텀해도 된다.
# KML 출력 시 _to_kml_abgr_color()에서 AABBGGRR 형식으로 자동 변환한다.

"""         return {
            0: "#6B7280",   # Manual       (gray, 회색)
            1: "#F97316",   # Altctl       (orange, 주황)
            2: "#2563EB",   # Posctl       (blue, 파랑)
            3: "#22C55E",   # Mission      (green, 초록)
            4: "#C4B5FD",   # Loiter/Hold  (lavender, 연보라)
            5: "#8B5CF6",   # RTL          (violet, 보라)
            6: "#0EA5E9",   # PositionSlow
            10: "#DC2626",  # Acro         (was orange; moved → red to avoid Altctl 충돌)
            12: "#F59E0B",  # Descend
            13: "#111827",  # Termination
            14: "#0891B2",  # Offboard     (was green; moved → teal to avoid Mission 충돌)
            15: "#EAB308",  # Stab         (yellow, 노랑)
            17: "#F97316",  # Takeoff (orange — Altctl 와 동일 색이지만 Takeoff 시 동시 출현 가능성 매우 낮음)
            18: "#84CC16",  # Land
            19: "#06B6D4",  # Follow
            20: "#A855F7",  # Precland
            21: "#EC4899",  # Orbit
            22: "#FB923C",  # VTOL Takeoff """
MODE_COLORS_ABGR = {
    0: "#6B7280",   # Manual       (gray, 회색)
    1: "#F97316",   # Altctl       (orange, 주황)
    2: "#2563EB",   # Posctl       (blue, 파랑)
    3: "#22C55E",   # Mission      (green, 초록)
    4: "#C4B5FD",   # Loiter/Hold  (lavender, 연보라)
    5: "#8B5CF6",   # RTL          (violet, 보라)
    6: "#0EA5E9",   # PositionSlow
    10: "#DC2626",  # Acro         (was orange; moved → red to avoid Altctl 충돌)
    12: "#F59E0B",  # Descend
    13: "#111827",  # Termination
    14: "#0891B2",  # Offboard     (was green; moved → teal to avoid Mission 충돌)
    15: "#EAB308",  # Stab         (yellow, 노랑)
    17: "#F97316",  # Takeoff (orange — Altctl 와 동일 색이지만 Takeoff 시 동시 출현 가능성 매우 낮음)
    18: "#84CC16",  # Land
    19: "#06B6D4",  # Follow
    20: "#A855F7",  # Precland
    21: "#EC4899",  # Orbit
    22: "#FB923C",  # VTOL Takeoff
}
MODE_COLOR_DEFAULT = "ffffffff"  # Unknown: white



def _to_kml_abgr_color(color, fallback="ffffffff"):
    """색상 문자열을 Google Earth KML <color>용 AABBGGRR로 변환.

    지원 입력:
      - "#RRGGBB"   : 일반 웹 색상. 예: "#2563EB"
      - "RRGGBB"    : 일반 웹 색상. 예: "2563EB"
      - "AABBGGRR"  : 이미 KML ABGR 형식. 예: "ffff8000"

    KML 색상 순서는 일반 RGB가 아니라 AABBGGRR이다.
    잘못된 값이 들어오면 fallback을 사용한다.
    """
    try:
        if color is None:
            return fallback

        s = str(color).strip()
        if s.startswith("#"):
            s = s[1:]

        if len(s) == 6:
            # RRGGBB -> ffBBGGRR
            rr = s[0:2]
            gg = s[2:4]
            bb = s[4:6]
            int(rr + gg + bb, 16)  # validation
            return f"ff{bb}{gg}{rr}".lower()

        if len(s) == 8:
            # 이미 KML AABBGGRR로 들어온 값은 그대로 사용.
            int(s, 16)  # validation
            return s.lower()

    except Exception:
        pass

    return fallback


def _safe_col(df, col):
    try:
        if df is None or col not in df.columns:
            return None
        arr = df[col].to_numpy()
        return arr if arr is not None and len(arr) else None
    except Exception:
        return None


def _first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_lat_lon_alt(df):
    """DataFrame에서 lat/lon/alt를 읽어 KML용 deg/m 단위 ndarray로 정규화."""
    lat_col = _first_col(df, ("lat", "latitude_deg"))
    lon_col = _first_col(df, ("lon", "longitude_deg"))
    if not lat_col or not lon_col:
        return None, None, None

    lat = _safe_col(df, lat_col)
    lon = _safe_col(df, lon_col)
    if lat is None or lon is None:
        return None, None, None

    lat = lat.astype("float64")
    lon = lon.astype("float64")

    # PX4 vehicle_global_position.lat/lon 또는 vehicle_gps_position.lat/lon은
    # int * 1e7 형식일 수 있으므로 자동 감지.
    if np.nanmax(np.abs(lat)) > 1000:
        lat = lat / 1e7
        lon = lon / 1e7

    alt_col = _first_col(df, ("alt", "altitude_m", "altitude"))
    alt = _safe_col(df, alt_col) if alt_col else None
    if alt is not None:
        alt = alt.astype("float64")
        # vehicle_gps_position.alt가 mm 단위인 경우 자동 감지.
        if np.nanmax(np.abs(alt)) > 100000:
            alt = alt / 1000.0
    else:
        alt = np.zeros_like(lat)

    return lat, lon, alt


def _get_global_position_topic(dataset):
    if dataset is None or not hasattr(dataset, "topics"):
        return None
    topic = dataset.topics.get("vehicle_global_position_0")
    if topic is None:
        topic = dataset.topics.get("vehicle_gps_position_0")
    return topic


def extract_flight_path_with_time(dataset):
    """비행 경로 좌표와 timestamp를 반환.

    반환:
      coords: ndarray (N, 3), 좌표 순서 (lon, lat, alt_m)
      ts_us:  ndarray (N,), PX4 timestamp [us]

    alt_m은 아직 원 로그 기준 고도이며, 이후 convert_to_takeoff_agl()에서
    이륙고도 기준 AGL로 변환한다.
    """
    topic = _get_global_position_topic(dataset)
    if topic is None:
        return None, None

    df = topic.dataframe
    lat, lon, alt = _normalize_lat_lon_alt(df)
    ts = _safe_col(df, "timestamp")
    if lat is None or lon is None or alt is None or ts is None:
        return None, None

    valid = (
        np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt)
        & (lat != 0) & (lon != 0)
    )
    if not valid.any():
        return None, None

    lat = lat[valid]
    lon = lon[valid]
    alt = alt[valid]
    ts = ts.astype("float64")[valid]

    coords = np.stack([lon, lat, alt], axis=1)
    return coords, ts


def extract_flight_path(dataset):
    """기존 호출 호환용: 좌표만 반환."""
    coords, _ = extract_flight_path_with_time(dataset)
    return coords


def thin_path(coords, max_points=MAX_PATH_POINTS):
    n = len(coords)
    if n <= max_points:
        return coords
    step = int(math.ceil(n / max_points))
    return coords[::step]


def thin_path_with_aux(coords, aux, max_points=MAX_PATH_POINTS):
    """좌표와 동일 길이의 보조 배열(nav_state 등)을 같은 step으로 thinning."""
    n = len(coords)
    if n <= max_points:
        return coords, aux
    step = int(math.ceil(n / max_points))
    return coords[::step], aux[::step]


def convert_to_takeoff_agl(coords, takeoff=None):
    """원 고도값을 이륙고도 기준 AGL로 변환.

    - 기준 고도는 우선 takeoff placemark의 alt 사용
    - takeoff가 없으면 coords 첫 번째 유효점의 alt 사용
    - Google Earth relativeToGround와 함께 쓰기 위해 지면 기준 상대고도처럼 표시
    """
    if coords is None or len(coords) == 0:
        return coords, None

    coords_agl = coords.astype("float64", copy=True)

    if takeoff is not None and len(takeoff) >= 3 and np.isfinite(takeoff[2]):
        home_alt_m = float(takeoff[2])
    else:
        home_alt_m = float(coords_agl[0, 2])

    coords_agl[:, 2] = coords_agl[:, 2] - home_alt_m

    # Google Earth에서 지형 아래로 파고드는 현상을 줄이기 위한 방어.
    # 실제 로그가 이륙지점보다 낮아지는 구간을 보고 싶으면 아래 줄을 주석 처리한다.
    coords_agl[:, 2] = np.maximum(coords_agl[:, 2], 0.0)

    return coords_agl, home_alt_m


def detect_takeoff_landing(dataset, fallback_path=None):
    """actuator_armed.armed True 첫/마지막 시점의 (lon, lat, alt). 없으면 path의 첫/끝."""
    takeoff = None
    landing = None

    if dataset is not None and hasattr(dataset, "topics"):
        armed_topic = dataset.topics.get("actuator_armed_0")
        gp_topic = dataset.topics.get("vehicle_global_position_0")
        if armed_topic is not None and gp_topic is not None:
            ad = armed_topic.dataframe
            gd = gp_topic.dataframe
            armed = _safe_col(ad, "armed")
            ats = _safe_col(ad, "timestamp")
            if armed is not None and ats is not None and len(ats) >= 1:
                armed_bool = armed.astype("bool")
                armed_idx = np.where(armed_bool)[0]
                if len(armed_idx) >= 1:
                    t_takeoff = ats[armed_idx[0]]
                    t_landing = ats[armed_idx[-1]]
                    takeoff = _sample_global_at_time(gd, t_takeoff)
                    landing = _sample_global_at_time(gd, t_landing)

    if (takeoff is None or landing is None) and fallback_path is not None and len(fallback_path) >= 2:
        if takeoff is None:
            takeoff = tuple(fallback_path[0])
        if landing is None:
            landing = tuple(fallback_path[-1])

    return takeoff, landing


def _sample_global_at_time(gd, t_us):
    lat, lon, alt = _normalize_lat_lon_alt(gd)
    if lat is None or lon is None or alt is None:
        return None

    ts = _safe_col(gd, "timestamp")
    if ts is None:
        return None

    idx = int(np.argmin(np.abs(ts.astype("float64") - float(t_us))))
    if not (np.isfinite(lat[idx]) and np.isfinite(lon[idx]) and np.isfinite(alt[idx])):
        return None

    return (float(lon[idx]), float(lat[idx]), float(alt[idx]))


def get_nav_state_for_times(dataset, target_ts_us):
    """global_position timestamp마다 vehicle_status.nav_state를 매칭.

    매칭 방식:
      - vehicle_status.timestamp가 target timestamp 이하인 가장 최근 nav_state 사용
      - target이 첫 vehicle_status보다 빠르면 첫 nav_state 사용
      - vehicle_status/nav_state가 없으면 None 반환
    """
    if dataset is None or not hasattr(dataset, "topics") or target_ts_us is None:
        return None

    topic = dataset.topics.get("vehicle_status_0")
    if topic is None:
        return None

    df = topic.dataframe
    status_ts = _safe_col(df, "timestamp")
    nav_state = _safe_col(df, "nav_state")
    if status_ts is None or nav_state is None or len(status_ts) == 0:
        return None

    status_ts = status_ts.astype("float64")
    nav_state = nav_state.astype("int64")

    # timestamp 정렬 보장. 로그가 보통 정렬되어 있지만 방어적으로 처리한다.
    order = np.argsort(status_ts)
    status_ts = status_ts[order]
    nav_state = nav_state[order]

    idx = np.searchsorted(status_ts, target_ts_us.astype("float64"), side="right") - 1
    idx = np.clip(idx, 0, len(nav_state) - 1)
    return nav_state[idx]


# ---- KML 생성 -----------------------------------------------------------

def _coords_to_str(coords):
    # KML <coordinates>: 'lon,lat,alt lon,lat,alt ...' (공백 또는 줄바꿈).
    parts = [f"{lon:.7f},{lat:.7f},{alt:.2f}" for lon, lat, alt in coords]
    return "\n          ".join(parts)


def _nav_name(nav_value):
    try:
        nav_int = int(nav_value)
    except Exception:
        return "Unknown"
    return NAV_STATE_NAMES.get(nav_int, f"Unknown({nav_int})")


def _style_id_for_nav(nav_value):
    try:
        return f"modeNav{int(nav_value)}"
    except Exception:
        return "modeNavUnknown"


def _split_coords_by_nav_state(coords, nav_states):
    """nav_state 변경 지점 기준으로 LineString segment 목록 생성.

    반환: [(nav_state, segment_coords), ...]
    - 모드 전환 지점의 좌표는 이전/다음 구간에 중복 포함하여 경로가 끊겨 보이지 않게 한다.
    """
    if coords is None or len(coords) < 2:
        return []

    if nav_states is None or len(nav_states) != len(coords):
        return [(None, coords)]

    segments = []
    start = 0

    for i in range(1, len(coords)):
        if int(nav_states[i]) != int(nav_states[i - 1]):
            seg = coords[start:i + 1]
            if len(seg) >= 2:
                segments.append((int(nav_states[i - 1]), seg))
            start = i

    seg = coords[start:]
    if len(seg) >= 2:
        segments.append((int(nav_states[-1]), seg))

    return segments


def _build_mode_styles(nav_states):
    """KML Style 문자열 목록 생성."""
    parts = []
    observed = []
    if nav_states is not None and len(nav_states):
        observed = sorted({int(v) for v in nav_states if np.isfinite(v)})

    if not observed:
        observed = [None]

    for nav in observed:
        style_id = _style_id_for_nav(nav)
        raw_color = MODE_COLORS_ABGR.get(nav, MODE_COLOR_DEFAULT) if nav is not None else LINE_COLOR_ABGR
        color = _to_kml_abgr_color(raw_color, fallback=MODE_COLOR_DEFAULT)
        parts += [
            f'    <Style id="{style_id}">',
            '      <LineStyle>',
            f'        <color>{color}</color>',
            '        <width>3</width>',
            '      </LineStyle>',
            '    </Style>',
        ]
    return parts


def _build_mode_path_placemarks(segments):
    """모드별 LineString Placemark 생성."""
    parts = [
        '    <Folder>',
        '      <name>Flight Path by Mode</name>',
    ]

    if not segments:
        parts += [
            '    </Folder>',
        ]
        return parts

    for idx, (nav_state, seg_coords) in enumerate(segments, start=1):
        mode_name = _nav_name(nav_state)
        style_id = _style_id_for_nav(nav_state)
        point_count = len(seg_coords)
        parts += [
            '      <Placemark>',
            f'        <name>{xml_escape(mode_name)} #{idx} ({point_count} pts)</name>',
            f'        <styleUrl>#{style_id}</styleUrl>',
            '        <LineString>',
            '          <extrude>0</extrude>',
            '          <tessellate>1</tessellate>',
            f'          <altitudeMode>{KML_ALTITUDE_MODE}</altitudeMode>',
            '          <coordinates>',
            f'          {_coords_to_str(seg_coords)}',
            '          </coordinates>',
            '        </LineString>',
            '      </Placemark>',
        ]

    parts += [
        '    </Folder>',
    ]
    return parts


def _mode_summary_lines(nav_states):
    if nav_states is None or len(nav_states) == 0:
        return ["Mode coloring: unavailable (vehicle_status_0/nav_state not found)"]

    unique, counts = np.unique(nav_states.astype("int64"), return_counts=True)
    lines = ["Mode coloring: enabled (vehicle_status_0/nav_state)", "Mode point counts:"]
    for nav, count in zip(unique, counts):
        lines.append(f"  - {_nav_name(nav)} ({int(nav)}): {int(count)} points")
    return lines


def _summary_text(
    file_name,
    n_points,
    n_sampled,
    dist_2d_m,
    alt_min_agl,
    alt_max_agl,
    duration_s,
    home_alt_m,
    nav_states_sampled,
):
    def fmt_or(v, suffix):
        return f"{v:.1f} {suffix}" if v is not None and np.isfinite(v) else "N/A"

    lines = [
        f"Source ulg: {file_name}",
        f"Raw path points: {n_points}",
        f"Exported points: {n_sampled}",
        f"Flight duration: {fmt_or(duration_s, 's') if duration_s else 'N/A'}",
        f"2D distance: {fmt_or(dist_2d_m, 'm')}",
        f"Takeoff reference altitude: {fmt_or(home_alt_m, 'm')}",
        f"Altitude mode: {KML_ALTITUDE_MODE}",
        f"AGL altitude range: {fmt_or(alt_min_agl, 'm')} ~ {fmt_or(alt_max_agl, 'm')}",
    ]
    lines.extend(_mode_summary_lines(nav_states_sampled))
    return "\n".join(lines)


def build_kml_document(dataset, file_name="flight.ulg"):
    coords_raw, ts_raw = extract_flight_path_with_time(dataset)
    if coords_raw is None or ts_raw is None or len(coords_raw) < 2:
        raise ValueError(
            "vehicle_global_position 토픽이 없거나 유효한 lat/lon 이 없습니다 (GPS fix 부족)."
        )

    n_raw = len(coords_raw)

    # 2D 거리: raw path 기준 — 더 정확.
    R = 6378137.0
    deg2rad = math.pi / 180.0
    lat_rad = coords_raw[:, 1] * deg2rad
    dlat = np.diff(coords_raw[:, 1]) * deg2rad
    dlon = np.diff(coords_raw[:, 0]) * deg2rad * np.cos(lat_rad[:-1])
    seg = np.sqrt((R * dlat) ** 2 + (R * dlon) ** 2)
    seg = seg[np.isfinite(seg)]
    dist_2d = float(np.sum(seg)) if len(seg) else None

    # takeoff/landing 검출은 원 고도 기준에서 먼저 수행.
    takeoff_raw, landing_raw = detect_takeoff_landing(dataset, fallback_path=coords_raw)

    # 이륙고도 기준 AGL로 변환한 뒤 Google Earth relativeToGround로 export.
    coords_agl, home_alt_m = convert_to_takeoff_agl(coords_raw, takeoff=takeoff_raw)

    # 각 위치 timestamp에 대해 vehicle_status.nav_state를 매칭한다.
    nav_states = get_nav_state_for_times(dataset, ts_raw)

    # 좌표와 nav_state를 같은 step으로 thinning한다.
    if nav_states is not None and len(nav_states) == len(coords_agl):
        sampled, nav_sampled = thin_path_with_aux(coords_agl, nav_states)
    else:
        sampled = thin_path(coords_agl)
        nav_sampled = None

    n_sampled = len(sampled)

    alts_agl = sampled[:, 2]
    alt_min_agl = float(np.min(alts_agl)) if len(alts_agl) else None
    alt_max_agl = float(np.max(alts_agl)) if len(alts_agl) else None

    # Pin 고도도 relativeToGround 기준으로 변환.
    takeoff = None
    landing = None
    if takeoff_raw is not None:
        takeoff_alt_agl = max(float(takeoff_raw[2]) - float(home_alt_m), 0.0)
        takeoff = (float(takeoff_raw[0]), float(takeoff_raw[1]), takeoff_alt_agl)
    if landing_raw is not None:
        landing_alt_agl = max(float(landing_raw[2]) - float(home_alt_m), 0.0)
        landing = (float(landing_raw[0]), float(landing_raw[1]), landing_alt_agl)

    path_segments = _split_coords_by_nav_state(sampled, nav_sampled)

    summary = _summary_text(
        file_name,
        n_raw,
        n_sampled,
        dist_2d,
        alt_min_agl,
        alt_max_agl,
        duration_s=None,
        home_alt_m=home_alt_m,
        nav_states_sampled=nav_sampled,
    )

    # KML XML 직접 작성 (ET가 namespace 처리할 때 prefix가 지저분해질 수 있어 문자열로).
    doc_name = xml_escape(f"ULG Flight Path - {file_name}")
    desc_cdata = summary  # CDATA 안이라 escape 불요.

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<kml xmlns="{KML_NS}">',
        '  <Document>',
        f'    <name>{doc_name}</name>',
        f'    <description><![CDATA[{desc_cdata}]]></description>',
        '    <Style id="flightPathLine">',
        '      <LineStyle>',
        f'        <color>{_to_kml_abgr_color(LINE_COLOR_ABGR)}</color>',
        '        <width>3</width>',
        '      </LineStyle>',
        '    </Style>',
    ]

    parts += _build_mode_styles(nav_sampled)

    parts += [
        '    <Style id="takeoffPin">',
        '      <IconStyle>',
        f'        <Icon><href>{TAKEOFF_ICON}</href></Icon>',
        '      </IconStyle>',
        '    </Style>',
        '    <Style id="landingPin">',
        '      <IconStyle>',
        f'        <Icon><href>{LANDING_ICON}</href></Icon>',
        '      </IconStyle>',
        '    </Style>',
    ]

    parts += _build_mode_path_placemarks(path_segments)

    if takeoff is not None:
        parts += [
            '    <Placemark>',
            '      <name>Takeoff</name>',
            '      <styleUrl>#takeoffPin</styleUrl>',
            '      <Point>',
            f'        <altitudeMode>{KML_ALTITUDE_MODE}</altitudeMode>',
            f'        <coordinates>{takeoff[0]:.7f},{takeoff[1]:.7f},{takeoff[2]:.2f}</coordinates>',
            '      </Point>',
            '    </Placemark>',
        ]

    if landing is not None:
        parts += [
            '    <Placemark>',
            '      <name>Landing</name>',
            '      <styleUrl>#landingPin</styleUrl>',
            '      <Point>',
            f'        <altitudeMode>{KML_ALTITUDE_MODE}</altitudeMode>',
            f'        <coordinates>{landing[0]:.7f},{landing[1]:.7f},{landing[2]:.2f}</coordinates>',
            '      </Point>',
            '    </Placemark>',
        ]

    parts += [
        '  </Document>',
        '</kml>',
    ]

    mode_counts = {}
    if nav_sampled is not None and len(nav_sampled):
        unique, counts = np.unique(nav_sampled.astype("int64"), return_counts=True)
        mode_counts = {int(nav): int(count) for nav, count in zip(unique, counts)}

    return "\n".join(parts), {
        "n_raw": n_raw,
        "n_sampled": n_sampled,
        "dist_2d_m": dist_2d,
        "takeoff_reference_alt_m": home_alt_m,
        "alt_min_agl_m": alt_min_agl,
        "alt_max_agl_m": alt_max_agl,
        "altitude_mode": KML_ALTITUDE_MODE,
        "mode_coloring": nav_sampled is not None,
        "mode_counts": mode_counts,
        "summary": summary,
    }


def export_kmz(dataset, output_path, file_name="flight.ulg"):
    """비행 데이터 → KMZ 파일 저장. 반환: 통계 dict."""
    kml_text, stats = build_kml_document(dataset, file_name=file_name)

    # KMZ = doc.kml 한 개만 들어있는 ZIP.
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml_text.encode("utf-8"))

    return stats
