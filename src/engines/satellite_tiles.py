# src/engines/satellite_tiles.py
# (2026-05-28) Esri World Imagery 위성 타일 다운로드 + stitch.
# 사용: 2D Flight Path 바닥에 비행 영역 위성 사진을 깔기 (pyqtgraph ImageItem).
# 외부 타일 사용은 공급자 약관을 확인해야 하며 요청 좌표와 IP가 공급자에게 전달될 수 있다.
#
# 기술 요점:
#   - Web Mercator (EPSG:3857) tile scheme. ArcGIS REST: /tile/{z}/{y}/{x}.
#   - 자동 zoom 선택: 비행 영역이 ~4 타일 정도 채우도록.
#   - 디스크 캐시: 같은 z/x/y 는 한 번만 다운로드.
#   - QImage 로 PNG/JPEG 디코드 → numpy RGBA.
#   - 새 외부 의존성 없음 (PySide6 + numpy 만 사용).

import math
import os
import urllib.request
import urllib.error

import numpy as np


TILE_PIXEL = 256  # ArcGIS / OSM 표준
MIN_ZOOM = 10
# (2026-05-28) Esri World Imagery 는 지역에 따라 zoom 18+ 에서 "Map data not yet available"
# placeholder 가 떨어짐 (특히 한국 시골/외곽). 17 로 cap 해야 안정적인 위성 사진 확보.
MAX_ZOOM = 17


def cache_root():
    home = os.path.expanduser("~")
    d = os.path.join(home, ".log_ansys_cache", "satellite_tiles", "esri_world_imagery")
    os.makedirs(d, exist_ok=True)
    return d


# ---- 좌표 / Tile 변환 ----------------------------------------------------

def lat_lon_to_tile_xy(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_xy_to_lat_lon(x, y, zoom):
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def best_zoom(lat_span_deg, lat_center, target_tile_count=4):
    earth_circ_m = 40075016.686
    span_m = max(earth_circ_m * lat_span_deg / 360.0, 1.0)
    cos_lat = math.cos(math.radians(lat_center))
    z = math.log2(earth_circ_m * cos_lat * target_tile_count / span_m)
    return int(max(MIN_ZOOM, min(MAX_ZOOM, round(z))))


def compute_tile_range(lat_min, lat_max, lon_min, lon_max, zoom):
    x_a, y_a = lat_lon_to_tile_xy(lat_min, lon_min, zoom)
    x_b, y_b = lat_lon_to_tile_xy(lat_max, lon_max, zoom)
    x_min = int(math.floor(min(x_a, x_b)))
    x_max = int(math.floor(max(x_a, x_b)))
    y_min = int(math.floor(min(y_a, y_b)))
    y_max = int(math.floor(max(y_a, y_b)))
    return x_min, x_max, y_min, y_max


# ---- 다운로드 + 캐시 -----------------------------------------------------

def _tile_url(z, x, y):
    return (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        f"World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )


def fetch_tile(z, x, y, cache_dir=None, timeout=10):
    if cache_dir is None:
        cache_dir = cache_root()
    cache_file = os.path.join(cache_dir, f"z{z}_x{x}_y{y}.jpg")
    if os.path.isfile(cache_file) and os.path.getsize(cache_file) > 0:
        try:
            with open(cache_file, "rb") as f:
                return f.read()
        except Exception:
            pass
    url = _tile_url(z, x, y)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Log-ansys/1.0 (flight-log analysis)",
            "Accept": "image/jpeg,image/png,*/*",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except (urllib.error.URLError, OSError, TimeoutError):
        return None
    try:
        with open(cache_file, "wb") as f:
            f.write(data)
    except Exception:
        pass
    return data


# ---- Stitch + numpy RGBA -------------------------------------------------

def stitch_tiles(tile_range, zoom, cache_dir=None, progress_cb=None):
    """타일 범위의 모든 타일을 다운로드/캐시 로드 후 QImage 로 합성.

    반환: (qimage, missing_count, bbox_latlon)
      bbox_latlon = (lat_max, lat_min, lon_min, lon_max) — 합성 이미지가 cover 하는 정확한 위경도.
                     주의: lat_max (north) 가 image row 0 (위쪽).
    """
    from PySide6.QtGui import QImage, QPainter

    x_min, x_max, y_min, y_max = tile_range
    n_x = x_max - x_min + 1
    n_y = y_max - y_min + 1
    big = QImage(n_x * TILE_PIXEL, n_y * TILE_PIXEL, QImage.Format_ARGB32)
    big.fill(0)
    painter = QPainter(big)
    total = n_x * n_y
    done = 0
    missing = 0
    try:
        for dy, y in enumerate(range(y_min, y_max + 1)):
            for dx, x in enumerate(range(x_min, x_max + 1)):
                data = fetch_tile(zoom, x, y, cache_dir=cache_dir)
                done += 1
                if progress_cb is not None:
                    try:
                        progress_cb(done, total)
                    except Exception:
                        pass
                if data is None:
                    missing += 1
                    continue
                tile_img = QImage()
                if not tile_img.loadFromData(data):
                    missing += 1
                    continue
                painter.drawImage(dx * TILE_PIXEL, dy * TILE_PIXEL, tile_img)
    finally:
        painter.end()

    lat_north, lon_west = tile_xy_to_lat_lon(x_min, y_min, zoom)
    lat_south, lon_east = tile_xy_to_lat_lon(x_max + 1, y_max + 1, zoom)
    return big, missing, (lat_north, lat_south, lon_west, lon_east)


def qimage_to_rgba_numpy(qimg):
    """QImage → numpy uint8 (H, W, 4). pyqtgraph ImageItem 이 기대하는 형식 (W, H, 4)
    로의 변환은 호출자가 transpose 로 처리 — 여기선 자연스러운 (H, W, 4) 반환."""
    from PySide6.QtGui import QImage

    img = qimg.convertToFormat(QImage.Format_RGBA8888)
    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()
    ptr = img.constBits()
    raw = bytes(ptr)[: bpl * h]
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, bpl // 4, 4)[:, :w, :]
    return arr.copy()


# ---- 비행 영역 bbox ------------------------------------------------------

def compute_flight_bbox(dataset, pad_ratio=0.2, min_pad_deg=0.0005, min_extent_km=None):
    """비행 영역 위경도 bbox + (옵션) 최소 폭 강제.

    min_extent_km 가 주어지면 결과 bbox 의 가로/세로 폭이 min_extent_km 이상이 되도록 확장.
    비행 path 가 GPS fix 일부 구간만 cover 하거나 멀리까지 갔을 때 위성 영역을 충분히 잡기 위함.
    """
    if dataset is None or not hasattr(dataset, "topics"):
        return None
    topic = dataset.topics.get("vehicle_global_position_0")
    if topic is None:
        return None
    df = topic.dataframe
    lat_col = "lat" if "lat" in df.columns else ("latitude_deg" if "latitude_deg" in df.columns else None)
    lon_col = "lon" if "lon" in df.columns else ("longitude_deg" if "longitude_deg" in df.columns else None)
    if not lat_col or not lon_col:
        return None
    lat = df[lat_col].to_numpy().astype("float64")
    lon = df[lon_col].to_numpy().astype("float64")
    valid = np.isfinite(lat) & np.isfinite(lon) & (lat != 0) & (lon != 0)
    if not valid.any():
        return None
    lat = lat[valid]
    lon = lon[valid]
    lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    lat_pad = max((lat_max - lat_min) * pad_ratio, min_pad_deg)
    lon_pad = max((lon_max - lon_min) * pad_ratio, min_pad_deg)
    lat_min -= lat_pad
    lat_max += lat_pad
    lon_min -= lon_pad
    lon_max += lon_pad
    # (2026-05-28) 비행 path 가 멀리까지 가는 경우 위성도 함께 cover 하도록 영역 강제 확장.
    if min_extent_km and min_extent_km > 0:
        lat_center = (lat_min + lat_max) / 2.0
        lon_center = (lon_min + lon_max) / 2.0
        # 1 deg lat ≈ 111 km. 1 deg lon ≈ 111 km × cos(lat).
        min_half_lat_deg = (float(min_extent_km) * 1000.0 / 2.0) / 111000.0
        cos_lat = max(math.cos(math.radians(lat_center)), 0.01)
        min_half_lon_deg = min_half_lat_deg / cos_lat
        if (lat_max - lat_min) < 2 * min_half_lat_deg:
            lat_min = lat_center - min_half_lat_deg
            lat_max = lat_center + min_half_lat_deg
        if (lon_max - lon_min) < 2 * min_half_lon_deg:
            lon_min = lon_center - min_half_lon_deg
            lon_max = lon_center + min_half_lon_deg
    return (lat_min, lat_max, lon_min, lon_max)


# ---- 좌표 변환 (lat/lon → ENU) -------------------------------------------

def lla_to_enu(lat, lon, lat0, lon0):
    """위경도 → ENU (m). 비행 영역 (~수 km) 에서 충분히 정확한 equirectangular 근사."""
    R = 6378137.0
    deg2rad = math.pi / 180.0
    dlat = (lat - lat0) * deg2rad
    dlon = (lon - lon0) * deg2rad
    north = R * dlat
    east = R * dlon * math.cos(lat0 * deg2rad)
    return east, north


# ---- 고수준 진입점 -------------------------------------------------------

def compute_flight_bbox_from_local(
    dataset, ref_lat, ref_lon,
    pad_ratio=0.25, min_pad_m=200.0,
    square=False, expand_factor=1.0,
):
    """(2026-05-28) vehicle_local_position 의 NED (x=N, y=E) 범위 → 위경도 bbox.

    비행 path 가 local 모드로 그려질 때 vehicle_global_position 의 GPS fix 가 일부만 있어도
    NED 좌표는 EKF dead-reckoning 으로 끝까지 cover → 비행 path 실제 영역을 더 정확히 측정.

    square=True: 가로/세로 중 큰 쪽 기준 정사각형으로 확장 — 위성을 모든 방향으로 cover.
    expand_factor: 추가 확장 배수 (1.5 면 영역 1.5배). 위성 영역을 viewport 보다 크게 잡아
                    어떤 viewport 비율에서도 화면이 위성으로 가득 차게 함.
    """
    if dataset is None or not hasattr(dataset, "topics") or ref_lat is None or ref_lon is None:
        return None
    topic = dataset.topics.get("vehicle_local_position_0")
    if topic is None:
        return None
    df = topic.dataframe
    if "x" not in df.columns or "y" not in df.columns:
        return None
    x = df["x"].to_numpy().astype("float64")  # NED north (m)
    y = df["y"].to_numpy().astype("float64")  # NED east (m)
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return None
    x = x[valid]
    y = y[valid]
    north_min, north_max = float(np.min(x)), float(np.max(x))
    east_min, east_max = float(np.min(y)), float(np.max(y))
    pad_n = max((north_max - north_min) * pad_ratio, min_pad_m)
    pad_e = max((east_max - east_min) * pad_ratio, min_pad_m)
    north_min -= pad_n
    north_max += pad_n
    east_min -= pad_e
    east_max += pad_e
    if square:
        center_n = (north_min + north_max) / 2.0
        center_e = (east_min + east_max) / 2.0
        half = max(north_max - north_min, east_max - east_min) / 2.0 * float(expand_factor)
        north_min = center_n - half
        north_max = center_n + half
        east_min = center_e - half
        east_max = center_e + half
    elif expand_factor != 1.0:
        center_n = (north_min + north_max) / 2.0
        center_e = (east_min + east_max) / 2.0
        half_n = (north_max - north_min) / 2.0 * float(expand_factor)
        half_e = (east_max - east_min) / 2.0 * float(expand_factor)
        north_min = center_n - half_n
        north_max = center_n + half_n
        east_min = center_e - half_e
        east_max = center_e + half_e
    # ENU(m) → 위경도 (m → deg) — equirectangular.
    R = 6378137.0
    deg_per_rad = 180.0 / math.pi
    cos_lat0 = max(math.cos(math.radians(ref_lat)), 0.01)
    lat_min = ref_lat + (north_min / R) * deg_per_rad
    lat_max = ref_lat + (north_max / R) * deg_per_rad
    lon_min = ref_lon + (east_min / (R * cos_lat0)) * deg_per_rad
    lon_max = ref_lon + (east_max / (R * cos_lat0)) * deg_per_rad
    return (lat_min, lat_max, lon_min, lon_max)


def build_flight_area_satellite(dataset, progress_cb=None, ref_lat=None, ref_lon=None):
    """비행 영역 자동 감지 → 위성 타일 stitch → numpy RGBA + 위경도 bbox.

    ref_lat/ref_lon 주어지면 vehicle_local_position 의 NED 범위를 우선 사용 (가장 정확).
    없으면 vehicle_global_position 의 lat/lon 범위 (GPS fix 구간만).

    반환: (rgba: np.ndarray(H, W, 4) uint8,
           bbox: (lat_north, lat_south, lon_west, lon_east),
           zoom, missing) 또는 None.
    """
    bbox = None
    if ref_lat is not None and ref_lon is not None:
        # (2026-05-28) 위성 영역을 정사각형 + 1.5배 확장 → 그래프 viewport 가 어떤 비율이든
        # 위성 사진이 화면 가득. 사용자가 줌아웃해도 어느 정도 cover 됨.
        bbox = compute_flight_bbox_from_local(
            dataset, ref_lat, ref_lon,
            pad_ratio=0.15, square=True, expand_factor=1.5,
        )
    if bbox is None:
        bbox = compute_flight_bbox(dataset)
    if bbox is None:
        return None
    lat_min, lat_max, lon_min, lon_max = bbox
    lat_span = lat_max - lat_min
    lat_center = (lat_min + lat_max) / 2.0
    zoom = best_zoom(lat_span, lat_center)
    tile_range = compute_tile_range(lat_min, lat_max, lon_min, lon_max, zoom)
    qimg, missing, latlon_bbox = stitch_tiles(tile_range, zoom, progress_cb=progress_cb)
    n_x = tile_range[1] - tile_range[0] + 1
    n_y = tile_range[3] - tile_range[2] + 1
    if missing >= n_x * n_y:
        return None
    rgba = qimage_to_rgba_numpy(qimg)
    return rgba, latlon_bbox, zoom, missing
