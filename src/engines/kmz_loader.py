# src/engines/kmz_loader.py
# (2026-05-28) KMZ/KML import — Analysis → Flight Path → Load Map Overlay (KMZ) 등 용도.
#
# 지원 요소:
#   - Placemark + Point      (마커 핀)
#   - Placemark + LineString (경로/계획)
#   - GroundOverlay          (이미지를 위경도 4 corner 에 깖) — 2D Path 위성 대체용 핵심
#
# 좌표계:
#   - KML 좌표는 WGS84 (lon, lat, alt).
#   - 2D Flight Path 는 비행 경로 원점(lat0, lon0) 을 기준으로 한 ENU (East=x, North=y) m.
#   - lla_to_enu() 가 equirectangular 근사로 변환 — 비행 영역(~수 km)에서 mm~m 오차.

import io
import math
import os
import zipfile
import xml.etree.ElementTree as ET

import numpy as np


KML_NS = "{http://www.opengis.net/kml/2.2}"


# ---- 파싱 ----------------------------------------------------------------

def load_kml_items(file_path):
    """KMZ/KML 파일 경로 → 항목 리스트 + (옵션) ZIP 핸들 보관용 dict.

    GroundOverlay 의 이미지 데이터(href) 는 KMZ 안에 있으면 raw bytes 까지 함께 담아 반환.
    호출자는 bytes 를 QImage 로 decode 해서 사용.

    반환 예:
      [
        {"type": "placemark", "lon": ..., "lat": ..., "alt": ..., "name": ..., "description": ...},
        {"type": "linestring", "coords": [(lon,lat,alt), ...], "name": ..., "description": ...},
        {"type": "groundoverlay",
         "name": ..., "description": ...,
         "north": ..., "south": ..., "east": ..., "west": ..., "rotation": 0.0,
         "image_bytes": bytes or None,        # KMZ 안 이미지면 raw bytes
         "image_href": "files/overlay.png",   # KML <Icon><href> 원본값
        },
      ]
    """
    file_path = os.fspath(file_path)
    if file_path.lower().endswith(".kmz"):
        with zipfile.ZipFile(file_path) as z:
            kml_entries = [n for n in z.namelist() if n.lower().endswith(".kml")]
            if not kml_entries:
                raise ValueError("KMZ 파일 안에 .kml 이 없습니다.")
            kml_entries.sort(
                key=lambda n: (0 if os.path.basename(n).lower() == "doc.kml" else 1, n)
            )
            kml_name = kml_entries[0]
            xml_data = z.read(kml_name)
            zip_namelist = z.namelist()
            zip_data = {n: z.read(n) for n in zip_namelist}
    else:
        with open(file_path, "rb") as f:
            xml_data = f.read()
        zip_data = None
    return parse_kml_bytes(xml_data, zip_data=zip_data)


def parse_kml_bytes(xml_data, zip_data=None):
    if xml_data.startswith(b"\xef\xbb\xbf"):
        xml_data = xml_data[3:]
    root = ET.fromstring(xml_data)
    return list(_iter_kml_items(root, zip_data=zip_data))


def _iter_kml_items(node, zip_data=None):
    # Placemark
    for pm in node.iter(KML_NS + "Placemark"):
        name = _text(pm.find(KML_NS + "name"))
        desc = _text(pm.find(KML_NS + "description"))

        point = pm.find(KML_NS + "Point")
        if point is not None:
            coords_el = point.find(KML_NS + "coordinates")
            if coords_el is not None and coords_el.text:
                coords = _parse_coord_list(coords_el.text)
                if coords:
                    lon, lat, alt = coords[0]
                    yield {
                        "type": "placemark",
                        "lon": lon, "lat": lat, "alt": alt,
                        "name": name or "(unnamed)", "description": desc,
                    }

        ls = pm.find(KML_NS + "LineString")
        if ls is not None:
            coords_el = ls.find(KML_NS + "coordinates")
            if coords_el is not None and coords_el.text:
                coords = _parse_coord_list(coords_el.text)
                if len(coords) >= 2:
                    yield {
                        "type": "linestring",
                        "coords": coords,
                        "name": name or "(unnamed)", "description": desc,
                    }

    # GroundOverlay — 별도 iter (Placemark 안에 들어있지 않음, Document/Folder 자식).
    for go in node.iter(KML_NS + "GroundOverlay"):
        name = _text(go.find(KML_NS + "name"))
        desc = _text(go.find(KML_NS + "description"))
        latlon_box = go.find(KML_NS + "LatLonBox")
        if latlon_box is None:
            # gx:LatLonQuad (회전된 사각형) 형식은 v1 에선 미지원. 향후 확장.
            continue
        try:
            north = float(_text(latlon_box.find(KML_NS + "north")))
            south = float(_text(latlon_box.find(KML_NS + "south")))
            east = float(_text(latlon_box.find(KML_NS + "east")))
            west = float(_text(latlon_box.find(KML_NS + "west")))
        except (TypeError, ValueError):
            continue
        rotation_text = _text(latlon_box.find(KML_NS + "rotation"))
        rotation = 0.0
        if rotation_text:
            try:
                rotation = float(rotation_text)
            except ValueError:
                pass
        icon = go.find(KML_NS + "Icon")
        href = _text(icon.find(KML_NS + "href")) if icon is not None else ""
        image_bytes = None
        if href and zip_data is not None:
            # KMZ 내부 상대 경로일 가능성 — 정규화해서 매칭.
            cand_keys = (href, href.lstrip("./").lstrip("/"))
            for key in cand_keys:
                if key in zip_data:
                    image_bytes = zip_data[key]
                    break
            if image_bytes is None:
                # 마지막 케이스: basename 일치 (드물지만 일부 KMZ).
                base = os.path.basename(href)
                for k, v in zip_data.items():
                    if os.path.basename(k) == base:
                        image_bytes = v
                        break
        yield {
            "type": "groundoverlay",
            "name": name or "(unnamed overlay)", "description": desc,
            "north": north, "south": south, "east": east, "west": west,
            "rotation": rotation,
            "image_bytes": image_bytes, "image_href": href,
        }


def _text(el):
    if el is None or el.text is None:
        return ""
    return el.text.strip()


def _parse_coord_list(text):
    tokens = text.replace("\n", " ").replace("\t", " ").split()
    out = []
    for tok in tokens:
        parts = tok.split(",")
        if len(parts) < 2:
            continue
        try:
            lon = float(parts[0])
            lat = float(parts[1])
            alt = float(parts[2]) if len(parts) >= 3 else 0.0
            out.append((lon, lat, alt))
        except ValueError:
            continue
    return out


# ---- 좌표 변환 -----------------------------------------------------------

def lla_to_enu(lat, lon, lat0, lon0):
    """위경도 (deg) → ENU (m). equirectangular 근사."""
    R = 6378137.0
    deg2rad = math.pi / 180.0
    dlat = (lat - lat0) * deg2rad
    dlon = (lon - lon0) * deg2rad
    north = R * dlat
    east = R * dlon * math.cos(lat0 * deg2rad)
    return east, north


# ---- GroundOverlay 이미지 디코드 ----------------------------------------

def decode_groundoverlay_image(item):
    """GroundOverlay item 의 image_bytes 를 numpy RGBA (H, W, 4) 로 변환.
    PySide6.QImage 사용 — PNG/JPEG 모두 처리."""
    from PySide6.QtGui import QImage

    data = item.get("image_bytes")
    if not data:
        return None
    qimg = QImage()
    if not qimg.loadFromData(data):
        return None
    qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()
    bpl = qimg.bytesPerLine()
    ptr = qimg.constBits()
    raw = bytes(ptr)[: bpl * h]
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, bpl // 4, 4)[:, :w, :]
    return arr.copy()
