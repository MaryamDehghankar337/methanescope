"""Sentinel-2 methane candidate screening app - stable final version.

Windows PowerShell:
    $env:CDSE_USERNAME="your CDSE email"
    $env:CDSE_PASSWORD="your CDSE password"
    python -m streamlit run sentinel_methane_app_final.py

The algorithm is a relative MBMP screening workflow. It is not a physical
methane concentration or emission-rate retrieval.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import folium
import numpy as np
import pandas as pd
import requests
import rasterio
import streamlit as st
from folium.plugins import Draw
from scipy.ndimage import binary_dilation, gaussian_filter, label
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from skimage.morphology import disk
from streamlit_folium import st_folium

STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
BANDS = ["B03", "B04", "B08", "B11", "B12"]
RESOLUTION = 20
CACHE_DIR = Path.home() / ".sentinel_methane_cache"
RESULT_DIR = Path.home() / ".sentinel_methane_results"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

PARAMS = {
    "b03_quantile": 0.05,
    "swir_saturation": 1.0,
    "ndwi_threshold": 0.20,
    "ndvi_threshold": 0.30,
    "ndbi_threshold": 0.20,
    "ndsi_threshold": 0.42,
    "lrad_dilation": 1,
    "gaussian_sigma": 0.85,
    "threshold_sigma": 1.0,
    "min_component_pixels": 2,
    "final_dilation": 3,
}


def as_dict(item):
    if isinstance(item, dict):
        return item
    if hasattr(item, "to_dict"):
        return item.to_dict()
    return dict(item)


def get_properties(item):
    return as_dict(item).get("properties", {}) or {}


def get_datetime(item) -> Optional[datetime]:
    data = as_dict(item)
    props = get_properties(data)
    value = props.get("datetime") or props.get("start_datetime")
    if value:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass
    match = re.search(r"_(\d{8}T\d{6})_", data.get("id", "").upper())
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S") if match else None


def get_tile(item):
    data = as_dict(item)
    props = get_properties(data)
    for key in ("mgrs:tile", "s2:mgrs_tile", "tile"):
        if props.get(key):
            return str(props[key]).upper()
    match = re.search(r"_(T\d{2}[A-Z]{3})_", data.get("id", "").upper())
    return match.group(1) if match else None


def get_cloud(item):
    props = get_properties(item)
    for key in ("eo:cloud_cover", "cloudCover", "cloud_cover"):
        try:
            return float(props[key])
        except Exception:
            continue
    return 100.0


def normalize_geometry(obj):
    if obj is None:
        return None
    if hasattr(obj, "__geo_interface__"):
        obj = obj.__geo_interface__
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "Feature":
        return normalize_geometry(obj.get("geometry"))
    if obj.get("type") == "FeatureCollection":
        geometries = []
        for feature in obj.get("features", []):
            geometry = normalize_geometry(feature.get("geometry"))
            if geometry:
                geometries.append(shape(geometry))
        return mapping(unary_union(geometries)) if geometries else None
    try:
        geometry = shape(obj)
        return mapping(geometry) if not geometry.is_empty else None
    except Exception:
        return None


def ensure_aoi(obj):
    return normalize_geometry(obj) or mapping(box(48.0, 29.0, 49.0, 30.0))


def search_scenes(aoi, start, end, max_cloud):
    payload = {
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{start.isoformat()}Z/{end.isoformat()}Z",
        "intersects": ensure_aoi(aoi),
        "query": {"eo:cloud_cover": {"lt": float(max_cloud)}},
        "limit": 100,
    }
    response = requests.post(f"{STAC_URL}search", json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("features", [])


def authenticate_cdse(username: str, password: str, totp: str = ""):
    """Authenticate one Streamlit browser session against Copernicus CDSE.

    The password is sent only to the official CDSE identity endpoint over HTTPS
    and is not stored in the app session. Only the short-lived access token and
    refresh token are kept in this browser session.
    """
    username = username.strip()
    password = password.strip()
    totp = totp.strip()
    if not username or not password:
        raise RuntimeError("Please enter your Copernicus email and password.")

    form = {
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": username,
        "password": password,
    }
    if totp:
        form["totp"] = totp

    response = requests.post(TOKEN_URL, data=form, timeout=90)
    if response.status_code >= 400:
        try:
            detail = response.json().get("error_description") or response.json().get("error")
        except Exception:
            detail = None
        raise RuntimeError(detail or f"Copernicus login failed (HTTP {response.status_code}).")

    data = response.json()
    access_token = data.get("access_token")
    if not access_token:
        raise RuntimeError("Copernicus did not return an access token.")

    now = time.time()
    return {
        "access_token": access_token,
        "refresh_token": data.get("refresh_token", ""),
        "expires_at": now + int(data.get("expires_in", 600)),
        "username": username,
    }


def refresh_cdse_session(auth):
    refresh_token = auth.get("refresh_token", "")
    if not refresh_token:
        return None

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "client_id": "cdse-public",
            "refresh_token": refresh_token,
        },
        timeout=90,
    )
    if response.status_code >= 400:
        return None

    data = response.json()
    token = data.get("access_token")
    if not token:
        return None

    auth["access_token"] = token
    auth["refresh_token"] = data.get("refresh_token", refresh_token)
    auth["expires_at"] = time.time() + int(data.get("expires_in", 600))
    return auth


def get_access_token():
    """Return a valid token for the current browser session."""
    auth = st.session_state.get("cdse_auth")
    if not auth:
        raise RuntimeError("Please log in to Copernicus first.")

    if time.time() < float(auth.get("expires_at", 0)) - 60:
        return auth["access_token"]

    refreshed = refresh_cdse_session(auth)
    if refreshed:
        st.session_state.cdse_auth = refreshed
        return refreshed["access_token"]

    st.session_state.pop("cdse_auth", None)
    raise RuntimeError("Your Copernicus session expired. Please log in again.")


def evalscript():
    return """//VERSION=3
function setup() {
  return {
    input: [{bands: [\"B03\",\"B04\",\"B08\",\"B11\",\"B12\"], units: \"REFLECTANCE\"}],
    output: {bands: 5, sampleType: \"FLOAT32\"}
  };
}
function evaluatePixel(sample) {
  return [sample.B03, sample.B04, sample.B08, sample.B11, sample.B12];
}
"""


def download_scene(item, aoi, access_token):
    item = as_dict(item)
    aoi = ensure_aoi(aoi)
    cache_id = hashlib.sha256(json.dumps([item.get("id"), aoi, RESOLUTION], sort_keys=True).encode()).hexdigest()[:24]
    folder = CACHE_DIR / cache_id
    output_path = folder / "bands.tif"
    metadata_path = folder / "metadata.json"
    if output_path.exists() and metadata_path.exists():
        return output_path
    folder.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = shape(aoi).bounds
    latitude = math.radians((miny + maxy) / 2.0)
    width = max(1, min(2500, int(abs(maxx - minx) * 111320 * math.cos(latitude) / RESOLUTION)))
    height = max(1, min(2500, int(abs(maxy - miny) * 111320 / RESOLUTION)))
    acquisition = get_datetime(item)
    if acquisition is None:
        raise RuntimeError("Could not read acquisition date.")
    payload = {
        "input": {
            "bounds": {"bbox": [minx, miny, maxx, maxy], "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": acquisition.strftime("%Y-%m-%dT00:00:00Z"), "to": (acquisition + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")}, "mosaickingOrder": "leastCC"}}],
        },
        "output": {"width": width, "height": height, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": evalscript(),
    }
    response = requests.post(PROCESS_URL, headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}, json=payload, timeout=900)
    if response.status_code >= 400:
        try:
            detail = response.json()
        except Exception:
            detail = response.text[:1000]
        raise RuntimeError(f"CDSE Process API failed ({response.status_code}): {detail}")
    output_path.write_bytes(response.content)
    metadata_path.write_text(json.dumps(item, indent=2), encoding="utf-8")
    return output_path


def read_stack(path):
    with rasterio.open(path) as source:
        array = source.read().astype(np.float32)
        profile = source.profile.copy()
    return {band: array[index] for index, band in enumerate(BANDS)}, profile


def normalized_difference(first, second):
    output = np.full(first.shape, np.nan, dtype=np.float32)
    denominator = first + second
    valid = np.isfinite(first) & np.isfinite(second) & (np.abs(denominator) > 1e-12)
    output[valid] = (first[valid] - second[valid]) / denominator[valid]
    return output


def calculate_lrad(bands, q_value):
    finite = np.logical_and.reduce([np.isfinite(bands[band]) for band in BANDS])
    artifact = ((bands["B11"] >= PARAMS["swir_saturation"]) & (bands["B12"] >= PARAMS["swir_saturation"]))
    artifact |= bands["B03"] <= q_value
    artifact |= normalized_difference(bands["B03"], bands["B08"]) >= PARAMS["ndwi_threshold"]
    artifact |= normalized_difference(bands["B08"], bands["B04"]) >= PARAMS["ndvi_threshold"]
    artifact |= normalized_difference(bands["B11"], bands["B08"]) >= PARAMS["ndbi_threshold"]
    artifact |= normalized_difference(bands["B03"], bands["B11"]) >= PARAMS["ndsi_threshold"]
    artifact |= ~finite
    artifact = binary_dilation(artifact, structure=disk(PARAMS["lrad_dilation"]))
    return finite & ~artifact


def calculate_c(b11, b12, valid):
    use = valid & np.isfinite(b11) & np.isfinite(b12) & (b11 > 0) & (b12 > 0)
    if use.sum() < 100:
        return 1.0
    x, y = b11[use].astype(np.float64), b12[use].astype(np.float64)
    return float(np.sum(x * y) / max(np.sum(x * x), 1e-20))


def calculate_mbsp(b11, b12, c, valid):
    output = np.full(b11.shape, np.nan, dtype=np.float32)
    use = valid & np.isfinite(b11) & np.isfinite(b12) & (np.abs(b12) > 1e-12)
    output[use] = c * (b12[use] - b11[use]) / b12[use]
    return output


def run_algorithm(target, reference):
    target_q = float(np.nanquantile(target["B03"], PARAMS["b03_quantile"]))
    reference_q = float(np.nanquantile(reference["B03"], PARAMS["b03_quantile"]))
    valid = calculate_lrad(target, target_q) & calculate_lrad(reference, reference_q)
    target_mbsp = calculate_mbsp(target["B11"], target["B12"], calculate_c(target["B11"], target["B12"], valid), valid)
    reference_mbsp = calculate_mbsp(reference["B11"], reference["B12"], calculate_c(reference["B11"], reference["B12"], valid), valid)
    relative = target_mbsp - reference_mbsp
    relative[~valid] = np.nan
    finite = np.isfinite(relative)
    if not finite.any():
        raise RuntimeError("LRAD removed all pixels. Reduce artifact thresholds or use a smaller valid AOI.")
    values = relative[finite].astype(np.float64)
    mean_value = float(np.mean(values))
    std_value = float(np.std(values))
    fill_value = mean_value
    gaussian = gaussian_filter(np.where(finite, relative, fill_value), sigma=PARAMS["gaussian_sigma"])
    threshold = mean_value + PARAMS["threshold_sigma"] * std_value
    initial = valid & np.isfinite(gaussian) & (gaussian > threshold)
    labels, _ = label(initial, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    retained = np.where(sizes >= PARAMS["min_component_pixels"])[0]
    retained = retained[retained != 0]
    connected = np.isin(labels, retained)
    final = binary_dilation(connected, structure=disk(PARAMS["final_dilation"])) & valid
    return {"relative": relative, "gaussian": gaussian, "valid": valid, "initial": initial, "connected": connected, "final": final, "mean": mean_value, "std": std_value, "threshold": threshold, "regions": int(len(retained)), "valid_count": int(valid.sum()), "initial_count": int(initial.sum()), "final_count": int(final.sum())}


def save_raster(path, array, profile, mask=False):
    output_profile = profile.copy()
    output_profile.update(count=1, dtype="uint8" if mask else "float32", nodata=255 if mask else -9999, compress="deflate", tiled=False, BIGTIFF="IF_SAFER")
    output = np.where(array, 1, 0).astype(np.uint8) if mask else np.where(np.isfinite(array), array, -9999).astype(np.float32)
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(output, 1)


def create_map(aoi):
    geometry = shape(ensure_aoi(aoi))
    fmap = folium.Map([geometry.centroid.y, geometry.centroid.x], zoom_start=7, tiles="OpenStreetMap")
    folium.GeoJson(mapping(geometry), style_function=lambda _: {"color": "blue", "fill": False}).add_to(fmap)
    Draw(export=True, draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False}).add_to(fmap)
    return fmap


def image_png(array, mask=False):
    from PIL import Image
    data = np.asarray(array)
    if mask:
        rgb = np.zeros((*data.shape, 3), dtype=np.uint8)
        rgb[data.astype(bool)] = [220, 30, 30]
    else:
        finite = np.isfinite(data)
        rgb = np.full((*data.shape, 3), 255, dtype=np.uint8)
        if finite.any():
            values = data[finite]
            low, high = float(np.percentile(values, 2)), float(np.percentile(values, 98))
            if high <= low:
                low, high = float(values.min()), float(values.max())
            if high > low:
                normalized = np.clip((np.nan_to_num(data, nan=low) - low) / (high - low), 0, 1)
                import matplotlib.pyplot as plt
                rgb = (plt.get_cmap("RdBu_r")(normalized)[:, :, :3] * 255).astype(np.uint8)
                rgb[~finite] = 255
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def legend_html(kind):
    """Readable standalone legend rendered outside each output image."""
    if kind == "mask":
        rows = [("#dc1e1e", "Methane candidate"), ("#000000", "Background / non-candidate")]
    elif kind == "valid":
        rows = [("#dc1e1e", "Valid pixels"), ("#000000", "Invalid / masked pixels")]
    else:
        rows = [("#b43232", "Higher anomaly"), ("#3250b4", "Lower anomaly"), ("#ffffff", "No data")]
    items = "".join(f'<div class="legend-row"><span class="legend-swatch" style="background:{c};"></span><span>{t}</span></div>' for c,t in rows)
    return f'<div class="result-legend"><div class="legend-heading">Legend</div>{items}</div>'


def create_png_worldfile(profile, array_shape):
    # PNG pixel coordinates are tied to a world file (.pgw).
    transform = profile["transform"]
    width = int(array_shape[1])
    height = int(array_shape[0])
    xres = transform.a
    yres = transform.e
    x_center = transform.c + xres / 2.0
    y_center = transform.f + yres / 2.0
    pgw = f"{xres:.12f}\n0.0\n0.0\n{yres:.12f}\n{x_center:.12f}\n{y_center:.12f}\n"
    crs_text = profile.get("crs")
    prj = crs_text.to_wkt() if crs_text else ""
    return pgw.encode("utf-8"), prj.encode("utf-8")


def georeferenced_png_package(array, profile, mask=False, legend_kind=None):
    import zipfile
    png_data = image_png(array, mask=mask)
    pgw_data, prj_data = create_png_worldfile(profile, np.asarray(array).shape)
    package = io.BytesIO()
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("output.png", png_data)
        archive.writestr("output.pgw", pgw_data)
        if prj_data:
            archive.writestr("output.prj", prj_data)
    return package.getvalue()


# ============================================================
# User Interface
# ============================================================

# ============================================================
# Compact App-like User Interface
# ============================================================
st.set_page_config(
    page_title="Sentinel-2 Methane",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root {
    --red: #e63946;
    --honeydew: #f1faee;
    --frost: #a8dadc;
    --blue: #457b9d;
    --navy: #1d3557;
    --black: #111111;
    --white: #ffffff;
    --border: #d8e6e8;
    --muted: #4f5d63;
}

/* ---------- Main application ---------- */
.stApp {
    background: #f1faee;
    color: #111111 !important;
}

/* Keep Streamlit's top toolbar separate from the application header. */
[data-testid="stHeader"] {
    background: #f1faee !important;
    height: 3.25rem !important;
}

[data-testid="stSidebar"] {
    display: none;
}

/* Push the custom header below Streamlit's top toolbar. */
.block-container {
    max-width: 1700px;
    padding-top: 3.9rem !important;
    padding-bottom: 0.8rem;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}

/* ---------- Header ---------- */
.app-header {
    position: relative;
    z-index: 10;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.75rem 1rem;
    margin-top: 0.15rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 2px 10px rgba(29,53,87,0.05);
}

.app-title {
    color: #111111 !important;
    font-size: 1.45rem;
    font-weight: 850;
    line-height: 1.1;
}

.app-subtitle {
    color: #111111 !important;
    font-size: 0.78rem;
    margin-top: 0.15rem;
}

.status-pill {
    background: #f1faee;
    color: #111111 !important;
    border: 1px solid #a8dadc;
    border-radius: 999px;
    padding: 0.35rem 0.7rem;
    font-size: 0.72rem;
    font-weight: 750;
    white-space: nowrap;
}

/* ---------- Cards ---------- */
.app-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 0.75rem;
    box-shadow: 0 2px 10px rgba(29,53,87,0.04);
    height: 100%;
    color: #111111 !important;
}

.card-title {
    color: #111111 !important;
    font-size: 1rem;
    font-weight: 800;
    margin-bottom: 0.1rem;
}

.card-caption {
    color: #111111 !important;
    font-size: 0.73rem;
    margin-bottom: 0.45rem;
}

.section-label {
    display: inline-block;
    background: #a8dadc;
    color: #111111 !important;
    border-radius: 999px;
    padding: 0.2rem 0.55rem;
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    margin-bottom: 0.35rem;
}

/* ---------- Force native Streamlit text to black ---------- */
.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp div,
.stApp small,
.stApp strong,
.stApp em,
.stApp li,
.stApp td,
.stApp th,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] span,
.stApp [data-testid="stMarkdownContainer"] li {
    color: #111111 !important;
}

/* Restore white text only where the UI intentionally needs it. */
.stButton > button[kind="primary"],
.stButton > button[kind="primary"] *,
.stDownloadButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] * {
    color: #ffffff !important;
}

/* ---------- Readable standalone legends ---------- */
.result-legend {
    background: #ffffff; border: 1px solid #d7e4e7; border-radius: 10px;
    padding: 0.75rem 0.7rem; min-height: 96px; box-sizing: border-box;
    display: flex; flex-direction: column; justify-content: center; gap: 0.42rem;
}
.result-legend .legend-heading { color: #111111 !important; font-size: 0.88rem; font-weight: 800; }
.result-legend .legend-row { display:flex; align-items:center; gap:0.45rem; color:#111111 !important; font-size:0.82rem; line-height:1.25; }
.result-legend .legend-row span:last-child { color:#111111 !important; }
.legend-swatch { width:18px; height:14px; min-width:18px; border:1px solid #555; border-radius:2px; display:inline-block; }

/* ---------- Inputs ---------- */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-radius: 9px;
}

/* Date and number input text */
.stDateInput input,
.stNumberInput input,
div[data-baseweb="input"] input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important;
}

/* Placeholder text */
input::placeholder,
textarea::placeholder {
    color: #4f5d63 !important;
    opacity: 1 !important;
}

/* Selectbox selected value and dropdown text */
div[data-baseweb="select"] * {
    color: #111111 !important;
}

[role="listbox"] *,
[role="option"] * {
    color: #111111 !important;
}

/* Dark dropdown popup: Target scene options must be white and readable */
div[data-baseweb="popover"] [role="listbox"],
div[data-baseweb="popover"] [role="option"],
div[data-baseweb="popover"] [role="option"] *,
ul[role="listbox"],
li[role="option"],
li[role="option"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

div[data-baseweb="popover"] [role="listbox"],
div[data-baseweb="popover"] ul[role="listbox"] {
    background: #111318 !important;
}

div[data-baseweb="popover"] [role="option"],
div[data-baseweb="popover"] li[role="option"] {
    background: #111318 !important;
}

div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] li[role="option"]:hover {
    background: #2b2e38 !important;
}

.stDateInput, .stSlider, .stNumberInput, .stSelectbox {
    margin-bottom: 0.15rem;
}

.stSlider > div {
    padding-top: 0.05rem;
    padding-bottom: 0.05rem;
}

/* Slider value / labels */
.stSlider label,
.stSlider [data-testid="stTickBar"] *,
.stSlider [data-testid="stThumbValue"] * {
    color: #111111 !important;
}

/* ---------- Dark native controls: keep their values white ---------- */
/* Streamlit/BaseWeb may render date, number and threshold controls with a dark field. */
div[data-baseweb="input"] input,
.stNumberInput input,
.stDateInput input,
.stNumberInput [data-baseweb="input"] input,
.stDateInput [data-baseweb="input"] input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}"""Sentinel-2 methane candidate screening app - stable final version.

Windows PowerShell:
    $env:CDSE_USERNAME="your CDSE email"
    $env:CDSE_PASSWORD="your CDSE password"
    python -m streamlit run sentinel_methane_app_final.py

The algorithm is a relative MBMP screening workflow. It is not a physical
methane concentration or emission-rate retrieval.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import folium
import numpy as np
import pandas as pd
import requests
import rasterio
import streamlit as st
from folium.plugins import Draw
from scipy.ndimage import binary_dilation, gaussian_filter, label
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from skimage.morphology import disk
from streamlit_folium import st_folium

STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
BANDS = ["B03", "B04", "B08", "B11", "B12"]
RESOLUTION = 20
CACHE_DIR = Path.home() / ".sentinel_methane_cache"
RESULT_DIR = Path.home() / ".sentinel_methane_results"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

PARAMS = {
    "b03_quantile": 0.05,
    "swir_saturation": 1.0,
    "ndwi_threshold": 0.20,
    "ndvi_threshold": 0.30,
    "ndbi_threshold": 0.20,
    "ndsi_threshold": 0.42,
    "lrad_dilation": 1,
    "gaussian_sigma": 0.85,
    "threshold_sigma": 1.0,
    "min_component_pixels": 2,
    "final_dilation": 3,
}


def as_dict(item):
    if isinstance(item, dict):
        return item
    if hasattr(item, "to_dict"):
        return item.to_dict()
    return dict(item)


def get_properties(item):
    return as_dict(item).get("properties", {}) or {}


def get_datetime(item) -> Optional[datetime]:
    data = as_dict(item)
    props = get_properties(data)
    value = props.get("datetime") or props.get("start_datetime")
    if value:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass
    match = re.search(r"_(\d{8}T\d{6})_", data.get("id", "").upper())
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S") if match else None


def get_tile(item):
    data = as_dict(item)
    props = get_properties(data)
    for key in ("mgrs:tile", "s2:mgrs_tile", "tile"):
        if props.get(key):
            return str(props[key]).upper()
    match = re.search(r"_(T\d{2}[A-Z]{3})_", data.get("id", "").upper())
    return match.group(1) if match else None


def get_cloud(item):
    props = get_properties(item)
    for key in ("eo:cloud_cover", "cloudCover", "cloud_cover"):
        try:
            return float(props[key])
        except Exception:
            continue
    return 100.0


def normalize_geometry(obj):
    if obj is None:
        return None
    if hasattr(obj, "__geo_interface__"):
        obj = obj.__geo_interface__
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "Feature":
        return normalize_geometry(obj.get("geometry"))
    if obj.get("type") == "FeatureCollection":
        geometries = []
        for feature in obj.get("features", []):
            geometry = normalize_geometry(feature.get("geometry"))
            if geometry:
                geometries.append(shape(geometry))
        return mapping(unary_union(geometries)) if geometries else None
    try:
        geometry = shape(obj)
        return mapping(geometry) if not geometry.is_empty else None
    except Exception:
        return None


def ensure_aoi(obj):
    return normalize_geometry(obj) or mapping(box(48.0, 29.0, 49.0, 30.0))


def search_scenes(aoi, start, end, max_cloud):
    payload = {
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{start.isoformat()}Z/{end.isoformat()}Z",
        "intersects": ensure_aoi(aoi),
        "query": {"eo:cloud_cover": {"lt": float(max_cloud)}},
        "limit": 100,
    }
    response = requests.post(f"{STAC_URL}search", json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("features", [])


def authenticate_cdse(username: str, password: str, totp: str = ""):
    """Authenticate one Streamlit browser session against Copernicus CDSE.

    The password is sent only to the official CDSE identity endpoint over HTTPS
    and is not stored in the app session. Only the short-lived access token and
    refresh token are kept in this browser session.
    """
    username = username.strip()
    password = password.strip()
    totp = totp.strip()
    if not username or not password:
        raise RuntimeError("Please enter your Copernicus email and password.")

    form = {
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": username,
        "password": password,
    }
    if totp:
        form["totp"] = totp

    response = requests.post(TOKEN_URL, data=form, timeout=90)
    if response.status_code >= 400:
        try:
            detail = response.json().get("error_description") or response.json().get("error")
        except Exception:
            detail = None
        raise RuntimeError(detail or f"Copernicus login failed (HTTP {response.status_code}).")

    data = response.json()
    access_token = data.get("access_token")
    if not access_token:
        raise RuntimeError("Copernicus did not return an access token.")

    now = time.time()
    return {
        "access_token": access_token,
        "refresh_token": data.get("refresh_token", ""),
        "expires_at": now + int(data.get("expires_in", 600)),
        "username": username,
    }


def refresh_cdse_session(auth):
    refresh_token = auth.get("refresh_token", "")
    if not refresh_token:
        return None

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "client_id": "cdse-public",
            "refresh_token": refresh_token,
        },
        timeout=90,
    )
    if response.status_code >= 400:
        return None

    data = response.json()
    token = data.get("access_token")
    if not token:
        return None

    auth["access_token"] = token
    auth["refresh_token"] = data.get("refresh_token", refresh_token)
    auth["expires_at"] = time.time() + int(data.get("expires_in", 600))
    return auth


def get_access_token():
    """Return a valid token for the current browser session."""
    auth = st.session_state.get("cdse_auth")
    if not auth:
        raise RuntimeError("Please log in to Copernicus first.")

    if time.time() < float(auth.get("expires_at", 0)) - 60:
        return auth["access_token"]

    refreshed = refresh_cdse_session(auth)
    if refreshed:
        st.session_state.cdse_auth = refreshed
        return refreshed["access_token"]

    st.session_state.pop("cdse_auth", None)
    raise RuntimeError("Your Copernicus session expired. Please log in again.")


def evalscript():
    return """//VERSION=3
function setup() {
  return {
    input: [{bands: [\"B03\",\"B04\",\"B08\",\"B11\",\"B12\"], units: \"REFLECTANCE\"}],
    output: {bands: 5, sampleType: \"FLOAT32\"}
  };
}
function evaluatePixel(sample) {
  return [sample.B03, sample.B04, sample.B08, sample.B11, sample.B12];
}
"""


def download_scene(item, aoi, access_token):
    item = as_dict(item)
    aoi = ensure_aoi(aoi)
    cache_id = hashlib.sha256(json.dumps([item.get("id"), aoi, RESOLUTION], sort_keys=True).encode()).hexdigest()[:24]
    folder = CACHE_DIR / cache_id
    output_path = folder / "bands.tif"
    metadata_path = folder / "metadata.json"
    if output_path.exists() and metadata_path.exists():
        return output_path
    folder.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = shape(aoi).bounds
    latitude = math.radians((miny + maxy) / 2.0)
    width = max(1, min(2500, int(abs(maxx - minx) * 111320 * math.cos(latitude) / RESOLUTION)))
    height = max(1, min(2500, int(abs(maxy - miny) * 111320 / RESOLUTION)))
    acquisition = get_datetime(item)
    if acquisition is None:
        raise RuntimeError("Could not read acquisition date.")
    payload = {
        "input": {
            "bounds": {"bbox": [minx, miny, maxx, maxy], "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": acquisition.strftime("%Y-%m-%dT00:00:00Z"), "to": (acquisition + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")}, "mosaickingOrder": "leastCC"}}],
        },
        "output": {"width": width, "height": height, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": evalscript(),
    }
    response = requests.post(PROCESS_URL, headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}, json=payload, timeout=900)
    if response.status_code >= 400:
        try:
            detail = response.json()
        except Exception:
            detail = response.text[:1000]
        raise RuntimeError(f"CDSE Process API failed ({response.status_code}): {detail}")
    output_path.write_bytes(response.content)
    metadata_path.write_text(json.dumps(item, indent=2), encoding="utf-8")
    return output_path


def read_stack(path):
    with rasterio.open(path) as source:
        array = source.read().astype(np.float32)
        profile = source.profile.copy()
    return {band: array[index] for index, band in enumerate(BANDS)}, profile


def normalized_difference(first, second):
    output = np.full(first.shape, np.nan, dtype=np.float32)
    denominator = first + second
    valid = np.isfinite(first) & np.isfinite(second) & (np.abs(denominator) > 1e-12)
    output[valid] = (first[valid] - second[valid]) / denominator[valid]
    return output


def calculate_lrad(bands, q_value):
    finite = np.logical_and.reduce([np.isfinite(bands[band]) for band in BANDS])
    artifact = ((bands["B11"] >= PARAMS["swir_saturation"]) & (bands["B12"] >= PARAMS["swir_saturation"]))
    artifact |= bands["B03"] <= q_value
    artifact |= normalized_difference(bands["B03"], bands["B08"]) >= PARAMS["ndwi_threshold"]
    artifact |= normalized_difference(bands["B08"], bands["B04"]) >= PARAMS["ndvi_threshold"]
    artifact |= normalized_difference(bands["B11"], bands["B08"]) >= PARAMS["ndbi_threshold"]
    artifact |= normalized_difference(bands["B03"], bands["B11"]) >= PARAMS["ndsi_threshold"]
    artifact |= ~finite
    artifact = binary_dilation(artifact, structure=disk(PARAMS["lrad_dilation"]))
    return finite & ~artifact


def calculate_c(b11, b12, valid):
    use = valid & np.isfinite(b11) & np.isfinite(b12) & (b11 > 0) & (b12 > 0)
    if use.sum() < 100:
        return 1.0
    x, y = b11[use].astype(np.float64), b12[use].astype(np.float64)
    return float(np.sum(x * y) / max(np.sum(x * x), 1e-20))


def calculate_mbsp(b11, b12, c, valid):
    output = np.full(b11.shape, np.nan, dtype=np.float32)
    use = valid & np.isfinite(b11) & np.isfinite(b12) & (np.abs(b12) > 1e-12)
    output[use] = c * (b12[use] - b11[use]) / b12[use]
    return output


def run_algorithm(target, reference):
    target_q = float(np.nanquantile(target["B03"], PARAMS["b03_quantile"]))
    reference_q = float(np.nanquantile(reference["B03"], PARAMS["b03_quantile"]))
    valid = calculate_lrad(target, target_q) & calculate_lrad(reference, reference_q)
    target_mbsp = calculate_mbsp(target["B11"], target["B12"], calculate_c(target["B11"], target["B12"], valid), valid)
    reference_mbsp = calculate_mbsp(reference["B11"], reference["B12"], calculate_c(reference["B11"], reference["B12"], valid), valid)
    relative = target_mbsp - reference_mbsp
    relative[~valid] = np.nan
    finite = np.isfinite(relative)
    if not finite.any():
        raise RuntimeError("LRAD removed all pixels. Reduce artifact thresholds or use a smaller valid AOI.")
    values = relative[finite].astype(np.float64)
    mean_value = float(np.mean(values))
    std_value = float(np.std(values))
    fill_value = mean_value
    gaussian = gaussian_filter(np.where(finite, relative, fill_value), sigma=PARAMS["gaussian_sigma"])
    threshold = mean_value + PARAMS["threshold_sigma"] * std_value
    initial = valid & np.isfinite(gaussian) & (gaussian > threshold)
    labels, _ = label(initial, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    retained = np.where(sizes >= PARAMS["min_component_pixels"])[0]
    retained = retained[retained != 0]
    connected = np.isin(labels, retained)
    final = binary_dilation(connected, structure=disk(PARAMS["final_dilation"])) & valid
    return {"relative": relative, "gaussian": gaussian, "valid": valid, "initial": initial, "connected": connected, "final": final, "mean": mean_value, "std": std_value, "threshold": threshold, "regions": int(len(retained)), "valid_count": int(valid.sum()), "initial_count": int(initial.sum()), "final_count": int(final.sum())}


def save_raster(path, array, profile, mask=False):
    output_profile = profile.copy()
    output_profile.update(count=1, dtype="uint8" if mask else "float32", nodata=255 if mask else -9999, compress="deflate", tiled=False, BIGTIFF="IF_SAFER")
    output = np.where(array, 1, 0).astype(np.uint8) if mask else np.where(np.isfinite(array), array, -9999).astype(np.float32)
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(output, 1)


def create_map(aoi):
    geometry = shape(ensure_aoi(aoi))
    fmap = folium.Map([geometry.centroid.y, geometry.centroid.x], zoom_start=7, tiles="OpenStreetMap")
    folium.GeoJson(mapping(geometry), style_function=lambda _: {"color": "blue", "fill": False}).add_to(fmap)
    Draw(export=True, draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False}).add_to(fmap)
    return fmap


def image_png(array, mask=False):
    from PIL import Image
    data = np.asarray(array)
    if mask:
        rgb = np.zeros((*data.shape, 3), dtype=np.uint8)
        rgb[data.astype(bool)] = [220, 30, 30]
    else:
        finite = np.isfinite(data)
        rgb = np.full((*data.shape, 3), 255, dtype=np.uint8)
        if finite.any():
            values = data[finite]
            low, high = float(np.percentile(values, 2)), float(np.percentile(values, 98))
            if high <= low:
                low, high = float(values.min()), float(values.max())
            if high > low:
                normalized = np.clip((np.nan_to_num(data, nan=low) - low) / (high - low), 0, 1)
                import matplotlib.pyplot as plt
                rgb = (plt.get_cmap("RdBu_r")(normalized)[:, :, :3] * 255).astype(np.uint8)
                rgb[~finite] = 255
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def legend_html(kind):
    """Readable standalone legend rendered outside each output image."""
    if kind == "mask":
        rows = [("#dc1e1e", "Methane candidate"), ("#000000", "Background / non-candidate")]
    elif kind == "valid":
        rows = [("#dc1e1e", "Valid pixels"), ("#000000", "Invalid / masked pixels")]
    else:
        rows = [("#b43232", "Higher anomaly"), ("#3250b4", "Lower anomaly"), ("#ffffff", "No data")]
    items = "".join(f'<div class="legend-row"><span class="legend-swatch" style="background:{c};"></span><span>{t}</span></div>' for c,t in rows)
    return f'<div class="result-legend"><div class="legend-heading">Legend</div>{items}</div>'


def create_png_worldfile(profile, array_shape):
    # PNG pixel coordinates are tied to a world file (.pgw).
    transform = profile["transform"]
    width = int(array_shape[1])
    height = int(array_shape[0])
    xres = transform.a
    yres = transform.e
    x_center = transform.c + xres / 2.0
    y_center = transform.f + yres / 2.0
    pgw = f"{xres:.12f}\n0.0\n0.0\n{yres:.12f}\n{x_center:.12f}\n{y_center:.12f}\n"
    crs_text = profile.get("crs")
    prj = crs_text.to_wkt() if crs_text else ""
    return pgw.encode("utf-8"), prj.encode("utf-8")


def georeferenced_png_package(array, profile, mask=False, legend_kind=None):
    import zipfile
    png_data = image_png(array, mask=mask)
    pgw_data, prj_data = create_png_worldfile(profile, np.asarray(array).shape)
    package = io.BytesIO()
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("output.png", png_data)
        archive.writestr("output.pgw", pgw_data)
        if prj_data:
            archive.writestr("output.prj", prj_data)
    return package.getvalue()


# ============================================================
# User Interface
# ============================================================

# ============================================================
# Compact App-like User Interface
# ============================================================
st.set_page_config(
    page_title="Sentinel-2 Methane",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root {
    --red: #e63946;
    --honeydew: #f1faee;
    --frost: #a8dadc;
    --blue: #457b9d;
    --navy: #1d3557;
    --black: #111111;
    --white: #ffffff;
    --border: #d8e6e8;
    --muted: #4f5d63;
}

/* ---------- Main application ---------- */
.stApp {
    background: #f1faee;
    color: #111111 !important;
}

/* Keep Streamlit's top toolbar separate from the application header. */
[data-testid="stHeader"] {
    background: #f1faee !important;
    height: 3.25rem !important;
}

[data-testid="stSidebar"] {
    display: none;
}

/* Push the custom header below Streamlit's top toolbar. */
.block-container {
    max-width: 1700px;
    padding-top: 3.9rem !important;
    padding-bottom: 0.8rem;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}

/* ---------- Header ---------- */
.app-header {
    position: relative;
    z-index: 10;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.75rem 1rem;
    margin-top: 0.15rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 2px 10px rgba(29,53,87,0.05);
}

.app-title {
    color: #111111 !important;
    font-size: 1.45rem;
    font-weight: 850;
    line-height: 1.1;
}

.app-subtitle {
    color: #111111 !important;
    font-size: 0.78rem;
    margin-top: 0.15rem;
}

.status-pill {
    background: #f1faee;
    color: #111111 !important;
    border: 1px solid #a8dadc;
    border-radius: 999px;
    padding: 0.35rem 0.7rem;
    font-size: 0.72rem;
    font-weight: 750;
    white-space: nowrap;
}

/* ---------- Cards ---------- */
.app-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 0.75rem;
    box-shadow: 0 2px 10px rgba(29,53,87,0.04);
    height: 100%;
    color: #111111 !important;
}

.card-title {
    color: #111111 !important;
    font-size: 1rem;
    font-weight: 800;
    margin-bottom: 0.1rem;
}

.card-caption {
    color: #111111 !important;
    font-size: 0.73rem;
    margin-bottom: 0.45rem;
}

.section-label {
    display: inline-block;
    background: #a8dadc;
    color: #111111 !important;
    border-radius: 999px;
    padding: 0.2rem 0.55rem;
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    margin-bottom: 0.35rem;
}

/* ---------- Force native Streamlit text to black ---------- */
.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp div,
.stApp small,
.stApp strong,
.stApp em,
.stApp li,
.stApp td,
.stApp th,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] span,
.stApp [data-testid="stMarkdownContainer"] li {
    color: #111111 !important;
}

/* Restore white text only where the UI intentionally needs it. */
.stButton > button[kind="primary"],
.stButton > button[kind="primary"] *,
.stDownloadButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] * {
    color: #ffffff !important;
}

/* ---------- Readable standalone legends ---------- */
.result-legend {
    background: #ffffff; border: 1px solid #d7e4e7; border-radius: 10px;
    padding: 0.75rem 0.7rem; min-height: 96px; box-sizing: border-box;
    display: flex; flex-direction: column; justify-content: center; gap: 0.42rem;
}
.result-legend .legend-heading { color: #111111 !important; font-size: 0.88rem; font-weight: 800; }
.result-legend .legend-row { display:flex; align-items:center; gap:0.45rem; color:#111111 !important; font-size:0.82rem; line-height:1.25; }
.result-legend .legend-row span:last-child { color:#111111 !important; }
.legend-swatch { width:18px; height:14px; min-width:18px; border:1px solid #555; border-radius:2px; display:inline-block; }

/* ---------- Inputs ---------- */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-radius: 9px;
}

div[data-baseweb="input"] input,
div[data-baseweb="select"] input,
div[data-baseweb="select"] [role="combobox"],
.stTextInput input,
.stNumberInput input,
.stDateInput input {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
}

/* Placeholder text */
input::placeholder,
textarea::placeholder {
    color: #4f5d63 !important;
    opacity: 1 !important;
}

/* Selectbox selected value and dropdown text */
div[data-baseweb="select"] * {
    color: #111111 !important;
}

[role="listbox"] *,
[role="option"] * {
    color: #111111 !important;
}

/* Dark dropdown popup: Target scene options must be white and readable */
div[data-baseweb="popover"] [role="listbox"],
div[data-baseweb="popover"] [role="option"],
div[data-baseweb="popover"] [role="option"] *,
ul[role="listbox"],
li[role="option"],
li[role="option"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

div[data-baseweb="popover"] [role="listbox"],
div[data-baseweb="popover"] ul[role="listbox"] {
    background: #111318 !important;
}

div[data-baseweb="popover"] [role="option"],
div[data-baseweb="popover"] li[role="option"] {
    background: #111318 !important;
}

div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] li[role="option"]:hover {
    background: #2b2e38 !important;
}

.stDateInput, .stSlider, .stNumberInput, .stSelectbox {
    margin-bottom: 0.15rem;
}

.stSlider > div {
    padding-top: 0.05rem;
    padding-bottom: 0.05rem;
}

/* Slider value / labels */
.stSlider label,
.stSlider [data-testid="stTickBar"] *,
.stSlider [data-testid="stThumbValue"] * {
    color: #111111 !important;
}

/* ---------- Dark native controls: keep their values white ---------- */
/* Streamlit/BaseWeb may render date, number and threshold controls with a dark field. */
div[data-baseweb="input"] input,
.stNumberInput input,
.stDateInput input,
.stNumberInput [data-baseweb="input"] input,
.stDateInput [data-baseweb="input"] input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Threshold/slider numeric value when it is displayed inside a dark thumb/track */
.stSlider [data-testid="stThumbValue"],
.stSlider [data-testid="stThumbValue"] *,
.stSlider [data-baseweb="slider"] [role="slider"] + div,
.stSlider [data-baseweb="slider"] [role="slider"] + div * {
    color: #ffffff !important;
}

/* Calendar/date picker values and controls when Streamlit opens a dark popup */
[data-baseweb="calendar"] *,
[data-baseweb="popover"] [data-baseweb="calendar"] *,
[data-baseweb="calendar"] button {
    color: #ffffff !important;
}

/* Keep dark input fields readable even when browser autofill is active */
input:-webkit-autofill,
input:-webkit-autofill:hover,
input:-webkit-autofill:focus {
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important;
}

/* Final override for date fields */
div[data-testid="stDateInput"] input,
div[data-testid="stDateInput"] [data-baseweb="input"] input,
.stDateInput input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important;
}

/* ---------- Copernicus login ---------- */
.auth-card {
    background: #f8fbfb;
    border: 1px solid #d7e4e7;
    border-radius: 11px;
    padding: 0.65rem 0.75rem;
    margin-top: 0.45rem;
}
.auth-status {
    background: #e8f7ea;
    border: 1px solid #9ed2a4;
    color: #155724 !important;
    border-radius: 9px;
    padding: 0.45rem 0.6rem;
    font-size: 0.76rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
}
.auth-help {
    color: #111111 !important;
    font-size: 0.72rem;
    line-height: 1.45;
    margin: 0.2rem 0 0.45rem 0;
}

/* ---------- Buttons ---------- */
.stButton > button,
.stDownloadButton > button {
    border-radius: 9px;
    min-height: 2.15rem;
    font-weight: 750;
    font-size: 0.78rem;
    color: #111111 !important;
}

.stButton > button[kind="primary"] {
    background: #e63946;
    border-color: #e63946;
    color: #ffffff !important;
}

.stButton > button[kind="primary"]:hover {
    background: #c92f3b;
    border-color: #c92f3b;
    color: #ffffff !important;
}

.stDownloadButton > button {
    background: #ffffff;
    color: #111111 !important;
    border: 1px solid #a8dadc;
}

.stDownloadButton > button:hover {
    background: #f1faee;
    border-color: #457b9d;
    color: #111111 !important;
}

/* ---------- Tables / dataframes ---------- */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
}

div[data-testid="stDataFrame"] * {
    color: #111111 !important;
}

/* ---------- Download labels / map ---------- */
.download-label {
    color: #111111 !important;
    font-size: 0.62rem;
    font-weight: 700;
    margin: 0.2rem 0 0.12rem 0;
}

.map-frame {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

/* ---------- Streamlit chrome ---------- */
footer { visibility: hidden; }

/* Tight vertical rhythm */
.stMarkdown { margin-bottom: 0.1rem; }
.element-container { margin-bottom: 0.15rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">🛰️ Sentinel-2 Methane Screening</div>
        <div class="app-subtitle">CDSE STAC + Process API &nbsp;|&nbsp; Relative MBMP candidate detection</div>
    </div>
    <div class="status-pill">20 m processing &nbsp;•&nbsp; Light dashboard</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Top dashboard: map + controls/search
# ============================================================
if "aoi" not in st.session_state:
    st.session_state.aoi = mapping(box(48.0, 29.0, 49.0, 30.0))

map_col, control_col = st.columns([1.65, 1.0], gap="small")

with map_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">01 · STUDY AREA</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Area of Interest</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption">Draw or edit the study area directly on the map.</div>', unsafe_allow_html=True)

    map_data = st_folium(
        create_map(st.session_state.aoi),
        height=385,
        width=1000,
        key="aoi_map"
    )

    if map_data and map_data.get("all_drawings"):
        new_aoi = normalize_geometry({
            "type": "FeatureCollection",
            "features": map_data["all_drawings"]
        })
        if new_aoi:
            st.session_state.aoi = new_aoi

    st.markdown('</div>', unsafe_allow_html=True)

with control_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 · SEARCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Scene Search</div>', unsafe_allow_html=True)

    d1, d2 = st.columns(2, gap="small")
    with d1:
        start_date = st.date_input(
            "Start date",
            datetime.now().date() - timedelta(days=30),
            key="start_date"
        )
    with d2:
        end_date = st.date_input(
            "End date",
            datetime.now().date(),
            key="end_date"
        )

    s1, s2 = st.columns(2, gap="small")
    with s1:
        max_cloud = st.slider(
            "Cloud cover (%)", 0.0, 100.0, 50.0, key="max_cloud"
        )
    with s2:
        reference_days = st.slider(
            "Reference window (days)", 1, 90, 60, key="reference_days"
        )

    if st.button("🔎  Search Sentinel-2 scenes", type="primary", use_container_width=True):
        try:
            with st.spinner("Searching CDSE STAC..."):
                st.session_state.items = search_scenes(
                    st.session_state.aoi,
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.max.time()),
                    max_cloud
                )
            st.session_state.pop("target", None)
            st.success(f"{len(st.session_state.items)} scene(s) found")
        except Exception as error:
            st.exception(error)

    items = st.session_state.get("items", [])

    if items:
        scene_table = pd.DataFrame([
            {
                "date": get_datetime(item),
                "tile": get_tile(item),
                "cloud": get_cloud(item)
            }
            for item in items
        ]).sort_values(["date", "cloud"], ascending=[True, True], na_position="last")

        st.dataframe(
            scene_table,
            use_container_width=True,
            height=112,
            hide_index=True,
            column_config={
                "date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
                "cloud": st.column_config.NumberColumn("Cloud %", format="%.1f")
            }
        )

        item_ids = [as_dict(item).get("id") for item in items]
        selected_id = st.selectbox(
            "Target scene",
            item_ids,
            format_func=lambda value: (
                f"{get_datetime(next(x for x in items if as_dict(x).get('id') == value)).strftime('%Y-%m-%d')}  |  "
                f"{get_tile(next(x for x in items if as_dict(x).get('id') == value))}  |  "
                f"cloud {get_cloud(next(x for x in items if as_dict(x).get('id') == value)):.1f}%"
            ),
            key="target_scene_select"
        )

        st.session_state.target = next(
            item for item in items if as_dict(item).get("id") == selected_id
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# Detection settings + action bar
# ============================================================
st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
settings_col, action_col = st.columns([1.65, 1.0], gap="small")

with settings_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 · DETECTION</div>', unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3, gap="small")
    with p1:
        PARAMS["threshold_sigma"] = st.number_input(
            "Threshold multiplier",
            min_value=0.1,
            max_value=5.0,
            value=float(PARAMS["threshold_sigma"]),
            step=0.1,
            key="threshold_sigma"
        )
    with p2:
        PARAMS["min_component_pixels"] = st.number_input(
            "Minimum candidate pixels",
            min_value=2,
            max_value=1000,
            value=50,
            step=5,
            key="min_component_pixels"
        )
    with p3:
        PARAMS["final_dilation"] = st.number_input(
            "Final dilation radius",
            min_value=0,
            max_value=20,
            value=int(PARAMS["final_dilation"]),
            step=1,
            key="final_dilation"
        )

    estimated_area_m2 = int(PARAMS["min_component_pixels"]) * RESOLUTION * RESOLUTION
    st.markdown(
        f'<div class="card-caption">Minimum connected region ≈ {estimated_area_m2:,} m² at {RESOLUTION} m resolution.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with action_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 · PROCESS</div>', unsafe_allow_html=True)
    items = st.session_state.get("items", [])

    # --------------------------------------------------------
    # Copernicus authentication
    # --------------------------------------------------------
    cdse_auth = st.session_state.get("cdse_auth")
    if cdse_auth:
        st.markdown(
            f'<div class="auth-status">✓ Copernicus connected · {cdse_auth.get("username", "")}</div>',
            unsafe_allow_html=True
        )
        logout_col, _ = st.columns([1, 2])
        with logout_col:
            if st.button("Log out", use_container_width=True, key="cdse_logout"):
                st.session_state.pop("cdse_auth", None)
                st.rerun()
    else:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Copernicus login</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="auth-help">Log in once in this browser session. Your password is sent directly to the official Copernicus identity service; the app keeps only the temporary API token.</div>',
            unsafe_allow_html=True
        )
        st.link_button(
            "🌐 Open Copernicus website",
            "https://dataspace.copernicus.eu/",
            use_container_width=True
        )
        with st.form("cdse_login_form", clear_on_submit=True):
            login_user = st.text_input("Copernicus email", placeholder="your-email@example.com")
            login_password = st.text_input("Copernicus password", type="password")
            login_totp = st.text_input("2FA code (optional)", max_chars=8, placeholder="Only if your account uses 2FA")
            login_submitted = st.form_submit_button("🔐 Login & connect", type="primary", use_container_width=True)

        if login_submitted:
            try:
                with st.spinner("Connecting to Copernicus…"):
                    st.session_state.cdse_auth = authenticate_cdse(login_user, login_password, login_totp)
                st.success("Copernicus login successful. You can now run the detection.")
                st.rerun()
            except Exception as login_error:
                st.error(str(login_error))
        st.markdown('</div>', unsafe_allow_html=True)

    if items and "target" in st.session_state:
        target = st.session_state.target
        st.markdown(
            f'<div class="card-title">Ready to detect</div>'
            f'<div class="card-caption">Target: {get_datetime(target).strftime("%Y-%m-%d")} · {get_tile(target)}</div>',
            unsafe_allow_html=True
        )

        detect_clicked = st.button(
            "🛰️  Download AOI & Detect Methane",
            type="primary",
            use_container_width=True,
            key="detect_button",
            disabled=not bool(st.session_state.get("cdse_auth"))
        )

        if not st.session_state.get("cdse_auth"):
            st.markdown(
                '<div class="card-caption">Please connect your Copernicus account above before downloading Sentinel-2 data.</div>',
                unsafe_allow_html=True
            )

        if detect_clicked:
            progress = st.progress(0, text="Preparing methane detection…")
            progress_status = st.empty()
            try:
                progress_status.markdown('<div class="card-caption">Step 1 of 5 · Connecting to CDSE and preparing the target scene…</div>', unsafe_allow_html=True)
                progress.progress(8, text="Preparing target scene…")
                access_token = get_access_token()
                target = st.session_state.target
                target_date = get_datetime(target)
                target_tile = get_tile(target)

                all_candidates = [
                    item for item in items
                    if as_dict(item).get("id") != as_dict(target).get("id")
                    and get_datetime(item)
                    and abs((get_datetime(item) - target_date).total_seconds()) / 86400 <= reference_days
                ]

                same_tile = [item for item in all_candidates if get_tile(item) == target_tile]
                references = same_tile if same_tile else all_candidates

                if not references:
                    st.error(
                        "No reference scene exists in the selected time window. "
                        "Increase the date range or reference window."
                    )
                    st.stop()

                progress_status.markdown('<div class="card-caption">Step 2 of 5 · Downloading target image bands and preparing the AOI…</div>', unsafe_allow_html=True)
                progress.progress(25, text="Downloading target bands…")
                target_bands, profile = read_stack(
                    download_scene(target, st.session_state.aoi, access_token)
                )

                best_reference = None
                best_correlation = -np.inf
                reference_rows = []

                total_refs = max(1, len(references))
                for ref_index, reference in enumerate(references, start=1):
                    pct = 30 + int(40 * (ref_index - 1) / total_refs)
                    progress_status.markdown(
                        f'<div class="card-caption">Step 3 of 5 · Downloading and comparing reference scene {ref_index} of {total_refs}…</div>',
                        unsafe_allow_html=True
                    )
                    progress.progress(pct, text=f"Reference scene {ref_index} of {total_refs}…")
                    reference_bands, _ = read_stack(
                        download_scene(reference, st.session_state.aoi, access_token)
                    )

                    valid_pixels = (
                        np.isfinite(target_bands["B04"])
                        & np.isfinite(reference_bands["B04"])
                    )

                    correlation = (
                        float(np.corrcoef(
                            target_bands["B04"][valid_pixels],
                            reference_bands["B04"][valid_pixels]
                        )[0, 1])
                        if valid_pixels.sum() > 100 else np.nan
                    )

                    reference_rows.append({
                        "id": as_dict(reference).get("id"),
                        "date": get_datetime(reference),
                        "tile": get_tile(reference),
                        "b4_correlation": correlation,
                        "valid_b4_pixels": int(valid_pixels.sum())
                    })

                    if np.isfinite(correlation) and correlation > best_correlation:
                        best_correlation = correlation
                        best_reference = reference_bands

                st.session_state.reference_table = pd.DataFrame(reference_rows)

                if best_reference is None:
                    st.error(
                        "Reference scenes were downloaded, but B4 correlation could not be calculated. "
                        "Check valid pixels and cloud cover."
                    )
                    st.stop()

                progress_status.markdown('<div class="card-caption">Step 4 of 5 · Running relative MBMP anomaly detection and candidate cleanup…</div>', unsafe_allow_html=True)
                progress.progress(78, text="Running methane detection…")
                # --------------------------------------------------------
                # IMPORTANT: detection calculation is unchanged.
                # --------------------------------------------------------
                result = run_algorithm(target_bands, best_reference)
                result["b4_correlation"] = best_correlation
                result["date"] = target_date.strftime("%Y-%m-%d")

                output_folder = RESULT_DIR / target_date.strftime("%Y%m%d")
                output_folder.mkdir(parents=True, exist_ok=True)

                paths = {}
                for key in ("relative", "gaussian", "final", "valid"):
                    paths[key] = output_folder / f"{key}.tif"
                    save_raster(
                        paths[key],
                        result[key],
                        profile,
                        key in ("final", "valid")
                    )

                st.session_state.result = result
                st.session_state.paths = paths
                st.session_state.output_profile = profile
                st.session_state.png_outputs = {
                    "relative": image_png(result["relative"]),
                    "gaussian": image_png(result["gaussian"]),
                    "final": image_png(result["final"], mask=True),
                    "valid": image_png(result["valid"], mask=True),
                }
                st.session_state.legend_outputs = {
                    "relative": legend_html("continuous"),
                    "gaussian": legend_html("continuous"),
                    "final": legend_html("mask"),
                    "valid": legend_html("valid"),
                }
                progress_status.markdown('<div class="card-caption">Step 5 of 5 · Saving georeferenced outputs and preparing downloads…</div>', unsafe_allow_html=True)
                progress.progress(100, text="Ready to detect · outputs are ready")
                st.success("Processing completed")
            except Exception as error:
                st.exception(error)
    else:
        st.markdown(
            '<div class="card-title">Select scenes first</div>'
            '<div class="card-caption">Search for Sentinel-2 scenes, select a target, then run the detection.</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# Results dashboard — compact, all together
# ============================================================
if "result" in st.session_state:
    result = st.session_state.result
    png_outputs = st.session_state.get("png_outputs", {})
    legend_outputs = st.session_state.get("legend_outputs", {})
    profile = st.session_state.get("output_profile")

    st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">05 · RESULTS</div>', unsafe_allow_html=True)

    metrics = st.columns(6, gap="small")
    metrics[0].metric("B4 correlation", f"{result['b4_correlation']:.3f}")
    metrics[1].metric("Valid pixels", f"{result['valid_count']:,}")
    metrics[2].metric("Initial", f"{result['initial_count']:,}")
    metrics[3].metric("Final", f"{result['final_count']:,}")
    metrics[4].metric("Regions", result["regions"])
    metrics[5].metric("Threshold", f"{result['threshold']:.5f}")

    result_items = [
        ("relative", "Relative MBMP", "Anomaly"),
        ("gaussian", "Gaussian filtered", "Smoothed"),
        ("final", "Methane candidates", "Final mask"),
        ("valid", "Valid pixels", "Validity mask"),
    ]

    result_cols = st.columns(4, gap="small")
    for col, (key, title, tag) in zip(result_cols, result_items):
        with col:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="result-tag">{tag}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-name">{title}</div>', unsafe_allow_html=True)
            preview_col, legend_col = st.columns([3.6, 1.0], gap="small")
            with preview_col:
                st.image(
                    png_outputs[key],
                    use_container_width=True,
                    output_format="PNG"
                )
            with legend_col:
                 st.markdown('<div style="padding-top:0.35rem;"></div>', unsafe_allow_html=True)
                 st.markdown(legend_html("mask" if key == "final" else "valid" if key == "valid" else "continuous"), unsafe_allow_html=True)

            path = st.session_state.paths[key]
            format_choice = st.selectbox(
                "Download format",
                ["GeoTIFF (georeferenced)", "PNG + World File (georeferenced)"],
                key=f"format_choice_{key}"
            )
            if format_choice == "GeoTIFF (georeferenced)":
                st.download_button(
                    "⬇ Download GeoTIFF", path.read_bytes(),
                    file_name=path.name, mime="image/tiff",
                    key=f"download_tif_compact_{key}", use_container_width=True
                )
            else:
                # A PNG cannot store raster CRS/geotransform like GeoTIFF; the ZIP
                # keeps PNG + PGW + PRJ together so GIS software can georeference it.
                png_package = georeferenced_png_package(
                    result[key], profile, mask=key in ("final", "valid")
                )
                st.download_button(
                    "⬇ Download Georeferenced PNG package", png_package,
                    file_name=f"{key}_georeferenced_png.zip", mime="application/zip",
                    key=f"download_png_compact_{key}", use_container_width=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

    removed_pixels = max(0, int(result["initial_count"]) - int(result["final_count"]))
    st.markdown(
        f'<div class="result-note"><b>Candidate cleanup:</b> regions smaller than '
        f'<b>{int(PARAMS["min_component_pixels"]):,} pixels</b> were excluded. '
        f'Initial: <b>{result["initial_count"]:,}</b> → Final: <b>{result["final_count"]:,}</b> '
        f'· Removed: <b>{removed_pixels:,}</b> · {result["regions"]:,} connected regions.</div>',
        unsafe_allow_html=True
    )

    d1, d2 = st.columns([1, 3], gap="small")
    with d1:
        st.download_button(
            "⬇ Reference table CSV",
            st.session_state.reference_table.to_csv(index=False),
            file_name="reference_selection.csv",
            mime="text/csv",
            key="download_reference_csv_compact",
            use_container_width=True
        )
    with d2:
        st.markdown(
            '<div class="card-caption" style="margin-top:0.55rem;">'
            'Relative MBMP anomaly and candidate mask are screening outputs, not physical methane concentration or emission rate.'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


/* Threshold/slider numeric value when it is displayed inside a dark thumb/track */
.stSlider [data-testid="stThumbValue"],
.stSlider [data-testid="stThumbValue"] *,
.stSlider [data-baseweb="slider"] [role="slider"] + div,
.stSlider [data-baseweb="slider"] [role="slider"] + div * {
    color: #ffffff !important;
}

/* Calendar/date picker values and controls when Streamlit opens a dark popup */
[data-baseweb="calendar"] *,
[data-baseweb="popover"] [data-baseweb="calendar"] *,
[data-baseweb="calendar"] button {
    color: #ffffff !important;
}

/* ---------- Start date / End date text ---------- */
.stDateInput input,
.stDateInput input[type="text"] {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important;
}

/* Keep dark input fields readable even when browser autofill is active */
input:-webkit-autofill,
input:-webkit-autofill:hover,
input:-webkit-autofill:focus {
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important;
}

/* Final override for dark date and number fields */
.stDateInput input, .stNumberInput input, div[data-baseweb="input"] input {
    color:#ffffff !important; -webkit-text-fill-color:#ffffff !important; caret-color:#ffffff !important;
}

/* ---------- Copernicus login ---------- */
.auth-card {
    background: #f8fbfb;
    border: 1px solid #d7e4e7;
    border-radius: 11px;
    padding: 0.65rem 0.75rem;
    margin-top: 0.45rem;
}
.auth-status {
    background: #e8f7ea;
    border: 1px solid #9ed2a4;
    color: #155724 !important;
    border-radius: 9px;
    padding: 0.45rem 0.6rem;
    font-size: 0.76rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
}
.auth-help {
    color: #111111 !important;
    font-size: 0.72rem;
    line-height: 1.45;
    margin: 0.2rem 0 0.45rem 0;
}

/* ---------- Buttons ---------- */
.stButton > button,
.stDownloadButton > button {
    border-radius: 9px;
    min-height: 2.15rem;
    font-weight: 750;
    font-size: 0.78rem;
    color: #111111 !important;
}

.stButton > button[kind="primary"] {
    background: #e63946;
    border-color: #e63946;
    color: #ffffff !important;
}

.stButton > button[kind="primary"]:hover {
    background: #c92f3b;
    border-color: #c92f3b;
    color: #ffffff !important;
}

.stDownloadButton > button {
    background: #ffffff;
    color: #111111 !important;
    border: 1px solid #a8dadc;
}

.stDownloadButton > button:hover {
    background: #f1faee;
    border-color: #457b9d;
    color: #111111 !important;
}

/* ---------- Tables / dataframes ---------- */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
}

div[data-testid="stDataFrame"] * {
    color: #111111 !important;
}

/* ---------- Download labels / map ---------- */
.download-label {
    color: #111111 !important;
    font-size: 0.62rem;
    font-weight: 700;
    margin: 0.2rem 0 0.12rem 0;
}

.map-frame {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

/* ---------- Streamlit chrome ---------- */
footer { visibility: hidden; }

/* Tight vertical rhythm */
.stMarkdown { margin-bottom: 0.1rem; }
.element-container { margin-bottom: 0.15rem; }


</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">🛰️ Sentinel-2 Methane Screening</div>
        <div class="app-subtitle">CDSE STAC + Process API &nbsp;|&nbsp; Relative MBMP candidate detection</div>
    </div>
    <div class="status-pill">20 m processing &nbsp;•&nbsp; Light dashboard</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Top dashboard: map + controls/search
# ============================================================
if "aoi" not in st.session_state:
    st.session_state.aoi = mapping(box(48.0, 29.0, 49.0, 30.0))

map_col, control_col = st.columns([1.65, 1.0], gap="small")

with map_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">01 · STUDY AREA</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Area of Interest</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-caption">Draw or edit the study area directly on the map.</div>', unsafe_allow_html=True)

    map_data = st_folium(
        create_map(st.session_state.aoi),
        height=385,
        width=1000,
        key="aoi_map"
    )

    if map_data and map_data.get("all_drawings"):
        new_aoi = normalize_geometry({
            "type": "FeatureCollection",
            "features": map_data["all_drawings"]
        })
        if new_aoi:
            st.session_state.aoi = new_aoi

    st.markdown('</div>', unsafe_allow_html=True)

with control_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 · SEARCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Scene Search</div>', unsafe_allow_html=True)

    d1, d2 = st.columns(2, gap="small")
    with d1:
        start_date = st.date_input(
            "Start date",
            datetime.now().date() - timedelta(days=30),
            key="start_date"
        )
    with d2:
        end_date = st.date_input(
            "End date",
            datetime.now().date(),
            key="end_date"
        )

    s1, s2 = st.columns(2, gap="small")
    with s1:
        max_cloud = st.slider(
            "Cloud cover (%)", 0.0, 100.0, 50.0, key="max_cloud"
        )
    with s2:
        reference_days = st.slider(
            "Reference window (days)", 1, 90, 60, key="reference_days"
        )

    if st.button("🔎  Search Sentinel-2 scenes", type="primary", use_container_width=True):
        try:
            with st.spinner("Searching CDSE STAC..."):
                st.session_state.items = search_scenes(
                    st.session_state.aoi,
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.max.time()),
                    max_cloud
                )
            st.session_state.pop("target", None)
            st.success(f"{len(st.session_state.items)} scene(s) found")
        except Exception as error:
            st.exception(error)

    items = st.session_state.get("items", [])

    if items:
        scene_table = pd.DataFrame([
            {
                "date": get_datetime(item),
                "tile": get_tile(item),
                "cloud": get_cloud(item)
            }
            for item in items
        ]).sort_values(["date", "cloud"], ascending=[True, True], na_position="last")

        st.dataframe(
            scene_table,
            use_container_width=True,
            height=112,
            hide_index=True,
            column_config={
                "date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
                "cloud": st.column_config.NumberColumn("Cloud %", format="%.1f")
            }
        )

        item_ids = [as_dict(item).get("id") for item in items]
        selected_id = st.selectbox(
            "Target scene",
            item_ids,
            format_func=lambda value: (
                f"{get_datetime(next(x for x in items if as_dict(x).get('id') == value)).strftime('%Y-%m-%d')}  |  "
                f"{get_tile(next(x for x in items if as_dict(x).get('id') == value))}  |  "
                f"cloud {get_cloud(next(x for x in items if as_dict(x).get('id') == value)):.1f}%"
            ),
            key="target_scene_select"
        )

        st.session_state.target = next(
            item for item in items if as_dict(item).get("id") == selected_id
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# Detection settings + action bar
# ============================================================
st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
settings_col, action_col = st.columns([1.65, 1.0], gap="small")

with settings_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 · DETECTION</div>', unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3, gap="small")
    with p1:
        PARAMS["threshold_sigma"] = st.number_input(
            "Threshold multiplier",
            min_value=0.1,
            max_value=5.0,
            value=float(PARAMS["threshold_sigma"]),
            step=0.1,
            key="threshold_sigma"
        )
    with p2:
        PARAMS["min_component_pixels"] = st.number_input(
            "Minimum candidate pixels",
            min_value=2,
            max_value=1000,
            value=50,
            step=5,
            key="min_component_pixels"
        )
    with p3:
        PARAMS["final_dilation"] = st.number_input(
            "Final dilation radius",
            min_value=0,
            max_value=20,
            value=int(PARAMS["final_dilation"]),
            step=1,
            key="final_dilation"
        )

    estimated_area_m2 = int(PARAMS["min_component_pixels"]) * RESOLUTION * RESOLUTION
    st.markdown(
        f'<div class="card-caption">Minimum connected region ≈ {estimated_area_m2:,} m² at {RESOLUTION} m resolution.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with action_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 · PROCESS</div>', unsafe_allow_html=True)
    items = st.session_state.get("items", [])

    # --------------------------------------------------------
    # Copernicus authentication
    # --------------------------------------------------------
    cdse_auth = st.session_state.get("cdse_auth")
    if cdse_auth:
        st.markdown(
            f'<div class="auth-status">✓ Copernicus connected · {cdse_auth.get("username", "")}</div>',
            unsafe_allow_html=True
        )
        logout_col, _ = st.columns([1, 2])
        with logout_col:
            if st.button("Log out", use_container_width=True, key="cdse_logout"):
                st.session_state.pop("cdse_auth", None)
                st.rerun()
    else:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Copernicus login</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="auth-help">Log in once in this browser session. Your password is sent directly to the official Copernicus identity service; the app keeps only the temporary API token.</div>',
            unsafe_allow_html=True
        )
        st.link_button(
            "🌐 Open Copernicus website",
            "https://dataspace.copernicus.eu/",
            use_container_width=True
        )
        with st.form("cdse_login_form", clear_on_submit=True):
            login_user = st.text_input("Copernicus email", placeholder="your-email@example.com")
            login_password = st.text_input("Copernicus password", type="password")
            login_totp = st.text_input("2FA code (optional)", max_chars=8, placeholder="Only if your account uses 2FA")
            login_submitted = st.form_submit_button("🔐 Login & connect", type="primary", use_container_width=True)

        if login_submitted:
            try:
                with st.spinner("Connecting to Copernicus…"):
                    st.session_state.cdse_auth = authenticate_cdse(login_user, login_password, login_totp)
                st.success("Copernicus login successful. You can now run the detection.")
                st.rerun()
            except Exception as login_error:
                st.error(str(login_error))
        st.markdown('</div>', unsafe_allow_html=True)

    if items and "target" in st.session_state:
        target = st.session_state.target
        st.markdown(
            f'<div class="card-title">Ready to detect</div>'
            f'<div class="card-caption">Target: {get_datetime(target).strftime("%Y-%m-%d")} · {get_tile(target)}</div>',
            unsafe_allow_html=True
        )

        detect_clicked = st.button(
            "🛰️  Download AOI & Detect Methane",
            type="primary",
            use_container_width=True,
            key="detect_button",
            disabled=not bool(st.session_state.get("cdse_auth"))
        )

        if not st.session_state.get("cdse_auth"):
            st.markdown(
                '<div class="card-caption">Please connect your Copernicus account above before downloading Sentinel-2 data.</div>',
                unsafe_allow_html=True
            )

        if detect_clicked:
            progress = st.progress(0, text="Preparing methane detection…")
            progress_status = st.empty()
            try:
                progress_status.markdown('<div class="card-caption">Step 1 of 5 · Connecting to CDSE and preparing the target scene…</div>', unsafe_allow_html=True)
                progress.progress(8, text="Preparing target scene…")
                access_token = get_access_token()
                target = st.session_state.target
                target_date = get_datetime(target)
                target_tile = get_tile(target)

                all_candidates = [
                    item for item in items
                    if as_dict(item).get("id") != as_dict(target).get("id")
                    and get_datetime(item)
                    and abs((get_datetime(item) - target_date).total_seconds()) / 86400 <= reference_days
                ]

                same_tile = [item for item in all_candidates if get_tile(item) == target_tile]
                references = same_tile if same_tile else all_candidates

                if not references:
                    st.error(
                        "No reference scene exists in the selected time window. "
                        "Increase the date range or reference window."
                    )
                    st.stop()

                progress_status.markdown('<div class="card-caption">Step 2 of 5 · Downloading target image bands and preparing the AOI…</div>', unsafe_allow_html=True)
                progress.progress(25, text="Downloading target bands…")
                target_bands, profile = read_stack(
                    download_scene(target, st.session_state.aoi, access_token)
                )

                best_reference = None
                best_correlation = -np.inf
                reference_rows = []

                total_refs = max(1, len(references))
                for ref_index, reference in enumerate(references, start=1):
                    pct = 30 + int(40 * (ref_index - 1) / total_refs)
                    progress_status.markdown(
                        f'<div class="card-caption">Step 3 of 5 · Downloading and comparing reference scene {ref_index} of {total_refs}…</div>',
                        unsafe_allow_html=True
                    )
                    progress.progress(pct, text=f"Reference scene {ref_index} of {total_refs}…")
                    reference_bands, _ = read_stack(
                        download_scene(reference, st.session_state.aoi, access_token)
                    )

                    valid_pixels = (
                        np.isfinite(target_bands["B04"])
                        & np.isfinite(reference_bands["B04"])
                    )

                    correlation = (
                        float(np.corrcoef(
                            target_bands["B04"][valid_pixels],
                            reference_bands["B04"][valid_pixels]
                        )[0, 1])
                        if valid_pixels.sum() > 100 else np.nan
                    )

                    reference_rows.append({
                        "id": as_dict(reference).get("id"),
                        "date": get_datetime(reference),
                        "tile": get_tile(reference),
                        "b4_correlation": correlation,
                        "valid_b4_pixels": int(valid_pixels.sum())
                    })

                    if np.isfinite(correlation) and correlation > best_correlation:
                        best_correlation = correlation
                        best_reference = reference_bands

                st.session_state.reference_table = pd.DataFrame(reference_rows)

                if best_reference is None:
                    st.error(
                        "Reference scenes were downloaded, but B4 correlation could not be calculated. "
                        "Check valid pixels and cloud cover."
                    )
                    st.stop()

                progress_status.markdown('<div class="card-caption">Step 4 of 5 · Running relative MBMP anomaly detection and candidate cleanup…</div>', unsafe_allow_html=True)
                progress.progress(78, text="Running methane detection…")
                # --------------------------------------------------------
                # IMPORTANT: detection calculation is unchanged.
                # --------------------------------------------------------
                result = run_algorithm(target_bands, best_reference)
                result["b4_correlation"] = best_correlation
                result["date"] = target_date.strftime("%Y-%m-%d")

                output_folder = RESULT_DIR / target_date.strftime("%Y%m%d")
                output_folder.mkdir(parents=True, exist_ok=True)

                paths = {}
                for key in ("relative", "gaussian", "final", "valid"):
                    paths[key] = output_folder / f"{key}.tif"
                    save_raster(
                        paths[key],
                        result[key],
                        profile,
                        key in ("final", "valid")
                    )

                st.session_state.result = result
                st.session_state.paths = paths
                st.session_state.output_profile = profile
                st.session_state.png_outputs = {
                    "relative": image_png(result["relative"]),
                    "gaussian": image_png(result["gaussian"]),
                    "final": image_png(result["final"], mask=True),
                    "valid": image_png(result["valid"], mask=True),
                }
                st.session_state.legend_outputs = {
                    "relative": legend_html("continuous"),
                    "gaussian": legend_html("continuous"),
                    "final": legend_html("mask"),
                    "valid": legend_html("valid"),
                }
                progress_status.markdown('<div class="card-caption">Step 5 of 5 · Saving georeferenced outputs and preparing downloads…</div>', unsafe_allow_html=True)
                progress.progress(100, text="Ready to detect · outputs are ready")
                st.success("Processing completed")
            except Exception as error:
                st.exception(error)
    else:
        st.markdown(
            '<div class="card-title">Select scenes first</div>'
            '<div class="card-caption">Search for Sentinel-2 scenes, select a target, then run the detection.</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# Results dashboard — compact, all together
# ============================================================
if "result" in st.session_state:
    result = st.session_state.result
    png_outputs = st.session_state.get("png_outputs", {})
    legend_outputs = st.session_state.get("legend_outputs", {})
    profile = st.session_state.get("output_profile")

    st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">05 · RESULTS</div>', unsafe_allow_html=True)

    metrics = st.columns(6, gap="small")
    metrics[0].metric("B4 correlation", f"{result['b4_correlation']:.3f}")
    metrics[1].metric("Valid pixels", f"{result['valid_count']:,}")
    metrics[2].metric("Initial", f"{result['initial_count']:,}")
    metrics[3].metric("Final", f"{result['final_count']:,}")
    metrics[4].metric("Regions", result["regions"])
    metrics[5].metric("Threshold", f"{result['threshold']:.5f}")

    result_items = [
        ("relative", "Relative MBMP", "Anomaly"),
        ("gaussian", "Gaussian filtered", "Smoothed"),
        ("final", "Methane candidates", "Final mask"),
        ("valid", "Valid pixels", "Validity mask"),
    ]

    result_cols = st.columns(4, gap="small")
    for col, (key, title, tag) in zip(result_cols, result_items):
        with col:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="result-tag">{tag}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-name">{title}</div>', unsafe_allow_html=True)
            preview_col, legend_col = st.columns([3.6, 1.0], gap="small")
            with preview_col:
                st.image(
                    png_outputs[key],
                    use_container_width=True,
                    output_format="PNG"
                )
            with legend_col:
                 st.markdown('<div style="padding-top:0.35rem;"></div>', unsafe_allow_html=True)
                 st.markdown(legend_html("mask" if key == "final" else "valid" if key == "valid" else "continuous"), unsafe_allow_html=True)

            path = st.session_state.paths[key]
            format_choice = st.selectbox(
                "Download format",
                ["GeoTIFF (georeferenced)", "PNG + World File (georeferenced)"],
                key=f"format_choice_{key}"
            )
            if format_choice == "GeoTIFF (georeferenced)":
                st.download_button(
                    "⬇ Download GeoTIFF", path.read_bytes(),
                    file_name=path.name, mime="image/tiff",
                    key=f"download_tif_compact_{key}", use_container_width=True
                )
            else:
                # A PNG cannot store raster CRS/geotransform like GeoTIFF; the ZIP
                # keeps PNG + PGW + PRJ together so GIS software can georeference it.
                png_package = georeferenced_png_package(
                    result[key], profile, mask=key in ("final", "valid")
                )
                st.download_button(
                    "⬇ Download Georeferenced PNG package", png_package,
                    file_name=f"{key}_georeferenced_png.zip", mime="application/zip",
                    key=f"download_png_compact_{key}", use_container_width=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

    removed_pixels = max(0, int(result["initial_count"]) - int(result["final_count"]))
    st.markdown(
        f'<div class="result-note"><b>Candidate cleanup:</b> regions smaller than '
        f'<b>{int(PARAMS["min_component_pixels"]):,} pixels</b> were excluded. '
        f'Initial: <b>{result["initial_count"]:,}</b> → Final: <b>{result["final_count"]:,}</b> '
        f'· Removed: <b>{removed_pixels:,}</b> · {result["regions"]:,} connected regions.</div>',
        unsafe_allow_html=True
    )

    d1, d2 = st.columns([1, 3], gap="small")
    with d1:
        st.download_button(
            "⬇ Reference table CSV",
            st.session_state.reference_table.to_csv(index=False),
            file_name="reference_selection.csv",
            mime="text/csv",
            key="download_reference_csv_compact",
            use_container_width=True
        )
    with d2:
        st.markdown(
            '<div class="card-caption" style="margin-top:0.55rem;">'
            'Relative MBMP anomaly and candidate mask are screening outputs, not physical methane concentration or emission rate.'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)
