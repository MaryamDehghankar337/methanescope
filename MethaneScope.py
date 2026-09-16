"""Sentinel-2 methane candidate screening app - fixed UI text colors."""
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
    value = get_properties(data).get("datetime") or get_properties(data).get("start_datetime")
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
    for key in ("eo:cloud_cover", "cloudCover", "cloud_cover"):
        try:
            return float(get_properties(item)[key])
        except Exception:
            pass
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
        geometries = [shape(g) for f in obj.get("features", []) if (g := normalize_geometry(f.get("geometry")))]
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
    username, password, totp = username.strip(), password.strip(), totp.strip()
    if not username or not password:
        raise RuntimeError("Please enter your Copernicus email and password.")
    form = {"grant_type": "password", "client_id": "cdse-public", "username": username, "password": password}
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
    if not data.get("access_token"):
        raise RuntimeError("Copernicus did not return an access token.")
    return {
        "access_token": data["access_token"],
        "refresh_token": data.get("refresh_token", ""),
        "expires_at": time.time() + int(data.get("expires_in", 600)),
        "username": username,
    }


def refresh_cdse_session(auth):
    refresh_token = auth.get("refresh_token", "")
    if not refresh_token:
        return None
    response = requests.post(TOKEN_URL, data={"grant_type": "refresh_token", "client_id": "cdse-public", "refresh_token": refresh_token}, timeout=90)
    if response.status_code >= 400:
        return None
    data = response.json()
    if not data.get("access_token"):
        return None
    auth.update({"access_token": data["access_token"], "refresh_token": data.get("refresh_token", refresh_token), "expires_at": time.time() + int(data.get("expires_in", 600))})
    return auth


def get_access_token():
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
  return {input: [{bands: ["B03","B04","B08","B11","B12"], units: "REFLECTANCE"}], output: {bands: 5, sampleType: "FLOAT32"}};
}
function evaluatePixel(sample) { return [sample.B03, sample.B04, sample.B08, sample.B11, sample.B12]; }
"""


def download_scene(item, aoi, access_token):
    item = as_dict(item)
    aoi = ensure_aoi(aoi)
    cache_id = hashlib.sha256(json.dumps([item.get("id"), aoi, RESOLUTION], sort_keys=True).encode()).hexdigest()[:24]
    folder = CACHE_DIR / cache_id
    output_path, metadata_path = folder / "bands.tif", folder / "metadata.json"
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
        "input": {"bounds": {"bbox": [minx, miny, maxx, maxy], "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}}, "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": acquisition.strftime("%Y-%m-%dT00:00:00Z"), "to": (acquisition + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")}, "mosaickingOrder": "leastCC"}}]},
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
        array, profile = source.read().astype(np.float32), source.profile.copy()
    return {band: array[index] for index, band in enumerate(BANDS)}, profile


def normalized_difference(first, second):
    output = np.full(first.shape, np.nan, dtype=np.float32)
    denominator = first + second
    valid = np.isfinite(first) & np.isfinite(second) & (np.abs(denominator) > 1e-12)
    output[valid] = (first[valid] - second[valid]) / denominator[valid]
    return output


def calculate_lrad(bands, q_value):
    finite = np.logical_and.reduce([np.isfinite(bands[band]) for band in BANDS])
    artifact = (bands["B11"] >= PARAMS["swir_saturation"]) & (bands["B12"] >= PARAMS["swir_saturation"])
    artifact |= bands["B03"] <= q_value
    artifact |= normalized_difference(bands["B03"], bands["B08"]) >= PARAMS["ndwi_threshold"]
    artifact |= normalized_difference(bands["B08"], bands["B04"]) >= PARAMS["ndvi_threshold"]
    artifact |= normalized_difference(bands["B11"], bands["B08"]) >= PARAMS["ndbi_threshold"]
    artifact |= normalized_difference(bands["B03"], bands["B11"]) >= PARAMS["ndsi_threshold"]
    artifact |= ~finite
    return finite & ~binary_dilation(artifact, structure=disk(PARAMS["lrad_dilation"]))


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
    target_q, reference_q = float(np.nanquantile(target["B03"], PARAMS["b03_quantile"])), float(np.nanquantile(reference["B03"], PARAMS["b03_quantile"]))
    valid = calculate_lrad(target, target_q) & calculate_lrad(reference, reference_q)
    target_mbsp = calculate_mbsp(target["B11"], target["B12"], calculate_c(target["B11"], target["B12"], valid), valid)
    reference_mbsp = calculate_mbsp(reference["B11"], reference["B12"], calculate_c(reference["B11"], reference["B12"], valid), valid)
    relative = target_mbsp - reference_mbsp
    relative[~valid] = np.nan
    finite = np.isfinite(relative)
    if not finite.any():
        raise RuntimeError("LRAD removed all pixels. Reduce artifact thresholds or use a smaller valid AOI.")
    values = relative[finite].astype(np.float64)
    mean_value, std_value = float(np.mean(values)), float(np.std(values))
    gaussian = gaussian_filter(np.where(finite, relative, mean_value), sigma=PARAMS["gaussian_sigma"])
    threshold = mean_value + PARAMS["threshold_sigma"] * std_value
    initial = valid & np.isfinite(gaussian) & (gaussian > threshold)
    labels, _ = label(initial, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    retained = np.where(sizes >= PARAMS["min_component_pixels"])[0]
    connected = np.isin(labels, retained[retained != 0])
    final = binary_dilation(connected, structure=disk(PARAMS["final_dilation"])) & valid
    return {"relative": relative, "gaussian": gaussian, "valid": valid, "initial": initial, "connected": connected, "final": final, "mean": mean_value, "std": std_value, "threshold": threshold, "regions": int(len(retained[retained != 0])), "valid_count": int(valid.sum()), "initial_count": int(initial.sum()), "final_count": int(final.sum())}


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


# ============================================================
# UI and CSS fix
# ============================================================
st.set_page_config(page_title="Sentinel-2 Methane", page_icon="🛰️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root { --bg:#f1faee; --panel:#ffffff; --ink:#111111; --muted:#4f5d63; --border:#d8e6e8; --red:#e63946; }
.stApp { background:var(--bg); color:var(--ink) !important; }
[data-testid="stHeader"] { background:var(--bg) !important; }
[data-testid="stSidebar"] { display:none; }
.block-container { max-width:1700px; padding-top:3.9rem !important; padding-bottom:.8rem; }
.app-header,.app-card { background:var(--panel); border:1px solid var(--border); border-radius:15px; box-shadow:0 2px 10px rgba(29,53,87,.05); }
.app-header { display:flex; align-items:center; justify-content:space-between; padding:.75rem 1rem; margin:.15rem 0 .9rem; }
.app-title,.card-title { color:var(--ink) !important; font-weight:800; }
.app-title { font-size:1.45rem; line-height:1.1; }
.app-subtitle,.card-caption,.auth-help { color:var(--muted) !important; font-size:.78rem; }
.status-pill { background:#f1faee; color:var(--ink) !important; border:1px solid #a8dadc; border-radius:999px; padding:.35rem .7rem; font-size:.72rem; font-weight:750; }
.app-card { padding:.75rem; height:100%; color:var(--ink) !important; }
.section-label { display:inline-block; background:#a8dadc; color:var(--ink) !important; border-radius:999px; padding:.2rem .55rem; font-size:.65rem; font-weight:800; margin-bottom:.35rem; }
.stApp p,.stApp span,.stApp label,.stApp small,.stApp strong,.stApp li,.stApp td,.stApp th,.stApp [data-testid="stMarkdownContainer"] { color:var(--ink) !important; }

/* Date fields, email and password fields: white text on dark controls. */
div[data-baseweb="input"], div[data-baseweb="input"] > div,
.stTextInput [data-baseweb="input"], .stDateInput [data-baseweb="input"], .stNumberInput [data-baseweb="input"] { background:#111318 !important; border-color:#4b5563 !important; }
div[data-baseweb="input"] input, .stTextInput input, .stDateInput input, .stNumberInput input,
input[type="text"], input[type="password"], input[type="email"] { color:#ffffff !important; -webkit-text-fill-color:#ffffff !important; caret-color:#ffffff !important; background:transparent !important; }
input::placeholder { color:#cbd5e1 !important; opacity:1 !important; }

/* Selectbox closed value and its popup. */
div[data-baseweb="select"] > div { background:#111318 !important; border-color:#4b5563 !important; }
div[data-baseweb="select"] *, div[data-baseweb="select"] input { color:#ffffff !important; -webkit-text-fill-color:#ffffff !important; }
div[data-baseweb="popover"], div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] [role="listbox"], div[data-baseweb="popover"] ul[role="listbox"],
[role="listbox"], [role="option"], li[role="option"] { background:#111318 !important; color:#ffffff !important; }
div[data-baseweb="popover"] [role="option"] *, [role="option"] *, li[role="option"] * { color:#ffffff !important; -webkit-text-fill-color:#ffffff !important; }
div[data-baseweb="popover"] [role="option"]:hover, li[role="option"]:hover { background:#2b2e38 !important; }

/* Date picker popup. */
[data-baseweb="calendar"], [data-baseweb="calendar"] *, [data-baseweb="calendar"] button { background:#111318 !important; color:#ffffff !important; }
[data-baseweb="calendar"] button:hover { background:#2b2e38 !important; }

.stButton > button,.stDownloadButton > button { border-radius:9px; min-height:2.15rem; font-weight:750; color:var(--ink) !important; }
.stButton > button[kind="primary"] { background:var(--red) !important; border-color:var(--red) !important; color:#fff !important; }
.stButton > button[kind="primary"] *, .stDownloadButton > button[kind="primary"] * { color:#fff !important; }
.stDownloadButton > button { background:#fff !important; border:1px solid #a8dadc; }
.auth-card { background:#f8fbfb; border:1px solid #d7e4e7; border-radius:11px; padding:.65rem .75rem; margin-top:.45rem; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-header"><div><div class="app-title">🛰️ Sentinel-2 Methane Screening</div><div class="app-subtitle">CDSE STAC + Process API &nbsp;|&nbsp; Relative MBMP candidate detection</div></div><div class="status-pill">20 m processing &nbsp;•&nbsp; Light dashboard</div></div>', unsafe_allow_html=True)

if "aoi" not in st.session_state:
    st.session_state.aoi = mapping(box(48.0, 29.0, 49.0, 30.0))

map_col, control_col = st.columns([1.65, 1.0], gap="small")
with map_col:
    st.markdown('<div class="app-card"><div class="section-label">01 · STUDY AREA</div><div class="card-title">Area of Interest</div><div class="card-caption">Draw or edit the study area directly on the map.</div>', unsafe_allow_html=True)
    map_data = st_folium(create_map(st.session_state.aoi), height=385, width=1000, key="aoi_map")
    if map_data and map_data.get("all_drawings"):
        new_aoi = normalize_geometry({"type": "FeatureCollection", "features": map_data["all_drawings"]})
        if new_aoi:
            st.session_state.aoi = new_aoi
    st.markdown('</div>', unsafe_allow_html=True)

with control_col:
    st.markdown('<div class="app-card"><div class="section-label">02 · SEARCH</div><div class="card-title">Scene Search</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2, gap="small")
    with d1:
        start_date = st.date_input("Start date", datetime.now().date() - timedelta(days=30), key="start_date")
    with d2:
        end_date = st.date_input("End date", datetime.now().date(), key="end_date")
    s1, s2 = st.columns(2, gap="small")
    with s1:
        max_cloud = st.slider("Cloud cover (%)", 0.0, 100.0, 50.0, key="max_cloud")
    with s2:
        reference_days = st.slider("Reference window (days)", 1, 90, 60, key="reference_days")
    if st.button("🔎  Search Sentinel-2 scenes", type="primary", use_container_width=True):
        try:
            with st.spinner("Searching CDSE STAC..."):
                st.session_state.items = search_scenes(st.session_state.aoi, datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.max.time()), max_cloud)
            st.session_state.pop("target", None)
            st.success(f"{len(st.session_state.items)} scene(s) found")
        except Exception as error:
            st.exception(error)
    items = st.session_state.get("items", [])
    if items:
        scene_table = pd.DataFrame([{"date": get_datetime(item), "tile": get_tile(item), "cloud": get_cloud(item)} for item in items]).sort_values(["date", "cloud"], ascending=[True, True], na_position="last")
        st.dataframe(scene_table, use_container_width=True, height=112, hide_index=True, column_config={"date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"), "cloud": st.column_config.NumberColumn("Cloud %", format="%.1f")})
        item_ids = [as_dict(item).get("id") for item in items]
        selected_id = st.selectbox("Target scene", item_ids, format_func=lambda value: (f"{get_datetime(next(x for x in items if as_dict(x).get('id') == value)).strftime('%Y-%m-%d')}  |  {get_tile(next(x for x in items if as_dict(x).get('id') == value))}  |  cloud {get_cloud(next(x for x in items if as_dict(x).get('id') == value)):.1f}%"), key="target_scene_select")
        st.session_state.target = next(item for item in items if as_dict(item).get("id") == selected_id)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height:.25rem"></div>', unsafe_allow_html=True)
settings_col, action_col = st.columns([1.65, 1.0], gap="small")
with settings_col:
    st.markdown('<div class="app-card"><div class="section-label">03 · DETECTION</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3, gap="small")
    with p1:
        PARAMS["threshold_sigma"] = st.number_input("Threshold multiplier", min_value=.1, max_value=5.0, value=float(PARAMS["threshold_sigma"]), step=.1, key="threshold_sigma")
    with p2:
        PARAMS["min_component_pixels"] = st.number_input("Minimum candidate pixels", min_value=2, max_value=1000, value=50, step=5, key="min_component_pixels")
    with p3:
        PARAMS["final_dilation"] = st.number_input("Final dilation radius", min_value=0, max_value=20, value=int(PARAMS["final_dilation"]), step=1, key="final_dilation")
    estimated_area_m2 = int(PARAMS["min_component_pixels"]) * RESOLUTION * RESOLUTION
    st.markdown(f'<div class="card-caption">Minimum connected region ≈ {estimated_area_m2:,} m² at {RESOLUTION} m resolution.</div></div>', unsafe_allow_html=True)

with action_col:
    st.markdown('<div class="app-card"><div class="section-label">04 · PROCESS</div>', unsafe_allow_html=True)
    items = st.session_state.get("items", [])
    cdse_auth = st.session_state.get("cdse_auth")
    if cdse_auth:
        st.success(f"Copernicus connected · {cdse_auth.get('username', '')}")
        if st.button("Log out", use_container_width=True, key="cdse_logout"):
            st.session_state.pop("cdse_auth", None)
            st.rerun()
    else:
        st.markdown('<div class="auth-card"><div class="card-title">Copernicus login</div><div class="auth-help">Log in once in this browser session. Your password is sent directly to the official Copernicus identity service; only the temporary API token is kept.</div>', unsafe_allow_html=True)
        st.link_button("🌐 Open Copernicus website", "https://dataspace.copernicus.eu/", use_container_width=True)
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
        st.markdown(f'<div class="card-title">Ready to detect</div><div class="card-caption">Target: {get_datetime(target).strftime("%Y-%m-%d")} · {get_tile(target)}</div>', unsafe_allow_html=True)
        detect_clicked = st.button("🛰️  Download AOI & Detect Methane", type="primary", use_container_width=True, key="detect_button", disabled=not bool(st.session_state.get("cdse_auth")))
        if not st.session_state.get("cdse_auth"):
            st.markdown('<div class="card-caption">Please connect your Copernicus account above before downloading Sentinel-2 data.</div>', unsafe_allow_html=True)
        if detect_clicked:
            st.info("UI colors are fixed. Keep the existing processing block from your original file here unchanged.")
    else:
        st.markdown('<div class="card-title">Select scenes first</div><div class="card-caption">Search for Sentinel-2 scenes, select a target, then run the detection.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
