"""
Kraken — Vessel Detection Web UI
Run with:  streamlit run app.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st

# Inject Streamlit Cloud secrets into env before config.settings loads
for _key in ("COPERNICUS_USER", "COPERNICUS_PASSWORD", "GFW_API_TOKEN"):
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.detect.cfar_detect import detect_scene
from src.match.ais_match import run_ais_matching
from config.settings import PROCESSED_DIR, OUTPUTS_DIR, CFAR_FALSE_ALARM_RATE, AIS_DIR, GFW_API_TOKEN

# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Kraken · Vessel Detection",
    page_icon="🦑",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
[data-testid="stSidebar"] { background: #0D1117; }
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stFileUploader label { color: #6B7FA3 !important; font-size: 12px !important; }
.metric-card {
    background: #111827; border: 1px solid #1E3048;
    border-radius: 8px; padding: 16px 20px; text-align: center;
}
.metric-val { font-size: 32px; font-weight: 700; font-family: monospace; }
.metric-lbl { font-size: 12px; color: #6B7FA3; margin-top: 4px; }
.dark-val   { color: #FF4444; }
.match-val  { color: #22C55E; }
.total-val  { color: #00D4FF; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def _pfa_label(exp: int) -> str:
    return f"1e{exp}"


@st.cache_data(show_spinner=False)
def _list_processed_scenes() -> list[str]:
    tifs = sorted(PROCESSED_DIR.glob("*_processed.tif"))
    return [t.name for t in tifs]


@st.cache_data(show_spinner=False)
def _list_ais_files() -> list[str]:
    AIS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(f.name for f in AIS_DIR.glob("*.csv"))


def _build_map(detections: list[dict], center_lat: float, center_lon: float) -> folium.Map:
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        prefer_canvas=True,
    )
    for d in detections:
        lat, lon = d.get("lat"), d.get("lon")
        if lat is None or lon is None:
            continue
        dark     = d.get("dark_vessel", True)
        color    = "#FF4444" if dark else "#22C55E"
        v_class  = d.get("vessel_class", "unknown")
        conf     = d.get("conf")
        conf_str = f"{conf:.2f}" if conf is not None else "n/a"
        length   = d.get("length_m", 0)
        mmsi     = d.get("matched_vessel_id") or "—"
        name     = d.get("matched_vessel_name") or "—"

        popup_html = f"""
        <div style='font-family:monospace;font-size:12px;min-width:180px'>
          <b style='color:{color}'>{'🔴 DARK VESSEL' if dark else '🟢 AIS MATCHED'}</b><br>
          <hr style='margin:4px 0;border-color:#333'>
          Class: {v_class}<br>
          Length: {length:.0f} m<br>
          Conf: {conf_str}<br>
          MMSI: {mmsi}<br>
          Name: {name}<br>
          Lat: {lat:.5f} &nbsp; Lon: {lon:.5f}
        </div>"""

        folium.CircleMarker(
            location=[lat, lon],
            radius=7 if dark else 6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"{'DARK' if dark else 'MATCHED'} · {v_class} · {length:.0f} m",
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def _detections_to_df(detections: list[dict]) -> pd.DataFrame:
    rows = []
    for d in detections:
        rows.append({
            "lat":          round(d.get("lat", 0), 5),
            "lon":          round(d.get("lon", 0), 5),
            "vessel_class": d.get("vessel_class", ""),
            "length_m":     round(d.get("length_m", 0), 1),
            "width_m":      round(d.get("width_m", 0), 1),
            "conf":         round(d.get("conf") or 0, 3),
            "dark_vessel":  d.get("dark_vessel", True),
            "mmsi":         d.get("matched_vessel_id") or "",
            "vessel_name":  d.get("matched_vessel_name") or "",
            "flag":         d.get("matched_flag") or "",
        })
    return pd.DataFrame(rows)


def _df_to_geojson(df: pd.DataFrame) -> str:
    features = []
    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
            "properties": row.drop(["lat", "lon"]).to_dict(),
        })
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2)


# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🦑 KRAKEN")
    st.markdown("*SAR Vessel Detection*")
    st.divider()

    # ── scene selection ────────────────────────────────────────────────────────
    st.markdown("#### Scene")
    scene_source = st.radio(
        "Source",
        ["Existing processed scene", "Upload .tif"],
        label_visibility="collapsed",
    )

    uploaded_tif = None
    selected_scene_name = None

    if scene_source == "Upload .tif":
        uploaded_tif = st.file_uploader("Upload preprocessed GeoTIFF", type=["tif", "tiff"])
    else:
        scene_names = _list_processed_scenes()
        if scene_names:
            selected_scene_name = st.selectbox("Scene", scene_names)
        else:
            st.warning("No processed scenes found in data/processed/")

    st.divider()

    # ── detection params ───────────────────────────────────────────────────────
    st.markdown("#### Detection")
    pfa_exp = st.slider(
        "False alarm rate (PFA)",
        min_value=-9, max_value=-4, value=-6, step=1,
        format="1e%d",
        help="Lower = stricter = fewer but higher-confidence detections",
    )
    pfa = 10 ** pfa_exp

    land_mask = st.checkbox("Apply land mask", value=True,
                            help="Remove detections over land (recommended)")

    st.divider()

    # ── AIS matching ───────────────────────────────────────────────────────────
    st.markdown("#### AIS Matching")

    local_ais = _list_ais_files()
    _ais_options = []
    if GFW_API_TOKEN:
        _ais_options.append("🛰 GFW Live API")
    _ais_options += ["Local file (data/ais/)", "Upload CSV"]

    ais_source = st.radio(
        "AIS source",
        _ais_options,
        label_visibility="collapsed",
    )

    ais_file = None
    selected_ais_name = None

    if ais_source == "🛰 GFW Live API":
        st.success("Live AIS pulled automatically for each scene's time window.")
    elif ais_source == "Upload CSV":
        ais_file = st.file_uploader(
            "AIS CSV",
            type=["csv"],
            help="Columns: lat, lon, mmsi, vessel_name, flag, timestamp",
        )
    else:
        if local_ais:
            selected_ais_name = st.selectbox("AIS file", local_ais)
            st.caption("Drop CSVs into `data/ais/` — auto-loaded each run.")
        else:
            st.info("No CSVs in `data/ais/` yet.")
            ais_file = st.file_uploader(
                "AIS CSV",
                type=["csv"],
                help="Columns: lat, lon, mmsi, vessel_name, flag, timestamp",
            )

    if not GFW_API_TOKEN and not ais_file and not selected_ais_name:
        st.caption("No AIS source — all detections will be dark.")

    ais_radius = st.slider("Match radius (km)", 0.5, 10.0, 2.0, 0.5)

    st.divider()

    run_btn = st.button("▶  Run Detection", type="primary", use_container_width=True)


# ── main area ──────────────────────────────────────────────────────────────────

st.markdown("# Vessel Detection")
st.markdown(
    "Sentinel-1 SAR · CA-CFAR dual-band · AIS dark vessel cross-reference",
    help="Gulf of Guinea IUU fishing surveillance",
)

if "detections" not in st.session_state:
    st.session_state.detections = []
    st.session_state.scene_name = ""

# ── run ────────────────────────────────────────────────────────────────────────

if run_btn:
    scene_path = None

    scene_name_hint = None  # original filename for timestamp extraction
    if scene_source == "Upload .tif" and uploaded_tif:
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp.write(uploaded_tif.read())
        tmp.flush()
        scene_path = Path(tmp.name)
        scene_name_hint = Path(uploaded_tif.name)  # keep S1 naming for timestamp
        st.session_state.scene_name = uploaded_tif.name
    elif selected_scene_name:
        scene_path = PROCESSED_DIR / selected_scene_name
        st.session_state.scene_name = selected_scene_name
    else:
        st.error("Select or upload a scene first.")
        st.stop()

    ais_csv_path = None
    if ais_file:
        tmp_ais = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb")
        tmp_ais.write(ais_file.read())
        tmp_ais.flush()
        ais_csv_path = Path(tmp_ais.name)
    elif selected_ais_name:
        ais_csv_path = AIS_DIR / selected_ais_name

    with st.spinner("Running CFAR detection…"):
        try:
            detections = detect_scene(
                scene_path,
                pfa=pfa,
                use_land_mask=land_mask,
                out_dir=OUTPUTS_DIR,
                dual_band=True,
            )
        except Exception as e:
            st.error(f"Detection failed: {e}")
            st.stop()

    gfw_token = GFW_API_TOKEN or None
    ais_label = (
        "GFW live API" if (gfw_token and not ais_csv_path)
        else (ais_csv_path.name if ais_csv_path else "none — all vessels dark")
    )
    with st.spinner(f"Cross-referencing AIS ({ais_label})…"):
        detections = run_ais_matching(
            detections,
            scene_path=scene_name_hint or scene_path,  # use original name for timestamp
            radius_km=ais_radius,
            api_token=gfw_token,
            ais_csv_path=ais_csv_path,
        )

    st.session_state.detections = detections
    st.rerun()

# ── results ────────────────────────────────────────────────────────────────────

detections = st.session_state.detections

if not detections:
    st.info("Configure parameters in the sidebar and click **▶ Run Detection** to start.")
    st.stop()

n_total   = len(detections)
n_dark    = sum(1 for d in detections if d.get("dark_vessel"))
n_matched = n_total - n_dark
scene_label = st.session_state.scene_name

st.markdown(f"**Scene:** `{scene_label}`")

# metrics row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-val total-val'>{n_total}</div>
        <div class='metric-lbl'>Total detections</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-val dark-val'>{n_dark}</div>
        <div class='metric-lbl'>🔴 Dark vessels</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-val match-val'>{n_matched}</div>
        <div class='metric-lbl'>🟢 AIS matched</div></div>""", unsafe_allow_html=True)
with c4:
    pct = round(n_dark / n_total * 100) if n_total else 0
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-val' style='color:#F59E0B'>{pct}%</div>
        <div class='metric-lbl'>Dark rate</div></div>""", unsafe_allow_html=True)

st.markdown("")

# ── map ────────────────────────────────────────────────────────────────────────

lats = [d["lat"] for d in detections if d.get("lat")]
lons = [d["lon"] for d in detections if d.get("lon")]
center_lat = float(np.mean(lats)) if lats else 1.5
center_lon = float(np.mean(lons)) if lons else 2.5

m = _build_map(detections, center_lat, center_lon)
map_html = m._repr_html_()
st.components.v1.html(map_html, height=500, scrolling=False)

st.caption("🔴 Dark vessel (no AIS)  ·  🟢 AIS matched  ·  Click markers for details")

# ── table ──────────────────────────────────────────────────────────────────────

st.markdown("### Detections")

df = _detections_to_df(detections)

# filter controls
col_f1, col_f2, _ = st.columns([1, 1, 2])
with col_f1:
    show_dark = st.checkbox("Dark only", value=False)
with col_f2:
    classes = ["All"] + sorted(df["vessel_class"].unique().tolist())
    class_filter = st.selectbox("Class filter", classes, label_visibility="collapsed")

filtered = df.copy()
if show_dark:
    filtered = filtered[filtered["dark_vessel"]]
if class_filter != "All":
    filtered = filtered[filtered["vessel_class"] == class_filter]

st.dataframe(
    filtered.style.apply(
        lambda row: ["background-color: #2E0F0F" if row["dark_vessel"] else
                     "background-color: #0F2E1A"] * len(row),
        axis=1,
    ),
    use_container_width=True,
    height=300,
)

# ── downloads ──────────────────────────────────────────────────────────────────

st.markdown("### Download results")
dl1, dl2 = st.columns(2)

with dl1:
    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button(
        "⬇  Download CSV",
        data=csv_bytes,
        file_name=f"{Path(scene_label).stem}_detections.csv",
        mime="text/csv",
        use_container_width=True,
    )

with dl2:
    geojson_str = _df_to_geojson(filtered)
    st.download_button(
        "⬇  Download GeoJSON",
        data=geojson_str,
        file_name=f"{Path(scene_label).stem}_detections.geojson",
        mime="application/geo+json",
        use_container_width=True,
    )
