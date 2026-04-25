"""
app.py  —  KisanDrishti 🌾
---------------------------
Streamlit UI for Crop Health Monitoring
- GEE se Sentinel-2 data fetch
- Crop health + disease prediction
- Interactive Folium map (green/orange/red points)
- Stress points ke liye image upload → disease detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from PIL import Image
import io
import time
import pickle
import base64
from pathlib import Path
from datetime import datetime, timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KisanDrishti — Crop Health Monitor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0d1117;
    color: #e6edf3;
  }

  .main { background: #0d1117; }

  .kisan-header {
    background: linear-gradient(135deg, #1a3a2a 0%, #0d2b1a 50%, #0d1117 100%);
    border: 1px solid #2ea043;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .kisan-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(46,160,67,0.15) 0%, transparent 70%);
    pointer-events: none;
  }
  .kisan-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #3fb950;
    margin: 0 0 4px 0;
    letter-spacing: -1px;
  }
  .kisan-header p {
    color: #8b949e;
    font-size: 0.95rem;
    margin: 0;
  }

  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #3fb950; }
  .metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
  }
  .metric-card .label {
    font-size: 0.78rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
  }

  .green-val  { color: #3fb950; }
  .orange-val { color: #f0883e; }
  .red-val    { color: #f85149; }

  .step-badge {
    display: inline-block;
    background: #1f2d1f;
    border: 1px solid #2ea043;
    color: #3fb950;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
  }

  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 12px;
  }

  .disease-card {
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border-left: 4px solid;
  }
  .disease-detected { background: #2d1b1b; border-color: #f85149; }
  .no-disease       { background: #1b2d1b; border-color: #3fb950; }

  .point-row {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .dot-green  { background: #3fb950; }
  .dot-orange { background: #f0883e; }
  .dot-red    { background: #f85149; }

  div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
  }
  div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(46,160,67,0.3) !important;
  }

  .stDataFrame { border-radius: 10px; overflow: hidden; }

  .upload-zone {
    background: #161b22;
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .upload-zone:hover { border-color: #3fb950; }

  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-right: 16px;
    font-size: 0.85rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

@st.cache_resource
def load_ml_models():
    """Models ek baar load karo, cache karo."""
    try:
        with open("models/model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        with open("models/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        # with open("crop_desease_model.pkl", "rb") as f:
        #     disease_model = pickle.load(f)
        return rf_model, label_encoder #,  disease_model, None
    except FileNotFoundError as e:
        return None, None #, None, str(e)


def get_map_color(health: str, disease: str) -> str:
    h = health.lower()
    if "stress" in h:
        return "red"
    elif "moderate" in h:
        return "orange"
    return "green"


def predict_disease_from_image(disease_model, uploaded_image) -> dict:
    """
    Uploaded crop image se disease predict karo.
    Model image features expect karta hai — yahan hum
    simple color-based proxy features use karte hain.
    (Apne model ke actual preprocessing se replace karna)
    """
    img = Image.open(uploaded_image).convert("RGB").resize((64, 64))
    arr = np.array(img).astype(np.float32) / 255.0

    # Simple color features as proxy
    r, g, b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
    # Vegetation-like indices from RGB
    ndvi_proxy = (g - r) / (g + r + 1e-6)
    yellowness  = r - b

    features = np.array([[ndvi_proxy, yellowness, r, g, b, r-g]])

    try:
        pred = disease_model.predict(features)[0]
        prob = disease_model.predict_proba(features)[0].max() * 100
        p_str = str(pred).lower()
        is_disease = p_str in ["1", "true", "disease", "yes", "diseased"]
        label = "Disease Detected ⚠️" if is_disease else "No Disease ✅"
        return {
            "label": label,
            "is_disease": is_disease,
            "confidence": f"{prob:.1f}%",
        }
    except Exception as ex:
        return {"label": f"Error: {ex}", "is_disease": False, "confidence": "N/A"}


def build_farm_map(df: pd.DataFrame) -> folium.Map:
    """Folium map banao with colored markers."""
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
    )

    color_map = {"red": "#f85149", "orange": "#f0883e", "green": "#3fb950"}
    icon_map  = {"red": "exclamation-triangle", "orange": "exclamation-circle", "green": "check-circle"}

    for _, row in df.iterrows():
        clr = row["map_color"]
        popup_html = f"""
        <div style="font-family:sans-serif; min-width:160px;">
          <b style="font-size:14px;">📍 {row['point_id']}</b><br/>
          <hr style="margin:4px 0;"/>
          <b>Crop Health:</b> {row['crop_health']}<br/>
          <b>Disease:</b> {row['disease']}<br/>
          <hr style="margin:4px 0;"/>
          <small>NDVI: {row['NDVI']:.3f} | NDWI: {row['NDWI']:.3f}</small><br/>
          <small>EVI: {row['EVI']:.3f}  | SAVI: {row['SAVI']:.3f}</small>
        </div>
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=14,
            color=color_map[clr],
            fill=True,
            fill_color=color_map[clr],
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{row['point_id']}: {row['crop_health']}",
        ).add_to(m)

        # Point ID label
        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.DivIcon(
                html=f'<div style="color:white;font-weight:700;font-size:11px;'
                     f'text-shadow:0 0 3px black;margin-top:-8px;margin-left:18px;">'
                     f'{row["point_id"]}</div>',
                icon_size=(30, 20),
            ),
        ).add_to(m)

    # Farm boundary
    bbox = [
        [30.920, 74.600], [30.920, 74.625],
        [30.945, 74.625], [30.945, 74.600],
        [30.920, 74.600],
    ]
    folium.PolyLine(
        bbox,
        color="#3fb950",
        weight=2.5,
        opacity=0.7,
        dash_array="6 4",
        tooltip="Firozpur Farm Boundary",
    ).add_to(m)

    return m


# ── Session state init ────────────────────────────────────────────────────────
if "predictions_df"  not in st.session_state: st.session_state.predictions_df  = None
if "fetch_done"      not in st.session_state: st.session_state.fetch_done      = False
if "gee_project"     not in st.session_state: st.session_state.gee_project     = ""


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px;">
      <span style="font-size:2.5rem;">🌾</span>
      <h2 style="font-family:'Syne',sans-serif; color:#3fb950; margin:4px 0;">KisanDrishti</h2>
      <p style="color:#8b949e; font-size:0.8rem;">Satellite-powered Crop Health Monitor</p>
    </div>
    <hr style="border-color:#30363d; margin: 12px 0;"/>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ GEE Settings")
    gee_project = st.text_input(
        "GEE Project ID",
        placeholder="your-gee-project-id",
        help="Google Earth Engine project ID jo tumne register kiya hai",
    )
    days_back = st.slider("📅 Days back (data range)", 30, 90, 50, 5,
                          help="Kitne din peeche tak ka Sentinel-2 data fetch karna hai")
    cloud_pct = st.slider("☁️ Max Cloud Cover %", 5, 50, 20, 5)

    st.markdown("---")
    st.markdown("### 🗺️ Farm Info")
    st.markdown("""
    <div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px; font-size:0.82rem;">
      <b>📍 Location:</b> Firozpur, Punjab<br/>
      <b>🛰️ Satellite:</b> Sentinel-2 (10m)<br/>
      <b>📐 Points:</b> 10 sample locations<br/>
      <b>🌿 Indices:</b> NDVI, NDWI, EVI, SAVI
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem; color:#8b949e; text-align:center;">
      🟢 Healthy &nbsp;|&nbsp; 🟡 Moderate &nbsp;|&nbsp; 🔴 Stress
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="kisan-header">
  <h2>🌾 KisanDrishti</h2>
  <p>Firozpur Farm · Punjab · Sentinel-2 Satellite · Last {days_back} days of data</p>
</div>
""", unsafe_allow_html=True)

# Models status
rf_model, label_encoder, disease_model, model_err = load_ml_models()
if model_err:
    st.error(f"⚠️ Models load nahi hue: {model_err}\n\nMake sure `rf_model.pkl`, `label_encoder.pkl`, `crop_desease_model.pkl` isi folder mein hain.")
else:
    st.success("✅ Teenon models load ho gaye hain!")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FETCH DATA
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-badge">STEP 1</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">📡 Sentinel-2 Data Fetch (GEE)</div>', unsafe_allow_html=True)

col_btn1, col_btn2 = st.columns([1, 3])
with col_btn1:
    fetch_clicked = st.button("🛰️ GEE se Data Fetch Karo", use_container_width=True)

with col_btn2:
    upload_csv = st.file_uploader(
        "Ya pehle se download ki CSV upload karo",
        type=["csv"],
        label_visibility="collapsed",
    )

if upload_csv:
    df_uploaded = pd.read_csv(upload_csv)
    required = {"point_id", "lat", "lon", "NDVI", "NDWI", "EVI", "SAVI"}
    if required.issubset(df_uploaded.columns):
        st.session_state.features_df = df_uploaded
        st.success(f"✅ CSV load ho gayi — {len(df_uploaded)} points milein!")
    else:
        st.error(f"❌ CSV mein ye columns chahiye: {required}")

if fetch_clicked:
    if not gee_project.strip():
        st.error("❌ Sidebar mein GEE Project ID daalo pehle!")
    else:
        with st.spinner("🛰️ Google Earth Engine se data aa raha hai... (1-2 min lag sakta hai)"):
            try:
                import ee
                from fetch_sentinel import fetch_sentinel2_data, authenticate_gee, SAMPLE_POINTS

                # Project ID dynamically set karo
                import fetch_sentinel as fs_module
                fs_module.authenticate_gee = lambda: (
                    ee.Authenticate() or ee.Initialize(project=gee_project)
                )

                df_fetched = fetch_sentinel2_data(
                    days_back=days_back,
                    cloud_cover=cloud_pct,
                )
                st.session_state.features_df = df_fetched
                st.success(f"✅ {len(df_fetched)} points ka data fetch ho gaya!")
            except ImportError:
                st.error("❌ `earthengine-api` install nahi hai. Chalao: `pip install earthengine-api`")
            except Exception as e:
                st.error(f"❌ GEE error: {e}")

# Show raw features if available
if "features_df" in st.session_state:
    with st.expander("📊 Raw Spectral Data dekho", expanded=False):
        st.dataframe(
            st.session_state.features_df.style.background_gradient(
                subset=["NDVI", "EVI"], cmap="Greens"
            ).background_gradient(
                subset=["NDWI"], cmap="Blues"
            ),
            use_container_width=True,
        )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — RUN PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-badge">STEP 2</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔮 Crop Health & Disease Prediction</div>', unsafe_allow_html=True)

predict_clicked = st.button(
    "🌿 Predictions Chalao",
    disabled=("features_df" not in st.session_state or rf_model is None),
    use_container_width=False,
)

if predict_clicked and "features_df" in st.session_state:
    from predict import predict_crop_health, predict_disease, get_map_color

    df = st.session_state.features_df.copy()
    X  = df[["NDVI", "NDWI", "EVI", "SAVI"]].values.astype(np.float32)

    with st.spinner("🔮 Models predictions kar rahe hain..."):
        time.sleep(0.5)  # UX ke liye
        df["crop_health"]  = predict_crop_health(rf_model, label_encoder, X)
        df["disease"]      = predict_disease(disease_model, X)
        df["map_color"]    = [get_map_color(h, d) for h, d in zip(df["crop_health"], df["disease"])]

        health_score_map = {"healthy": 100, "moderate": 60, "stress": 20}
        df["health_score"] = df["crop_health"].apply(
            lambda x: health_score_map.get(x.lower(), 50)
        )

    st.session_state.predictions_df = df
    df.to_csv("predictions.csv", index=False)
    st.success("✅ Predictions complete!")

# ─── Metrics row ──────────────────────────────────────────────────────────────
if st.session_state.predictions_df is not None:
    df = st.session_state.predictions_df
    n_healthy  = (df["map_color"] == "green").sum()
    n_moderate = (df["map_color"] == "orange").sum()
    n_stress   = (df["map_color"] == "red").sum()
    n_disease  = df["disease"].str.contains("Detected", case=False, na=False).sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"""<div class="metric-card"><div class="value green-val">{n_healthy}</div><div class="label">🟢 Healthy Points</div></div>""", unsafe_allow_html=True)
    m2.markdown(f"""<div class="metric-card"><div class="value orange-val">{n_moderate}</div><div class="label">🟡 Moderate Points</div></div>""", unsafe_allow_html=True)
    m3.markdown(f"""<div class="metric-card"><div class="value red-val">{n_stress}</div><div class="label">🔴 Stress Points</div></div>""", unsafe_allow_html=True)
    m4.markdown(f"""<div class="metric-card"><div class="value red-val">{n_disease}</div><div class="label">⚠️ Disease Detected</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3 — MAP
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="step-badge">STEP 3</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗺️ Farm Health Map — Firozpur, Punjab</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:12px; font-size:0.85rem;">
      <span class="legend-item"><span class="dot dot-green"></span> Healthy</span>
      <span class="legend-item"><span class="dot dot-orange"></span> Moderate Stress</span>
      <span class="legend-item"><span class="dot dot-red"></span> High Stress / Disease Risk</span>
      <span style="color:#8b949e; font-size:0.8rem; margin-left:8px;">👆 Kisi bhi point pe click karo details dekhne ke liye</span>
    </div>
    """, unsafe_allow_html=True)

    farm_map = build_farm_map(df)
    map_output = st_folium(farm_map, width="100%", height=480, returned_objects=[])

    # Points list
    with st.expander("📋 Sabhi Points ki Details", expanded=True):
        for _, row in df.iterrows():
            clr = row["map_color"]
            dot_class = f"dot-{clr}"
            st.markdown(f"""
            <div class="point-row">
              <span class="dot {dot_class}"></span>
              <div style="flex:1;">
                <b>{row['point_id']}</b> &nbsp;
                <span style="color:#8b949e; font-size:0.82rem;">{row['lat']:.4f}°N, {row['lon']:.4f}°E</span>
              </div>
              <div style="font-size:0.85rem; color:#e6edf3; min-width:100px;">{row['crop_health']}</div>
              <div style="font-size:0.85rem; min-width:150px; color:{'#f85149' if 'Detected' in str(row['disease']) else '#3fb950'};">{row['disease']}</div>
              <div style="font-size:0.78rem; color:#8b949e;">NDVI {row['NDVI']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

    # CSV download
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Results CSV Download Karo",
        data=csv_data,
        file_name=f"farm_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4 — DISEASE DETECTION FROM IMAGE (Stress points only)
    # ═══════════════════════════════════════════════════════════════════════════
    stress_points = df[df["map_color"] == "red"]["point_id"].tolist()
    moderate_points = df[df["map_color"] == "orange"]["point_id"].tolist()
    upload_candidates = stress_points + moderate_points

    st.markdown('<div class="step-badge">STEP 4</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📸 Stress Points ke Crops ki Photo Upload Karo</div>', unsafe_allow_html=True)

    if not upload_candidates:
        st.success("🎉 Koi bhi stress/moderate point nahi hai! Saari fasal theek hai.")
    else:
        st.markdown(f"""
        <div style="background:#1a1f2d; border:1px solid #3d4a6b; border-radius:10px; padding:14px 18px; margin-bottom:16px;">
          ⚠️ <b style="color:#f0883e;">{len(stress_points)} Red</b> aur
          <b style="color:#f0883e;">{len(moderate_points)} Orange</b> points hain.<br/>
          <span style="color:#8b949e; font-size:0.85rem;">
            Khet mein jaake in points ki fasal ki photo kheecho aur yahan upload karo —
            AI disease detect karega.
          </span>
        </div>
        """, unsafe_allow_html=True)

        selected_point = st.selectbox(
            "📍 Kaunse point ki photo upload kar rahe ho?",
            options=upload_candidates,
            format_func=lambda p: f"{p} — {'🔴 Stress' if p in stress_points else '🟡 Moderate'}",
        )

        uploaded_img = st.file_uploader(
            f"'{selected_point}' ki crop photo upload karo",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"img_{selected_point}",
        )

        if uploaded_img:
            col_img, col_result = st.columns([1, 1])

            with col_img:
                st.image(uploaded_img, caption=f"📍 {selected_point} — Uploaded Photo", use_container_width=True)

            with col_result:
                with st.spinner("🔬 Disease analysis ho rahi hai..."):
                    time.sleep(0.8)
                    result = predict_disease_from_image(disease_model, uploaded_img)

                card_class = "disease-detected" if result["is_disease"] else "no-disease"
                icon = "⚠️" if result["is_disease"] else "✅"
                title = "Disease Detected!" if result["is_disease"] else "No Disease Found"
                advice = (
                    "🌿 Agronomist se consult karo. Pesticide/fungicide spray ki zaroorat ho sakti hai. "
                    "Nearby plants bhi check karo."
                    if result["is_disease"]
                    else "✅ Is point pe disease nahi dikh rahi. Stress kisi aur wajah se ho sakta hai — "
                         "soil moisture ya nutrient deficiency check karo."
                )

                st.markdown(f"""
                <div class="disease-card {card_class}">
                  <h3 style="margin:0 0 8px; font-family:'Syne',sans-serif;">{icon} {title}</h3>
                  <div style="font-size:0.85rem; color:#8b949e; margin-bottom:10px;">
                    Confidence: <b style="color:#e6edf3;">{result['confidence']}</b>
                  </div>
                  <hr style="border-color:#30363d; margin: 8px 0;"/>
                  <p style="font-size:0.88rem; line-height:1.5; margin:0;">{advice}</p>
                </div>
                """, unsafe_allow_html=True)

                # Point ka data bhi dikhao
                pt_data = df[df["point_id"] == selected_point].iloc[0]
                st.markdown(f"""
                <div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px; font-size:0.82rem; margin-top:8px;">
                  <b>Satellite Data — {selected_point}</b><br/>
                  NDVI: <b>{pt_data['NDVI']:.3f}</b> &nbsp;|&nbsp;
                  NDWI: <b>{pt_data['NDWI']:.3f}</b><br/>
                  EVI: <b>{pt_data['EVI']:.3f}</b> &nbsp;|&nbsp;
                  SAVI: <b>{pt_data['SAVI']:.3f}</b>
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:#161b22; border:1px dashed #30363d; border-radius:12px;
                padding:40px; text-align:center; color:#8b949e; margin-top:24px;">
      <div style="font-size:3rem; margin-bottom:12px;">🛰️</div>
      <b style="font-size:1.1rem; color:#e6edf3;">Abhi tak koi prediction nahi hui</b><br/>
      <span style="font-size:0.88rem;">Step 1: GEE se data fetch karo &nbsp;→&nbsp; Step 2: Predictions chalao</span>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:24px 0 8px; color:#484f58; font-size:0.78rem; border-top:1px solid #21262d; margin-top:32px;">
  KisanDrishti · Sentinel-2 + GEE + Random Forest · Made for Punjab Farmers 🌾
</div>
""", unsafe_allow_html=True)