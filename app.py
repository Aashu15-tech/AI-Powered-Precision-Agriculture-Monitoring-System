# =============================================================================
#  app.py — Streamlit UI for Crop Stress + Disease Prediction
#
#  Run:
#      streamlit run app.py
#
#  Features:
#      1. Fetch last 50 days Sentinel-2 data for Firozpur plot via GEE
#      2. Preprocess → predict crop stress (RF model)
#      3. Show Folium map with color-coded stress labels per location
#      4. Upload crop photo → predict disease (model.keras)
#      5. Results table + NDVI time series chart
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io

warnings.filterwarnings('ignore')

from config.config         import (CONFIG, FIROZPUR_PLOT, FETCH_DAYS,CLASS_NAMES, IMG_SIZE)
from gee_fetch       import fetch_data
from predict         import run_prediction
from disease_predict import predict_disease

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title = "Crop Stress & Disease Monitor — Firozpur",
    page_icon  = "🌾",
    layout     = "wide",
)

LABEL_COLORS = {
    'HEALTHY'    : '#2ecc71',
    'MILD_STRESS': '#f39c12',
    'STRESSED'   : '#e74c3c',
}
LABEL_ICONS = {
    'HEALTHY'    : '✅',
    'MILD_STRESS': '⚠️',
    'STRESSED'   : '🚨',
}

# =============================================================================
# SESSION STATE INIT
# =============================================================================

for key, default in {
    'raw_csv'     : 'data/Sentinel2_data.csv',
    'pred_csv'    : 'data/firozpur_predictions.csv',
    'result_df'   : None,
    'ts_store'    : None,
    'sample_points': None,
    'fetch_done'  : False,
    'pred_done'   : False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# =============================================================================
# HEADER
# =============================================================================

st.title("🌾 Crop Stress & Disease Monitor")
st.caption("Firozpur, Punjab — Sentinel-2 | GEE | Random Forest | Keras")
st.divider()

# =============================================================================
# LAYOUT — TWO COLUMNS
# =============================================================================

left_col, right_col = st.columns([1, 2], gap="large")

# =============================================================================
# LEFT COLUMN — CONTROLS
# =============================================================================

with left_col:

    # ── Section 1: GEE Fetch ──────────────────────────────────────────────────
    st.subheader("📡 Step 1 — Fetch Sentinel-2 Data")
    st.markdown(f"""
    - **Plot**: Firozpur Agricultural Plot
    - **Center**: `{FIROZPUR_PLOT['center_lat']}°N, {FIROZPUR_PLOT['center_lon']}°E`
    - **Buffer**: {FIROZPUR_PLOT['buffer_m']}m radius
    - **Locations**: {FIROZPUR_PLOT['n_points']} sample points
    - **Period**: Last **{FETCH_DAYS} days**
    """)

    fetch_btn = st.button("🛰️ Fetch GEE Data", type="primary",
                           use_container_width=True)

    if fetch_btn:
        with st.spinner("Fetching GEE data....(may take 1-3 min)"):
            try:
                df_raw = fetch_data()
                st.session_state['sample_points'] = df_raw[['lat','lon']].drop_duplicates()
                st.session_state['fetch_done']    = True
                st.success(f"✅ Data fetched! {len(df_raw):,} rows, "
                           f"{df_raw[['lat','lon']].drop_duplicates().shape[0]} locations")
            except Exception as e:
                st.error(f"❌ GEE fetch failed:\n{e}")

    # Show if already fetched
    if st.session_state['fetch_done']:
        if os.path.exists(st.session_state['raw_csv']):
            df_peek = pd.read_csv(st.session_state['raw_csv'])
            with st.expander("📄 Raw Data Preview"):
                st.dataframe(df_peek.head(10), use_container_width=True)

    st.divider()

    # ── Section 2: Predict Stress ─────────────────────────────────────────────
    st.subheader("🤖 Step 2 — Predict Crop Stress")

    model_ok   = os.path.exists('models/model.pkl')
    encoder_ok = os.path.exists('models/label_encoder.pkl')

    if not model_ok:
        st.warning("⚠️ `model.pkl` not found. run Main.py or check name.")
    if not encoder_ok:
        st.warning("⚠️ `label_encoder.pkl` not found. run Main.py or check name.")

    predict_btn = st.button(
        "🔍 Run Stress Prediction",
        type      = "primary",
        disabled  = not (st.session_state['fetch_done'] and model_ok and encoder_ok),
        use_container_width = True,
    )

    if predict_btn:
        with st.spinner("Preprocessing and prediction ..."):
            try:
                result_df, ts_store = run_prediction(
                    raw_csv_path = st.session_state['raw_csv'],
                    model_path   = 'models/model.pkl',
                    encoder_path = 'models/label_encoder.pkl',
                    output_csv   = st.session_state['pred_csv'],
                )
                st.session_state['result_df'] = result_df
                st.session_state['ts_store']  = ts_store
                st.session_state['pred_done'] = True
                st.success("✅ Prediction complete!")
            except Exception as e:
                st.error(f"❌ Prediction failed:\n{e}")

    st.divider()

    # ── Section 3: Disease Prediction ─────────────────────────────────────────
    st.subheader("🔬 Step 3 — Disease Detection")
    st.markdown("Upload Stressed location crop photo :")

    uploaded_photo = st.file_uploader(
        "Upload crop photo",
        type    = ['jpg', 'jpeg', 'png'],
        help    = f"Photo resize to {IMG_SIZE}×{IMG_SIZE}px automatically"
    )

    if uploaded_photo is not None:
        st.image(uploaded_photo, caption="Uploaded crop photo",
                 use_container_width=True)

        disease_btn = st.button("🧪 Detect Disease", type="primary",
                                use_container_width=True)

        if disease_btn:
            if not os.path.exists('models/model.keras'):
                st.error("❌ `model.keras` not found in project root.")
            else:
                with st.spinner("Disease model predicting..."):
                    try:
                        result = predict_disease(uploaded_photo, 'models/model.keras')

                        # Display result
                        if result['label'] == 'Disease':
                            st.error(
                                f"🚨 **{result['label']}** detected!\n\n"
                                f"Confidence: **{result['confidence']}%**"
                            )
                        else:
                            st.success(
                                f"✅ **{result['label']}**\n\n"
                                f"Confidence: **{result['confidence']}%**"
                            )

                        # Probability bar
                        st.markdown("**Probabilities:**")
                        prob_df = pd.DataFrame({
                            'Class'      : list(result['raw_probs'].keys()),
                            'Probability': list(result['raw_probs'].values())
                        })
                        st.bar_chart(prob_df.set_index('Class'))

                    except Exception as e:
                        st.error(f"❌ Disease prediction failed:\n{e}")

# =============================================================================
# RIGHT COLUMN — MAP + RESULTS
# =============================================================================

with right_col:

    # ── TAB layout ────────────────────────────────────────────────────────────
    tab_map, tab_table, tab_chart = st.tabs([
        "🗺️ Stress Map", "📊 Results Table", "📈 NDVI Time Series"
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — FOLIUM MAP
    # ─────────────────────────────────────────────────────────────────────────
    with tab_map:

        if st.session_state['result_df'] is not None:
            result_df = st.session_state['result_df']

            # Build Folium map centered on plot
            m = folium.Map(
                location = [FIROZPUR_PLOT['center_lat'],
                             FIROZPUR_PLOT['center_lon']],
                zoom_start = 15,
                tiles      = 'Esri.WorldImagery',   # satellite view
            )

            # Add plot boundary circle
            folium.Circle(
                location = [FIROZPUR_PLOT['center_lat'],
                             FIROZPUR_PLOT['center_lon']],
                radius   = FIROZPUR_PLOT['buffer_m'],
                color    = '#ffffff',
                weight   = 2,
                fill     = False,
                tooltip  = 'Plot boundary'
            ).add_to(m)

            # Add center marker
            folium.Marker(
                location = [FIROZPUR_PLOT['center_lat'],
                             FIROZPUR_PLOT['center_lon']],
                tooltip  = 'Plot Center',
                icon     = folium.Icon(color='blue', icon='home')
            ).add_to(m)

            # Add one marker per location
            for _, row in result_df.iterrows():
                label   = row['stress_label']
                color   = LABEL_COLORS.get(label, 'gray')
                icon_sym= LABEL_ICONS.get(label, '❓')

                # Color mapping for folium icons
                folium_color = {
                    'HEALTHY'    : 'green',
                    'MILD_STRESS': 'orange',
                    'STRESSED'   : 'red',
                }.get(label, 'gray')

                popup_html = f"""
                <div style="font-family:Arial; min-width:180px">
                    <b>{icon_sym} {label.replace('_',' ')}</b><br>
                    <hr style="margin:4px 0">
                    <b>Confidence:</b> {row['stress_confidence']}%<br>
                    <b>NDVI mean:</b>  {row['ndvi_mean']:.3f}<br>
                    <b>NDWI mean:</b>  {row['ndwi_mean']:.3f}<br>
                    <b>EVI mean:</b>   {row['evi_mean']:.3f}<br>
                    <b>SAVI mean:</b>  {row['savi_mean']:.3f}<br>
                    <b>Lat:</b> {row['lat']:.6f}<br>
                    <b>Lon:</b> {row['lon']:.6f}
                </div>
                """

                folium.CircleMarker(
                    location      = [row['lat'], row['lon']],
                    radius        = 12,
                    color         = color,
                    fill          = True,
                    fill_color    = color,
                    fill_opacity  = 0.85,
                    tooltip       = f"{icon_sym} {label} ({row['stress_confidence']}%)",
                    popup         = folium.Popup(popup_html, max_width=220),
                ).add_to(m)

            # Legend
            legend_html = """
            <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
                        background:white; padding:12px 16px; border-radius:8px;
                        border:1px solid #ccc; font-family:Arial; font-size:13px;
                        box-shadow: 2px 2px 6px rgba(0,0,0,0.3)">
                <b>🌾 Crop Stress</b><br>
                <span style="color:#2ecc71">●</span> Healthy<br>
                <span style="color:#f39c12">●</span> Mild Stress<br>
                <span style="color:#e74c3c">●</span> Stressed
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))

            st_folium(m, width=None, height=550, returned_objects=[])

            # Label count summary below map
            st.markdown("**Label Summary:**")
            counts = result_df['stress_label'].value_counts()
            cols   = st.columns(len(counts))
            for i, (label, cnt) in enumerate(counts.items()):
                icon = LABEL_ICONS.get(label, '❓')
                cols[i].metric(f"{icon} {label.replace('_',' ')}", cnt)

        else:
            # Placeholder map before prediction
            m = folium.Map(
                location   = [FIROZPUR_PLOT['center_lat'],
                               FIROZPUR_PLOT['center_lon']],
                zoom_start = 14,
                tiles      = 'Esri.WorldImagery',
            )
            folium.Circle(
                location = [FIROZPUR_PLOT['center_lat'],
                             FIROZPUR_PLOT['center_lon']],
                radius   = FIROZPUR_PLOT['buffer_m'],
                color    = '#f39c12', weight=2, fill=False,
                tooltip  = 'Firozpur Agricultural Plot'
            ).add_to(m)
            folium.Marker(
                location = [FIROZPUR_PLOT['center_lat'],
                             FIROZPUR_PLOT['center_lon']],
                tooltip  = 'Plot Center — Run prediction to see results',
                icon     = folium.Icon(color='orange', icon='info-sign')
            ).add_to(m)
            st_folium(m, width=None, height=550, returned_objects=[])
            st.info("ℹ️ GEE data fetch karo aur prediction chalao — map pe stress labels aayenge.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — RESULTS TABLE
    # ─────────────────────────────────────────────────────────────────────────
    with tab_table:

        if st.session_state['result_df'] is not None:
            result_df = st.session_state['result_df']

            # Filter controls
            filter_label = st.multiselect(
                "Filter by stress label:",
                options  = ['HEALTHY', 'MILD_STRESS', 'STRESSED'],
                default  = ['HEALTHY', 'MILD_STRESS', 'STRESSED'],
            )
            filtered = result_df[result_df['stress_label'].isin(filter_label)]

            # Style table
            def color_label(val):
                colors = {
                    'HEALTHY'    : 'background-color:#d5f5e3; color:#1a7a3a',
                    'MILD_STRESS': 'background-color:#fef9e7; color:#b7770d',
                    'STRESSED'   : 'background-color:#fadbd8; color:#a93226',
                }
                return colors.get(val, '')

            display_cols = ['lat', 'lon', 'stress_label', 'stress_confidence',
                            'ndvi_mean', 'ndwi_mean', 'evi_mean', 'savi_mean',
                            'dual_stress_fraction']

            styled = (
                filtered[display_cols]
                .style
                .applymap(color_label, subset=['stress_label'])
                .format({
                    'lat'                 : '{:.6f}',
                    'lon'                 : '{:.6f}',
                    'stress_confidence'   : '{:.1f}%',
                    'ndvi_mean'           : '{:.3f}',
                    'ndwi_mean'           : '{:.3f}',
                    'evi_mean'            : '{:.3f}',
                    'savi_mean'           : '{:.3f}',
                    'dual_stress_fraction': '{:.2f}',
                })
            )
            st.dataframe(styled, use_container_width=True, height=400)

            # Download button
            csv_bytes = filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label     = "⬇️ Download Predictions CSV",
                data      = csv_bytes,
                file_name = "firozpur_predictions.csv",
                mime      = "text/csv",
            )
        else:
            st.info("ℹ️ Prediction complete hone ke baad results yahaan dikhenge.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — NDVI TIME SERIES CHART
    # ─────────────────────────────────────────────────────────────────────────
    with tab_chart:

        if st.session_state['ts_store'] is not None and \
           st.session_state['result_df'] is not None:

            ts_store  = st.session_state['ts_store']
            result_df = st.session_state['result_df']

            # Location selector
            location_options = [
                f"({row['lat']:.5f}, {row['lon']:.5f}) — {row['stress_label']}"
                for _, row in result_df.iterrows()
            ]
            selected = st.selectbox("Location select karo:", location_options)

            # Parse selected lat/lon
            sel_idx = location_options.index(selected)
            sel_row = result_df.iloc[sel_idx]
            sel_key = (sel_row['lat'], sel_row['lon'])

            # Find closest key in ts_store (float precision matching)
            def find_closest_key(ts_store, target_lat, target_lon):
                best_key  = None
                best_dist = float('inf')
                for (lat, lon) in ts_store.keys():
                    dist = abs(lat - target_lat) + abs(lon - target_lon)
                    if dist < best_dist:
                        best_dist = dist
                        best_key  = (lat, lon)
                return best_key

            ts_key = find_closest_key(ts_store, sel_row['lat'], sel_row['lon'])

            if ts_key and ts_key in ts_store:
                ts    = ts_store[ts_key]
                dates = ts['dates']
                label = sel_row['stress_label']
                color = LABEL_COLORS.get(label, 'gray')

                # Plot all 4 indices
                fig, axes = plt.subplots(2, 2, figsize=(12, 7),
                                         constrained_layout=True)
                fig.suptitle(
                    f"Time Series — ({ts_key[0]:.5f}, {ts_key[1]:.5f})\n"
                    f"Label: {LABEL_ICONS.get(label,'')} {label.replace('_',' ')}",
                    fontsize=13, fontweight='bold', color=color
                )

                index_axes = [('NDVI', axes[0][0]),
                               ('NDWI', axes[0][1]),
                               ('EVI',  axes[1][0]),
                               ('SAVI', axes[1][1])]

                for idx_name, ax in index_axes:
                    raw = ts.get(idx_name, [])
                    smt = ts.get(f'{idx_name}_smooth', [])

                    if raw:
                        ax.plot(dates, raw, color='lightgray', lw=1,
                                alpha=0.8, label='Raw')
                    if smt:
                        ax.plot(dates, smt, color=color, lw=2.5,
                                label='Smoothed')

                    ax.axhline(0, color='black', lw=0.5, ls='--', alpha=0.5)

                    # Threshold lines
                    if idx_name == 'NDVI':
                        ax.axhline(CONFIG['ndvi_stress_thresh'],
                                   color='red', lw=1, ls=':', alpha=0.7,
                                   label=f"Stress thresh ({CONFIG['ndvi_stress_thresh']})")
                    if idx_name == 'NDWI':
                        ax.axhline(CONFIG['ndwi_stress_thresh'],
                                   color='red', lw=1, ls=':', alpha=0.7,
                                   label=f"Stress thresh ({CONFIG['ndwi_stress_thresh']})")

                    ax.set_title(idx_name, fontweight='bold')
                    ax.set_ylim(-1, 1)
                    ax.grid(alpha=0.25)
                    ax.legend(fontsize=8)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

                # Render in Streamlit
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
                buf.seek(0)
                st.image(buf, use_container_width=True)
                plt.close(fig)

        else:
            st.info("ℹ️ Prediction complete hone ke baad time series yahaan dikhegi.")
