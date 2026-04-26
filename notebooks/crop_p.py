"""
=============================================================================
  CROP STRESS LABELING PIPELINE
  Data: Sentinel-2 | Features: NDVI, NDWI, EVI, SAVI | No Crop Type Labels
=============================================================================

"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# STEP 0 — CONFIGURATION
# =============================================================================

CONFIG = {
    'indices'          : ['NDVI', 'NDWI', 'EVI', 'SAVI'],
    'n_clusters'       : 3,                  # Healthy / Mild Stress / Stressed
    'savgol_window'    : 5,                  # Savitzky-Golay smoothing window
    'savgol_polyorder' : 2,
    'ndwi_stress_thresh': 0.0,              # NDWI below this = water stressed
    'ndvi_stress_thresh': 0.35,             # NDVI below this = vegetation stressed
    'min_valid_obs'    : 10,                 # Min observations to keep a location
    'random_state'     : 42,
}


# =============================================================================
# STEP 1 — LOAD & VALIDATE DATA
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV. Expected columns:
        lat, lon, date, NDVI, NDWI, EVI, SAVI
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    required_cols = ['lat', 'lon', 'date'] + CONFIG['indices']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Clip valid index ranges
    for col in CONFIG['indices']:
        df[col] = df[col].clip(-1, 1)

    print(f"[✓] Loaded {len(df):,} rows | "
          f"{df[['lat','lon']].drop_duplicates().shape[0]:,} unique locations | "
          f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


# =============================================================================
# STEP 2 — GROUP BY LOCATION, SORT BY DATE, BUILD TIME SERIES
# =============================================================================

def build_time_series(df: pd.DataFrame) -> dict:
    """
    Group by (lat, lon), sort each group by date,
    store time series for NDVI, NDWI, EVI, SAVI.

    Returns:
        ts_store = {
            (lat, lon): {
                'dates': [...],
                'NDVI' : [...],
                'NDWI' : [...],
                'EVI'  : [...],
                'SAVI' : [...],
            },
            ...
        }
    """
    ts_store = {}

    grouped = df.groupby(['lat', 'lon'])

    for (lat, lon), group in grouped:
        # ── Sort by date ──────────────────────────────────────────────────────
        group = group.sort_values('date').reset_index(drop=True)

        # ── Skip locations with too few observations ───────────────────────────
        if len(group) < CONFIG['min_valid_obs']:
            continue

        ts_store[(lat, lon)] = {
            'dates': group['date'].tolist(),
            'NDVI' : group['NDVI'].tolist(),
            'NDWI' : group['NDWI'].tolist(),
            'EVI'  : group['EVI'].tolist(),
            'SAVI' : group['SAVI'].tolist(),
        }

    print(f"[✓] Built time series for {len(ts_store):,} locations "
          f"(min {CONFIG['min_valid_obs']} obs required)")
    return ts_store


# =============================================================================
# STEP 3 — SMOOTH & GAP-FILL EACH TIME SERIES
# =============================================================================

def smooth_series(values: list, dates: list) -> np.ndarray:
    """
    1. Convert dates to day-of-year integers
    2. Drop NaN observations
    3. Interpolate missing gaps linearly
    4. Apply Savitzky-Golay smoothing
    """
    doys   = np.array([(d - dates[0]).days for d in dates], dtype=float)
    vals   = np.array(values, dtype=float)

    # Keep only valid (non-NaN) points
    valid  = ~np.isnan(vals)
    if valid.sum() < 3:
        return vals                          # not enough points to smooth

    # Interpolate onto full time grid
    interp_fn  = interp1d(doys[valid], vals[valid],
                          kind='linear', bounds_error=False,
                          fill_value=(vals[valid][0], vals[valid][-1]))
    vals_filled = interp_fn(doys)

    # Savitzky-Golay smoothing
    win = min(CONFIG['savgol_window'], len(vals_filled))
    win = win if win % 2 == 1 else win - 1   # must be odd
    if win >= 3:
        vals_smooth = savgol_filter(vals_filled, win, CONFIG['savgol_polyorder'])
    else:
        vals_smooth = vals_filled

    return vals_smooth


def smooth_all(ts_store: dict) -> dict:
    """Apply smoothing to every index in every location."""
    for key in ts_store:
        dates = ts_store[key]['dates']
        for idx in CONFIG['indices']:
            raw    = ts_store[key][idx]
            smooth = smooth_series(raw, dates)
            ts_store[key][f'{idx}_smooth'] = smooth.tolist()

    print(f"[✓] Smoothing complete for all locations")
    return ts_store


# =============================================================================
# STEP 4 — EXTRACT TEMPORAL FEATURES PER LOCATION
# =============================================================================

def extract_features(ts_store: dict) -> pd.DataFrame:
    """
    For each location extract stress-indicative temporal features
    from smoothed index time series.
    """
    records = []

    for (lat, lon), ts in ts_store.items():
        ndvi = np.array(ts['NDVI_smooth'])
        ndwi = np.array(ts['NDWI_smooth'])
        evi  = np.array(ts['EVI_smooth'])
        savi = np.array(ts['SAVI_smooth'])
        n    = len(ndvi)

        # ── NDVI features ────────────────────────────────────────────────────
        ndvi_diff    = np.diff(ndvi)
        peak_idx     = int(np.argmax(ndvi))

        ndvi_max          = float(np.max(ndvi))
        ndvi_min          = float(np.min(ndvi))
        ndvi_mean         = float(np.mean(ndvi))
        ndvi_std          = float(np.std(ndvi))
        ndvi_range        = ndvi_max - ndvi_min
        ndvi_max_decline  = float(np.min(ndvi_diff))        # sharpest single-step drop
        ndvi_recovery     = float(ndvi[-1] - ndvi_min)      # did it recover at end?
        ndvi_below_thresh = int(np.sum(ndvi < CONFIG['ndvi_stress_thresh']))  # stressed days count
        peak_timing       = float(peak_idx / n)             # 0=early peak, 1=late peak

        # ── NDWI features ────────────────────────────────────────────────────
        ndwi_mean           = float(np.mean(ndwi))
        ndwi_min            = float(np.min(ndwi))
        ndwi_below_thresh   = int(np.sum(ndwi < CONFIG['ndwi_stress_thresh']))
        ndwi_stress_fraction= float(ndwi_below_thresh / n)

        # ── EVI features ─────────────────────────────────────────────────────
        evi_mean  = float(np.mean(evi))
        evi_max   = float(np.max(evi))
        evi_std   = float(np.std(evi))

        # ── SAVI features ────────────────────────────────────────────────────
        savi_mean = float(np.mean(savi))
        savi_max  = float(np.max(savi))

        # ── Combined / derived features ───────────────────────────────────────
        # NDVI–NDWI correlation: healthy crops show positive co-movement
        ndvi_ndwi_corr = float(np.corrcoef(ndvi, ndwi)[0, 1]) if n > 2 else 0.0

        # Stress window: fraction of season where BOTH NDVI & NDWI are low
        dual_stress_fraction = float(
            np.sum((ndvi < CONFIG['ndvi_stress_thresh']) &
                   (ndwi < CONFIG['ndwi_stress_thresh'])) / n
        )

        records.append({
            'lat': lat, 'lon': lon,
            # NDVI
            'ndvi_max'          : ndvi_max,
            'ndvi_min'          : ndvi_min,
            'ndvi_mean'         : ndvi_mean,
            'ndvi_std'          : ndvi_std,
            'ndvi_range'        : ndvi_range,
            'ndvi_max_decline'  : ndvi_max_decline,
            'ndvi_recovery'     : ndvi_recovery,
            'ndvi_below_thresh' : ndvi_below_thresh,
            'peak_timing'       : peak_timing,
            # NDWI
            'ndwi_mean'             : ndwi_mean,
            'ndwi_min'              : ndwi_min,
            'ndwi_stress_fraction'  : ndwi_stress_fraction,
            # EVI
            'evi_mean'  : evi_mean,
            'evi_max'   : evi_max,
            'evi_std'   : evi_std,
            # SAVI
            'savi_mean' : savi_mean,
            'savi_max'  : savi_max,
            # Combined
            'ndvi_ndwi_corr'        : ndvi_ndwi_corr,
            'dual_stress_fraction'  : dual_stress_fraction,
        })

    feat_df = pd.DataFrame(records)
    print(f"[✓] Extracted {len(feat_df.columns)-2} features "
          f"for {len(feat_df):,} locations")
    return feat_df


# =============================================================================
# STEP 5 — CLUSTER INTO STRESS LABELS
# =============================================================================

FEATURE_COLS = [
    'ndvi_max', 'ndvi_min', 'ndvi_mean', 'ndvi_std',
    'ndvi_range', 'ndvi_max_decline', 'ndvi_recovery', 'ndvi_below_thresh',
    'ndwi_mean', 'ndwi_min', 'ndwi_stress_fraction',
    'evi_mean', 'evi_max',
    'savi_mean', 'savi_max',
    'ndvi_ndwi_corr', 'dual_stress_fraction',
]


def cluster_stress(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    KMeans clustering → map cluster IDs → stress labels.
    Cluster interpretation is automatic based on cluster center
    values of key health indicators.
    """
    X = feat_df[FEATURE_COLS].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(
        n_clusters=CONFIG['n_clusters'],
        random_state=CONFIG['random_state'],
        n_init=20,
        max_iter=500,
    )
    cluster_ids = kmeans.fit_predict(X_scaled)
    feat_df['cluster'] = cluster_ids

    # Silhouette score (quality check, 1.0 = perfect separation)
    score = silhouette_score(X_scaled, cluster_ids)
    print(f"[✓] KMeans silhouette score: {score:.3f}  (>0.3 is acceptable)")

    # ── Interpret clusters automatically ──────────────────────────────────────
    centers = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=FEATURE_COLS
    )
    # Health rank: highest ndvi_mean + ndwi_mean = healthiest cluster
    centers['health_score'] = centers['ndvi_mean'] + centers['ndwi_mean']
    rank = centers['health_score'].rank(ascending=True).astype(int)  # 1=worst

    label_map = {}
    for cluster_id, r in rank.items():
        if r == 1:
            label_map[cluster_id] = 'STRESSED'
        elif r == 2:
            label_map[cluster_id] = 'MILD_STRESS'
        else:
            label_map[cluster_id] = 'HEALTHY'

    feat_df['stress_label'] = feat_df['cluster'].map(label_map)

    # Print cluster summary
    print("\n── Cluster Summary ──────────────────────────────────────")
    summary = feat_df.groupby('stress_label')[
        ['ndvi_mean', 'ndwi_mean', 'evi_mean', 'savi_mean',
         'ndwi_stress_fraction', 'dual_stress_fraction']
    ].mean().round(3)
    print(summary.to_string())
    print("─────────────────────────────────────────────────────────\n")

    return feat_df

# =============================================================================
# STEP 6 — PHENOLOGY: DETECT GROWING SEASON
# =============================================================================

def detect_phenology(ts: dict, threshold: float = 0.3) -> dict:
    """
    Detect green-up start, peak, and senescence end from NDVI.
    Used to contextualize stress within the crop growth stage.
    """
    ndvi     = np.array(ts['NDVI_smooth'])
    dates    = ts['dates']
    peak_idx = int(np.argmax(ndvi))

    above = np.where(ndvi > threshold)[0]
    green_up_idx    = int(above[0])  if len(above) > 0 else 0
    senescence_idx  = int(above[-1]) if len(above) > 0 else len(ndvi) - 1

    return {
        'green_up_date'   : dates[green_up_idx],
        'peak_date'       : dates[peak_idx],
        'senescence_date' : dates[senescence_idx],
        'peak_ndvi'       : float(ndvi[peak_idx]),
        'season_length'   : senescence_idx - green_up_idx,
    }


# =============================================================================
# STEP 7 — VISUALIZATION
# =============================================================================

def plot_sample_locations(ts_store: dict, feat_df: pd.DataFrame,
                          n_samples: int = 6, save_path: str = None):
    """
    Plot smoothed time series for sample locations,
    colored by stress label.
    """
    label_colors = {
        'HEALTHY'    : '#2ecc71',
        'MILD_STRESS': '#f39c12',
        'STRESSED'   : '#e74c3c',
    }

    keys        = list(ts_store.keys())
    sample_keys = keys[:n_samples]

    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 3.5 * n_samples),constrained_layout=True)
    if n_samples == 1:
        axes = [axes]

    fig.suptitle('Crop Stress Time Series — Sample Locations',fontsize=16, fontweight='bold')

    for row, key in enumerate(sample_keys):
        ts    = ts_store[key]
        dates = ts['dates']

        row_feat = feat_df[(feat_df['lat'] == key[0]) &
                           (feat_df['lon'] == key[1])]
        label = row_feat['stress_label'].values[0] if len(row_feat) else 'UNKNOWN'
        color = label_colors.get(label, 'gray')

        for col, idx_name in enumerate(CONFIG['indices']):
            ax  = axes[row][col]
            raw = ts[idx_name]
            smt = ts[f'{idx_name}_smooth']

            ax.plot(dates, raw, color='lightgray', lw=1, label='Raw')
            ax.plot(dates, smt, color=color, lw=2,  label='Smoothed')
            ax.axhline(0,    color='black', lw=0.5, linestyle='--')
            ax.set_title(f'{idx_name} | ({key[0]:.3f}, {key[1]:.3f})',
                         fontsize=9)
            ax.set_ylabel(idx_name)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
            ax.set_ylim(-1, 1)
            ax.grid(alpha=0.3)

            if col == 0:
                ax.set_ylabel(f'Label: {label}\n{idx_name}',
                              color=color, fontweight='bold')
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Time series plot saved → {save_path}")
    plt.show()


def plot_stress_distribution(feat_df: pd.DataFrame, save_path: str = None):
    """Bar chart of stress label distribution."""
    counts = feat_df['stress_label'].value_counts()
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    order  = ['HEALTHY', 'MILD_STRESS', 'STRESSED']
    counts = counts.reindex([o for o in order if o in counts.index])

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=colors[:len(counts)], edgecolor='white', width=0.5)

    for bar, val in zip(bars, counts.values):
        pct = val / len(feat_df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{val:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title('Crop Stress Label Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Locations')
    ax.set_xlabel('Stress Label')
    ax.set_ylim(0, counts.max() * 1.2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Distribution plot saved → {save_path}")
    plt.show()


def plot_feature_heatmap(feat_df: pd.DataFrame, save_path: str = None):
    """Heatmap of mean feature values per stress label."""
    agg = feat_df.groupby('stress_label')[
        ['ndvi_mean', 'ndvi_std', 'ndvi_max_decline',
         'ndwi_mean', 'ndwi_stress_fraction',
         'evi_mean', 'savi_mean', 'dual_stress_fraction']
    ].mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    scaled  = (agg - agg.min()) / (agg.max() - agg.min() + 1e-9)
    im      = ax.imshow(scaled.values, cmap='RdYlGn', aspect='auto',
                        vmin=0, vmax=1)

    ax.set_xticks(range(len(agg.columns)))
    ax.set_xticklabels(agg.columns, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(agg.index)))
    ax.set_yticklabels(agg.index, fontsize=11, fontweight='bold')
    ax.set_title('Feature Means per Stress Label (normalized 0–1)',
                 fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Normalized Value')

    # Annotate cells
    for i in range(len(agg.index)):
        for j in range(len(agg.columns)):
            ax.text(j, i, f'{agg.values[i, j]:.2f}',
                    ha='center', va='center', fontsize=8, color='black')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Heatmap saved → {save_path}")
    plt.show()


# =============================================================================
# STEP 8 — EXPORT RESULTS
# =============================================================================

def export_results(feat_df: pd.DataFrame,ts_store: dict,output_csv: str = 'crop_stress_labels.csv'):
    """
    Add phenology columns to feat_df and save final labeled CSV.
    """
    pheno_records = []
    for key, ts in ts_store.items():
        p = detect_phenology(ts)
        p['lat'] = key[0]
        p['lon'] = key[1]
        pheno_records.append(p)

    pheno_df = pd.DataFrame(pheno_records)
    final_df = feat_df.merge(pheno_df, on=['lat', 'lon'], how='left')
    final_df.to_csv(output_csv, index=False)

    print(f"[✓] Final labeled data saved → {output_csv}")
    print(f"    Shape: {final_df.shape}")
    print(f"\n── Label counts ──")
    print(final_df['stress_label'].value_counts().to_string())
    return final_df


# =============================================================================
# STEP 9 — OPTIONAL: TRAIN SUPERVISED MODEL ON PSEUDO-LABELS
# =============================================================================

def train_supervised_model(feat_df: pd.DataFrame):
    """
    Once clustering gives pseudo-labels,
    train a Random Forest for fast inference on new data.
    """
    from sklearn.ensemble         import RandomForestClassifier
    from sklearn.model_selection  import cross_val_score
    from sklearn.preprocessing    import LabelEncoder

    le = LabelEncoder()
    X  = feat_df[FEATURE_COLS].fillna(0).values
    y  = le.fit_transform(feat_df['stress_label'])

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=CONFIG['random_state'],
        class_weight='balanced',
    )

    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='f1_macro')
    print(f"[✓] RF cross-val F1 (macro): {cv_scores.mean():.3f} "
          f"± {cv_scores.std():.3f}")

    rf.fit(X, y)

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    print("\n── Top 10 Important Features ──")
    print(importances.sort_values(ascending=False).head(10).round(4).to_string())

    return rf, le


# =============================================================================
# MAIN — RUN FULL PIPELINE
# =============================================================================

def run_pipeline(filepath: str,
                 output_csv: str  = 'crop_stress_labels.csv',
                 plot_samples: int = 6,
                 train_rf: bool    = True):

    print("\n" + "="*65)
    print("  CROP STRESS LABELING PIPELINE  —  Sentinel-2 Indices")
    print("="*65 + "\n")

    # 1. Load
    df = load_data(filepath)

    # 2. Build time series (grouped by lat/lon, sorted by date)
    ts_store = build_time_series(df)

    # 3. Smooth & gap-fill
    ts_store = smooth_all(ts_store)

    # 4. Extract temporal features
    feat_df = extract_features(ts_store)

    # 5. Cluster → stress labels
    feat_df = cluster_stress(feat_df)

    # 6. Export results
    final_df = export_results(feat_df, ts_store, output_csv)

    # 7. Visualize
    plot_stress_distribution(feat_df)
    plot_feature_heatmap(feat_df)
    if plot_samples > 0:
        plot_sample_locations(ts_store, feat_df, n_samples=plot_samples)

    # 8. (Optional) Train supervised RF on pseudo-labels
    if train_rf:
        rf_model, label_encoder = train_supervised_model(feat_df)
    else:
        rf_model, label_encoder = None, None

    print("\n[✓] Pipeline complete!\n")
    return final_df, ts_store, rf_model, label_encoder


# =============================================================================
# USAGE EXAMPLE  (replace path with your actual CSV)
# =============================================================================

if __name__ == '__main__':

    # ── Your CSV must have these columns: ─────────────────────────────────────
    #   lat | lon | date | NDVI | NDWI | EVI | SAVI
    # ──────────────────────────────────────────────────────────────────────────

    final_df, ts_store, rf_model, le = run_pipeline(
        filepath     = 'data/raw/crop_health_assesment_dataset.csv',   # ← change this
        output_csv   = 'data/processed/crop_stress_labels.csv',
        plot_samples = 6,
        train_rf     = True,
    )

    # ── Quick preview of output ────────────────────────────────────────────────
    print(final_df[['lat', 'lon', 'ndvi_mean', 'ndwi_mean',
                     'stress_label', 'peak_date']].head(10))

    # ── Predict on new data ────────────────────────────────────────────────────
    # new_features = extract_features(build_time_series(new_df))
    # preds = le.inverse_transform(
    #     rf_model.predict(new_features[FEATURE_COLS].fillna(0))
    # )