# =============================================================================
#  config.py — Shared configuration for the entire pipeline
# =============================================================================

CONFIG = {
    'indices'            : ['NDVI', 'NDWI', 'EVI', 'SAVI'],
    'n_clusters'         : 3,        # Healthy / Mild Stress / Stressed
    'savgol_window'      : 5,        # Savitzky-Golay smoothing window (must be odd)
    'savgol_polyorder'   : 2,
    'ndwi_stress_thresh' : 0.0,      # NDWI below this = water stressed
    'ndvi_stress_thresh' : 0.35,     # NDVI below this = vegetation stressed
    'min_valid_obs'      : 10,       # Min observations required to keep a location
    'random_state'       : 42,
}

# Features used for clustering and supervised model training
FEATURE_COLS = [
    'ndvi_max', 'ndvi_min', 'ndvi_mean', 'ndvi_std',
    'ndvi_range', 'ndvi_max_decline', 'ndvi_recovery', 'ndvi_below_thresh',
    'ndwi_mean', 'ndwi_min', 'ndwi_stress_fraction',
    'evi_mean', 'evi_max',
    'savi_mean', 'savi_max',
    'ndvi_ndwi_corr', 'dual_stress_fraction',
]

# ── GEE / Firozpur plot config ────────────────────────────────────────────────
# Specific agricultural plot near Firozpur, Punjab
FIROZPUR_PLOT = {
    'center_lat'  : 30.9252,
    'center_lon'  : 74.6157,
    'buffer_m'    : 500,          # 500m buffer around center = ~1 sq km plot
    'n_points'    : 20,           # number of sample locations inside plot
}

# Last N days of Sentinel-2 data to fetch
FETCH_DAYS    = 100               # ~1 month 20 days
CLOUD_COVER   = 20               # max cloud cover % allowed




# ── Disease model (model.keras) image input config ───────────────────────────
IMG_SIZE      = 224          # resize crop photo to IMG_SIZE x IMG_SIZE
INPUT_SHAPE   = (224, 224, 3)
CLASS_NAMES   = ['Early_Blight', 'Late_Blight', 'Healthy']  



DISEASE_THRESHOLD = 0.5     # sigmoid output above this → Disease
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
DATASET_PATH = "dataset/"
MODEL_PATH = "models/model.keras"
