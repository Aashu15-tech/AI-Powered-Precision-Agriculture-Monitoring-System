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
