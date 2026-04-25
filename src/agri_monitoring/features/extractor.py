# =============================================================================
#  features/extractor.py — Extract temporal stress features per location
# =============================================================================

import numpy as np
import pandas as pd
from config.config import CONFIG


def extract_features(ts_store: dict) -> pd.DataFrame:
    """
    For each location, compute stress-indicative temporal features
    from the smoothed index time series.

    Args:
        ts_store (dict): Output of smooth_all(). Each entry must have
                         NDVI_smooth, NDWI_smooth, EVI_smooth, SAVI_smooth keys.

    Returns:
        pd.DataFrame: One row per location with columns:
            lat, lon,
            [NDVI features], [NDWI features], [EVI features],
            [SAVI features], [combined features]
    """
    records = []

    for (lat, lon), ts in ts_store.items():
        ndvi = np.array(ts['NDVI_smooth'])
        ndwi = np.array(ts['NDWI_smooth'])
        evi  = np.array(ts['EVI_smooth'])
        savi = np.array(ts['SAVI_smooth'])
        n    = len(ndvi)

        # ── NDVI features ─────────────────────────────────────────────────────
        ndvi_diff         = np.diff(ndvi)
        peak_idx          = int(np.argmax(ndvi))

        ndvi_max          = float(np.max(ndvi))
        ndvi_min          = float(np.min(ndvi))
        ndvi_mean         = float(np.mean(ndvi))
        ndvi_std          = float(np.std(ndvi))
        ndvi_range        = ndvi_max - ndvi_min
        ndvi_max_decline  = float(np.min(ndvi_diff))   # sharpest single-step drop
        ndvi_recovery     = float(ndvi[-1] - ndvi_min) # did it bounce back?
        ndvi_below_thresh = int(np.sum(ndvi < CONFIG['ndvi_stress_thresh']))
        peak_timing       = float(peak_idx / n)        # 0=early peak, 1=late peak

        # ── NDWI features ─────────────────────────────────────────────────────
        ndwi_below_thresh    = int(np.sum(ndwi < CONFIG['ndwi_stress_thresh']))
        ndwi_mean            = float(np.mean(ndwi))
        ndwi_min             = float(np.min(ndwi))
        ndwi_stress_fraction = float(ndwi_below_thresh / n)

        # ── EVI features ──────────────────────────────────────────────────────
        evi_mean = float(np.mean(evi))
        evi_max  = float(np.max(evi))
        evi_std  = float(np.std(evi))

        # ── SAVI features ─────────────────────────────────────────────────────
        savi_mean = float(np.mean(savi))
        savi_max  = float(np.max(savi))

        # ── Combined / derived features ───────────────────────────────────────
        # NDVI–NDWI correlation: healthy crops show positive co-movement
        ndvi_ndwi_corr = float(np.corrcoef(ndvi, ndwi)[0, 1]) if n > 2 else 0.0

        # Fraction of season where BOTH NDVI & NDWI are below stress thresholds
        dual_stress_fraction = float(
            np.sum(
                (ndvi < CONFIG['ndvi_stress_thresh']) &
                (ndwi < CONFIG['ndwi_stress_thresh'])
            ) / n
        )

        records.append({
            'lat'  : lat,
            'lon'  : lon,
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
            'ndvi_ndwi_corr'       : ndvi_ndwi_corr,
            'dual_stress_fraction' : dual_stress_fraction,
        })

    feat_df = pd.DataFrame(records)
    print(f"[✓] Extracted {len(feat_df.columns) - 2} features "
          f"for {len(feat_df):,} locations")
    return feat_df
