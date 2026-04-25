# =============================================================================
#  data/time_series.py — Group by location, sort by date, build time series
# =============================================================================

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from config.config import CONFIG


def build_time_series(df: pd.DataFrame) -> dict:
    """
    Group raw dataframe by (lat, lon), sort each group by date,
    and store index time series per location.

    Args:
        df (pd.DataFrame): Cleaned dataframe from loader.py

    Returns:
        ts_store (dict): {
            (lat, lon): {
                'dates': [datetime, ...],   # sorted list of datetimes
                'NDVI' : [float, ...],
                'NDWI' : [float, ...],
                'EVI'  : [float, ...],
                'SAVI' : [float, ...],
            },
            ...
        }
    """
    ts_store = {}
    grouped  = df.groupby(['lat', 'lon'])

    for (lat, lon), group in grouped:

        # Sort by date ascending
        group = group.sort_values('date').reset_index(drop=True)

        # Skip locations with too few observations
        if len(group) < CONFIG['min_valid_obs']:
            continue

        ts_store[(lat, lon)] = {
            'dates' : group['date'].tolist(),
            'NDVI'  : group['NDVI'].tolist(),
            'NDWI'  : group['NDWI'].tolist(),
            'EVI'   : group['EVI'].tolist(),
            'SAVI'  : group['SAVI'].tolist(),
        }

    print(f"[✓] Built time series for {len(ts_store):,} locations "
          f"(min {CONFIG['min_valid_obs']} obs required)")
    return ts_store


def smooth_series(values: list, dates: list) -> np.ndarray:
    """
    Gap-fill and smooth a single index time series.

    Steps:
        1. Convert dates to elapsed days (integer axis)
        2. Drop NaN observations
        3. Linearly interpolate gaps onto full time grid
        4. Apply Savitzky-Golay smoothing

    Args:
        values (list): Raw index values (may contain NaNs).
        dates  (list): Corresponding datetime objects (sorted).

    Returns:
        np.ndarray: Smoothed values, same length as input.
    """
    doys = np.array([(d - dates[0]).days for d in dates], dtype=float)
    vals = np.array(values, dtype=float)

    valid = ~np.isnan(vals)
    if valid.sum() < 3:
        return vals                    # not enough points to smooth

    # Interpolate onto the full time grid
    interp_fn   = interp1d(doys[valid], vals[valid],
                            kind='linear', bounds_error=False,
                            fill_value=(vals[valid][0], vals[valid][-1]))
    vals_filled = interp_fn(doys)

    # Savitzky-Golay: window must be odd and >= 3
    win = min(CONFIG['savgol_window'], len(vals_filled))
    win = win if win % 2 == 1 else win - 1
    if win >= 3:
        vals_smooth = savgol_filter(vals_filled, win,
                                    CONFIG['savgol_polyorder'])
    else:
        vals_smooth = vals_filled

    return vals_smooth


def smooth_all(ts_store: dict) -> dict:
    """
    Apply gap-filling and smoothing to every index in every location.
    Adds '{INDEX}_smooth' keys to each ts_store entry in-place.

    Args:
        ts_store (dict): Output of build_time_series().

    Returns:
        dict: Same ts_store with added smooth keys:
              'NDVI_smooth', 'NDWI_smooth', 'EVI_smooth', 'SAVI_smooth'
    """
    for key in ts_store:
        dates = ts_store[key]['dates']
        for idx in CONFIG['indices']:
            raw    = ts_store[key][idx]
            smooth = smooth_series(raw, dates)
            ts_store[key][f'{idx}_smooth'] = smooth.tolist()

    print(f"[✓] Smoothing complete for all locations")
    return ts_store
