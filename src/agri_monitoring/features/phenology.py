# =============================================================================
#  features/phenology.py — Detect crop growing season from NDVI shape
# =============================================================================


def detect_phenology(ts: dict, threshold: float = 0.3) -> dict:
    """
    Detect key phenological events from the smoothed NDVI time series:
        - Green-up  : first date NDVI crosses above threshold
        - Peak      : date of maximum NDVI
        - Senescence: last date NDVI is still above threshold

    Args:
        ts (dict): Single location entry from ts_store.
                   Must contain 'NDVI_smooth' and 'dates' keys.
        threshold (float): NDVI value marking start/end of growing season.
                           Default 0.3 works well for most crops.

    Returns:
        dict: {
            'green_up_date'  : datetime,  # when crop started growing
            'peak_date'      : datetime,  # day of maximum vegetation
            'senescence_date': datetime,  # when crop started dying back
            'peak_ndvi'      : float,     # maximum NDVI achieved
            'season_length'  : int,       # number of days from green-up to senescence
        }
    """
    import numpy as np

    ndvi     = np.array(ts['NDVI_smooth'])
    dates    = ts['dates']
    peak_idx = int(np.argmax(ndvi))

    above          = np.where(ndvi > threshold)[0]
    green_up_idx   = int(above[0])  if len(above) > 0 else 0
    senescence_idx = int(above[-1]) if len(above) > 0 else len(ndvi) - 1

    return {
        'green_up_date'   : dates[green_up_idx],
        'peak_date'       : dates[peak_idx],
        'senescence_date' : dates[senescence_idx],
        'peak_ndvi'       : float(ndvi[peak_idx]),
        # Real calendar days (not index count)
        'season_length'   : (dates[senescence_idx] - dates[green_up_idx]).days,
    }
