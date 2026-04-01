import pandas as pd
import logging

logger = logging.getLogger(__name__)


def create_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-series features for each location (lat_r, lon_r)
    """

    group = group.sort_values("date").copy()

    # -----------------------------
    # Rolling Mean (Trend smoothing)
    # -----------------------------
    group["ndvi_roll_mean"] = group["NDVI"].rolling(window=3).mean()

    # -----------------------------
    # Rolling Std (Variability)
    # -----------------------------
    group["ndvi_roll_std"] = group["NDVI"].rolling(window=3).std()

    # -----------------------------
    # NDVI Difference (Change)
    # -----------------------------
    group["ndvi_diff"] = group["NDVI"].diff()

    # -----------------------------
    # NDVI Trend (Direction over window)
    # -----------------------------
    group["ndvi_trend"] = group["NDVI"].rolling(window=3).apply(
        lambda x: x.iloc[-1] - x.iloc[0], raw=False
    )

    logger.info("Feature engineering completed for one group")

    return group