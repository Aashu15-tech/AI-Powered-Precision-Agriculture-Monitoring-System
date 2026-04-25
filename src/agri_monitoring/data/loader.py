# =============================================================================
#  data/loader.py — Load & validate raw Sentinel-2 CSV data
# =============================================================================

import pandas as pd
from config.config import CONFIG


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV and validate required columns.

    Expected CSV columns:
        lat | lon | date | NDVI | NDWI | EVI | SAVI

    Steps:
        1. Read CSV
        2. Parse date column to datetime
        3. Check all required columns exist
        4. Clip index values to valid range [-1, 1]

    Args:
        filepath (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Cleaned dataframe ready for time series building.
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    required_cols = ['lat', 'lon', 'date'] + CONFIG['indices']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Replace out-of-range values with NaN, then clip as safety net
    for col in CONFIG['indices']:
        df[col] = df[col].where(df[col].between(-1, 1), other=float('nan'))
        df[col] = df[col].clip(-1, 1)

    print(f"[✓] Loaded {len(df):,} rows | "
          f"{df[['lat','lon']].drop_duplicates().shape[0]:,} unique locations | "
          f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df
