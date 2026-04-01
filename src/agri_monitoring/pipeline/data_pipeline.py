import os
import pandas as pd
import logging

from src.agri_monitoring.data.data_loader import load_data
from src.agri_monitoring.features.feature_engineering import create_features
from src.agri_monitoring.features.labeling import process_group, smooth_labels
logger = logging.getLogger(__name__)

from pathlib import Path

def run_pipeline(input_path: Path, output_path: Path):
    """
    Full pipeline:
    Load → Process → Label → Save
    """

    logger.info("Starting data pipeline...")

    # Load data
    df = load_data(input_path)

    # Convert date
    df['date'] = pd.to_datetime(df['date'])

    # Round lat/lon
    df['lat_r'] = df['lat'].round(3)
    df['lon_r'] = df['lon'].round(3)

    # Sort
    df = df.sort_values(by=['lat_r', 'lon_r', 'date'])

    # Step 1: Feature Engineering
    df = (
        df.groupby(['lat_r', 'lon_r'])
        .apply(create_features)
        .reset_index(drop=True)
    )

    # Step 2: Labeling
    df_labeled = (
        df.groupby(['lat_r', 'lon_r'])
        .apply(process_group)
        .reset_index(drop=True)
    )

    # Step 3: Remove boundary rows (due to shift/rolling)
    df_labeled = df_labeled.dropna(subset=['prev_ndvi', 'next_ndvi'])

    # Step 4: Smoothing
    df_labeled = (
        df_labeled.groupby(['lat_r', 'lon_r'])
        .apply(smooth_labels)
        .reset_index(drop=True)
    )

    # Step 5: Final cleanup (important)
    df_labeled = df_labeled.dropna() 
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_csv(output_path, index=False)

    logger.info(f"Pipeline completed. Saved at: {output_path}")