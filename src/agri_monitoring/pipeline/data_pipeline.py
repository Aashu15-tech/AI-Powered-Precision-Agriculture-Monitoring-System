import os
import pandas as pd
import logging

from src.agri_monitoring.data.data_loader import load_data
from src.agri_monitoring.features.feature_engineering import create_features
from src.agri_monitoring.features.labeling import process_group, smooth_labels
from src.agri_monitoring.data.weather_loader import get_weather_data

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

    # -----------------------------
    # Add Weather Data (Simple Version)
    # -----------------------------

    # Take one sample location
    sample_lat = df['lat'].iloc[0]
    sample_lon = df['lon'].iloc[0]

    # Get weather data
    weather_df = get_weather_data(
        lat=sample_lat,
        lon=sample_lon,
        start_date="20230101",
        end_date="20231231"
    )

    # Convert date format
    df['date'] = pd.to_datetime(df['date'])
    weather_df['date'] = pd.to_datetime(weather_df['date'])

    # Merge with main dataset
    df = pd.merge(df, weather_df, on='date', how='left')

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
