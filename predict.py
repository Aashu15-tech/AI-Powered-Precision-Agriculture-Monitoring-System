# =============================================================================
#  predict.py — Preprocess new GEE Sentinel-2 data and predict stress labels
#
#  Uses:
#     model.pkl       → crop stress prediction (HEALTHY/MILD_STRESS/STRESSED)
#      label_encoder.pkl  → int → label string
#
#  Input  : raw CSV from gee_fetch.py  (lat, lon, date, NDVI, NDWI, EVI, SAVI)
#  Output : predictions DataFrame      (lat, lon, stress_label, confidence, ...)
# =============================================================================

import os
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from config.config           import CONFIG, FEATURE_COLS
from src.agri_monitoring.data.loader      import load_data
from src.agri_monitoring.data.time_series import build_time_series, smooth_all
from src.agri_monitoring.features.extractor import extract_features


# =============================================================================
# LOAD RF MODEL + LABEL ENCODER
# =============================================================================

def load_stress_model(model_path   : str = 'models/model.pkl',
                      encoder_path : str = 'models/label_encoder.pkl'):
    """
    Load saved RandomForest model and LabelEncoder from disk.

    Args:
        model_path   (str): Path to rf_model.pkl
        encoder_path (str): Path to label_encoder.pkl

    Returns:
        rf_model      (RandomForestClassifier)
        label_encoder (LabelEncoder)
    """
    for path in [model_path, encoder_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n[✗] File not found: '{path}'\n"
                f"    Pehle main.py chalao aur model save karo:\n"
                f"        import pickle\n"
                f"        with open('rf_model.pkl','wb') as f: pickle.dump(rf_model, f)\n"
                f"        with open('label_encoder.pkl','wb') as f: pickle.dump(le, f)"
            )

    with open(model_path,   'rb') as f:
        rf_model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    print(f"[✓] RF model loaded      → {model_path}")
    print(f"[✓] Label encoder loaded → {encoder_path}")
    return rf_model, label_encoder


# =============================================================================
# PREPROCESS NEW DATA
# =============================================================================

def preprocess_new_data(raw_csv_path: str) -> tuple:
    """
    Full preprocessing pipeline on new GEE-fetched CSV.

    Steps (same as training pipeline):
        1. Load & validate CSV
        2. Group by (lat, lon), sort by date → time series
        3. Gap-fill + Savitzky-Golay smoothing
        4. Extract 17 temporal features per location

    Args:
        raw_csv_path (str): Path to raw CSV from gee_fetch.py

    Returns:
        feat_df  (pd.DataFrame): Feature matrix — one row per location
        ts_store (dict)        : Smoothed time series — for plotting in Streamlit
    """
    print("\n── Preprocessing new data ───────────────────────────────")

    # Step 1 — Load
    df = load_data(raw_csv_path)

    # Step 2 — Build time series (grouped by lat/lon, sorted by date)
    ts_store = build_time_series(df)

    if len(ts_store) == 0:
        raise ValueError(
            "[✗] No valid locations found after time series building.\n"
            f"    Each location needs at least {CONFIG['min_valid_obs']} observations.\n"
            "    Try increasing FETCH_DAYS in config.py"
        )

    # Step 3 — Smooth & gap-fill
    ts_store = smooth_all(ts_store)

    # Step 4 — Extract features
    feat_df = extract_features(ts_store)

    print(f"[✓] Preprocessing complete: {len(feat_df)} locations ready")
    return feat_df, ts_store


# =============================================================================
# PREDICT STRESS LABELS
# =============================================================================

def predict_stress(feat_df      : pd.DataFrame,
                   rf_model,
                   label_encoder) -> pd.DataFrame:
    """
    Predict crop stress label for each location using the trained RF model.

    Args:
        feat_df       (pd.DataFrame): Output of preprocess_new_data()
        rf_model      : Loaded RandomForestClassifier
        label_encoder : Loaded LabelEncoder

    Returns:
        pd.DataFrame: feat_df with added columns:
            'stress_label'      — HEALTHY / MILD_STRESS / STRESSED
            'stress_confidence' — probability of predicted class (0–1)
            'healthy_prob'      — P(HEALTHY)
            'mild_stress_prob'  — P(MILD_STRESS)
            'stressed_prob'     — P(STRESSED)
    """
    X = feat_df[FEATURE_COLS].fillna(0).values

    # Predict class indices
    pred_indices = rf_model.predict(X)

    # Predict probabilities for all classes
    pred_proba = rf_model.predict_proba(X)   # shape: (n_locations, n_classes)

    # Map int → label strings
    pred_labels = label_encoder.inverse_transform(pred_indices)

    # Confidence = probability of the predicted class
    confidence = pred_proba[np.arange(len(pred_indices)), pred_indices]

    # Map class names to probability columns
    classes = label_encoder.classes_    # ordered list of class names
    result_df = feat_df[['lat', 'lon',
                          'ndvi_mean', 'ndwi_mean',
                          'evi_mean',  'savi_mean',
                          'ndvi_max',  'ndvi_min',
                          'dual_stress_fraction',
                          'ndwi_stress_fraction']].copy()

    result_df['stress_label']      = pred_labels
    result_df['stress_confidence'] = np.round(confidence * 100, 1)  # as %

    # Individual class probabilities
    for i, cls in enumerate(classes):
        col_name = f'{cls.lower()}_prob'
        result_df[col_name] = np.round(pred_proba[:, i] * 100, 1)

    print(f"\n── Prediction Results ───────────────────────────────────")
    print(result_df[['lat', 'lon', 'stress_label',
                      'stress_confidence', 'ndvi_mean']].to_string(index=False))
    print(f"\n── Label Distribution ───────────────────────────────────")
    print(result_df['stress_label'].value_counts().to_string())
    print("─────────────────────────────────────────────────────────\n")

    return result_df


# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

def save_predictions(result_df  : pd.DataFrame,
                     output_csv : str = 'data/firozpur_predictions.csv'):
    """Save prediction results to CSV."""
    result_df.to_csv(output_csv, index=False)
    print(f"[✓] Predictions saved → {output_csv}")


# =============================================================================
# FULL PREDICT PIPELINE (called from Streamlit)
# =============================================================================

def run_prediction(raw_csv_path : str = 'data/sentinel2_data.csv',
                   model_path   : str = 'models/model.pkl',
                   encoder_path : str = 'models/label_encoder.pkl',
                   output_csv   : str = 'data/firozpur_predictions.csv'):
    """
    End-to-end prediction pipeline for new Sentinel-2 data.

    Args:
        raw_csv_path (str): Raw CSV from gee_fetch.py
        model_path   (str): Path to rf_model.pkl
        encoder_path (str): Path to label_encoder.pkl
        output_csv   (str): Where to save predictions

    Returns:
        result_df (pd.DataFrame): Predictions with lat/lon/label/confidence
        ts_store  (dict)        : Smoothed time series for Streamlit plots
    """
    # Load model
    rf_model, label_encoder = load_stress_model(model_path, encoder_path)

    # Preprocess
    feat_df, ts_store = preprocess_new_data(raw_csv_path)

    # Predict
    result_df = predict_stress(feat_df, rf_model, label_encoder)

    # Save
    save_predictions(result_df, output_csv)

    return result_df, ts_store


# =============================================================================
# STANDALONE RUN
# =============================================================================

if __name__ == '__main__':
    result_df, ts_store = run_prediction(
        raw_csv_path = 'data/Sentinel2_data.csv',
        model_path   = 'models/model.pkl',
        encoder_path = 'models/label_encoder.pkl',
        output_csv   = 'data/firozpur_predictions.csv',
    )
    print(result_df.head())
