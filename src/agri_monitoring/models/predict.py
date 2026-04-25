
#  Your new CSV must have columns:
#      lat | lon | date | NDVI | NDWI | EVI | SAVI

import os
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from src.agri_monitoring.features.phenology import extract_features, detect_phenology
from config.config           import FEATURE_COLS

from src.agri_monitoring.data.loader import load_data
from src.agri_monitoring.data.time_series import build_time_series, smooth_all
from src.agri_monitoring.features.extractor import extract_features

# =============================================================================
#  CONFIGURATION — change these paths
# =============================================================================

NEW_DATA_CSV      = 'predict_dataset/new_sentinel2_data.csv'   # ← your new input CSV
OUTPUT_CSV        = 'predict_dataset/new_data_predictions.csv' # ← where predictions are saved
MODEL_PATH        = 'model/model.pkl'             # ← saved trained RF model
ENCODER_PATH      = 'label_encoder.pkl'        # ← saved label encoder


# =============================================================================
#  STEP A — Save model after training (run this once after main.py)
# =============================================================================

def save_model(model, label_encoder,
               model_path  : str = MODEL_PATH,
               encoder_path: str = ENCODER_PATH):
    """
    Serialize and save the trained RF model and label encoder to disk.

    Call this once right after running run_pipeline() in main.py:

        final_df, ts_store, model, label_encoder = run_pipeline(...)
        save_model(rf_model, label_encoder)

    Args:
        model         : Trained RandomForestClassifier from classifier.py
        label_encoder : Fitted LabelEncoder from classifier.py
        model_path    : Filepath to save .pkl model
        encoder_path  : Filepath to save .pkl encoder
    """
    with open(model_path,   'wb') as f:
        pickle.dump(model, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    print(f"[✓] Model saved   → {model_path}")
    print(f"[✓] Encoder saved → {encoder_path}")


# =============================================================================
#  STEP B — Load saved model
# =============================================================================

def load_model(model_path  : str = MODEL_PATH,
               encoder_path: str = ENCODER_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            f"Run main.py first to train and save the model via save_model()."
        )
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Encoder not found at '{encoder_path}'.\n"
            f"Run main.py first to train and save the encoder via save_model()."
        )

    with open(model_path,   'rb') as f:
        rf_model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    print(f"[✓] Model loaded   ← {model_path}")
    print(f"[✓] Encoder loaded ← {encoder_path}")
    return rf_model, label_encoder


# =============================================================================
#  STEP C — Preprocess new data
# =============================================================================

def preprocess_new_data(filepath: str) -> tuple[dict, pd.DataFrame]:
    """
    Run full preprocessing pipeline on a new Sentinel-2 CSV:
        load → build time series → smooth → extract features

    Args:
        filepath (str): Path to new input CSV.
                        Required columns: lat, lon, date, NDVI, NDWI, EVI, SAVI

    Returns:
        ts_store (dict)        : Smoothed time series per location.
        feat_df  (pd.DataFrame): Extracted temporal features per location.
    """
    print("\n── Preprocessing new data ────────────────────────────────")

    df = load_data(filepath)

    # 2. Group by (lat, lon), sort by date, build time series dict
    ts_store = build_time_series(df)

    # 3. Gap-fill NaNs + Savitzky-Golay smoothing
    ts_store = smooth_all(ts_store)

    # 4. Extract same temporal features as training
    feat_df = extract_features(ts_store)

    print(f"[✓] Preprocessing done — {len(feat_df):,} locations ready for prediction")
    return ts_store, feat_df


# =============================================================================
#  STEP D — Predict stress labels
# =============================================================================

def predict_stress(feat_df     : pd.DataFrame,
                   rf_model,
                   label_encoder) -> pd.DataFrame:
    """
    Predict stress labels for preprocessed feature dataframe.

    Args:
        feat_df       (pd.DataFrame): Output of extract_features().
                                      Must contain all FEATURE_COLS.
        rf_model                    : Loaded RandomForestClassifier.
        label_encoder               : Loaded LabelEncoder.

    Returns:
        pd.DataFrame: feat_df with three new columns:
            'stress_label'       (str)   — HEALTHY / MILD_STRESS / STRESSED
            'stress_label_code'  (int)   — numeric class (0/1/2)
            'confidence'         (float) — model's max class probability (0–1)
    """
    print("\n── Predicting stress labels ──────────────────────────────")

    X = feat_df[FEATURE_COLS].fillna(0).values

    # Numeric predictions
    label_codes = rf_model.predict(X)

    # Decode to string labels
    stress_labels = label_encoder.inverse_transform(label_codes)

    # Confidence = probability of the predicted class
    proba      = rf_model.predict_proba(X)          # shape: (n_locations, n_classes)
    confidence = proba[np.arange(len(proba)), label_codes]

    result_df = feat_df.copy()
    result_df['stress_label']      = stress_labels
    result_df['stress_label_code'] = label_codes
    result_df['confidence']        = confidence.round(3)

    # Summary
    print(f"[✓] Predictions complete for {len(result_df):,} locations")
    print(f"\n── Prediction counts ─────────────────────────────────────")
    print(result_df['stress_label'].value_counts().to_string())
    print(f"\n── Mean confidence per label ─────────────────────────────")
    print(result_df.groupby('stress_label')['confidence'].mean().round(3).to_string())
    print("──────────────────────────────────────────────────────────\n")

    return result_df


# =============================================================================
#  STEP E — Attach phenology + export
# =============================================================================

def export_predictions(result_df  : pd.DataFrame,
                       ts_store   : dict,
                       output_csv : str = OUTPUT_CSV) -> pd.DataFrame:
    """
    Attach phenology columns (green_up_date, peak_date, senescence_date,
    peak_ndvi, season_length) and save final prediction CSV.

    Args:
        result_df  (pd.DataFrame): Output of predict_stress().
        ts_store   (dict)        : Smoothed time series from preprocess_new_data().
        output_csv (str)         : Filepath to save final CSV.

    Returns:
        pd.DataFrame: Final dataframe with predictions + phenology.
    """
    # Build phenology for each new location
    pheno_records = []
    for (lat, lon), ts in ts_store.items():
        p        = detect_phenology(ts)
        p['lat'] = lat
        p['lon'] = lon
        pheno_records.append(p)

    pheno_df = pd.DataFrame(pheno_records)
    final_df = result_df.merge(pheno_df, on=['lat', 'lon'], how='left')

    # Reorder columns — identifiers first, label second, rest after
    priority_cols = ['lat', 'lon', 'stress_label', 'confidence',
                     'stress_label_code', 'peak_date',
                     'green_up_date', 'senescence_date',
                     'peak_ndvi', 'season_length']
    other_cols    = [c for c in final_df.columns if c not in priority_cols]
    final_df      = final_df[priority_cols + other_cols]

    final_df.to_csv(output_csv, index=False)
    print(f"[✓] Predictions saved → {output_csv}")
    print(f"    Shape: {final_df.shape}")

    return final_df


# =============================================================================
#  STEP F — Quick confidence check
# =============================================================================

def flag_low_confidence(final_df    : pd.DataFrame,
                        threshold   : float = 0.5,
                        output_csv  : str   = 'low_confidence_locations.csv'):
    """
    Flag locations where the model is uncertain (low confidence).
    These may need manual review or re-inspection.

    Args:
        final_df   (pd.DataFrame): Output of export_predictions().
        threshold  (float)       : Confidence below this = flagged. Default 0.5.
        output_csv (str)         : Path to save flagged locations.

    Returns:
        pd.DataFrame: Subset of flagged low-confidence rows.
    """
    low_conf = final_df[final_df['confidence'] < threshold].copy()

    if len(low_conf) == 0:
        print(f"[✓] No low-confidence predictions (threshold={threshold})")
    else:
        low_conf.to_csv(output_csv, index=False)
        pct = len(low_conf) / len(final_df) * 100
        print(f"[!] {len(low_conf):,} locations ({pct:.1f}%) have confidence "
              f"< {threshold} → saved to {output_csv}")

    return low_conf


# =============================================================================
#  MAIN — Run full prediction pipeline
# =============================================================================

def run_predict(new_data_csv  : str   = NEW_DATA_CSV,
                output_csv    : str   = OUTPUT_CSV,
                model_path    : str   = MODEL_PATH,
                encoder_path  : str   = ENCODER_PATH,
                conf_threshold: float = 0.5):
    """
    Full end-to-end prediction pipeline for new Sentinel-2 data.

    Args:
        new_data_csv   (str)  : Path to new input CSV.
        output_csv     (str)  : Path to save prediction CSV.
        model_path     (str)  : Path to saved rf_model.pkl.
        encoder_path   (str)  : Path to saved label_encoder.pkl.
        conf_threshold (float): Flag locations below this confidence.

    Returns:
        final_df (pd.DataFrame): Predictions with phenology columns.
    """
    print("\n" + "=" * 65)
    print("  CROP STRESS PREDICTION  —  New Sentinel-2 Dataset")
    print("=" * 65)

    # B. Load model
    rf_model, label_encoder = load_model(model_path, encoder_path)

    # C. Preprocess new data
    ts_store, feat_df = preprocess_new_data(new_data_csv)

    # D. Predict
    result_df = predict_stress(feat_df, rf_model, label_encoder)

    # E. Attach phenology + export
    final_df = export_predictions(result_df, ts_store, output_csv)

    # F. Flag uncertain predictions
    flag_low_confidence(final_df, threshold=conf_threshold)

    print("[✓] Prediction pipeline complete!\n")
    return final_df


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == '__main__':

    final_df = run_predict(
        new_data_csv   = NEW_DATA_CSV,
        output_csv     = OUTPUT_CSV,
        model_path     = MODEL_PATH,
        encoder_path   = ENCODER_PATH,
        conf_threshold = 0.5,
    )

    # Preview top results
    print("── Preview ───────────────────────────────────────────────────")
    print(final_df[['lat', 'lon', 'stress_label',
                     'confidence', 'peak_date',
                     'ndvi_mean', 'ndwi_mean']].head(10).to_string(index=False))


# =============================================================================
#  HOW TO SAVE MODEL FROM main.py  (add these lines at end of main.py)
# =============================================================================
#
#   from predict import save_model
#
#   final_df, ts_store, rf_model, label_encoder = run_pipeline(...)
#   save_model(rf_model, label_encoder)
#
# =============================================================================