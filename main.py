# =============================================================================
#  main.py — Entry point for the Crop Stress Labeling Pipeline
#
#  Usage:
#      python main.py
#
#  Your CSV must have columns:
#      lat | lon | date | NDVI | NDWI | EVI | SAVI
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

from src.agri_monitoring.pipeline.runner import run_pipeline
from config.config import FEATURE_COLS


if __name__ == '__main__':

    # ── Configure paths ────────────────────────────────────────────────────────
    INPUT_CSV  = 'data/raw/crop_health_assesment_dataset.csv'   # ← change to your file path
    OUTPUT_CSV = 'data/processed/crop_stress_labels.csv'

    # ── Run full pipeline ──────────────────────────────────────────────────────
    final_df, ts_store, model, label_encoder = run_pipeline(
        filepath     = INPUT_CSV,
        output_csv   = OUTPUT_CSV,
        plot_samples = 6,       # number of sample locations to plot
        train_rf     = True,    # set False to skip Random Forest training
    )

    # ── Preview output ─────────────────────────────────────────────────────────
    print("── Sample Output ─────────────────────────────────────────────")
    print(final_df[[
        'lat', 'lon', 'ndvi_mean', 'ndwi_mean',
        'stress_label', 'peak_date', 'season_length'
    ]].head(10).to_string(index=False))

    # ── Predict on new data (after training rf_model) ─────────────────────────
    # from data           import load_data, build_time_series, smooth_all
    # from features       import extract_features
    #
    # new_df       = load_data('new_data.csv')
    # new_ts       = smooth_all(build_time_series(new_df))
    # new_feat_df  = extract_features(new_ts)
    # new_X        = new_feat_df[FEATURE_COLS].fillna(0).values
    # predictions  = label_encoder.inverse_transform(rf_model.predict(new_X))
    # new_feat_df['stress_label'] = predictions
    # new_feat_df.to_csv('new_data_predictions.csv', index=False)
    import pickle

    with open('models/model.pkl',      'wb') as f:
        pickle.dump(model, f)

    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    print("[✓] Model and encoder saved")