from src.agri_monitoring.data.loader import load_data
from src.agri_monitoring.data.time_series import build_time_series, smooth_all
from src.agri_monitoring.features.extractor import extract_features
from src.agri_monitoring.models.cluster import cluster_stress
from src.agri_monitoring.models.classifier import train_supervised_model
from src.agri_monitoring.models.exporter import  export_results
from src.agri_monitoring.visualization.plots import (plot_stress_distribution,plot_feature_heatmap,plot_sample_locations)


def run_pipeline(filepath     : str,
                 output_csv   : str  = 'data/processed/crop_stress_labels.csv',
                 plot_samples : int  = 6,
                 train_rf     : bool = True):
    """
    Run the full crop stress labeling pipeline:
        1. Load & validate CSV
        2. Build time series per location (grouped + sorted by date)
        3. Smooth & gap-fill each index time series
        4. Extract temporal features
        5. Cluster into stress labels (HEALTHY / MILD_STRESS / STRESSED)
        6. Export final labeled CSV (with phenology columns)
        7. Visualize distributions and sample time series
        8. (Optional) Train Random Forest on pseudo-labels

    Args:
        filepath     (str)  : Path to input Sentinel-2 CSV.
        output_csv   (str)  : Path for the output labeled CSV.
        plot_samples (int)  : Number of sample locations to plot (0 = skip).
        train_rf     (bool) : Whether to train supervised RF on pseudo-labels.

    Returns:
        final_df      (pd.DataFrame)          : Labeled dataset.
        ts_store      (dict)                  : Smoothed time series store.
        rf_model      (RandomForestClassifier): Trained model (or None).
        label_encoder (LabelEncoder)          : Label encoder (or None).
    """
    print("\n" + "=" * 65)
    print("  CROP STRESS LABELING PIPELINE  —  Sentinel-2 Indices")
    print("=" * 65 + "\n")

    # Step 1 — Load
    df = load_data(filepath)

    # Step 2 — Build time series (grouped by lat/lon, sorted by date)
    ts_store = build_time_series(df)

    # Step 3 — Smooth & gap-fill
    ts_store = smooth_all(ts_store)

    # Step 4 — Extract temporal features
    feat_df = extract_features(ts_store)

    # Step 5 — Cluster → stress labels
    feat_df = cluster_stress(feat_df)

    # Step 6 — Export (merges phenology + saves CSV)
    final_df = export_results(feat_df, ts_store, output_csv)

    # Step 7 — Visualize
    plot_stress_distribution(feat_df)
    plot_feature_heatmap(feat_df)
    if plot_samples > 0:
        plot_sample_locations(ts_store, feat_df, n_samples=plot_samples)

    # Step 8 — (Optional) Supervised model on pseudo-labels
    if train_rf:
        rf_model, label_encoder = train_supervised_model(feat_df)
    else:
        rf_model, label_encoder = None, None

    print("[✓] Pipeline complete!\n")
    return final_df, ts_store, rf_model, label_encoder
