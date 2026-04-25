# =============================================================================
#  models/exporter.py — Merge phenology info and export final labeled CSV
# =============================================================================

import pandas as pd
from src.agri_monitoring.features.phenology import detect_phenology


def export_results(feat_df: pd.DataFrame,
                   ts_store: dict,
                   output_csv: str = 'crop_stress_labels.csv') -> pd.DataFrame:
    """
    Attach phenology columns (green_up_date, peak_date, senescence_date,
    peak_ndvi, season_length) to feat_df, then save to CSV.

    Args:
        feat_df    (pd.DataFrame): Output of cluster_stress().
        ts_store   (dict)        : Output of smooth_all().
        output_csv (str)         : Filepath for the output CSV.

    Returns:
        pd.DataFrame: Final dataframe with all features + stress_label
                      + phenology columns, saved to output_csv.
    """
    pheno_records = []
    for (lat, lon), ts in ts_store.items():
        p        = detect_phenology(ts)
        p['lat'] = lat
        p['lon'] = lon
        pheno_records.append(p)

    pheno_df = pd.DataFrame(pheno_records)
    final_df = feat_df.merge(pheno_df, on=['lat', 'lon'], how='left')
    final_df.to_csv(output_csv, index=False)

    print(f"[✓] Final labeled data saved → {output_csv}")
    print(f"    Shape: {final_df.shape}")
    print(f"\n── Label counts ──────────────────────────────────────────")
    print(final_df['stress_label'].value_counts().to_string())
    print("─────────────────────────────────────────────────────────\n")

    return final_df
