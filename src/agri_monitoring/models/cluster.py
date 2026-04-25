# =============================================================================
#  models/cluster.py — KMeans clustering to assign stress labels
# =============================================================================

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster       import KMeans
from sklearn.metrics       import silhouette_score
from config.config import CONFIG, FEATURE_COLS


def cluster_stress(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster locations into stress categories using KMeans,
    then automatically map cluster IDs to human-readable labels
    based on the cluster center values of NDVI and NDWI.

    Label mapping logic:
        Highest ndvi_mean + ndwi_mean → HEALTHY
        Middle                        → MILD_STRESS
        Lowest                        → STRESSED

    Args:
        feat_df (pd.DataFrame): Output of extract_features().
                                Must contain all FEATURE_COLS.

    Returns:
        pd.DataFrame: Same dataframe with two new columns:
            'cluster'      (int)  — raw KMeans cluster ID
            'stress_label' (str)  — HEALTHY / MILD_STRESS / STRESSED
    """
    X        = feat_df[FEATURE_COLS].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(
        n_clusters   = CONFIG['n_clusters'],
        random_state = CONFIG['random_state'],
        n_init       = 20,
        max_iter     = 500,
    )
    cluster_ids = kmeans.fit_predict(X_scaled)
    feat_df     = feat_df.copy()
    feat_df['cluster'] = cluster_ids

    # Quality check
    score = silhouette_score(X_scaled, cluster_ids)
    print(f"[✓] KMeans silhouette score: {score:.3f}  (>0.3 is acceptable)")

    # ── Auto-interpret clusters by health score ────────────────────────────────

    centers = pd.DataFrame(kmeans.cluster_centers_, columns=FEATURE_COLS)

    centers['health_score'] = (
        0.5 * centers['ndvi_mean'] +0.1 * centers['savi_mean'] +0.2 * centers['evi_mean'] +
        0.2 * centers['ndwi_mean'] - 0.3 * centers['ndwi_stress_fraction'] - 0.4 * centers['dual_stress_fraction']
        )

    # rank: 1 = least healthy (STRESSED), 3 = most healthy (HEALTHY)
    rank = centers['health_score'].rank(ascending=True).astype(int)

    label_map = {}
    for cluster_id, r in rank.items():
        if r == 1:
            label_map[cluster_id] = 'STRESSED'
        elif r == 2:
            label_map[cluster_id] = 'MILD_STRESS'
        else:
            label_map[cluster_id] = 'HEALTHY'

    feat_df['stress_label'] = feat_df['cluster'].map(label_map)

    # Print per-cluster summary
    print("\n── Cluster Summary ──────────────────────────────────────")
    summary = feat_df.groupby('stress_label')[
        ['ndvi_mean', 'ndwi_mean', 'evi_mean', 'savi_mean',
         'ndwi_stress_fraction', 'dual_stress_fraction']
    ].mean().round(3)
    print(summary.to_string())
    print("─────────────────────────────────────────────────────────\n")

    return feat_df
