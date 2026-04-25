# =============================================================================
#  models/classifier.py — Supervised Random Forest on pseudo-labels
# =============================================================================

import pandas as pd
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing   import LabelEncoder
from config.config import CONFIG, FEATURE_COLS


def train_supervised_model(feat_df: pd.DataFrame):
    """
    Train a Random Forest classifier using the cluster-derived
    stress labels as pseudo ground-truth.

    Purpose:
        Once clustering labels every location, use those labels to
        train a supervised model so future/new data can be predicted
        instantly — without re-running the full clustering pipeline.

    Args:
        feat_df (pd.DataFrame): Output of cluster_stress().
                                Must have 'stress_label' column.

    Returns:
        rf    (RandomForestClassifier): Trained model.
        le    (LabelEncoder)          : Encoder to map int → label string.

    Usage after training:
        new_X  = new_feat_df[FEATURE_COLS].fillna(0).values
        preds  = le.inverse_transform(rf.predict(new_X))
    """
    le = LabelEncoder()
    X  = feat_df[FEATURE_COLS].fillna(0).values
    y  = le.fit_transform(feat_df['stress_label'])

    rf = RandomForestClassifier(
        n_estimators = 200,
        max_depth    = 8,
        random_state = CONFIG['random_state'],
        class_weight = 'balanced',
    )

    # 5-fold cross-validation on pseudo-labels
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='f1_macro')
    print(f"[✓] RF cross-val F1 (macro): {cv_scores.mean():.3f} "
          f"± {cv_scores.std():.3f}")

    rf.fit(X, y)

    # Top feature importances
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    print("\n── Top 10 Important Features ──────────────────────────────")
    print(importances.sort_values(ascending=False).head(10).round(4).to_string())
    print("────────────────────────────────────────────────────────────\n")

    return rf, le
