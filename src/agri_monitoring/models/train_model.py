
import logging
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def train_model(data_path: Path):
    import pandas as pd
    logger.info("Loading processed data...")
    df = pd.read_csv(data_path)

    # -----------------------------
    # Feature Selection (IMPORTANT)
    # -----------------------------
    feature_cols = [
        "NDVI",
        "ndvi_roll_mean",
        "ndvi_roll_std",
        "ndvi_diff",
        "ndvi_trend",
        "prev_ndvi",
        "next_ndvi",
        "ndvi_drop",
        "temperature",
        "humidity",
        "precipitation"
    ]

    X = df[feature_cols]
    y = df["label"]

    logger.info(f"Using features: {feature_cols}")

    # -----------------------------
    # Train Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Model Training
    # -----------------------------
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    # -----------------------------
    # Evaluation
    # -----------------------------
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    logger.info(f"Accuracy: {acc}")
    logger.info(f"Precision: {prec}")
    logger.info(f"Recall: {rec}")

    #confusion matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    #feature imp
    import pandas as pd

    importance = model.feature_importances_
    feature_importance = pd.Series(importance, index=feature_cols).sort_values(ascending=False)

    print("\nFeature Importance:\n", feature_importance)

    # -----------------------------
    # Save Metrics
    # -----------------------------
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec)
    }

    Path("reports").mkdir(exist_ok=True)

    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Metrics saved to reports/metrics.json")

    # -----------------------------
    # Save Model
    # -----------------------------
    Path("models").mkdir(exist_ok=True)

    import joblib
    joblib.dump(model, "models/model.pkl")

    logger.info("Model saved to models/model.pkl")

    return model

