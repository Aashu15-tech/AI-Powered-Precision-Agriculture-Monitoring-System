"""
predict.py
----------
Models load karke 10 sample points ki:
  1. Crop health predict karta hai  (rf_model.pkl + label_encoder.pkl)
  2. Disease detection karta hai    (crop_desease_model.pkl)
  3. Results CSV mein save karta hai
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


# ─── Model paths ──────────────────────────────────────────────────────────────
MODEL_DIR = Path(".")   # Same folder mein rakho sab models

RF_MODEL_PATH      = MODEL_DIR / "rf_model.pkl"
LABEL_ENC_PATH     = MODEL_DIR / "label_encoder.pkl"
DISEASE_MODEL_PATH = MODEL_DIR / "crop_desease_model.pkl"

# ─── Feature columns (GEE se aane wale) ──────────────────────────────────────
FEATURE_COLS = ["NDVI", "NDWI", "EVI", "SAVI"]


def load_models():
    """Teenon models aur label encoder load karo."""
    print("📦 Models load ho rahe hain...")

    with open(RF_MODEL_PATH, "rb") as f:
        rf_model = pickle.load(f)
    print(f"   ✅ rf_model loaded — {type(rf_model).__name__}")

    with open(LABEL_ENC_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    print(f"   ✅ label_encoder loaded — classes: {list(label_encoder.classes_)}")

    with open(DISEASE_MODEL_PATH, "rb") as f:
        disease_model = pickle.load(f)
    print(f"   ✅ disease_model loaded — {type(disease_model).__name__}")

    return rf_model, label_encoder, disease_model


def predict_crop_health(rf_model, label_encoder, features: np.ndarray) -> list[str]:
    """
    Crop health predict karo.
    Returns: ['Healthy', 'Moderate', 'Stress', ...]
    """
    preds_encoded = rf_model.predict(features)
    preds_labels  = label_encoder.inverse_transform(preds_encoded)
    return list(preds_labels)


def predict_disease(disease_model, features: np.ndarray) -> list[str]:
    """
    Disease detection karo.
    Returns: ['Disease Detected', 'No Disease'] list
    """
    preds = disease_model.predict(features)

    # Binary output handle karo (0/1 ya class names)
    result = []
    for p in preds:
        p_str = str(p).lower()
        if p_str in ["1", "true", "disease", "yes", "diseased"]:
            result.append("Disease Detected")
        elif p_str in ["0", "false", "no disease", "no", "healthy"]:
            result.append("No Disease")
        else:
            result.append(str(p))   # model ka raw output use karo
    return result


def get_map_color(health_label: str, disease_label: str) -> str:
    """
    Map pe point ka color decide karo:
      - Red:    Stress (chahe disease ho ya na ho)
      - Orange: Moderate
      - Green:  Healthy + No Disease
    """
    h = health_label.lower()
    if "stress" in h:
        return "red"
    elif "moderate" in h:
        return "orange"
    else:
        return "green"


def run_predictions(features_csv: str = "sentinel2_features.csv") -> pd.DataFrame:
    """
    Main prediction pipeline.

    Args:
        features_csv: GEE se fetch ki gayi CSV file path

    Returns:
        DataFrame with predictions added
    """
    # ── Data load ──
    if not Path(features_csv).exists():
        raise FileNotFoundError(
            f"❌ '{features_csv}' nahi mila!\n"
            f"   Pehle fetch_sentinel.py chalao."
        )

    df = pd.read_csv(features_csv)
    print(f"📊 {len(df)} points loaded from '{features_csv}'")
    print(df[FEATURE_COLS].describe())

    # ── Models load ──
    rf_model, label_encoder, disease_model = load_models()

    # ── Feature matrix ──
    X = df[FEATURE_COLS].values.astype(np.float32)

    # ── Predict ──
    print("\n🔮 Predictions chal rahi hain...")
    health_preds  = predict_crop_health(rf_model, label_encoder, X)
    disease_preds = predict_disease(disease_model, X)

    # ── Results add karo ──
    df["crop_health"]  = health_preds
    df["disease"]      = disease_preds
    df["map_color"]    = [
        get_map_color(h, d)
        for h, d in zip(health_preds, disease_preds)
    ]

    # ── Health score for display ──
    health_score_map = {"healthy": 100, "moderate": 60, "stress": 20}
    df["health_score"] = df["crop_health"].apply(
        lambda x: health_score_map.get(x.lower(), 50)
    )

    # ── Save ──
    output_path = "predictions.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Predictions saved: {output_path}")

    # ── Summary ──
    print("\n📋 Summary:")
    print(f"   🟢 Healthy:  {sum(1 for h in health_preds if 'healthy'  in h.lower())}")
    print(f"   🟡 Moderate: {sum(1 for h in health_preds if 'moderate' in h.lower())}")
    print(f"   🔴 Stress:   {sum(1 for h in health_preds if 'stress'   in h.lower())}")
    print(f"   ⚠️  Disease:  {sum(1 for d in disease_preds if 'detected' in d.lower())}")

    return df


if __name__ == "__main__":
    results = run_predictions("sentinel2_features.csv")
    print("\n🎯 Final Results:")
    print(results[["point_id", "lat", "lon", "NDVI", "crop_health", "disease", "map_color"]].to_string(index=False))