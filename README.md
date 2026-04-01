# 🌾 AI-Powered Precision Agriculture Monitoring System

## 📌 Overview
This project aims to build an AI system for monitoring crop health using satellite data (NDVI) and time-series analysis. The goal is to move from rule-based agricultural monitoring to a machine learning-based intelligent system.

---

## 🎯 Objectives
- Analyze satellite imagery (NDVI)
- Detect crop health conditions
- Build time-series based labeling system
- Train ML model for crop health prediction
- Prepare system for integration with weather and soil data

---


## ⚙️ Pipeline Flow

1. Load raw satellite data
2. Perform feature engineering (rolling mean, trend, etc.)
3. Apply time-series labeling
4. Smooth labels to remove noise
5. Generate processed dataset

---

## 🧠 Feature Engineering

Features used:
- NDVI
- Rolling mean
- Rolling standard deviation
- NDVI difference
- NDVI trend
- Previous & next NDVI
- NDVI drop

---

## 🏷 Labeling Strategy

Label is generated using:
- NDVI threshold
- NDVI drop percentage
- Temporal consistency

Label smoothing is applied to remove noise in time-series.

---

## 🤖 Model

- Algorithm: Random Forest Classifier
- Input: Engineered NDVI features
- Output: Crop health label (0 = unhealthy, 1 = healthy)

---

## 📊 Results (Baseline)

- Accuracy: ~99%
- Precision: ~99%
- Recall: 1.0

⚠️ Note: This is a baseline model trained on rule-based labels, not real ground truth.

---

## 🚧 Limitations

- Labels are synthetic (rule-based)
- Model heavily depends on NDVI
- No weather or soil data yet

---

## 🚀 Future Work

- Integrate weather data (temperature, rainfall, humidity)
- Add soil health data
- Improve labeling using real-world datasets
- Build deep learning models (LSTM for time-series)
- Deploy dashboard for visualization

---

## 🧪 How to Run

```bash
python main.py
