# 🌾 AI-Powered Agriculture Monitoring System

An end-to-end intelligent crop health monitoring and disease prediction system that leverages **real-time Sentinel-2 satellite imagery**, **machine learning**, and **deep learning** to help farmers and agronomists detect crop stress early and identify plant diseases with high accuracy.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Pipeline](#-pipeline)
  - [Phase 1: Satellite Data Collection](#phase-1-satellite-data-collection--preprocessing)
  - [Phase 2: Crop Health Classification](#phase-2-crop-health-classification)
  - [Phase 3: Disease Prediction](#phase-3-disease-prediction)
- [Models Used](#-models-used)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Future Work](#-future-work)
- [Contributing](#-contributing)

---

## 🌍 Overview

Traditional crop monitoring relies on manual field inspections, which are time-consuming, expensive, and often too slow to prevent large-scale crop loss. This system automates the entire process:

1. **Collects** 50 days of real-time Sentinel-2 multispectral satellite data via Google Earth Engine (GEE).
2. **Analyzes** spectral indices (NDVI, NDRE, EVI, etc.) using Random Forest and K-Means Clustering to classify crop health as **Healthy**, **Moderate**, or **Stressed**.
3. **Predicts** the specific disease affecting stressed crops using a fine-tuned **EfficientNetB0** deep learning model trained on the **PlantVillage** dataset — identifying **Early Blight**, **Late Blight**, or **Healthy** conditions.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SATELLITE DATA LAYER                         │
│   Google Earth Engine → Sentinel-2 (50 Days Time Series)       │
│   Bands: B2, B3, B4, B5, B8, B11 + Spectral Indices           │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                CROP HEALTH MONITORING LAYER                     │
│                                                                 │
│   ┌──────────────────────┐   ┌──────────────────────────────┐  │
│   │   K-Means Clustering │   │     Random Forest            │  │
│   │  (Unsupervised Zone  │   │  (Supervised Classification) │  │
│   │     Segmentation)    │   │                              │  │
│   └──────────────────────┘   └──────────────────────────────┘  │
│                                                                 │
│          Output: HEALTHY | MODERATE | STRESSED                  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
              ┌─────────▼──────────┐
              │  Is Crop Stressed? │
              └────────┬───────────┘
           NO ◄────────┤
           (Monitor)   │ YES
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                 DISEASE PREDICTION LAYER                        │
│                                                                 │
│   PlantVillage Dataset + EfficientNetB0 (Transfer Learning)    │
│                                                                 │
│          Output: EARLY BLIGHT | LATE BLIGHT | HEALTHY          │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- 🛰️ **Real-Time Satellite Monitoring** — Automated ingestion of 50-day Sentinel-2 time-series data from Google Earth Engine
- 📊 **Multi-Index Analysis** — Computes NDVI, EVI, NDRE, SAVI, and other vegetation indices for comprehensive health assessment
- 🤖 **Dual-Model Pipeline** — Combines unsupervised clustering + supervised classification for robust crop health detection
- 🔬 **Precision Disease Identification** — Deep learning-based disease diagnosis triggered only for stressed zones (efficient & targeted)
- 🗺️ **Spatial Mapping** — Zone-level spatial visualization of crop health and disease distribution
- ⚡ **Automated Alerts** — Flags stressed crop zones for immediate agronomic attention
- 📈 **Temporal Trends** — Tracks vegetation index changes over 50 days to detect progressive stress

---

## 🛠 Tech Stack

| Category | Technology |
|---|---|
| Satellite Data | Google Earth Engine (GEE) Python API |
| ML Framework | Scikit-Learn (Random Forest, K-Means) |
| DL Framework | TensorFlow / Keras |
| CNN Architecture | EfficientNetB0 (Transfer Learning) |
| Data Processing | NumPy, Pandas,  GeoPandas |
| Visualization | Matplotlib, Seaborn, Folium |
| Environment | Python 3.8+, Jupyter Notebook |
| Streamlit ,Streamlit-folium|

---

## 📂 Dataset

### 1. Sentinel-2 Satellite Data (Google Earth Engine)
| Parameter | Details |
|---|---|
| Satellite | ESA Sentinel-2 (Level-2A Surface Reflectance) |
| Temporal Coverage | 80 days (real-time rolling window) |
| Spatial Resolution | 10m – 20m per pixel |
| Bands Used | B2 (Blue), B3 (Green), B4 (Red), B5 (RedEdge), B8 (NIR), B11 (SWIR) |
| Derived Indices | NDVI, EVI, NDWI, SAVI |
| Cloud Masking |

### 2. PlantVillage Dataset (Disease Prediction)
| Parameter | Details |
|---|---|
| Source | PlantVillage (open-access benchmark dataset) |
| Total Images | ~540+ labeled leaf images |
| Classes Used | `Early_Blight`, `Late_Blight`, `Healthy` |
| Image Format | RGB, 256×256 pixels |
| Split | 80% Train / 20% Validation|

---

## 🔄 Pipeline

### Phase 1: Satellite Data Collection & Preprocessing

```python
# Google Earth Engine — Sentinel-2 Data Fetch
import ee
ee.Initialize()

sentinel2 = (ee.ImageCollection('COPERNICUS/S2_SR')
    .filterDate(start_date, end_date)          # 50-day window
    .filterBounds(roi)                          # Region of Interest
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    .map(mask_clouds)                           # QA60 cloud masking
    .median())                                  # Temporal composite

# Compute Vegetation Indices
ndvi = sentinel2.normalizedDifference(['B8', 'B4']).rename('NDVI')
evi  = sentinel2.expression(
    '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))',
    {'NIR': sentinel2.select('B8'), 'RED': sentinel2.select('B4'),
     'BLUE': sentinel2.select('B2')}).rename('EVI')
```

**Steps:**
- Define Region of Interest (ROI) as field polygons
- Fetch 50 days of Sentinel-2 imagery with cloud masking
- Compute a temporal median composite to reduce noise
- Extract spectral bands and calculate vegetation indices
- Export feature arrays for ML models

---

### Phase 2: Crop Health Classification

Two complementary ML approaches are used:

#### K-Means Clustering (Unsupervised — Zone Segmentation)
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(feature_array)
# Cluster 0 → Healthy | Cluster 1 → Moderate | Cluster 2 → Stressed
```
Used to segment the field into spatial zones without prior labeling, helping identify naturally distinct crop health regions.

#### Random Forest Classifier (Supervised — Health Labeling)
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf_model.fit(X_train, y_train)
# Classes: 0=Healthy | 1=Moderate | 2=Stressed
```
Trained on labeled spectral + index features to classify each pixel/zone into one of three health categories with high precision.

**Input Features:** `[
    'ndvi_max', 'ndvi_min', 'ndvi_mean', 'ndvi_std',
    'ndvi_range', 'ndvi_max_decline', 'ndvi_recovery', 'ndvi_below_thresh',
    'ndwi_mean', 'ndwi_min', 'ndwi_stress_fraction',
    'evi_mean', 'evi_max',
    'savi_mean', 'savi_max',
    'ndvi_ndwi_corr', 'dual_stress_fraction',
] `

**Output Labels:**
| Label | 
|---|---|
| 🟢 Healthy | 
| 🟡 Moderate |
| 🔴 Stressed |

---

### Phase 3: Disease Prediction

Triggered **only for Stressed zones** — an efficient design that minimizes unnecessary inference.

#### EfficientNetB0 — Transfer Learning
```python
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
output = layers.Dense(3, activation='softmax')(x)  # 3 disease classes

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

**Training Configuration:**
| Parameter | Value |
|---|---|
| Base Model | EfficientNetB0 (ImageNet weights) |
| Input Size | 224 × 224 × 3 |
| Epochs | 40 (with early stopping) |
| Batch Size | 32 |
| Optimizer | Adam (lr=1e-4) |
| Augmentation | Flip, Rotate, Zoom, Contrast , Brightness |

**Output Classes:**
| Class |
|---|---|
| 🍂 Early Blight | 
| 🟫 Late Blight |
| ✅ Healthy | 

---

## 🤖 Models Used

| Model | Type | Purpose | Input |
|---|---|---|---|
| **K-Means Clustering** | Unsupervised ML | Spatial field zone segmentation | Spectral feature vectors |
| **Random Forest** | Supervised ML | Crop health classification (Healthy / Moderate / Stressed) | Extracted features from Vegitation Indices |
| **EfficientNetB0** | Deep Learning (CNN) | Disease type prediction | RGB crop leaf images (224×224) |

---

## 📊 Results

### Crop Health Classification (Random Forest)
| Metric | Score |
|---|---|
| F1 Score | ~95% |

── Top 10 Important Features ──────────────────────────────
evi_mean      0.1652
evi_max       0.1481
ndwi_min      0.1298
ndvi_max      0.1249
savi_max      0.1150
ndvi_range    0.0672
ndvi_mean     0.0505
savi_mean     0.0505
ndvi_std      0.0426
ndvi_min      0.0319


### Disease Prediction (EfficientNetB0)
| Class | Prediction|
|---|---|
| **Overall Accuracy** | | | **~95%** |

> **Note:** Results may vary depending on region, crop type, season, and dataset size. Fine-tune thresholds and retrain on local ground truth data for best performance.



## ⚙️ Installation

### Prerequisites
- Python 3.8+
- Google Earth Engine account ([Sign up here](https://earthengine.google.com/))
- GEE authenticated on your machine

### 1. Clone the Repository
```bash
git clone https://github.com/Aashu15-tech/AI-Powered-Precision-Agriculture-Monitoring-System
cd Ai-Powered-Precision-Agriculture-Monitoring-System
```

### 2. Create Virtual Environment
```bash
conda activate agri-ai
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Authenticate Google Earth Engine
```bash
earthengine authenticate
```

### 5. Download PlantVillage Dataset
```bash
kaggle https://www.kaggle.com/datasets/nettemkrishna/plantvillage-dataset
```

---

## 🚀 Usage

### Step 1 — Configure Your Region & Dates
```
Edit `config`:
roi:
  type: "polygon"
  coordinates: [[lon1, lat1], [lon2, lat2], ...]  # Your field boundary

date_range: last 50-day from present

model_paths:
  random_forest: "models/model.pkl"
  efficientnet: "models/model.keras"
```
###step 2 - run the full pipeline for model creation of crop health and disease prediction
```bash
python main.py
```

### Step 3 — Run app in localhost
```bash
streamlit run app.py
```
---

## 🔮 Future Work

- [ ] Expand disease classes beyond Early/Late Blight (Leaf Mold, Mosaic Virus, etc.)
- [ ] Integrate weather/IoT sensor data for multi-modal analysis
- [ ] Deploy as a web dashboard using Streamlit or FastAPI
- [ ] Add support for multiple crop types (Wheat, Rice, Maize)
- [ ] Implement LSTM-based temporal stress forecasting
- [ ] Mobile app integration for field-level alerts
- [ ] Fine-tuning EfficientNetB0 with locally collected crop images

---


---

## 🙏 Acknowledgements
- [Google Earth Engine](https://earthengine.google.com/) — Cloud-based geospatial analysis platform
- [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) — Open-source crop disease image dataset
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019

---

<p align="center">
  Made with ❤️ for smarter, sustainable agriculture
</p>
