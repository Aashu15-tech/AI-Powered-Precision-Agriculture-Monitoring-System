import ee
import geemap
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config.config import CONFIG, FIROZPUR_PLOT, FETCH_DAYS, CLOUD_COVER

import ee
import pandas as pd
from datetime import datetime, timedelta
import os

# ── AUTH ────────────────────────────────────────────────
os.environ["EARTHENGINE_CREDENTIALS"] = "/Users/aashuanand/earthengine_token.json"

try:
    ee.Initialize(project='extractearthengine')
except:
    ee.Authenticate(auth_mode='localhost')
    ee.Initialize(project='extractearthengine')

# ── DATE RANGE (50 DAYS) ────────────────────────────────
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=FETCH_DAYS)

start_str = start_date.strftime('%Y-%m-%d')
end_str   = end_date.strftime('%Y-%m-%d')

# ── AOI (FIROZPUR) ──────────────────────────────────────
FIROZPUR_PLOT = {
    "center_lat": 30.933,
    "center_lon": 74.622,
    "buffer_m": 5000
}

center = ee.Geometry.Point([
    FIROZPUR_PLOT['center_lon'],
    FIROZPUR_PLOT['center_lat']
])

aoi = center.buffer(FIROZPUR_PLOT['buffer_m'])

# ── SENTINEL-2 ──────────────────────────────────────────
s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start_str, end_str)
        .filterBounds(aoi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']))

# ── ADD INDICES + DATE ──────────────────────────────────
def add_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')

    evi = image.expression(
        '2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))', {
            'NIR': image.select('B8'),
            'RED': image.select('B4'),
            'BLUE': image.select('B2')
        }).rename('EVI')

    savi = image.expression(
        '1.5*((NIR-RED)/(NIR+RED+0.5))', {
            'NIR': image.select('B8'),
            'RED': image.select('B4')
        }).rename('SAVI')

    date = ee.Date(image.date()).format('YYYY-MM-dd')

    return image.addBands([ndvi, ndwi, evi, savi]).set('date', date)

s2 = s2.map(add_indices)

# ── 20 RANDOM POINTS ────────────────────────────────────
points = ee.FeatureCollection.randomPoints(
    region=aoi,
    points=20,
    seed=42
)

# ── SAMPLE EACH IMAGE ───────────────────────────────────
def sample_image(image):
    sampled = image.sampleRegions(
        collection=points,
        scale=100,
        geometries=True   # ✅ needed for .geo
    )
    return sampled.map(lambda f: f.set('date', image.get('date')))

samples = s2.map(sample_image).flatten()

# ── ADD LAT/LON ─────────────────────────────────────────
def add_lat_lon(feature):
    coords = feature.geometry().coordinates()
    return feature.set({
        'lon': coords.get(0),
        'lat': coords.get(1)
    })

samples = samples.map(add_lat_lon)

# # ── EXPORT CSV ──────────────────────────────────────────
# task = ee.batch.Export.table.toDrive(
#     collection=samples,
#     description='Firozpur_full_features',
#     folder='AgriZsquad',
#     fileNamePrefix='S2_full_dataset',
#     fileFormat='CSV'
# )

# task.start()


import geemap
import os

# ── 9. Convert samples to pandas dataframe ───────────────

df = geemap.ee_to_df(samples)

# ── 10. Save to local data folder ────────────────────────
os.makedirs("data", exist_ok=True)


def fetch_data(file_path: str="data/sentinel2_data.csv")->pd.DataFrame:

    file_path = "data/sentinel2_data.csv"

    df.to_csv(file_path, index=False)

    print(f"✅ Data saved locally at: {file_path}")

    return df






# import gdown
# url="https://drive.google.com/file/d/1Im_i92y-Z0N5-REwT8-Jxqx5R_6pTnb-/view?usp=drive_link"

# import zipfile 

# file_id=url.split("/")[-2]
# print(file_id)


# def fetch_firozpur_data() -> pd.DataFrame:
#     os.makedirs("data", exist_ok=True)

#     zip_path = "data/new_data.zip"

#     # 🧹 Clean old files
#     if os.path.exists(zip_path):
#         os.remove(zip_path)

#     # ── 1. Download ─────────────────────────────
#     url = f"https://drive.google.com/uc?id={file_id}"
#     gdown.download(url, zip_path, quiet=False)

#     # ── 2. Extract ──────────────────────────────
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall("data/")
#         extracted_files = zip_ref.namelist()

#     # ── 3. Find CSV file dynamically ───────────
#     csv_file = None
#     for file in extracted_files:
#         if file.endswith(".csv"):
#             csv_file = os.path.join("data", file)
#             break

#     if csv_file is None:
#         raise Exception("❌ No CSV file found in ZIP")

#     # ── 4. Load DataFrame ──────────────────────
#     df = pd.read_csv(csv_file)

#     print(f"✅ Data loaded from → {csv_file}")

#     return df


