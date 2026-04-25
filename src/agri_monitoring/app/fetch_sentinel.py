"""
fetch_sentinel.py
-----------------
Google Earth Engine se Firozpur farm ka Sentinel-2 data fetch karta hai.
Last 50 days ka median composite banata hai aur 10 sample points ke liye
NDVI, NDWI, EVI, SAVI indices extract karta hai.
"""

import ee
import json
import pandas as pd
from datetime import datetime, timedelta


# ─── Firozpur farm bounding box ───────────────────────────────────────────────
# Firozpur district, Punjab — ek typical wheat/paddy farm plot
FARM_BBOX = {
    "lon_min": 74.600,
    "lat_min": 30.920,
    "lon_max": 74.625,
    "lat_max": 30.945,
}

# 10 evenly distributed sample points inside the bounding box
SAMPLE_POINTS = [
    {"id": f"P{i+1}", "lat": lat, "lon": lon}
    for i, (lat, lon) in enumerate([
        (30.922, 74.602),
        (30.925, 74.608),
        (30.928, 74.614),
        (30.931, 74.620),
        (30.934, 74.603),
        (30.937, 74.609),
        (30.940, 74.615),
        (30.935, 74.621),
        (30.929, 74.606),
        (30.942, 74.611),
    ])
]


def authenticate_gee():
    """GEE authentication — browser login."""
    try:
        ee.Initialize(project='your-gee-project-id')  # Replace with your project ID
        print("✅ GEE already initialized.")
    except Exception:
        print("🔐 GEE authentication required...")
        ee.Authenticate()
        ee.Initialize(project='your-gee-project-id')  # Replace with your project ID
        print("✅ GEE authenticated and initialized.")


def get_date_range(days_back: int = 50):
    """Last N days ka date range return karta hai."""
    end = datetime.utcnow()
    start = end - timedelta(days=days_back)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def compute_indices(image):
    """
    Sentinel-2 image se vegetation indices compute karta hai.
    B4=Red, B8=NIR, B3=Green, B11=SWIR1
    """
    red  = image.select("B4")
    nir  = image.select("B8")
    green = image.select("B3")
    swir = image.select("B11")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
    evi  = nir.subtract(red).divide(
        nir.add(red.multiply(6)).subtract(swir.multiply(7.5)).add(1)
    ).multiply(2.5).rename("EVI")
    savi = nir.subtract(red).divide(
        nir.add(red).add(0.5)
    ).multiply(1.5).rename("SAVI")

    return image.addBands([ndvi, ndwi, evi, savi])


def fetch_sentinel2_data(days_back: int = 50, cloud_cover: int = 20) -> pd.DataFrame:
    """
    Main function: GEE se Sentinel-2 data fetch karke
    10 sample points ke indices return karta hai.

    Args:
        days_back:    Kitne days peeche tak ka data chahiye (default 50)
        cloud_cover:  Max cloud cover % (default 20)

    Returns:
        DataFrame with columns: point_id, lat, lon, NDVI, NDWI, EVI, SAVI
    """
    authenticate_gee()

    start_date, end_date = get_date_range(days_back)
    print(f"📅 Date range: {start_date} → {end_date}")

    # Farm bounding box as GEE geometry
    farm_region = ee.Geometry.BBox(
        FARM_BBOX["lon_min"],
        FARM_BBOX["lat_min"],
        FARM_BBOX["lon_max"],
        FARM_BBOX["lat_max"],
    )

    # Sentinel-2 Surface Reflectance collection
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(farm_region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_cover))
        .map(compute_indices)
    )

    count = collection.size().getInfo()
    print(f"🛰️  {count} Sentinel-2 scenes mili (cloud < {cloud_cover}%)")

    if count == 0:
        raise ValueError(
            f"❌ Koi scene nahi mili! Cloud cover badhaao ya dates change karo.\n"
            f"   Tried: {start_date} → {end_date}, cloud < {cloud_cover}%"
        )

    # Median composite (cloud noise kam karta hai)
    composite = collection.median()

    # Har sample point ke liye indices extract karo
    records = []
    for pt in SAMPLE_POINTS:
        point = ee.Geometry.Point([pt["lon"], pt["lat"]])
        vals = composite.select(["NDVI", "NDWI", "EVI", "SAVI"]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10,
            maxPixels=1e9,
        ).getInfo()

        records.append({
            "point_id": pt["id"],
            "lat":      pt["lat"],
            "lon":      pt["lon"],
            "NDVI":     round(vals.get("NDVI", 0) or 0, 4),
            "NDWI":     round(vals.get("NDWI", 0) or 0, 4),
            "EVI":      round(vals.get("EVI",  0) or 0, 4),
            "SAVI":     round(vals.get("SAVI", 0) or 0, 4),
        })
        print(f"   ✓ {pt['id']} → NDVI={records[-1]['NDVI']:.3f}")

    df = pd.DataFrame(records)
    df.to_csv("sentinel2_features.csv", index=False)
    print(f"\n💾 Data saved: sentinel2_features.csv ({len(df)} points)")
    return df


if __name__ == "__main__":
    df = fetch_sentinel2_data(days_back=50)
    print("\n📊 Fetched Data:")
    print(df.to_string(index=False))