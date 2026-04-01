import requests
import pandas as pd


def get_weather_data(lat, lon, start_date, end_date):
    """
    Fetch weather data from NASA POWER API
    """

    url = f"https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": "T2M,RH2M,PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }

    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data["properties"]["parameter"])

    df = df.reset_index().rename(columns={"index": "date"})

    # Rename columns
    df = df.rename(columns={
        "T2M": "temperature",
        "RH2M": "humidity",
        "PRECTOTCORR": "precipitation"
    })

    return df