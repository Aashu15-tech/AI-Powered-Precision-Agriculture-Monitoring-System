from pathlib import Path
import logging

from src.agri_monitoring.pipeline.data_pipeline import run_pipeline
from src.agri_monitoring.models.train_model import train_model
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s'
)

if __name__ == "__main__":

    INPUT_PATH = Path("data/raw/Data.csv")
    OUTPUT_PATH = Path("data/processed/labeled_dataset.csv")

    run_pipeline(INPUT_PATH, OUTPUT_PATH)
    train_model(OUTPUT_PATH)