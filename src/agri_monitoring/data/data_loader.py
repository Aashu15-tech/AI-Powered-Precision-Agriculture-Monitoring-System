import pandas as pd
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

def load_data(file_path: Path)->pd.DataFrame:
    """
    Load dataset from CSV file
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)

        logger.info(f"Data loaded successfully with shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise