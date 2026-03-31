import os
from pathlib import Path
import logging

# -------------------------------
# Logging Configuration
# -------------------------------
BASE_DIR = os.getcwd()  # current project root
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "running_logs.log")

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.INFO,
    format='[%(asctime)s]: %(levelname)s - %(message)s'
)

logging.info("Logging is working!")
print("Log file location:", LOG_FILE)
# -------------------------------
# Project Name
# -------------------------------
project_name = "agri_monitoring"

# -------------------------------
# File Structure (Customized)
# -------------------------------
list_of_files = [

    # Git
    ".github/workflows/.gitkeep",

    # Core Source Code
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/data/__init__.py",
    f"src/{project_name}/features/__init__.py",
    f"src/{project_name}/models/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/visualization/__init__.py",

    # Core Modules
    f"src/{project_name}/data/data_loader.py",
    f"src/{project_name}/features/feature_engineering.py",
    f"src/{project_name}/features/labeling.py",
    f"src/{project_name}/models/train_model.py",
    f"src/{project_name}/models/predict.py",
    f"src/{project_name}/pipeline/data_pipeline.py",
    f"src/{project_name}/visualization/visualize.py",

    # Config
    "config/config.yaml",

    # Data folders (tracked with gitkeep)
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",

    # Models & Reports
    "models/.gitkeep",
    "reports/.gitkeep",

    # Notebook
    "notebooks/experiments.ipynb",

    # Project Files
    "requirements.txt",
    "README.md",
    ".gitignore"
]

# -------------------------------
# Create Structure
# -------------------------------
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")