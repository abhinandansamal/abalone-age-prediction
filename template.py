import os
from pathlib import Path
import logging

# logging string
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = 'abalone-age-prediction'

list_of_files = [
    ".github/workflows/.gitkeep",
    "data/",
    "logs/"
    "notebooks/",
    "models/",
    "src/__init__.py",
    "src/data_preparation.py",
    "src/exploratory_data_analysis.py",
    "src/data_preprocessing.py",
    "src/feature_engineering.py",
    "src/model_training.py",
    "src/evaluation.py",
    "src/config_loader.py",
    "src/logging_config.py",
    "config/config.yaml",
    "environment.yml",
    "Dockerfile",
    "main.py",
    "requirements.txt",
    "setup.py",

]

for filepath in list_of_files:
    # Convert to Path object
    path = Path(filepath)
    
    # If path ends with a '/', treat it as a directory
    if filepath.endswith('/'):
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Creating directory: {filepath}")
        else:
            logging.info(f"Directory {filepath} already exists.")
    else:
        # Ensure parent directory exists
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Creating directory: {path.parent}")

        # Create the file if it doesn't exist or is empty
        if not path.exists() or path.stat().st_size == 0:
            path.touch()
            logging.info(f"Creating empty file: {filepath}")
        else:
            logging.info(f"File {filepath} already exists or is not empty.")

logging.info("Project structure created successfully.")