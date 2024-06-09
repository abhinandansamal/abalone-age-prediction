import pandas as pd
import logging
from src.config_loader import load_config

def load_data(train_path, test_path):
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logging.info(f"Data loaded successfully: {train_path}, {test_path}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def main():
    config = load_config('config/config.yaml')
    train_data, test_data = load_data(config['data']['train_path'], config['data']['test_path'])
    return train_data, test_data

if __name__ == "__main__":
    main()
