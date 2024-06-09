import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from src.data_preparation import load_data
from src.config_loader import load_config

def eda(data):
    try:
        logging.info("Starting EDA")

        logging.info("Displaying first few rows of the dataset")
        print(data.head())

        logging.info("Displaying dataset information")
        print(data.info())

        logging.info("Displaying descriptive statistics")
        print(data.describe())

        logging.info("Plotting histograms of numerical features")
        data.hist(bins=20, figsize=(15, 10))
        plt.suptitle('Histograms of Numerical Features')
        plt.show()

        logging.info("Plotting pair plots to visualize relationships between features")
        sns.pairplot(data.sample(1000), diag_kind='kde')
        plt.suptitle('Pair Plots of Features (Sample)')
        plt.show()

        logging.info("Plotting histogram of Rings")
        sns.histplot(data['Rings'], kde=True)
        plt.title('Histogram of Rings')
        plt.show()

        logging.info("Calculating and plotting Heatmap for the correlation matrix")
        corr_matrix = data[data.select_dtypes(include=[np.number]).columns].corr()
        print("Correlation Matrix:", corr_matrix)
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()

        logging.info("EDA completed successfully")
    except Exception as e:
        logging.error(f"Error during EDA: {e}")
        raise

def main():
    config = load_config('config/config.yaml')
    train_data, _ = load_data(config['data']['train_path'], config['data']['test_path'])
    eda(train_data)

if __name__ == "__main__":
    main()
