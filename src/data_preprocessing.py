import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import mstats
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split
from src.config_loader import load_config
from src.data_preparation import load_data

def clean_data(data):
    try:
        logging.info("Cleaning data: renaming columns and dropping 'id' column")
        data = data.rename({
            "Whole weight": "Whole_weight",
            "Whole weight.1": "Shucked_weight",
            "Whole weight.2": "Viscera_weight",
            "Shell weight": "Shell_weight"
        }, axis=1)
        data = data.drop(columns=["id"], axis=1)
        return data
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise

def detect_outliers(data):
    try:
        logging.info("Detecting outliers")
        numerical_cols = data.select_dtypes(include=[np.number]).columns

        # calculate Q1 (25th percentile) and Q3 (75th percentile) for each numerical column
        Q1 = data[numerical_cols].quantile(0.25)
        Q3 = data[numerical_cols].quantile(0.75)
        IQR = Q3 - Q1

        # define outliers as points outside the range [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        outliers = ((data[numerical_cols] < (Q1 - 1.5 * IQR)) | (data[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        outliers_count = outliers.sum()
        logging.info(f"Number of detected outliers using the IQR method: {outliers_count}")
        return outliers_count
    except Exception as e:
        logging.error(f"Error detecting outliers: {e}")
        raise

def check_skewness(data):
    try:
        logging.info("Checking skewness of numerical columns")
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        skewness = data[numerical_cols].skew()
        logging.info(f"Skewness of numerical columns: \n{skewness}")
        return skewness
    except Exception as e:
        logging.error(f"Error checking skewness: {e}")
        raise

def visualize_outliers(data):
    try:
        logging.info("Visualizing outliers using box plots")
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numerical_cols):
            plt.subplot(3, 3, i+1)
            sns.boxplot(x=data[col])
            plt.title(f'Box plot of {col}')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error visualizing outliers: {e}")
        raise

def apply_winsorization(data):
    try:
        logging.info("Applying Winsorization to numerical columns")
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            data[col] = mstats.winsorize(data[col], limits=[0.01, 0.01])
        logging.info("Winsorization applied successfully")
        return data
    except Exception as e:
        logging.error(f"Error applying Winsorization: {e}")
        raise

def data_splitting(train_data, target):
    try:
        logging.info("Splitting data into training and validation sets")
        X = train_data.drop(columns=[target])
        y = train_data[target]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        logging.info("Data splitting completed successfully")
        return X_train, X_val, y_train, y_val
    except Exception as e:
        logging.error(f"Error during data splitting: {e}")
        raise

def data_transformation(X_train, y_train):
    try:
        logging.info("Checking the data distribution visually")
        num_cols = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
        
        # Plot histogram and Q-Q plot for the target variable
        plt.figure(figsize=(15, 20))
        num_plots = len(num_cols) + 1  # Additional plot for the target variable
        for i, col in enumerate(num_cols):
            plt.subplot(num_plots, 2, 2*i+1)
            sns.histplot(X_train[col], kde=True)
            plt.title(f'Histogram of {col}')
            plt.subplot(num_plots, 2, 2*i+2)
            stats.probplot(X_train[col], dist='norm', plot=plt)
            plt.title(f'Q-Q Plot of {col}')
        plt.subplot(num_plots, 2, 2*num_plots-1)
        sns.histplot(y_train, kde=True)
        plt.title(f'Histogram of Rings')
        plt.subplot(num_plots, 2, 2*num_plots)
        stats.probplot(y_train, dist='norm', plot=plt)
        plt.title(f'Q-Q Plot of Rings')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error during data transformation: {e}")
        raise

def main():
    config = load_config('config/config.yaml')
    train_data, test_data = load_data(config['data']['train_path'], config['data']['test_path'])
    
    train_data = clean_data(train_data)
    test_data = clean_data(test_data)
    
    detect_outliers(train_data)
    check_skewness(train_data)
    visualize_outliers(train_data)
    
    train_data = apply_winsorization(train_data)
    test_data = apply_winsorization(test_data)
    
    X_train, X_val, y_train, y_val = data_splitting(train_data, config['data']['target'])
    data_transformation(X_train, y_train)

    return X_train, X_val, y_train, y_val, test_data

if __name__ == "__main__":
    main()
