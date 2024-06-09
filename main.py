import numpy as np
import logging
from src.config_loader import load_config
from src.data_preparation import load_data
from src.data_preprocessing import clean_data, detect_outliers, check_skewness, visualize_outliers, apply_winsorization, data_splitting, data_transformation
from src.feature_engineering import feature_engineering
from src.model_training import train_model, hyperparameter_tuning, save_model
from src.evaluation import evaluate_model
from src.logging_config import setup_logging

def main():
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Set up logging
    logger = setup_logging(config['logging']['log_path'], config['logging']['level'])
    logger.info("Starting the main script")
    
    # Load data
    logger.info("Loading data")
    train_data, test_data = load_data(config['data']['train_path'], config['data']['test_path'])
    
    # Data cleaning
    logger.info("Cleaning data")
    train_data = clean_data(train_data)
    test_data = clean_data(test_data)
    
    # Detect and visualize outliers
    logger.info("Detecting outliers")
    detect_outliers(train_data)
    logger.info("Visualizing outliers")
    visualize_outliers(train_data)
    
    # Check skewness
    logger.info("Checking skewness")
    check_skewness(train_data)
    
    # Apply Winsorization
    logger.info("Applying Winsorization")
    train_data = apply_winsorization(train_data)
    test_data = apply_winsorization(test_data)
    
    # Data splitting
    logger.info("Splitting data")
    X_train, X_val, y_train, y_val = data_splitting(train_data, config['data']['target'])
    
    # Data transformation
    logger.info("Data transformation")
    data_transformation(X_train, y_train)
    
    # Feature engineering
    logger.info("Feature engineering")
    X_train_scaled, y_train_log, X_val_scaled, test_scaled = feature_engineering(X_train, X_val, test_data, y_train)
    
    # Train baseline model
    logger.info("Training baseline model")
    baseline_model = train_model(X_train_scaled, y_train_log)
    save_model(baseline_model, config['model']['baseline_model'])
    
    # Hyperparameter tuning to get the best model
    logger.info("Starting hyperparameter tuning")
    best_model = hyperparameter_tuning(X_train_scaled, y_train_log)
    save_model(best_model, config['model']['final_model'])
    
    # Evaluate baseline model
    logger.info("Evaluating baseline model")
    evaluate_model(baseline_model, X_val_scaled, y_val, "baseline validation")
    
    # Evaluate best model
    logger.info("Evaluating best model")
    val_rmsle, val_r2 = evaluate_model(best_model, X_val_scaled, y_val)
    logger.info(f"Best model validation RMSLE: {val_rmsle}")
    logger.info(f"Best model validation R^2: {val_r2}")
    
    # Final predictions on test data
    logger.info("Generating final predictions on test data")
    test_pred_log = best_model.predict(test_scaled)
    test_pred = np.expm1(test_pred_log)
    logger.info("Final predictions generated")
    
    logger.info("Main script completed")

if __name__ == "__main__":
    main()
