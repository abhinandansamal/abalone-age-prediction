from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import logging
from src.feature_engineering import main as feature_engineering_main
from src.config_loader import load_config

def train_model(X, y):
    try:
        logging.info("Starting baseline model training")
        # Initialize the baseline GradientBoostingRegressor with default hyperparameters
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X, y)
        logging.info("Baseline model training completed successfully")
        return model
    except Exception as e:
        logging.error(f"Error during baseline model training: {e}")
        raise

def hyperparameter_tuning(X, y):
    try:
        logging.info("Starting hyperparameter tuning")
        gbr = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_log_error')
        grid_search.fit(X, y)
        logging.info(f"Hyperparameter tuning completed. Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    except Exception as e:
        logging.error(f"Error during hyperparameter tuning: {e}")
        raise

def save_model(model, output_path):
    try:
        logging.info(f"Saving model to {output_path}")
        joblib.dump(model, output_path)
        logging.info("Model saved successfully")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def main():
    X_train_scaled, y_train_log, X_val_scaled, test_scaled, y_val = feature_engineering_main()

    # Train baseline model
    baseline_model = train_model(X_train_scaled, y_train_log)
    config = load_config('config/config.yaml')
    save_model(baseline_model, config['model']['baseline_model'])

    # Hyperparameter tuning to get the best model
    best_model = hyperparameter_tuning(X_train_scaled, y_train_log)
    save_model(best_model, config['model']['final_model'])

    return best_model, X_val_scaled, y_val, test_scaled

if __name__ == "__main__":
    main()
