import numpy as np
from sklearn.metrics import mean_squared_log_error, r2_score
import logging
import joblib
from src.model_training import main as model_training_main
from src.config_loader import load_config


def evaluate_model(model, X, y, dataset_type="validation"):
    try:
        logging.info(f"Evaluating model on {dataset_type} set")
        y_pred_log = model.predict(X)
        y_pred = np.expm1(y_pred_log)
        rmsle = np.sqrt(mean_squared_log_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        logging.info(f"{dataset_type.capitalize()} RMSLE: {rmsle}")
        logging.info(f"{dataset_type.capitalize()} R^2: {r2}")
        return rmsle, r2
    except Exception as e:
        logging.error(f"Error during model evaluation on {dataset_type} set: {e}")
        raise

def main():
    config = load_config('config/config.yaml')

    # Load the models
    baseline_model = joblib.load(config['model']['baseline_model'])
    best_model, X_val_scaled, y_val, test_scaled = model_training_main()

    # Evaluate baseline model
    evaluate_model(baseline_model, X_val_scaled, y_val, "baseline validation")

    # Evaluate best model
    val_rmsle, val_r2 = evaluate_model(best_model, X_val_scaled, y_val)

    logging.info(f"Best model validation RMSLE: {val_rmsle}")
    logging.info(f"Best model validation R^2: {val_r2}")

    # Final predictions on test data
    logging.info("Generating final predictions on test data")
    test_pred_log = best_model.predict(test_scaled)
    test_pred = np.expm1(test_pred_log)
    logging.info("Final predictions generated")

    return val_rmsle, val_r2, test_pred

if __name__ == "__main__":
    main()
