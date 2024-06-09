import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from src.data_preprocessing import main as preprocessing_main

def feature_engineering(X_train, X_val, test_data, y_train):
    try:
        logging.info("Starting feature engineering")

        # Log transformation
        log_transform_cols = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
        for col in log_transform_cols:
            X_train[col] = np.log1p(X_train[col])
            X_val[col] = np.log1p(X_val[col])
            test_data[col] = np.log1p(test_data[col])
        y_train_log = np.log1p(y_train)
        
        # One-hot encoding
        encoder = OneHotEncoder(drop='first')  # drop the first category to avoid multicollinearity
        encoded_sex_train = encoder.fit_transform(X_train[['Sex']]).toarray()
        encoded_sex_val = encoder.transform(X_val[['Sex']]).toarray()
        encoded_sex_test = encoder.transform(test_data[['Sex']]).toarray()

        encoded_sex_train_df = pd.DataFrame(encoded_sex_train, columns=encoder.get_feature_names_out(['Sex']))
        encoded_sex_val_df = pd.DataFrame(encoded_sex_val, columns=encoder.get_feature_names_out(['Sex']))
        encoded_sex_test_df = pd.DataFrame(encoded_sex_test, columns=encoder.get_feature_names_out(['Sex']))

        # Drop the original 'Sex' column and concatenate the encoded columns
        X_train = X_train.drop(columns=['Sex']).reset_index(drop=True)
        X_val = X_val.drop(columns=['Sex']).reset_index(drop=True)
        test_data = test_data.drop(columns=['Sex']).reset_index(drop=True)

        X_train = pd.concat([X_train, encoded_sex_train_df], axis=1)
        X_val = pd.concat([X_val, encoded_sex_val_df], axis=1)
        test_data = pd.concat([test_data, encoded_sex_test_df], axis=1)

        # Feature creation
        X_train['Length_Diameter'] = X_train['Length'] * X_train['Diameter']
        X_train['Whole_Shucked'] = X_train['Whole_weight'] * X_train['Shucked_weight']
        X_val['Length_Diameter'] = X_val['Length'] * X_val['Diameter']
        X_val['Whole_Shucked'] = X_val['Whole_weight'] * X_val['Shucked_weight']
        test_data['Length_Diameter'] = test_data['Length'] * test_data['Diameter']
        test_data['Whole_Shucked'] = test_data['Whole_weight'] * test_data['Shucked_weight']

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        test_scaled = scaler.transform(test_data)

        logging.info("Feature engineering completed successfully")
        return X_train_scaled, y_train_log, X_val_scaled, test_scaled
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        raise

def main():
    X_train, X_val, y_train, y_val, test_data = preprocessing_main()

    X_train_scaled, y_train_log, X_val_scaled, test_scaled = feature_engineering(X_train, X_val, test_data, y_train)

    return X_train_scaled, y_train_log, X_val_scaled, test_scaled, y_val

if __name__ == "__main__":
    main()
