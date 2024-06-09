# Abalone Age Prediction

## Description

This project aims to predict the age of abalone from various physical measurements. The goal is to use machine learning techniques to forecast the age accurately.

## Evaluation

The model's performance is evaluated using the Root Mean Squared Logarithmic Error (RMSLE).

## Dataset Description

• **train.csv:** The training dataset; Rings is the integer target.

• **test.csv:** The test dataset; your objective is to predict the value of Rings for each row.

## Usage

### Configuration and Logging
The script starts by loading the configuration from config/config.yaml and setting up logging.

### Data Loading
The data preprocessing steps are defined in the src/data_preparation.py module. It loads the training and testing data.

### Data Preprocessing
The data preprocessing steps are defined in the src/data_preprocessing.py module. The script cleans the data, detects and visualizes outliers, checks skewness, applies Winsorization, and splits the data into training and validation sets.

### Feature Engineering
Feature engineering steps are implemented in the src/feature_engineering.py module. It performs feature engineering on the training, validation, and test datasets.

### Model Training
The model training script is in the src/model_training.py module. The baseline model is trained and saved. Hyperparameter tuning is performed to find the best model, which is then saved.

### Model Evaluation
Model evaluation is performed in the src/evaluation.py module. Both the baseline and best models are evaluated on the validation set.

### Final Predictions:
The best model is used to generate predictions on the test dataset.


## Example Jupyter Notebook

Two Jupyter Notebooks are provided in the `notebooks/` directory (`abalone_gbr_model.ipynb` & `abalone_xgbreg_model.ipynb`). These notebook demonstrates the entire workflow from data loading, preprocessing, feature engineering, model training, and evaluation for two different models (`xgboost Regressor` & `Gradient Boosting Regressor`). Out of these two models, `Gradient Boosting Regressor` proformed better after hyperparameter tuning.


## Project Structure

```bash
abalone-age-prediction/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── logs/
│   └── app.log
│
├── models/
│   ├── baseline_model_abalone_gbr_model.joblib
│   └── abalone_gbr_model.joblib
│
├── notebooks/
│   ├── abalone_xgbreg_model.ipynb
│   └── abalone_gbr_model.ipynb
│
├── src/abalone-age-prediction/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── exploratory_data_analysis.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── config_loader.py
│   └── logging_config.py
│
├── submissions/
│
├── .gitignore
├── Dockerfile
├── environment.yml
├── LICENSE
├── README.md
├── main.py
├── requirements.txt
├── template.py
└── setup.py
```

• `data/`: Contains the raw input files (train data, test data).

• `models/`: Contains saved model files.

• `notebooks/`: Contains Jupyter notebooks for exploratory data analysis and model development.

• `src/`: Contains the source code for data preparation, feature engineering, model training, and evaluation.

## Setup Instructions

### Prerequisites

• Python 3.10.12

• Anaconda installed

• Git installed


### Setup Using Conda

1. Clone the Repository

```bash
git clone https://github.com/abhinandansamal/abalone-age-prediction.git
cd abalone-age-prediction
```

2. Create the Conda Environment

```bash
conda env create -f environment.yml
```

3. Activate the Environment

```bash
conda activate abalone-age-prediction
```

4. Run the Project

```bash
python main.py
```

## Notable Enhancements

### Model Training and Evaluation
• **Model:** Implemented Gradient Boosting Regressor model for prediction.

• **Error Metrics:** Calculated R-Squared and RMSLE to evaluate model performance.

• **Visualizations:** Generated visualizations to show histograms for numerical features, pair plots to visualize relationships between features, target variable distribution, heatmap for the correlation matrix, outliers using box plots for each numerical column, transformed features using box plots & data distribution using histograms and Q-Q plots for numerical columns.


### Future Work
• **Feature Integration:** Plan to create new features.

• **Algorithm Experimentation:** Explore different machine learning algorithms and ensemble methods to improve accuracy.

• **Hyperparameter Tuning:** Perform hyperparameter tuning to optimize model performance.


## Key Findings
• **Model Accuracy:** The model achieved an RMSLE of 0.1425 on the validation set.


## Conclusion

This notebook demonstrates the process of predicting the age of abalone from various physical measurements.

## Acknowledgments

• This project is based on data available on [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e4/data).

• Special thanks to the Python community for their support and contributions to the libraries used in this project.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/abhinandansamal/abalone-age-prediction/blob/main/LICENSE) file for details.