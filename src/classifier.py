import pandas as pd
import numpy as np
import os
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

def extract_last_features(df):
    return df.groupby('RecordID').last().reset_index()

def extract_tsfresh_features(df):
    # Define the settings for feature extraction
    extraction_settings = ComprehensiveFCParameters()
    
    # Extract features using tsfresh
    features = extract_features(df, column_id='RecordID', column_sort='Time', default_fc_parameters=extraction_settings)
    
    # Drop columns with NaN values
    features = features.dropna(axis=1)
    
    return features

if __name__ == '__main__':
    randomstate = 42

    logging.basicConfig(filename=os.path.join('logs', 'model_training_tsfresh.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    X = pd.read_parquet(os.path.join('loaded_data', 'a_patient_data_processed.parquet'), index=False)
    training_X = extract_tsfresh_features(X).drop(columns=['In-hospital_death', 'Time', 'RecordID'])
    
    training_Y = extract_tsfresh_features(X)['In-hospital_death']

    X_val = pd.read_parquet(os.path.join('loaded_data', 'b_patient_data_processed.parquet'), index=False)
    
    validation_X = extract_tsfresh_features(X_val).drop(columns=['In-hospital_death', 'Time', 'RecordID'])
    logging.info(f'Validation X columns: {validation_X.columns}')
    validation_Y = extract_tsfresh_features(X_val)['In-hospital_death']

    logging.info(f"Training X shape: {training_X.shape}")
    logging.info(f"Training Y shape: {training_Y.shape}")
    logging.info(f"Validation X shape: {validation_X.shape}")
    logging.info(f"Validation Y shape: {validation_Y.shape}")

    # Hyperparameter tuning for Logistic Regression
    log_reg = LogisticRegression(random_state=randomstate)
    log_reg_params = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']}
    log_reg_grid = GridSearchCV(log_reg, log_reg_params, cv=5, scoring='accuracy')
    log_reg_grid.fit(training_X, training_Y)
    log_reg_best = log_reg_grid.best_estimator_
    log_reg_predictions = log_reg_best.predict(validation_X)
    logging.info(f"Best Logistic Regression Params: {log_reg_grid.best_params_}")
    logging.info(f"Logistic Regression Accuracy: {accuracy_score(validation_Y, log_reg_predictions)}")
    logging.info(f"Logistic Regression Classification Report:\n{classification_report(validation_Y, log_reg_predictions)}")

    # Hyperparameter tuning for Random Forest
    rf = RandomForestClassifier(random_state=randomstate)
    rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy')
    rf_grid.fit(training_X, training_Y)
    rf_best = rf_grid.best_estimator_
    rf_predictions = rf_best.predict(validation_X)
    logging.info(f"Best Random Forest Params: {rf_grid.best_params_}")
    logging.info(f"Random Forest Accuracy: {accuracy_score(validation_Y, rf_predictions)}")
    logging.info(f"Random Forest Classification Report:\n{classification_report(validation_Y, rf_predictions)}")
