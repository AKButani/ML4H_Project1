import pandas as pd
import numpy as np
import os
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score

from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters

def extract_last_features(df):
    return df.groupby('RecordID').last().reset_index()

def extract_tsfresh_features(df):
    # Define the settings for feature extraction
    extraction_settings = EfficientFCParameters()
    
    # Extract features using tsfresh
    features = extract_features(df, column_id='RecordID', column_sort='Time', default_fc_parameters=extraction_settings)
    
    # Drop columns with NaN values
    features = features.dropna(axis=1)
    
    return features

def logistic_regression(training_X, training_Y, validation_X, validation_Y, randomstate=42 ) -> LogisticRegression:
    C_values = [0.01, 0.1, 1, 10, 100]
    penalties = ['l1', 'l2']

    best_accuracy = 0.0
    best_params = {}
    best_model = None

    # Loop through each combination of hyperparameters
    for C in C_values:
        for penalty in penalties:
            # l1 penalty is only supported by some solvers; here we use 'liblinear'

            model = LogisticRegression(
                C=C, 
                penalty=penalty,
                solver='liblinear',
                max_iter=100,
                random_state=randomstate
            )
            model.fit(training_X, training_Y)
            
            # Predict on the validation set
            predictions = model.predict(validation_X)
            accuracy = accuracy_score(validation_Y, predictions)
            #print(f"Validation accuracy for C={C}, penalty='{penalty}': {accuracy:.4f}")
            
            # Save best model based on validation accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'C': C, 'penalty': penalty}
                best_model = model
    return best_model

def random_forest(training_X, training_Y, validation_X, validation_Y, randomstate=42) -> RandomForestClassifier:
    # Define a grid of hyperparameters to search over
    n_estimators_values = [50, 100, 200]
    max_depth_values = [None, 5, 10]

    best_accuracy = 0.0
    best_params = {}
    best_model = None

    # Loop through each combination of hyperparameters
    for n_estimators in n_estimators_values:
        for max_depth in max_depth_values:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=randomstate
            )
            model.fit(training_X, training_Y)
            
            # Predict on the validation set
            predictions = model.predict(validation_X)
            accuracy = accuracy_score(validation_Y, predictions)
            print(f"Validation accuracy for n_estimators={n_estimators}, max_depth={max_depth}: {accuracy:.4f}")
            
            # Save best model based on validation accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
                best_model = model
    
    return best_model

def get_results(model, test_X, test_Y):
    # Predict on the test set
    test_predictions = model.predict(test_X)
    logging.info(f"Test Accuracy: {accuracy_score(test_Y, test_predictions)}")
    # Calculate AUROC
    auroc = roc_auc_score(test_Y, test_predictions)
    logging.info(f"Test AUROC: {auroc}")
    # Calculate AUPRC
    auprc = average_precision_score(test_Y, test_predictions)
    logging.info(f"Test AUPRC: {auprc}")

if __name__ == '__main__':
    randomstate = 42

    logging.basicConfig(filename=os.path.join('logs', f'model_training_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # choose feature extraction method
    feature_extractor = extract_tsfresh_features 
    logging.info(f"Using feature extractor: {feature_extractor.__name__}")

    set_a = pd.read_parquet(os.path.join('loaded_data', 'a_patient_data_processed.parquet'))
    training_X = feature_extractor(set_a).drop(columns=['In-hospital_death', 'Time', 'RecordID'])
    training_Y = feature_extractor(set_a)['In-hospital_death']

    set_b = pd.read_parquet(os.path.join('loaded_data', 'b_patient_data_processed.parquet'))
    validation_X = feature_extractor(set_b).drop(columns=['In-hospital_death', 'Time', 'RecordID'])
    validation_Y = feature_extractor(set_b)['In-hospital_death']

    set_c = pd.read_parquet(os.path.join('loaded_data', 'c_patient_data_processed.parquet'))
    test_X = feature_extractor(set_c).drop(columns=['In-hospital_death', 'Time', 'RecordID'])
    test_Y = feature_extractor(set_c)['In-hospital_death']

    logging.info(f"Training X shape: {training_X.shape}")
    logging.info(f"Training Y shape: {training_Y.shape}")
    logging.info(f"Validation X shape: {validation_X.shape}")
    logging.info(f"Validation Y shape: {validation_Y.shape}")
    logging.info(f"Test X shape: {test_X.shape}")
    logging.info(f"Test Y shape: {test_Y.shape}")

    logistic_model = logistic_regression(training_X, training_Y, validation_X, validation_Y)
    get_results(logistic_model, test_X, test_Y)

    rf_model = random_forest(training_X, training_Y, validation_X, validation_Y)
    get_results(rf_model, test_X, test_Y)
