import pandas as pd
import numpy as np
import os
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, fbeta_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters
from sklearn.impute import SimpleImputer

def extract_mean_features(df):
    return df.groupby('RecordID').mean().drop(columns=['Time', 'ICUType', 'In-hospital_death'])

def extract_last_features(df):
    return df.groupby('RecordID').last().drop(columns=['Time', 'ICUType', 'In-hospital_death'])

def get_target_variable(df):
    return df.groupby('RecordID').last()['In-hospital_death']

def logistic_regression(training_X, training_Y, validation_X, validation_Y, randomstate=42) -> LogisticRegression:
    C_values = [0.01, 0.1, 1, 10, 100]
    penalties = ['l1', 'l2']
    max_iter_values = [100, 200, 500, 1000, 2000, 5000]

    best_f2_score = 0.0
    best_params = {}
    best_model = None

    # Define class weights to penalize false negatives more
    class_weights = {0: 1, 1: 6.5}  # Adjust the weights as needed

    # Loop through each combination of hyperparameters
    for C in C_values:
        for max_iter in max_iter_values:
            for penalty in penalties:
                print(f"Training model with C={C}, max_iter={max_iter}, penalty={penalty}")
                
                pipeline = Pipeline([
                    #('scaler', StandardScaler()),
                    #('imputer', SimpleImputer(strategy='mean')),
                    ('classifier', LogisticRegression(
                        C=C, 
                        penalty=penalty,
                        solver='saga',
                        max_iter=max_iter,
                        random_state=randomstate,
                        class_weight=class_weights,  # Add class weights
                        n_jobs=-1
                    ))
                ])
                
                # model = LogisticRegression(
                #     C=C, 
                #     penalty=penalty,
                #     solver='saga',
                #     max_iter=max_iter,
                #     random_state=randomstate,
                #     class_weight=class_weights,  # Add class weights
                #     n_jobs=-1
                # )

                # Fit the model on the training data
                pipeline.fit(training_X, training_Y)
                
                # Predict on the validation set
                predictions = pipeline.predict(validation_X)
                f2 = fbeta_score(validation_Y, predictions, beta=2)
                
                # Save best model based on validation F2 score
                if f2 > best_f2_score:
                    best_f2_score = f2
                    best_params = {'C': C, 'penalty': penalty, 'max_iter': max_iter}
                    best_model = pipeline
                    
                    print(f"Best validation F2 score: {best_f2_score}")

    logging.info(f"Best model parameters: {best_params}")
    logging.info("Model parameters:")
    logging.info(best_model.get_params())
    logging.info(best_model.named_steps['classifier'].coef_)
    return best_model

def random_forest(training_X, training_Y, validation_X, validation_Y, randomstate=42) -> RandomForestClassifier:
    # Define a grid of hyperparameters to search over
    n_estimators_values = [100, 200, 300, 500]
    max_depth_values = [None, 5, 10]

    best_f2_score = 0.0
    best_params = {}
    best_model = None

    # Define class weights to penalize false negatives more
    class_weights = {0: 1, 1: 6.5}  # Adjust the weights as needed

    # Loop through each combination of hyperparameters
    for n_estimators in n_estimators_values:
        for max_depth in max_depth_values:

            pipeline = Pipeline([
                    #('scaler', StandardScaler()),
                    #('imputer', SimpleImputer(strategy='mean')),
                    ('classifier', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=randomstate,
                class_weight=class_weights  # Add class weights
                ))
            ])

            # model = RandomForestClassifier(
            #     n_estimators=n_estimators,
            #     max_depth=max_depth,
            #     random_state=randomstate,
            #     class_weight=class_weights  # Add class weights
            # )

            pipeline.fit(training_X, training_Y)
            # Use the model for prediction to ensure consistency
            predictions = pipeline.predict(validation_X)
            f2 = fbeta_score(validation_Y, predictions, beta=2)
            print(f"Validation F2 score for n_estimators={n_estimators}, max_depth={max_depth}: {f2:.4f}")
            
            # Save best model based on validation F2 score
            if f2 > best_f2_score:
                best_f2_score = f2
                best_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
                best_model = pipeline
    logging.info(f"Best model parameters: {best_params}")
    return best_model

def get_results(model, test_X, test_Y):
    # Use the pipeline for prediction to ensure consistency
    test_predictions = model.predict(test_X)
    logging.info(f"Test Accuracy: {accuracy_score(test_Y, test_predictions)}")
    # Calculate AUROC
    auroc = roc_auc_score(test_Y, test_predictions)
    logging.info(f"Test AUROC: {auroc}")
    # Calculate AUPRC
    auprc = average_precision_score(test_Y, test_predictions)
    logging.info(f"Test AUPRC: {auprc}")

    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(test_Y, test_predictions).ravel()
    logging.info(f"True Negatives: {tn}")
    logging.info(f"False Positives: {fp}")
    logging.info(f"False Negatives: {fn}")
    logging.info(f"True Positives: {tp}")

if __name__ == '__main__':
    randomstate = 42
    dataset = '_efficient_features'

    logging.basicConfig(filename=os.path.join('logs', f'model_training_mean_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # choose feature extraction method
    feature_extractor = extract_mean_features
    logging.info(f"Using feature extractor: {feature_extractor.__name__}")

    # Training set
    set_a = pd.read_parquet(os.path.join('loaded_data', 'a_patient_data_processed_cluster.parquet'))
    training_X = feature_extractor(set_a) 
    #training_X =  pd.read_parquet(os.path.join('extracted_features', 'training_X_clean_custom.parquet')) #
    
    training_Y = get_target_variable(set_a)


    # Validation set
    set_b = pd.read_parquet(os.path.join('loaded_data', 'b_patient_data_processed_cluster.parquet'))
    validation_X = feature_extractor(set_b)  #
    #validation_X = pd.read_parquet(os.path.join('extracted_features', 'validation_X_clean_custom.parquet'))[training_X.columns]  #
    validation_Y = get_target_variable(set_b)

    # Test set
    set_c = pd.read_parquet(os.path.join('loaded_data', 'c_patient_data_processed_cluster.parquet'))
    test_X = feature_extractor(set_c) #
    #test_X = pd.read_parquet(os.path.join('extracted_features', 'test_X_clean_custom.parquet'))[training_X.columns] #  #
    
    test_Y = get_target_variable(set_c)

    logging.info(f"Training X shape: {training_X.shape}")
    logging.info(f"Training Y shape: {training_Y.shape}")
    logging.info(f"Validation X shape: {validation_X.shape}")
    logging.info(f"Validation Y shape: {validation_Y.shape}")
    logging.info(f"Test X shape: {test_X.shape}")
    logging.info(f"Test Y shape: {test_Y.shape}")

    logging.info(f"Training X: {training_X.head()}")
    logging.info(f"Training Y: {training_Y.head()}")

    assert(training_X.index.equals(training_Y.index))
    assert(validation_X.index.equals(validation_Y.index)) 
    assert(test_X.index.equals(test_Y.index))

    logistic_model = logistic_regression(training_X, training_Y, validation_X, validation_Y)
    logging.info("Logistic Regression model trained")
    get_results(logistic_model, test_X, test_Y)

    rf_model = random_forest(training_X, training_Y, validation_X, validation_Y)
    logging.info("Random Forest model trained")
    get_results(rf_model, test_X, test_Y)
