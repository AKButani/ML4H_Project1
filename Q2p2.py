import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pyarrow  # Required for saving parquet files
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences

import random

seed = 42  # Set the seed for reproducibility
tf.random.set_seed(seed)
np.random.seed(seed)  # For NumPy (if you use NumPy anywhere)
random.seed(seed)  # For Python's built-in random module (if you use it anywhere)

# Load the datasets
def load_datasets():
    train_set = pd.read_parquet('loaded_data/a_patient_data_processed_cluster.parquet')
    test_set = pd.read_parquet('loaded_data/c_patient_data_processed_cluster.parquet')
    validation_set = pd.read_parquet('loaded_data/b_patient_data_processed_cluster.parquet')
    return train_set, test_set, validation_set

def prepare_lstm_data(df, target_column, time_column='Time'):
    if 'ICUType' in df.columns:
        df = df.drop(columns=['ICUType'])

    X, y = [], []
    for patient_id, group in df.groupby('RecordID'):
        group = group.sort_values(time_column)
        features = group.drop(columns=[target_column, 'RecordID', time_column]).values
        target = group[target_column].iloc[0]
        X.append(features)
        y.append(target)
    
    # Pad sequences so that all sequences have the same length
    X_padded = pad_sequences(X, padding='post', dtype='float32')
    return X_padded, np.array(y)

# Create Unidirectional LSTM Model
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create Bidirectional LSTM Model
def create_bidirectional_lstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Bidirectional(LSTM(32)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Aggregate LSTM outputs
def aggregate_lstm_outputs(predictions):
    mean_pred = np.mean(predictions, axis=0)
    max_pred = np.max(predictions, axis=0)
    last_pred = predictions[:, -1]
    return {
        'mean': mean_pred,
        'max': max_pred,
        'last': last_pred
    }

# Plot Loss and Accuracy
def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("Accuracy.png")
    plt.close()

def main():
    train_set, test_set, validation_set = load_datasets()
    X_train, y_train = prepare_lstm_data(train_set, target_column='In-hospital_death')
    X_test, y_test = prepare_lstm_data(test_set, target_column='In-hospital_death')
    X_val, y_val = prepare_lstm_data(validation_set, target_column='In-hospital_death')

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    # Alternatively, manually set class weights
    class_weight_dict = {0: 1.0, 1: 5.0}  # Penalize misclassifying deaths 5x more
    print("Class weights:", class_weight_dict)

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    unidirectional_model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    unidirectional_history = unidirectional_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        #validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        class_weight=class_weight_dict
    )

    unidirectional_results = unidirectional_model.evaluate(X_test, y_test)
    print(f"Unidirectional LSTM Test Loss: {unidirectional_results[0]}, Accuracy: {unidirectional_results[1]}")

    y_pred = (unidirectional_model.predict(X_test) > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"True Positives with Unidirectional (Deaths correctly predicted): {tp}")
    print(f"False Negatives with Unidirectional (Deaths incorrectly predicted): {fn}")

    y_pred_prob = unidirectional_model.predict(X_test).flatten()
    # Compute AUROC and AUPRC
    auroc = roc_auc_score(y_test, y_pred_prob)
    auprc = average_precision_score(y_test, y_pred_prob)
    print(f"AUROC: {auroc}")
    print(f"AUPRC: {auprc}")

    df_predictions = pd.DataFrame({
        'True_Value': y_test,
        "Prob_death": unidirectional_model.predict(X_test).flatten(),
        'Predicted_Value': y_pred.flatten()
    })
    df_predictions.to_parquet("unidirectional_predictions.parquet", index=False)
    print("Predictions saved to 'unidirectional_predictions.parquet'")

    plot_training_history(unidirectional_history, 'Unidirectional LSTM')

    bidirectional_model = create_bidirectional_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    bidirectional_history = bidirectional_model.fit(
        X_train, y_train,
        #validation_split=0.2,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        class_weight=class_weight_dict
    )

    bidirectional_results = bidirectional_model.evaluate(X_test, y_test)
    print("\nBidirectional LSTM Test Performance:")
    print(f"Loss: {bidirectional_results[0]}, Accuracy: {bidirectional_results[1]}")

    y_pred_bi = (bidirectional_model.predict(X_test) > 0.5).astype(int)
    df_predictions_bi = pd.DataFrame({
        'True_Value': y_test,
        "Prob_death": bidirectional_model.predict(X_test).flatten(),
        'Predicted_Value': y_pred_bi.flatten()
    })

    y_pred_prob_bi = bidirectional_model.predict(X_test).flatten()
    # Compute AUROC and AUPRC
    auroc_bi = roc_auc_score(y_test, y_pred_prob_bi)
    auprc_bi = average_precision_score(y_test, y_pred_prob_bi)
    print(f"AUROC Bidirectional: {auroc_bi}")
    print(f"AUPRC Bidirectional: {auprc_bi}")

    df_predictions_bi.to_parquet("bidirectional_predictions.parquet", index=False)
    print("Predictions saved to 'bidirectional_predictions.parquet'")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_bi).ravel()
    print(f"True Positives with bidirectional (Deaths correctly predicted): {tp}")
    print(f"False Negatives with bidirectional (Deaths incorrectly predicted): {fn}")

    unidirectional_predictions = unidirectional_model.predict(X_test)
    aggregated_outputs = aggregate_lstm_outputs(unidirectional_predictions)
    print("\nAggregated Output Strategies:")
    for method, prediction in aggregated_outputs.items():
        print(f"{method.capitalize()} Prediction: {prediction}")

if __name__ == "__main__":
    main()
