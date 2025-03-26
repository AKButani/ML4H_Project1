import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the datasets
def load_datasets():
    # Assuming parquet files for train and test sets
    train_set = pd.read_parquet('loaded_data/a_patient_data_processed_2.parquet')
    test_set = pd.read_parquet('loaded_data/c_patient_data_processed_2.parquet')
    print(train_set.columns)
    return train_set, test_set

# Prepare data for LSTM
def prepare_lstm_data(df, target_column, time_column='Time'):
    # Group by patient and time point
    grouped = df.groupby(['RecordID', time_column])
    
    # Create input features and target
    X = []
    y = []
    
    for (patient_id, time_point), group in grouped:
        # Extract features for this patient at this time point
        features = group.drop(columns=[target_column, 'RecordID', time_column]).values
        target = group[target_column].values[0]
        
        X.append(features)
        y.append(target)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# Create LSTM Model
def create_lstm_model(input_shape):
    model = Sequential([
        # Unidirectional LSTM
        LSTM(64, input_shape=input_shape, return_sequences=True),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Assuming binary classification
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Create Bidirectional LSTM Model
def create_bidirectional_lstm_model(input_shape):
    model = Sequential([
        # Bidirectional LSTM
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Bidirectional(LSTM(32)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Assuming binary classification
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Aggregate LSTM outputs
def aggregate_lstm_outputs(predictions):
    # Different aggregation strategies
    mean_pred = np.mean(predictions, axis=0)
    max_pred = np.max(predictions, axis=0)
    last_pred = predictions[:, -1]
    
    # Voting or weighted ensemble could be more sophisticated
    return {
        'mean': mean_pred,
        'max': max_pred,
        'last': last_pred
    }

def main():
    # Load datasets
    train_set, test_set = load_datasets()
    
    # Prepare data
    X_train, y_train = prepare_lstm_data(train_set, target_column='In-hospital_death')
    X_test, y_test = prepare_lstm_data(test_set, target_column='In-hospital_death')
    
    # Unidirectional LSTM
    unidirectional_model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    
    # Train unidirectional LSTM
    unidirectional_history = unidirectional_model.fit(
        X_train, y_train, 
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping]
    )
    
    # Evaluate on test set
    unidirectional_results = unidirectional_model.evaluate(X_test, y_test)
    print("Unidirectional LSTM Test Performance:")
    print(f"Loss: {unidirectional_results[0]}, Accuracy: {unidirectional_results[1]}")
    
    # Bidirectional LSTM
    bidirectional_model = create_bidirectional_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Train bidirectional LSTM
    bidirectional_history = bidirectional_model.fit(
        X_train, y_train, 
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping]
    )
    
    # Evaluate bidirectional model
    bidirectional_results = bidirectional_model.evaluate(X_test, y_test)
    print("\nBidirectional LSTM Test Performance:")
    print(f"Loss: {bidirectional_results[0]}, Accuracy: {bidirectional_results[1]}")
    
    # Aggregate outputs demonstration
    unidirectional_predictions = unidirectional_model.predict(X_test)
    aggregated_outputs = aggregate_lstm_outputs(unidirectional_predictions)
    print("\nAggregated Output Strategies:")
    for method, prediction in aggregated_outputs.items():
        print(f"{method.capitalize()} Prediction: {prediction}")

if __name__ == "__main__":
    main()