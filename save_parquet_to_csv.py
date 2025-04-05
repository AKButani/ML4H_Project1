import pandas as pd

# Load the dataset from the parquet file
file_path = 'loaded_data/a_patient_data_processed_cluster.parquet'  # Update the file path
data = pd.read_parquet(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Save the dataset to a CSV file
csv_file_path = 'patient_data.csv'  # Update this path if you'd like to specify a different location
data.to_csv(csv_file_path, index=False)

print(f"Data has been successfully saved to {csv_file_path}")
