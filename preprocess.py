import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler

def process_file(file_path):
    """
    Reads and processes a single txt file and returns a pivoted DataFrame.
    """
    # Read file and parse CSV data
    df = pd.read_csv(file_path, sep=',')
    
    # Convert 'Time' column: append ":00", convert to timedelta, then round up to the next hour.
    df['Time'] = pd.to_timedelta(df['Time'] + ":00")
    df['Time'] = pd.to_timedelta(np.ceil(df['Time'].dt.total_seconds() / 3600) * 3600, unit='s')
    
    # Define fixed parameters and extract their constant values
    fixed_parameters = ["RecordID", "Age", "Weight", "Gender", "Height", "ICUType"]
    fixed_values = df[df["Parameter"].isin(fixed_parameters)].set_index("Parameter")["Value"].to_dict()
    
    # Drop rows corresponding to fixed parameters and ICUType as they're not needed for pivoting
    df = df[~df["Parameter"].isin(fixed_parameters)]
    
    # Pivot the DataFrame so that each time point becomes a row and Parameters become columns
    df_pivot = pd.pivot_table(df, index=['Time'], columns='Parameter', values='Value', aggfunc='first').reset_index()
    
    # Add fixed values (e.g. RecordID, Age, etc.) as constant columns in each pivoted DataFrame
    for param, value in fixed_values.items():
        df_pivot[param] = value
        
    columns_order = ['Time'] + fixed_parameters + [col for col in df_pivot.columns if col not in ['Time'] + fixed_parameters]
    return df_pivot[columns_order]

def combine_files_in_folder(folder_path):
    """
    Processes all .txt files in the specified folder and combines the results into one DataFrame.
    """
    # Get list of all txt files in the folder
    all_files = os.listdir(folder_path)
    all_dfs = []
    
    for file in all_files:
        processed_df = process_file(os.path.join(folder_path, file))
        all_dfs.append(processed_df)
    
    # Concatenate all processed DataFrames into one final DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

def fill_null_values(df, groupby_col):
    cols_to_ffill = [col for col in patient_df.columns if col != groupby_col]
    df_imputed = df.copy().groupby(groupby_col)[cols_to_ffill].apply(lambda group: group.ffill())
    numeric_cols = df_imputed.select_dtypes(include=['number']).columns
    df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(df_imputed[numeric_cols].median())
    return df_imputed

def scale_values(df):
    scaler = StandardScaler()
    df_scaled = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    numeric_cols = numeric_cols[numeric_cols != 'Time']
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    return df_scaled

if __name__ == "__main__":

    print("Loading files")
    #patient_df = combine_files_in_folder(os.path.join('data', 'set-a'))
    #patient_df.to_parquet('patient_data.parquet', index=False)
    #outcomes_df = pd.read_csv(os.path.join('data', 'Outcomes-a.txt'), sep=',')

    patient_df = pd.read_parquet('patient_data.parquet')
    print("Processing files")
    patient_df_imputed = fill_null_values(patient_df, 'RecordID')
    patient_df_scaled = scale_values(patient_df_imputed)

    print("Saving processed files")
    patient_df_scaled.to_parquet('patient_data_processed.parquet', index=False)

