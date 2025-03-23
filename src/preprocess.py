import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler

def process_file(file_path):
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
    df_pivot = pd.pivot_table(df, index='Time', columns='Parameter', values='Value', aggfunc='first')
    df_pivot = df_pivot.reset_index()
    
    # Add fixed values (e.g. RecordID, Age, etc.) as constant columns in each pivoted DataFrame
    for param, value in fixed_values.items():
        df_pivot[param] = value

    if isinstance(df_pivot.columns, pd.MultiIndex):
        df_pivot.columns = [col if isinstance(col, str) else col[1] for col in df_pivot.columns]
    
    columns_order = ['Time'] + fixed_parameters + [col for col in df_pivot.columns if col not in (['Time'] + fixed_parameters)]
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
    cols_to_ffill = [col for col in df.columns if col != groupby_col]
    df_imputed = df.copy().groupby(groupby_col)[cols_to_ffill].apply(lambda group: group.ffill())

    numeric_cols = df_imputed.select_dtypes(include=['number']).columns
    df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(df_imputed[numeric_cols].median())

    return df_imputed

def scale_values(df, scaler=None):
    
    df_scaled = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    numeric_cols = numeric_cols.difference(['Time', 'RecordID'])
    if scaler is not None:
        df_scaled[numeric_cols] = scaler.transform(df_scaled[numeric_cols])
    else:
        scaler = StandardScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    return df_scaled, scaler
if __name__ == "__main__":

    mean, std = 0, 1

    for folder in ['a', 'b', 'c']:
        print("Loading files")
        patient_df = combine_files_in_folder(os.path.join('data', 'set-' + folder))

        outcomes_df = pd.read_csv(os.path.join('data', f'Outcomes-{folder}.txt'), sep=',')
        outcomes_df = outcomes_df[['RecordID', 'In-hospital_death']]
        

        #patient_df = pd.read_parquet(os.path.join('loaded_data', f'{folder}_patient_data.parquet'))
        print("Processing files")
        patient_df_imputed = fill_null_values(patient_df, 'RecordID')
        
        if folder == 'a':
            patient_df_scaled, scaler = scale_values(patient_df_imputed)
            print("scaler: ", scaler)
        else:
            patient_df_scaled, _ = scale_values(patient_df_imputed, scaler)

        patient_df_scaled = patient_df_scaled.merge(outcomes_df, on='RecordID', how='inner')

        print("Saving processed files")
        #Order columns alphabetically
        patient_df_scaled = patient_df_scaled.reindex(sorted(patient_df_scaled.columns), axis=1)
        patient_df_scaled.to_parquet(os.path.join('loaded_data', f'{folder}_patient_data_processed.parquet'), index=False)




