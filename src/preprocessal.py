import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def process_file(file_path):
    df = pd.read_csv(file_path, sep=',')
    
    # Convert 'Time' column: append ":00", convert to timedelta, then round up to the next hour.
    df['Time'] = pd.to_timedelta(df['Time'] + ":00")
    df['Time'] = np.ceil(df['Time'].dt.total_seconds() / 3600)  # Convert to numeric hours
    
    # Define fixed parameters and extract their constant values
    fixed_parameters = ["RecordID", "Age", "Weight", "Gender", "Height", "ICUType"]
    fixed_values = df[df["Parameter"].isin(fixed_parameters)].set_index("Parameter")["Value"].to_dict()
    
    # Drop rows corresponding to fixed parameters
    df = df[~df["Parameter"].isin(fixed_parameters)]
    
    # Pivot the DataFrame
    df_pivot = pd.pivot_table(df, index='Time', columns='Parameter', values='Value', aggfunc=lambda x: x.sum() if x.name == 'Urine' else x.mean()).reset_index()
    
    # Add fixed values
    for param, value in fixed_values.items():
        df_pivot[param] = value
    
    if isinstance(df_pivot.columns, pd.MultiIndex):
        df_pivot.columns = [col if isinstance(col, str) else col[1] for col in df_pivot.columns]
    
    columns_order = ['Time'] + fixed_parameters + [col for col in df_pivot.columns if col not in (['Time'] + fixed_parameters)]
    return df_pivot[columns_order]

def combine_files_in_folder(folder_path):
    all_files = os.listdir(folder_path)
    all_dfs = [process_file(os.path.join(folder_path, file)) for file in all_files]
    return pd.concat(all_dfs, ignore_index=True)

def address_special_cases(patient_df):
    patient_df.loc[~patient_df['Gender'].isin([0, 1]), 'Gender'] = np.nan
    patient_df.loc[(patient_df['Temp'] < 28) | (patient_df['Temp'] > 46), 'Temp'] = np.nan
    patient_df['MechVent'] = patient_df['MechVent'].fillna(0)
    patient_df.loc[patient_df['Weight'] < 5, 'Weight'] = np.nan
    patient_df.loc[(patient_df['Height'] < 25) | (patient_df['Height'] > 220), 'Height'] = np.nan
    return patient_df

def fill_null_values(df, groupby_col):
    cols_to_ffill = [col for col in df.columns if col != groupby_col]
    df.sort_values(by=['Time'], inplace=True)
    df_imputed = df.groupby(groupby_col)[cols_to_ffill].apply(lambda group: group.ffill())
    numeric_cols = df_imputed.select_dtypes(include=['number']).columns
    df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(df_imputed[numeric_cols].median())
    return df_imputed

def scale_values(df, scaler=None):
    df_scaled = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.difference(['Time', 'RecordID'])

    if scaler is None or not hasattr(scaler, "mean_"):  # Check if scaler is fitted
        scaler = StandardScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    else:
        df_scaled[numeric_cols] = scaler.transform(df_scaled[numeric_cols])
    
    return df_scaled, scaler

if __name__ == "__main__":
    mean, std = 0, 1
    for folder in ['a', 'b', 'c']:
        print("Loading files")
        patient_df = combine_files_in_folder(os.path.join('data', 'set-' + folder))
        patient_df = address_special_cases(patient_df)
        outcomes_df = pd.read_csv(os.path.join('data', f'Outcomes-{folder}.txt'), sep=',')[['RecordID', 'In-hospital_death']]
        print("Processing files")
        patient_df_imputed = fill_null_values(patient_df, 'RecordID')
        
        patient_df_imputed = patient_df_imputed.merge(outcomes_df, on='RecordID', how='inner')
        patient_df_imputed.to_parquet(os.path.join('loaded_data', f'{folder}_patient_data_NOT_scaled.parquet'), index=False)

        if folder == 'a':
            patient_df_scaled, scaler = scale_values(patient_df_imputed)
        else:
            patient_df_scaled, _ = scale_values(patient_df_imputed, scaler)
        
        print("Saving processed files")
        patient_df_scaled = patient_df_scaled.reindex(sorted(patient_df_scaled.columns), axis=1)
        patient_df_scaled.to_parquet(os.path.join('loaded_data', f'{folder}_patient_data_processed_cluster.parquet'), index=False)
