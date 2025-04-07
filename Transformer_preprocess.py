import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Define the fixed (patient-level) parameters.
fixed_parameters = ["RecordID", "Age", "Weight", "Gender", "Height", "ICUType"]

def get_variable_mapping_multiple(folder_paths, fixed_parameters=fixed_parameters):
    """
    Create a mapping from measurement variable names across multiple folders.
    """
    variables = set()
    for folder_path in folder_paths:
        for file in os.listdir(folder_path):
            # Optionally, filter for .csv files.
            if not file.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(folder_path, file), sep=',')
            df_meas = df[~df["Parameter"].isin(fixed_parameters)]
            variables.update(df_meas["Parameter"].unique())
    variables = sorted(list(variables))
    mapping = {var: idx for idx, var in enumerate(variables)}
    return mapping

def get_variable_mapping(folder_path, fixed_parameters=fixed_parameters):
    """
    Create a mapping from measurement variable names (from training data)
    to a unique integer. We assume that non-fixed parameters (measurements)
    in the training set yield 41 categories.
    """
    all_files = os.listdir(folder_path)
    variables = set()
    for file in all_files:
        df = pd.read_csv(os.path.join(folder_path, file), sep=',')
        # Only consider rows that are not fixed parameters.
        df_meas = df[~df["Parameter"].isin(fixed_parameters)]
        variables.update(df_meas["Parameter"].unique())
    variables = sorted(list(variables))
    mapping = {var: idx for idx, var in enumerate(variables)}
    return mapping

def address_special_cases(df, fixed_parameters=fixed_parameters):
    """
    Given a long-format DataFrame (with a "Parameter" column and a "Value" column),
    this function extracts the fixed parameters and applies filtering rules.
    If the fixed parameters do not meet the biological constraints, it returns an empty DataFrame.
    Otherwise, it returns the original DataFrame.
    """
    # Extract fixed parameter rows.
    fixed_df = df[df["Parameter"].isin(fixed_parameters)]
    fixed_dict = fixed_df.set_index("Parameter")["Value"].to_dict()

    try:
        # Attempt to convert relevant fixed parameter values to float.
        gender = float(fixed_dict.get("Gender", np.nan))
        temp = float(fixed_dict.get("Temp", np.nan))
        weight = float(fixed_dict.get("Weight", np.nan))
        height = float(fixed_dict.get("Height", np.nan))
    except Exception as e:
        # If conversion fails, skip this file.
        return pd.DataFrame()
    
    # Apply biological filtering.
    if gender not in [0, 1]:
        return pd.DataFrame()
    if not (28 <= temp <= 46):
        return pd.DataFrame()
    if weight < 5:
        return pd.DataFrame()
    if not (25 <= height <= 220):
        return pd.DataFrame()
    
    # Optionally, you can handle MechVent separately if needed.
    # For example, if "MechVent" is missing, you might want to set a default value.
    # Here we do not filter on it but you could extend this function as required.
    
    # If all conditions are met, return the original DataFrame.
    return df


def process_file_triplets(file_path, variable_mapping, fixed_parameters=fixed_parameters):
    """
    Process a single file into a long-format DataFrame with one row per measurement.
    Each row contains:
      - RecordID (extracted from the fixed parameters)
      - t_scaled: the time scaled to [0,1] (per patient)
      - z: the encoded measurement variable (using variable_mapping)
      - v: the observed value (to be scaled later)
    """
    df = pd.read_csv(file_path, sep=',')
    df = address_special_cases(df)
    
    fixed_rows = df[df["Parameter"].isin(fixed_parameters)]
    fixed_values = fixed_rows.set_index("Parameter")["Value"].to_dict()
    record_id = fixed_values.get("RecordID")

    df_meas = df[~df["Parameter"].isin(fixed_parameters)].copy()
    if df_meas.empty:
        return pd.DataFrame()  
    
    df_meas['Time'] = pd.to_timedelta(df_meas['Time'] + ":00")
    
    # Convert time to a numeric value (in seconds)
    df_meas['time_numeric'] = df_meas['Time'].dt.total_seconds()
    
    # Scale time for this patient to the range [0,1]
    t_min = df_meas['time_numeric'].min()
    t_max = df_meas['time_numeric'].max()
    if t_max - t_min > 0:
        df_meas['t_scaled'] = (df_meas['time_numeric'] - t_min) / (t_max - t_min)
    else:
        df_meas['t_scaled'] = 0.0  # If only one measurement
    
    # Map measurement variable to a category index using the provided mapping.
    df_meas['z'] = df_meas['Parameter'].map(variable_mapping)
    
    # Convert the "Value" column to numeric.
    df_meas['v'] = pd.to_numeric(df_meas['Value'], errors='coerce')
    df_meas = df_meas.dropna(subset=['v'])
    
    # Add the patient identifier.
    df_meas['RecordID'] = record_id
    
    # Return only the needed columns.
    return df_meas[['RecordID', 't_scaled', 'z', 'v']]

def combine_files_in_folder_triplets(folder_path, variable_mapping):
    """
    Process all files in a folder and concatenate the measurement triplets.
    """
    all_files = os.listdir(folder_path)
    dfs = []
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        df_triplet = process_file_triplets(file_path, variable_mapping)
        if not df_triplet.empty:
            dfs.append(df_triplet)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=['RecordID', 't_scaled', 'z', 'v'])


def scale_value_triplets_category_wise(df, scalers=None):
    """
    Scale the measurement values ('v') category-wise using StandardScaler.
    
    Parameters:
      - df: DataFrame containing the triplets with columns 'z' and 'v'.
      - scalers: Optional dictionary mapping category (z value) to a fitted StandardScaler.
                 If None, a new scaler is fitted for each category.
    
    Returns:
      - df_scaled: A DataFrame with the scaled 'v' values.
      - scalers: Dictionary of scalers used for each category.
    """
    df_scaled = df.copy()
    
    # If no scalers are provided, create an empty dictionary to hold them.
    if scalers is None:
        scalers = {}
    
    # Group the DataFrame by the category 'z'
    for z_val, group in df_scaled.groupby("z"):
        # Check if a scaler for this category exists
        if z_val not in scalers:
            scaler = StandardScaler()
            # Fit the scaler on the group and transform the 'v' column
            transformed = scaler.fit_transform(group[['v']])
            scalers[z_val] = scaler
        else:
            scaler = scalers[z_val]
            # Use the already fitted scaler to transform the values
            transformed = scaler.transform(group[['v']])
        
        # Assign the scaled values back to the appropriate rows in the DataFrame.
        df_scaled.loc[group.index, 'v'] = transformed.flatten()
    
    return df_scaled, scalers


if __name__ == "__main__":
    training_folder_path = os.path.join('data', 'set-a')
    test_folder_path = os.path.join('data', 'set-c')

    variable_mapping = get_variable_mapping_multiple([training_folder_path, test_folder_path])
    scaler = None
    for folder in ['a', 'b', 'c']:
        print("Loading files for folder", folder)
        
        folder_path = os.path.join('ml4h_data', 'p1', 'set-' + folder)
        outcomes_path = os.path.join('ml4h_data','p1', 'Outcomes-' + folder + '.txt')

        df_triplets = combine_files_in_folder_triplets(folder_path, variable_mapping)
        df_triplets.drop(columns=['ICUType'], inplace=True, errors='ignore')

        print("Scaling measurement values for folder", folder)
        #if folder == 'a':
            # For the training set, fit the scaler.
            #df_triplets_scaled, scaler = scale_value_triplets_category_wise(df_triplets, scaler)
        #else:
            # For validation/test, use the scaler fitted on training.
           # df_triplets_scaled, _ = scale_value_triplets_category_wise(df_triplets, scaler)
        
        # Load outcomes (which we assume contain RecordID and In-hospital_death)
        
        outcomes_df = pd.read_csv(outcomes_path, sep=',')[['RecordID', 'In-hospital_death']]
        
        # Merge outcomes (each patient appears in many triplet rows)
        df_triplets_scaled = df_triplets.merge(outcomes_df, on='RecordID', how='inner')
        
        print("Saving processed triplets for folder", folder)
        save_path = os.path.join('loaded_data', f'{folder}_patient_triplets_processed.parquet')
        df_triplets_scaled.to_parquet(save_path, index=False)