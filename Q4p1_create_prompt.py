# Create aggreagate measures

import pandas as pd

def create_summaries_from_dataset(dataset, features_of_interest=None, testing=False):
    """
    Creates text summaries from a given dataset, processing the first 48 hours of each patient's data.

    Args:
    - dataset: A pandas DataFrame containing patient data.
    - features_of_interest: A list of features to include in the summaries. If None, all columns are used.
    - testing: A boolean flag to control whether predictions should be appended. Default is False.

    Returns:
    - summaries: A pandas Series containing the text summaries for each patient.
    """
    # Filter the data to keep only the first 48 hours of each patient
    first_48_hours = dataset[dataset['Time'] <= 48]
    
    # Remove unwanted columns (for example 'ICUType' which we don't aggregate)
    first_48_hours = first_48_hours.drop(columns=['ICUType'], errors='ignore')

    # If no features are specified, use all columns
    if features_of_interest is None:
        features_of_interest = first_48_hours.columns
    
    # Aggregate statistics (max, min, mean) for these features
    aggregated_data = first_48_hours.groupby('RecordID')[features_of_interest].agg(['max', 'min', 'mean'])
    
    # Function to create a text summary for each patient
    def create_summary(row):
        summary = []
        for feature in features_of_interest:
            max_value = round(row[(feature, 'max')], 2)
            min_value = round(row[(feature, 'min')], 2)
            mean_value = round(row[(feature, 'mean')], 2)
            
            summary.append(f"Max {feature}: {max_value}, Min {feature}: {min_value}, Avg {feature}: {mean_value}")
        
        # Append prediction only if not testing
        if testing:
            summary.append("Prediction:")  # Just append "Prediction:" when testing
        else:
            # Predict in-hospital death based on max value
            in_hospital_death = 'died' if row[('In-hospital_death', 'max')] == 1 else 'survived'
            summary.append(f"Prediction: {in_hospital_death}")
        
        return " | ".join(summary)
    
    # Apply the function to create text summaries for each patient
    summaries = aggregated_data.apply(create_summary, axis=1)
    
    return summaries


training_df_path = 'loaded_data/a_patient_data_processed_cluster.parquet'
# read the parquet file
training_df = pd.read_parquet(training_df_path)

died_per_record = training_df.groupby('RecordID')['In-hospital_death'].any()
died_record_count = died_per_record.sum()
print(f"Number of RecordIDs with at least one death (1) entry: {died_record_count}")
summaries_training_df = create_summaries_from_dataset(training_df)

# View the first few summaries
print(summaries_training_df.head())
print(summaries_training_df.iloc[0])

testing_df_path = 'loaded_data/c_patient_data_processed_cluster.parquet'
# read the parquet file
testing_df = pd.read_parquet(testing_df_path)

died_per_record = testing_df.groupby('RecordID')['In-hospital_death'].any()
died_record_count = died_per_record.sum()
print(f"Number of RecordIDs with at least one death (1) entry: {died_record_count}")
summaries_testing_df = create_summaries_from_dataset(testing_df, testing=True)

# View the first few summaries
print(summaries_testing_df.head())
print(summaries_testing_df.iloc[0])
import random


# split the training summaries into patients who died and patients who survived
died_positions = []
for i in range(len(summaries_training_df)):
    if "Prediction: died" in summaries_training_df.iloc[i]:
        died_positions.append(i)

summaries_training_died_df = summaries_training_df.iloc[died_positions]
summaries_training_died_df.iloc[0]

surv_positions = []
for i in range(len(summaries_training_df)):
    if "Prediction: survived" in summaries_training_df.iloc[i]:
        surv_positions.append(i)
summaries_training_surv_df = summaries_training_df.iloc[surv_positions]        

def create_prompt_and_save_by_group(training_summaries_d, training_summaries_s, testing_summaries, group_size_train=5, group_size_test=1,  base_file_path='Q4p1_txt/Q4p1_prompt'):
    
    total_training_d = len(training_summaries_d)
    total_training_s = len(training_summaries_s)
    total_testing = len(testing_summaries)

    # Split training summaries into chunks of 30
    for i in range(0, total_testing, group_size_test):
        prompt = ""
        
        # get indices of patients who died by looking for "Prediction: died"
    
        sampled_train_indices_d = random.sample(range(total_training_d), group_size_train)

        for idx in sampled_train_indices_d:
            prompt += f"Example {idx+1}:\n{training_summaries_d.iloc[idx]}\n\n"

        sampled_train_indices_s = random.sample(range(total_training_s), group_size_train)

        for idx in sampled_train_indices_s:
            # + group_size_train to offset the index for dead patients
            prompt += f"Example {idx+1+group_size_train}:\n{training_summaries_s.iloc[idx]}\n\n"

        # Add testing summaries for the current group
        for j in range(i, min(i + group_size_test, total_testing)):
            prompt += f"Case {j+1}:\n{testing_summaries.iloc[j]}\n\n"
        
        # Save the prompt to a new text file
        file_path = f"{base_file_path}_{(i // group_size_test) + 1}.txt"
        with open(file_path, 'w') as file:
            file.write(prompt)
        
        print(f"Prompt saved to {file_path}")

# Use the function to create the prompt and save to different text files
create_prompt_and_save_by_group(
    training_summaries_s = summaries_training_surv_df,
    training_summaries_d = summaries_training_died_df, 
    testing_summaries = summaries_testing_df)
