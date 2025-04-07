import random
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
    
    # Remove unwanted columns (for example 'ICUType' which we don't aggregate)
    dataset = dataset.drop(columns=['ICUType'], errors='ignore')

    # If no features are specified, use all columns
    if features_of_interest is None:
        features_of_interest = dataset.columns
    # Remove 'RecordID' from features of interest if present
    features_of_interest = features_of_interest[features_of_interest != 'RecordID']
    
    # Aggregate statistics (max, min, mean) for these features
    aggregated_data = dataset.groupby('RecordID')[features_of_interest].agg(['max', 'min', 'mean','last'])
    
    # Function to create a text summary for each patient
    def create_summary(row):
        summary = []
        for feature in features_of_interest:
            
            if feature.lower() == 'in-hospital_death':
                continue
            mean_value = round(row[(feature, 'mean')], 2)
            last_value = row[(feature, 'last')] 
            summary.append(f"Avg {feature}: {mean_value}, Last {feature}: {last_value}")
            
            #summary.append(f"Max {feature}: {max_value}, Min {feature}: {min_value}, Avg {feature}: {mean_value}")
        
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

def create_prompt_and_save_by_group(training_summaries_d, training_summaries_s, testing_summaries, group_size_train=3, group_size_test=1):
    total_training_d = len(training_summaries_d)
    total_training_s = len(training_summaries_s)
    total_testing = len(testing_summaries)

    prompts = []  # List to store all the prompts
    
    # Split testing summaries into chunks of group_size_test
    for i in range(0, total_testing, group_size_test):
        prompt = ""

        # Get indices of patients who survived (Prediction: survived)
        sampled_train_indices_s = random.sample(range(total_training_s), group_size_train * 2)

        for idx in sampled_train_indices_s:
            prompt += f"Example {idx + 1 + group_size_train}:\n{training_summaries_s.iloc[idx]}\n\n"

        # Get indices of patients who died (Prediction: died)
        sampled_train_indices_d = random.sample(range(total_training_d), group_size_train)

        for idx in sampled_train_indices_d:
            prompt += f"Example {idx + 1}:\n{training_summaries_d.iloc[idx]}\n\n"

        # Add testing summaries for the current group
        for j in range(i, min(i + group_size_test, total_testing)):
            original_index = testing_summaries.index[j]
            prompt += f"Index: {original_index}:\n{testing_summaries.iloc[j]}\n\n"

        # Append the generated prompt to the list
        prompts.append(prompt)

    return prompts  # Return the list of generated prompts


training_df_path = 'loaded_data/a_patient_data_NOT_scaled.parquet'
# read the parquet file
training_df = pd.read_parquet(training_df_path)
training_df = training_df.drop(columns=['Time'], errors='ignore')

died_per_record = training_df.groupby('RecordID')['In-hospital_death'].any()
died_record_count = died_per_record.sum()
print(f"Number of RecordIDs with at least one death (1) entry: {died_record_count}")
summaries_training_df = create_summaries_from_dataset(training_df)

# Save the summaries_training_df to a CSV file for later use
summaries_training_df.to_csv('Q4p2/summaries_training_df.csv', index=True)

# View the first few summaries
print(summaries_training_df.head())
print(summaries_training_df.iloc[0])

testing_df_path = 'loaded_data/c_patient_data_NOT_scaled.parquet'
# read the parquet file
testing_df = pd.read_parquet(testing_df_path)

died_per_record = testing_df.groupby('RecordID')['In-hospital_death'].any()
died_record_count = died_per_record.sum()
print(f"Number of RecordIDs with at least one death (1) entry: {died_record_count}")

# Remove 'In-hospital_death' and 'Time' columns from testing_df
testing_df = testing_df.drop(columns=['In-hospital_death', 'Time'], errors='ignore')
summaries_testing_df = create_summaries_from_dataset(testing_df, testing=True)

# View the first few summaries
print(summaries_testing_df.head())
print(summaries_testing_df.iloc[0])
import random

# Save the summaries_training_df to a CSV file for later use
summaries_testing_df.to_csv('Q4p2/summaries_testing_df.csv', index=True)

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
summaries_training_surv_df.iloc[0]


# Use the function to create the prompts and store them in a list
prompts_list = create_prompt_and_save_by_group(
    training_summaries_s=summaries_training_surv_df,
    training_summaries_d=summaries_training_died_df,
    testing_summaries=summaries_testing_df
)

# Now, `prompts_list` contains all the generated prompts in a list format.
# You can print it or use it as needed.
print(prompts_list[:2])  # Print the first two prompts as a sample

import pickle

##############################################
################## Q4.1 #######################
##############################################

# Save as a pickle file
with open('Q4p1_prompts_list.pkl', 'wb') as file:
    pickle.dump(prompts_list, file)


##############################################
################## Q4.2 #######################
##############################################
import pandas as pd
import os
import pickle

def generate_patient_descriptions(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Rename the second column to 'Info' if necessary
    df.rename(columns={df.columns[1]: 'Info'}, inplace=True)

    # Strip any unwanted spaces from the 'Info' column
    df['Info'] = df['Info'].str.strip()

    # Initialize an empty list for patient descriptions
    patient_texts = []

    # Loop through each row in the dataframe and extract the information
    for index, row in df.iterrows():
        # Extract the RecordID
        patient_number = row['RecordID']
        
        # Extract the entire 'Info' column content
        info = row['Info']
        
        # Ensure info is not empty
        if pd.notnull(info) and info != '':
            # Create a patient description using the full 'Info' column
            description = f"Patient {patient_number}: {info}"
            # Append the description to the list
            patient_texts.append(description)

    # Return the list of patient descriptions
    return patient_texts

csv_file_path_test = 'Q4p2/summaries_testing_df.csv'
patient_descriptions_test = generate_patient_descriptions(csv_file_path_test)

csv_file_path_train = 'Q4p2/summaries_training_df.csv'
patient_descriptions_train = generate_patient_descriptions(csv_file_path_train)

# Check the result
print(patient_descriptions_train[0])
print(patient_descriptions_test[0])
print(len(patient_descriptions_test))
print(len(patient_descriptions_train))

def save_patient_descriptions_as_pkl(patient_descriptions, file_name):
    # Save the patient descriptions to a pickle file
    with open(file_name, 'wb') as file:
        pickle.dump(patient_descriptions, file)
    
    print(f"Saved {len(patient_descriptions)} descriptions in '{file_name}'.")

# Example usage:
# Save the training descriptions to a pickle file
save_patient_descriptions_as_pkl(patient_descriptions_train, file_name='Q4p2/patient_descriptions_train.pkl')

# Save the test descriptions to a pickle file
save_patient_descriptions_as_pkl(patient_descriptions_test, file_name='Q4p2/patient_descriptions_test.pkl')
