import pandas as pd

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

# Example usage:
csv_file_path_test = 'summaries_testing_df.csv'
patient_descriptions_test = generate_patient_descriptions(csv_file_path_test)

csv_file_path_train = 'summaries_training_df.csv'
patient_descriptions_train = generate_patient_descriptions(csv_file_path_train)

# Check the result
print(patient_descriptions_train[0])
print(len(patient_descriptions_test))

import os

def save_patient_descriptions(patient_descriptions, folder_name):

    # Loop through the list of descriptions and save each as a .txt file
    for i, description in enumerate(patient_descriptions):
        # Define the filename for each description
        file_name = f"Q4p2_prompt_{folder_name}/patient_{folder_name}_{i + 1}.txt"
        
        # Write the description to the text file
        with open(file_name, 'w') as file:
            file.write(description)
            
    print(f"Saved {len(patient_descriptions)} descriptions in '{folder_name}' folder.")

# Save descriptions for the training set in the 'Q4p2_prompt_train' folder
save_patient_descriptions(patient_descriptions_train, folder_name='train')

# Save descriptions for the test set in the 'Q4p2_prompt_test' folder
save_patient_descriptions(patient_descriptions_test, folder_name='test')
