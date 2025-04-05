import pandas as pd
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

folder_path = "Q4p1_results"
# Read all files in the folder and merge them into a single dataframe
# Add a column with the txt file name
# Read only the first row that contains either outcome: death or outcome: survival

all_data = []

def clean_file_name_2(file_name):
    return file_name.replace("Index:", "")

# Remove 'Q4p1_prompt_' and '.txt_response.txt' from the file_name
def clean_file_name(file_name):
    return file_name.replace("outcome:death", "").replace("outcome:survival", "")

for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            for line in file:
                if "outcome:death" in line or "outcome:survival" in line:
                    all_data.append({"file_name": file_name, "outcome": line.strip()})
                    break

df = pd.DataFrame(all_data)
df.head()
df['outcome'] = df['outcome'].apply(clean_file_name_2)
df['RecordID'] = df['outcome'].apply(clean_file_name)
df['outcome2'] = df['outcome'].apply(lambda x: 1 if "outcome:death" in x else 0)
df["outcome"]
df["outcome2"]
df['RecordID']

testing_df_path = 'loaded_data/c_patient_data_processed_cluster.parquet'
# read the parquet file
testing_df = pd.read_parquet(testing_df_path)

testing_df["In-hospital_death"].head()
testing_df["RecordID"].head()
# Ensure the 'RecordID' column is of integer type
testing_df["RecordID"] = testing_df["RecordID"].astype(int)

# remove blank spaces in the RecordID column
df["RecordID"]
df.head()
#df["RecordID"] = df["RecordID"].astype(str).str.replace(" ", "")  # Remove spaces
df["RecordID"] = df["RecordID"].str.replace(".0", "")  # Remove the decimal part
df["RecordID"] = df["RecordID"].astype(int)  # Convert to integer
df.head()

df_true_vals = testing_df[["In-hospital_death", "RecordID"]].drop_duplicates()

# Merge the two dataframes on the 'RecordID' column
merged_df = df_true_vals.merge(df, left_on="RecordID", right_on="RecordID", how="inner")
print(merged_df.head())

# Perform sensitivity analysis
y_true = merged_df["In-hospital_death"]
y_pred = merged_df["outcome2"]
y_pred

# Calculate sensitivity (recall for the positive class)
sensitivity = recall_score(y_true, y_pred)

# Print the sensitivity and other metrics
print("Sensitivity (Recall):", sensitivity)
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))

# Optional: Display confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)

