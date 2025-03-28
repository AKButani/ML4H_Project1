import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

def load_datasets():
    # Assuming parquet files for train and test sets
    train_set = pd.read_parquet('loaded_data/a_patient_data_processed_2.parquet')
    test_set = pd.read_parquet('loaded_data/c_patient_data_processed_2.parquet')
    print(train_set.columns)
    return train_set, test_set
# load_datasets 
x_train, x_test = load_datasets()

