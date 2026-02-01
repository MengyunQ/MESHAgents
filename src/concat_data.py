import pandas as pd
import os

data_dir = "data/clinical_measures_26k_collated_qced.csv"
participants_dir = "data/participant_characteristics_26k.csv"

data = pd.read_csv(data_dir)
participants = pd.read_csv(participants_dir)

merged_data = pd.merge(data, participants, on='ID', how='inner')
merged_data.to_csv('data/merged_data.csv', index=False)

print(merged_data.head())
print(merged_data.shape)

print(merged_data.columns)

print(merged_data.info())

print(merged_data.describe())

print(merged_data.isnull().sum())