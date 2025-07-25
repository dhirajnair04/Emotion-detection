
# Import necessary libraries
import os
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Load parameters from params.yaml
with open('params.yaml' , 'r') as file:
    params = yaml.safe_load(file)

# Read the dataset from the provided URL
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

# Drop the 'tweet_id' column as it is not needed for analysis
df.drop(columns=['tweet_id'], inplace=True)

# Filter the dataset to keep only 'happiness' and 'sadness' sentiments
final_df = df[df['sentiment'].isin(['happiness','sadness'])]

# Encode 'happiness' as 1 and 'sadness' as 0 for binary classification
final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)

# Split the data into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(final_df, test_size=params['data_ingestion']['test_size'], random_state=42)

# Create the directory to store raw data if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Save the training and testing data to CSV files
train_data.to_csv('data/raw/train_data.csv', index=False)
test_data.to_csv('data/raw/test_data.csv', index=False)
