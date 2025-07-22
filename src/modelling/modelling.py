# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier

# Load parameters from params.yaml
with open('params.yaml', 'r') as file:
    params=yaml.safe_load(file)

# Load the training data (Bag of Words features)
train_data = pd.read_csv("data/interim/train_bow.csv")

# Separate features and target variable
x_train = train_data.drop(columns=['label']).values
y_train = train_data['label'].values

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=params['modelling']['n_estimators'], max_depth=params['modelling']['max_depth'], random_state=42)

# Train the model on the training data
model.fit(x_train, y_train)

# Save the trained model to a file using pickle
pickle.dump(model, open("models/random_forest_model.pkl", "wb"))
