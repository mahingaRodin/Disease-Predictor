import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved model
model_path = r"C:\Users\user\OneDrive\Desktop\DP\HeartDiseasePredictor\heart_disease_model.pkl"
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Load the dataset
dataset_path = r"C:\Users\user\OneDrive\Desktop\DP\HeartDiseasePredictor\dataset.csv"
dataset = pd.read_csv(dataset_path)

# Preprocess the data
# 1. Encode categorical variables using get_dummies
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
dataset = pd.get_dummies(dataset, columns=categorical_columns)

# 2. Scale numeric features using StandardScaler
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])

# Select a row from the dataset as new input data (replace with actual row index or data)
new_sample = dataset.iloc[0].drop('target').values.reshape(1, -1)  # Drop 'target' column and reshape for prediction

# Convert new_sample to a DataFrame with the correct feature names
new_sample_df = pd.DataFrame(new_sample, columns=dataset.drop('target', axis=1).columns)

# Make predictions
prediction = loaded_model.predict(new_sample_df)
if prediction == 1:
    print("The person is predicted to have heart disease.")
else:
    print("The person is predicted to not have heart disease.")
