import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#loading the saved model 
model_path = r"C:\Users\user\OneDrive\Desktop\DP\HeartDiseasePredictor\heart_disease_model.pkl"
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

#load the dataset
dataset_path = r"C:\Users\user\OneDrive\Desktop\DP\HeartDiseasePredictor\dataset.csv"
dataset = pd.read_csv(dataset_path)

#preprocess the data
#1. Encode categorical variables using get_dummies
categorical_columns = ['sex','cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
dataset = pd.get_dummies(dataset, columns=categorical_columns)

# 2. scale numeric features using standardscalar
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])

# Example: Select a row from the dataset as new input data (replace with actual row index or data)
# Let's assume we are predicting for the first row in the dataset

new_sample = dataset.iloc[0].drop('target').values.reshape(1, -1)  # Drop 'target' column and reshape for prediction

#make predictions 
prediction = loaded_model.predict(new_sample)
print("Predicted class: ", prediction)
