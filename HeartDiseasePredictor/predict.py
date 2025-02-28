import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the saved model
model_path = r"C:\Users\user\OneDrive\Desktop\DP\HeartDiseasePredictor\heart_disease_model.pkl"
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print(f"Error: The model file '{model_path}' was not found. Please ensure it's in the correct path.")
    exit()

# Function to preprocess the input data (similar to preprocessing in training)
def preprocess_input(data, columns_to_scale, all_columns):
    # Scaling numerical features (apply the same scaling as in the training phase)
    scaler = StandardScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    # Convert categorical variables into dummy variables (as done during training)
    data = pd.get_dummies(data, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

    # Align columns with the model's training data (adding missing columns with zeros)
    missing_cols = set(all_columns) - set(data.columns)
    for c in missing_cols:
        data[c] = 0

    # Reorder columns to match the training data (if there are extra columns)
    data = data[all_columns]
    
    return data

# Function to make predictions
def make_prediction(input_data, columns_to_scale, all_columns):
    # Preprocess the input data
    processed_data = preprocess_input(input_data, columns_to_scale, all_columns)
    
    # Make a prediction using the loaded model
    prediction = model.predict(processed_data)
    return prediction

# Example: using the model to predict heart disease status
# You can replace this with input from users, or an example test dataset
# Example data: a sample entry with the same structure as the dataset
example_data = pd.DataFrame({
    'age': [63],
    'sex': [1],  # Male
    'cp': [3],   # Typical angina
    'trestbps': [145],
    'chol': [233],
    'fbs': [1],  # Fasting blood sugar > 120 mg/dl
    'restecg': [0],  # Normal
    'thalach': [150],
    'exang': [0],  # No exercise induced angina
    'oldpeak': [2.3],
    'slope': [3],  # Downsloping
    'ca': [0],  # No major vessels colored by fluoroscopy
    'thal': [2]   # Normal
})

# List of columns that were used for scaling in the training phase
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# All columns that were used during training (should match the columns of the training data)
all_columns = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex_0', 'sex_1', 'cp_0', 'cp_1', 'cp_2', 'cp_3',
    'fbs_0', 'fbs_1', 'restecg_0', 'restecg_1', 'restecg_2', 'exang_0', 'exang_1', 'slope_0', 'slope_1', 'slope_2',
    'ca_0', 'ca_1', 'ca_2', 'ca_3', 'ca_4', 'thal_0', 'thal_1', 'thal_2', 'thal_3'
]

# Make prediction
prediction = make_prediction(example_data, columns_to_scale, all_columns)

# Print result
if prediction[0] == 1:
    print("Prediction: Heart disease likely.")
else:
    print("Prediction: No heart disease detected.")
