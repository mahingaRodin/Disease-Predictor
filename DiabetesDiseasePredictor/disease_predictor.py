# Save as app.py or run with flask --app disease_predictor run
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load model (update path)
try:
    model = joblib.load(r"C:\Users\user\OneDrive\Desktop\DP\DiabetesDiseasePredictor\diabetes-model.pkl")
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Define EXACT feature order expected by model
        required_features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        # Create DataFrame with enforced column order
        df = pd.DataFrame([data], columns=required_features)
        
        prediction = model.predict(df)[0]
        return jsonify({"prediction": int(prediction)})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)