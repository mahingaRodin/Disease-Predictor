from flask import Flask, request, jsonify
import joblib  # For loading trained models
import pandas as pd

app = Flask(__name__)

# Load your trained model
model = joblib.load(r"C:/Users/user/OneDrive/Desktop/DP/DiabetesDiseasePredictor/diabetes-model.pkl")

# Home route
@app.route('/')
def home():
    return "Welcome to the Disease Predictor API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json()

        # Ensure the input data has the right columns
        required_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        # Validate if the data has all required columns
        if not all(col in data for col in required_columns):
            return jsonify({"error": "Missing required fields in the input data."}), 400

        # Convert the input data to a pandas DataFrame for prediction
        df = pd.DataFrame([data])

        # Make prediction using the model
        prediction = model.predict(df)[0]

        # Convert the prediction to a native Python type (int or float)
        prediction = int(prediction)  # Or float(prediction) if it's a float prediction

        # Return the prediction result
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Favicon route to avoid 404
@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content

from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Allow all origins

if __name__ == "__main__":
    app.run(debug=True)
