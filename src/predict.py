from sympy import false
from ucimlrepo import fetch_ucirepo
from utils import define_model, generate_advice
import torch


# Load the saved model and scaler
model = define_model(input_size=13, output_size=5)  # Update input and output sizes accordingly
model.load_state_dict(torch.load("../models/health_model.pth", weights_only=True))  # Use weights_only=True for safety
model.eval()

scaler_state = torch.load("../models/scaler.pth", weights_only=false)  # Use weights_only=True for safety
scaler = scaler_state["scaler"]

# Load test data (replace this with actual data input if available)
heart_disease = fetch_ucirepo(id=45)
X_test = heart_disease.data.features
y_test = heart_disease.data.targets.iloc[:, 0]  # Convert to Series

# Normalize the test data
X_test = scaler.transform(X_test)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Test predictions
with torch.no_grad():
    predictions = model(X_test_tensor).argmax(dim=1)

# Calculate accuracy
correct = (predictions == y_test_tensor).sum().item()
accuracy = correct / len(y_test_tensor)
print(f"Test Accuracy: {accuracy:.2%}")

# Display predictions and advice
for pred in predictions:
    print(f"Predicted Class: {pred.item()}, Advice: {generate_advice(pred.item())}")
