# Health Prediction AI Model

## Project Description

This project involves creating an AI-powered health prediction model using PyTorch. The model predicts potential diseases based on a person's health attributes and provides personalized health advice. The system is designed to handle structured health data (e.g., CSV datasets) and supports predictions for various diseases.

---

## Features

- **Disease Prediction**: Predicts potential diseases based on health attributes.
- **Health Advice**: Offers actionable advice tailored to the predicted disease.
- **CSV Data Support**: Processes structured health data from CSV files.
- **Scalable Model Architecture**: Neural network implemented using PyTorch.

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - **PyTorch**: For building and training the neural network model.
  - **Pandas**: For data manipulation and cleaning.
  - **NumPy**: For numerical computations.
  - **scikit-learn**: For preprocessing and train-test splitting.

---

## Setup and Installation

### Prerequisites

- Python 3.8 or later
- pip (Python package manager)

### Installation Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/health-prediction-ai.git
   cd health-prediction-ai
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare your dataset:
   - Ensure your CSV file has the necessary health attributes and a label column for diseases.
   - Place your dataset in the project directory.

---

## How to Use

### 1. Load Your Dataset

- Place your dataset (e.g., `health_data.csv`) in the root directory.
- Update the `data_path` in the script to point to your dataset.

### 2. Train the Model

Run the training script to train the neural network:

```bash
python train.py
```

### 3. Make Predictions

Use the trained model to make predictions for new patient data:

```bash
python predict.py --input "60,150,75,30,0,1"  # Example input features
```

The output will display the predicted disease and health advice.

---

## Project Structure

```
health-prediction-ai/
├── data/
│   └── health_data.csv        # Example dataset
├── models/
│   └── health_model.pth       # Trained model weights
├── src/
│   ├── train.py               # Training script
│   ├── predict.py             # Prediction script
│   └── utils.py               # Helper functions (e.g., preprocessing, advice generation)
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
```

---

## Key Concepts

### Neural Network

- **Architecture**: Fully connected layers with ReLU activations.
- **Loss Function**: CrossEntropyLoss for multi-class classification.
- **Optimizer**: Adam optimizer for efficient gradient updates.

### Preprocessing

- **Imputation**: Filling missing values with the median.
- **Normalization**: Scaling features using StandardScaler to standardize the range.

### Dataset Requirements

- Input features: Numerical data representing patient health attributes (e.g., age, blood pressure).
- Labels: Categorical disease labels for supervised training.

---

## Future Enhancements

1. Support for real-time data collection using IoT devices.
2. Expansion to handle unstructured data (e.g., images or text).
3. Deployment as a web or mobile app for broader accessibility.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push to the branch.
4. Submit a pull request for review.

---

## Contact

For any inquiries or feedback, feel free to contact:

- **Name**: Mahinga Rodin
- **Email**: [mahingarodin@gmail.com](mailto:mahingarodin@gmail.com)
- **GitHub**: [https://github.com/mahingaRodin/Disease-Predictor](https://github.com/mahingaRodin/Disease-Predictor)
