import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Define dataset path
dataset_path = r"C:/Users/user/OneDrive/Desktop/DP/HeartDiseasePredictor/dataset.csv"

# Check if file exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

# Import the dataset
dataset = pd.read_csv(dataset_path)
print(dataset.info())  # Display dataset structure
print(dataset.describe())  # Show dataset summary

# Set plot size
rcParams['figure.figsize'] = (20, 14)

# Correlation Matrix Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Histogram for Each Feature
dataset.hist(figsize=(12, 10), bins=20, edgecolor='black')
plt.suptitle("Feature Distributions")
plt.show()

# Bar Plot for Target Class Distribution
plt.figure(figsize=(8, 6))
plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color=['red', 'green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of Each Target Class')
plt.show()

# Convert Categorical Variables into Dummy Variables
dataset = pd.get_dummies(dataset, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Scaling Numerical Features
scaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])

# Splitting Dataset into Train and Test Sets
y = dataset['target']
X = dataset.drop(['target'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# K-Neighbors Classifier Performance
knn_scores = []
for k in range(1, 21):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))

# Plot K Neighbors Classifier Scores
plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), knn_scores, marker='o', linestyle='-', color='red')
for i in range(1, 21):
    plt.text(i, knn_scores[i-1], f"{knn_scores[i-1]:.2f}", fontsize=9, verticalalignment='bottom')
plt.xticks(range(1, 21))
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy Score')
plt.title('K Neighbors Classifier Accuracy for Different K Values')
plt.show()

# Print the max value reached
print(f"The score for K Neighbors Classifier is {knn_scores[7]*100:.2f}% with {8} neighbors.")

# SVC performance testing with kernels
svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    svc_classifier = SVC(kernel=kernel)
    svc_classifier.fit(X_train, y_train)
    svc_scores.append(svc_classifier.score(X_test, y_test))

# Plotting SVC Kernel Scores
colors = rainbow(np.linspace(0, 1, len(kernels)))
plt.bar(kernels, svc_scores, color=colors)
for i in range(len(kernels)):
    plt.text(i, svc_scores[i], f"{svc_scores[i]:.2f}")
plt.xlabel('Kernels')
plt.ylabel('Scores')
plt.title('Support Vector Classifier scores for different kernels')
plt.show()
print(f"The score for Support Vector Classifier is {svc_scores[0]*100:.2f}% with linear kernel.")

# Decision Tree Classifier performance testing
dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features=i, random_state=0)
    dt_classifier.fit(X_train, y_train)
    dt_scores.append(dt_classifier.score(X_test, y_test))

# Plotting Decision Tree Scores
plt.plot(range(1, len(X.columns) + 1), dt_scores, color='green')
for i in range(1, len(X.columns) + 1):
    plt.text(i, dt_scores[i-1], f"({i}, {dt_scores[i-1]:.2f})")
plt.xticks(range(1, len(X.columns) + 1))
plt.xlabel('Max Features')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier scores for different max features')
plt.show()
print(f"The score for Decision Tree Classifier is {dt_scores[17]*100:.2f}% with max features [2, 4, 18].")

# Random Forest Classifier performance testing
rf_scores = []
estimators = [10, 100, 200, 500, 1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators=i, random_state=0)
    rf_classifier.fit(X_train, y_train)
    rf_scores.append(rf_classifier.score(X_test, y_test))

# Plotting Random Forest Scores
colors = rainbow(np.linspace(0, 1, len(estimators)))
plt.bar(range(len(estimators)), rf_scores, color=colors, width=0.8)

# Correcting the indexing issue
for idx, score in enumerate(rf_scores):
    plt.text(idx, score, f"{score:.2f}")

plt.xticks(range(len(estimators)), labels=[str(estimator) for estimator in estimators])
plt.xlabel('Number of Estimators')
plt.ylabel('Scores')
plt.title('Random Forest Classifier scores for different estimators')
plt.show()

# Saving the best model using pickle
best_model = RandomForestClassifier(n_estimators=1000, random_state=0)
best_model.fit(X_train, y_train)

model_path = "heart_disease_model.pkl"
with open(model_path, 'wb') as file:
    pickle.dump(best_model, file)

print(f"Model saved at: {model_path}")
