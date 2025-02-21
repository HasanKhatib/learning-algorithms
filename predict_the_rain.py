"""
Lab 3: 3.2 Predict the Rain
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix

# Step 1: Load & Preprocess Data
print("Loading dataset...")
# Load dataset and drop date column data/seattle-weather.csv
df = pd.read_csv("data/seattle-weather.csv")
df.drop(columns=["date"], inplace=True)

# Encode categorical target variable (weather) into numeric labels
label_encoder = LabelEncoder()
df["weather"] = label_encoder.fit_transform(df["weather"])  # Convert "rain", "sun" to numbers

# Define input (X) and output (y)
X = df.drop(columns=["weather"])
y = df["weather"]

# improves SVM performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Dataset loaded and preprocessed.\n")

# Step 2: Train Random Forest
print("Training Random Forest model...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate Random Forest
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_mse = mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest -> Accuracy: {rf_accuracy:.2f}, MSE: {rf_mse:.2f}")

# Step 3: Train SVM
print("Training SVM model...")
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Evaluate SVM
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_mse = mean_squared_error(y_test, y_pred_svm)
print(f"SVM -> Accuracy: {svm_accuracy:.2f}, MSE: {svm_mse:.2f}")

# Step 4: Train Linear Regression (Modified for Classification)
print("Training Linear Regression model...")
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions (convert continuous output to nearest class)
y_pred_lr = np.round(lr.predict(X_test)).astype(int)
y_pred_lr = np.clip(y_pred_lr, 0, len(label_encoder.classes_) - 1)  # Ensure values stay within valid classes

# Evaluate Linear Regression
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_mse = mean_squared_error(y_test, y_pred_lr)
print(f"Linear Regression -> Accuracy: {lr_accuracy:.2f}, MSE: {lr_mse:.2f}")

# Step 5: Visualization

# Plot MSE for All Models
print("Plotting MSE comparison...")
models = ["Random Forest", "SVM", "Linear Regression"]
mse_values = [rf_mse, svm_mse, lr_mse]

plt.figure(figsize=(8, 5))
plt.bar(models, mse_values, color=["blue", "green", "red"])
plt.title("Mean Squared Error of Different Models")
plt.ylabel("MSE")
plt.show()

# Confusion Matrix for Random Forest
print("Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Random Forest")
plt.show()
