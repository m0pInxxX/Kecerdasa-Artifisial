import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# Preprocessing Function
def preprocess_serving_grams(data):
    def extract_serving(value):
        match = re.search(r'([\d.]+)', str(value))
        return float(match.group()) if match else 0.0

    def extract_grams(value):
        match = re.search(r'\(([\d.]+) g\)', str(value))
        return float(match.group(1)) if match else 0.0

    data['serving_numeric'] = data['serving'].apply(extract_serving)
    data['grams_in_serving'] = data['serving'].apply(extract_grams)
    data['grams_numeric'] = data['grams'].apply(lambda x: float(x) if pd.notna(x) and str(x).replace('.', '', 1).isdigit() else 0.0)
    data['interaction_term'] = data['serving_numeric'] * data['grams_numeric']
    return data

# Load and Train the Model
file_path_train = "Food_Dataset_train.csv"
data_train = pd.read_csv(file_path_train)

# Ensure correct data types and remove invalid rows
data_train.dropna(inplace=True)
data_train = preprocess_serving_grams(data_train)

# Standardize the Features
scaler = StandardScaler()
X_train = data_train[['serving_numeric', 'grams_in_serving', 'grams_numeric', 'interaction_term']].values.astype(float)
X_train_scaled = scaler.fit_transform(X_train)
y_train = data_train['calories'].values.astype(float)

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)

# Ridge Regression Model
model = Ridge(alpha=1.0)
model.fit(X_train_poly, y_train)

# Cross-validation Scores
cv_scores = cross_val_score(model, X_train_poly, y_train, cv=5, scoring='r2')

# Evaluate Model on Training Data
y_train_pred = model.predict(X_train_poly)
mae = mean_absolute_error(y_train, y_train_pred)
mse = mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)

print(f"Model Accuracy on Training Data:\n MAE: {mae:.2f}, MSE: {mse:.2f}, R^2: {r2:.2f}")
print(f"Cross-Validation R^2 Scores: {cv_scores}")
print(f"Average Cross-Validation R^2: {cv_scores.mean():.2f}\n")

# Load and Predict Test Data
file_path_test = "Food_Dataset_test.csv"
data_test = pd.read_csv(file_path_test)

# Ensure correct preprocessing and data types
data_test = preprocess_serving_grams(data_test)
X_test = data_test[['serving_numeric', 'grams_in_serving', 'grams_numeric', 'interaction_term']].values.astype(float)
X_test_scaled = scaler.transform(X_test)
X_test_poly = poly.transform(X_test_scaled)

data_test['calories'] = model.predict(X_test_poly)

# Save Predictions
output_file_path = "Food_Dataset_test_with_calories.csv"
data_test.to_csv(output_file_path, index=False)
print(f"Predicted calories saved to: {output_file_path}")

# Display Results
print(data_test.head())
