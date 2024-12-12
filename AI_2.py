import pandas as pd
import numpy as np
import re

# Preprocessing Function
def preprocess_serving_grams(data):
    def extract_serving(value):
        match = re.search(r'([\d.]+)', str(value))
        return float(match.group()) if match else 0.0

    def extract_grams(value):
        match = re.search(r'\((\d+(?:\.\d+)?)(?:\s*g|\s*ml)\)', str(value))
        return float(match.group(1)) if match else 0.0

    def extract_calories(value):
        match = re.search(r'([\d.]+)', str(value))
        return float(match.group()) if match else 0.0

    data['serving_numeric'] = data['serving'].apply(extract_serving)
    data['grams_in_serving'] = data['serving'].apply(extract_grams)
    data['grams_numeric'] = data['grams'].apply(lambda x: float(x) if pd.notna(x) and str(x).replace('.', '', 1).isdigit() else 0.0)
    data['calories'] = data['calories'].replace({',': ''}, regex=True).apply(lambda x: float(x))
    return data

# Manual Linear Regression Class
class LinearRegressionManual:
    def __init__(self):
        self.weights = None

    def fit(self, X, y, lambda_reg=0.0):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        I = np.eye(X_b.shape[1])
        try:
            self.weights = np.linalg.inv(X_b.T.dot(X_b) + lambda_reg * I).dot(X_b.T).dot(y)
        except np.linalg.LinAlgError:
            print("Matrix inversion failed. Check the data quality.")

    def predict(self, X):
        if self.weights is not None:
            X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
            return X_b.dot(self.weights)
        else:
            print("Model is not trained.")
            return np.zeros(X.shape[0])

# Evaluation Metrics

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Load and Train the Model
file_path_train = "Food_Dataset_train.csv"
data_train = pd.read_csv(file_path_train)

data_train.dropna(inplace=True)
def add_features(data):
    data['interaction_term'] = data['serving_numeric'] * data['grams_numeric']
    data['serving_squared'] = data['serving_numeric'] ** 2
    data['grams_squared'] = data['grams_numeric'] ** 2
    data['interaction_log'] = np.log(data['grams_numeric'] + 1)
    return data

# Apply the feature addition
data_train = preprocess_serving_grams(data_train)
data_train = add_features(data_train)

X_train = data_train[['serving_numeric', 'grams_in_serving', 'grams_numeric', 'interaction_term', 'serving_squared', 'grams_squared', 'interaction_log']].values.astype(float)
y_train = data_train['calories'].values.astype(float)

# Standardize the Data
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_train = (X_train - X_train_mean) / X_train_std

model = LinearRegressionManual()
model.fit(X_train, y_train, lambda_reg=1.0)

# Evaluate Model on Training Data
y_train_pred = model.predict(X_train)
mae = mean_absolute_error(y_train, y_train_pred)
mse = mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)

print(f"Manual Linear Regression Model Accuracy on Training Data:\n MAE: {mae:.2f}, MSE: {mse:.2f}, R^2: {r2:.2f}\n")

# Load and Predict Test Data
file_path_test = "Food_Dataset_test.csv"
data_test = pd.read_csv(file_path_test)

data_test = preprocess_serving_grams(data_test)
data_test = add_features(data_test)
X_test = data_test[['serving_numeric', 'grams_in_serving', 'grams_numeric', 'interaction_term', 'serving_squared', 'grams_squared', 'interaction_log']].values.astype(float)
X_test = (X_test - X_train_mean) / X_train_std

data_test['calories'] = np.round(model.predict(X_test), 2)

# Save Predictions
output_file_path = "Food_Dataset_test_with_calories.csv"
data_test.to_csv(output_file_path, index=False)
print(f"Predicted calories saved to: {output_file_path}")

# Display Results
print(data_test.head())
