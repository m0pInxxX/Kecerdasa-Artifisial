import pandas as pd
import numpy as np
import re

def preprocess_serving_grams(data):
    def extract_numeric(value):
        match = re.search(r'[\d.]+', str(value))  
        return float(match.group()) if match else 0.0
    
    data['serving_numeric'] = data['serving'].apply(extract_numeric)
    data['grams_numeric'] = data['grams'].apply(extract_numeric)
    return data

class LinearRegressionManual:
    def __init__(self):
        self.weights = None  
        self.bias = 0        

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]   
        self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  
        return X_b.dot(self.weights)

file_path_train = "Food_Dataset_train.csv"
data_train = pd.read_csv(file_path_train)

data_train.dropna(inplace=True)  
data_train = preprocess_serving_grams(data_train)
X_train = data_train[['serving_numeric', 'grams_numeric']].values
y_train = data_train['calories'].values

model = LinearRegressionManual()
model.fit(X_train, y_train)

file_path_test = "Food_Dataset_test.csv"
data_test = pd.read_csv(file_path_test)

data_test = preprocess_serving_grams(data_test)

X_test = data_test[['serving_numeric', 'grams_numeric']].values
data_test['calories'] = model.predict(X_test)

output_file_path = "Food_Dataset_test_with_calories.csv"
data_test.to_csv(output_file_path, index=False)
print(f"Predicted calories saved to: {output_file_path}")

print(data_test.head())
