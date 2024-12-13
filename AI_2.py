import pandas as pd
import numpy as np
import re
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler

# Fungsi Preprocessing
def preprocess_serving_grams(data):
    def extract_serving(value):
        match = re.search(r'([\d.]+)', str(value))
        return float(match.group()) if match else 0.0

    def extract_grams(value):
        match = re.search(r'\((\d+(?:\.\d+)?)(?:\s*g|\s*ml)\)', str(value))
        return float(match.group(1)) if match else 0.0

    data['serving_numeric'] = np.round(data['serving'].apply(extract_serving), 3)
    data['grams_in_serving'] = np.round(data['serving'].apply(extract_grams), 3)
    data['grams_numeric'] = np.round(data['grams'].apply(lambda x: float(x) if pd.notna(x) and str(x).replace('.', '', 1).isdigit() else 0.0), 3)
    data['calories'] = np.round(data['calories'].replace({',': ''}, regex=True).apply(lambda x: max(float(x), 0.0)), 3)
    return data

# Regresi Linear Manual
class LinearRegressionManual:
    def __init__(self):
        self.weights = None

    def fit(self, X, y, lambda_reg=0.0):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # menambahkan bias
        I = np.eye(X_b.shape[1])
        try:
            self.weights = np.linalg.inv(X_b.T.dot(X_b) + lambda_reg * I).dot(X_b.T).dot(y)
        except np.linalg.LinAlgError:
            print("Inversi matriks gagal. Periksa kualitas data.")

    def predict(self, X):
        if self.weights is not None:
            X_b = np.c_[np.ones((X.shape[0], 1)), X]  # menambahkan bias
            return X_b.dot(self.weights)
        else:
            print("Model belum dilatih.")
            return np.zeros(X.shape[0])

# Metrik Evaluasi
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Load dan train Model
file_path_train = "Food_Dataset_train.csv"
data_train = pd.read_csv(file_path_train)

data_train.dropna(inplace=True)
def add_features(data):
    data['interaction_term'] = np.round(data['serving_numeric'] * data['grams_numeric'], 3)
    data['serving_squared'] = np.round(data['serving_numeric'] ** 2, 3)
    data['grams_squared'] = np.round(data['grams_numeric'] ** 2, 3)
    data['interaction_log'] = np.round(np.log(data['grams_numeric'] + 1), 3)
    data['sqrt_grams'] = np.round(np.sqrt(data['grams_numeric']), 3)
    data['interaction_squared'] = np.round(data['interaction_term'] ** 2, 3)
    data['calories_log'] = np.round(np.log1p(data['calories']), 3)
    return data

data_train = preprocess_serving_grams(data_train)
data_train = add_features(data_train)

X_train = data_train[['serving_numeric', 'grams_in_serving', 'grams_numeric', 'interaction_term', 'serving_squared', 'grams_squared', 'interaction_log', 'sqrt_grams', 'interaction_squared']].values.astype(float)
y_train = data_train['calories_log'].values.astype(float)

# Standarisasi Data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Pencarian Grid untuk Regularisasi Terbaik
kf = KFold(n_splits=15, shuffle=True, random_state=42)
best_lambda = 0.0
best_score = -np.inf

for lambda_reg in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        model = LinearRegressionManual()
        model.fit(X_train[train_idx], y_train[train_idx], lambda_reg)
        y_val_pred = model.predict(X_train[val_idx])
        scores.append(r2_score(y_train[val_idx], y_val_pred))
    avg_score = np.mean(scores)
    if avg_score > best_score:
        best_score = avg_score
        best_lambda = lambda_reg

# Latih Model Akhir
model = LinearRegressionManual()
model.fit(X_train, y_train, lambda_reg=best_lambda)

# Evaluasi Model pada Data train
y_train_pred = model.predict(X_train)
mae = mean_absolute_error(y_train, y_train_pred)
mse = mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)

print(f"Akurasi Model Regresi Linear Manual pada Data Latih:\n MAE: {mae:.2f}, MSE: {mse:.2f}, R^2: {r2:.2f}\n")

# Load dan Prediksi Data test
file_path_test = "Food_Dataset_test.csv"
data_test = pd.read_csv(file_path_test)

data_test = preprocess_serving_grams(data_test)
data_test = add_features(data_test)
X_test = data_test[['serving_numeric', 'grams_in_serving', 'grams_numeric', 'interaction_term', 'serving_squared', 'grams_squared', 'interaction_log', 'sqrt_grams', 'interaction_squared']].values.astype(float)
X_test = scaler.transform(X_test)

# Prediksi
predictions = model.predict(X_test)
predictions = np.where(predictions < 0, 0, predictions)  # Ganti prediksi negatif dengan 0

# Transformasi Prediksi Kembali ke Skala Asli
data_test['calories'] = np.expm1(np.round(predictions, 2))

# Simpan Prediksi
output_file_path = "Food_Dataset_test_with_calories.csv"
data_test.to_csv(output_file_path, index=False)
print(f"Prediksi kalori disimpan ke: {output_file_path}")

# Tampilkan Hasil
print(data_test.head())
