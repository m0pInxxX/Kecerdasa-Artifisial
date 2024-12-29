import pandas as pd
import numpy as np
import re
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor

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
        I[0, 0] = 0  # Jangan regularisasi bias
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

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

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
    
    # Fitur tambahan lanjut
    data['grams_serving_ratio'] = np.round(data['grams_numeric'] / (data['serving_numeric'] + 1), 3)
    data['grams_interaction'] = np.round(data['grams_numeric'] * data['interaction_term'], 3)
    data['log_serving'] = np.round(np.log1p(data['serving_numeric']), 3)
    data['log_grams'] = np.round(np.log1p(data['grams_numeric']), 3)
    data['log_interaction'] = np.round(np.log1p(data['interaction_term']), 3)
    data['interaction_root'] = np.round(np.cbrt(data['interaction_term']), 3)
    data['grams_log_ratio'] = np.round(data['log_grams'] / (data['log_serving'] + 1), 3)
    data['interaction_cube'] = np.round(data['interaction_term'] ** 3, 3)
    data['serving_cube'] = np.round(data['serving_numeric'] ** 3, 3)
    data['grams_cube'] = np.round(data['grams_numeric'] ** 3, 3)
    data['interaction_exp'] = np.round(np.exp(data['interaction_log']), 3)
    return data

# Preprocessing dan Tambah Fitur
data_train = preprocess_serving_grams(data_train)
data_train = add_features(data_train)

X_train = data_train[['serving_numeric', 'grams_in_serving', 'grams_numeric', 'interaction_term', 'serving_squared', 'grams_squared', 'interaction_log', 'sqrt_grams', 'interaction_squared']].values.astype(float)
y_train = data_train['calories_log'].values.astype(float)

# Tambahkan Fitur Polinomial
poly = PolynomialFeatures(degree=6, include_bias=False)
X_train_poly = poly.fit_transform(X_train)

# Seleksi Fitur Terbaik
selector = SelectKBest(score_func=f_regression, k=10)
X_train_selected = selector.fit_transform(X_train_poly, y_train)

# Standarisasi Data
scaler = MinMaxScaler()
X_train_selected = scaler.fit_transform(X_train_selected)

# Model Linear dan Random Forest
model_linear = LinearRegressionManual()
model_rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)

# Latih Model
model_linear.fit(X_train_selected, y_train, lambda_reg=0.1)
model_rf.fit(X_train_selected, y_train)

# Evaluasi Model
y_train_pred_linear = model_linear.predict(X_train_selected)
y_train_pred_rf = model_rf.predict(X_train_selected)
y_train_pred = 0.5 * y_train_pred_linear + 0.5 * y_train_pred_rf
y_train_pred_original = np.expm1(y_train_pred)
y_train_original = np.expm1(y_train)
mae = mean_absolute_error(y_train_original, y_train_pred_original)
mse = mean_squared_error(y_train_original, y_train_pred_original)
r2 = r2_score(y_train_original, y_train_pred_original)
mape = mean_absolute_percentage_error(y_train_original, y_train_pred_original)

print(f"Akurasi Model Regresi Hybrid pada Data Latih:\n MAE: {mae:.2f}, MSE: {mse:.2f}, R^2: {r2:.2f}, MAPE : {mape:.2f}\n")

# Prediksi pada Data Uji
file_path_test = "Food_Dataset_test.csv"
data_test = pd.read_csv(file_path_test)
data_test = preprocess_serving_grams(data_test)
data_test = add_features(data_test)
X_test = data_test[['serving_numeric', 'grams_in_serving', 'grams_numeric', 'interaction_term', 'serving_squared', 'grams_squared', 'interaction_log', 'sqrt_grams', 'interaction_squared']].values.astype(float)
X_test_poly = poly.transform(X_test)
X_test_selected = selector.transform(X_test_poly)
X_test_selected = scaler.transform(X_test_selected)

# Prediksi
predictions_linear = model_linear.predict(X_test_selected)
predictions_rf = model_rf.predict(X_test_selected)
predictions = 0.5 * predictions_linear + 0.5 * predictions_rf
predictions_original = np.expm1(predictions)

# Simpan Prediksi
output_file_path = "Food_Dataset_test_with_calories.csv"
data_test['calories'] = np.round(predictions_original, 2)
data_test.to_csv(output_file_path, index=False)
print(f"Prediksi kalori disimpan ke: {output_file_path}")

print(data_test.head().round(2))
