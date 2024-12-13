import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_squared_error, r2_score
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

# Manual Decision Tree Implementation
class ManualDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        if len(set(y)) == 1 or len(y) < self.min_samples_split or depth >= self.max_depth:
            return np.mean(y)
        feature, threshold = self.best_split(X, y)
        if feature is None:
            return np.mean(y)
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold
        left = self.fit(X[left_idx], y[left_idx], depth + 1)
        right = self.fit(X[right_idx], y[right_idx], depth + 1)
        self.tree = (feature, threshold, left, right)
        return self.tree

    def best_split(self, X, y):
        best_feature, best_threshold, best_mse = None, None, float('inf')
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                mse = self.calculate_mse(y[left_idx], y[right_idx])
                if mse < best_mse:
                    best_feature, best_threshold, best_mse = feature, threshold, mse
        return best_feature, best_threshold

    def calculate_mse(self, y_left, y_right):
        left_mse = np.var(y_left) * len(y_left) if len(y_left) > 0 else 0
        right_mse = np.var(y_right) * len(y_right) if len(y_right) > 0 else 0
        return left_mse + right_mse

    def predict_sample(self, x, node):
        if not isinstance(node, tuple):
            return node
        feature, threshold, left, right = node
        if x[feature] <= threshold:
            return self.predict_sample(x, left)
        else:
            return self.predict_sample(x, right)

    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])

# Manual Random Forest Implementation
class ManualRandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree = ManualDecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_predictions, axis=0)

# Load Data train dan Tambah Fitur
file_path_train = "Food_Dataset_train.csv"
data_train = pd.read_csv(file_path_train)

data_train.dropna(inplace=True)

def add_features(data):
    data['interaction_term'] = np.round(data['serving_numeric'] * data['grams_numeric'], 3)
    data['serving_squared'] = np.round(data['serving_numeric'] ** 2, 3)
    data['grams_squared'] = np.round(data['grams_numeric'] ** 2, 3)
    data['interaction_log'] = np.round(np.log(data['grams_numeric'] + 1), 3)
    return data

data_train = preprocess_serving_grams(data_train)
data_train = add_features(data_train)

X_train = data_train[['serving_numeric', 'grams_in_serving', 'grams_numeric', 'interaction_term', 'serving_squared', 'grams_squared', 'interaction_log']].values.astype(float)
y_train = data_train['calories'].values.astype(float)

# Latih Model Random Forest Manual
model = ManualRandomForest(n_trees=100, max_depth=5, min_samples_split=2)
model.fit(X_train, y_train)

# Evaluasi Model
train_predictions = model.predict(X_train)
mse = round(mean_squared_error(y_train, train_predictions), 2)
r2 = round(r2_score(y_train, train_predictions), 2)

print(f"Akurasi Model pada Data Latih:\nMean Squared Error: {mse:.2f}\nR² Score: {r2:.2f}\n")

# Load Data test dan Tambah Fitur
file_path_test = "Food_Dataset_test.csv"
data_test = pd.read_csv(file_path_test)

data_test = preprocess_serving_grams(data_test)
data_test = add_features(data_test)
X_test = data_test[['serving_numeric', 'grams_in_serving', 'grams_numeric', 'interaction_term', 'serving_squared', 'grams_squared', 'interaction_log']].values.astype(float)

# Prediksi Data test
predictions = np.round(model.predict(X_test), 2)
data_test['rf_calories'] = np.round(predictions, 2)
data_test['calories'] = np.round(predictions, 2)

# Simpan Prediksi ke CSV
output_file_path = "Food_Dataset_test_with_calories.csv"
data_test.to_csv(output_file_path, index=False)
print(f"Prediksi kalori disimpan ke: {output_file_path}")

# Tampilkan Hasil Data Uji
print(data_test.head())
