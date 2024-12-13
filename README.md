# Kecerdasan-Artifisial

## Perubahan yang terjadi

---

## **1. Pembersihan dan Pemrosesan Data**
- **Fungsi `preprocess_serving_grams`**:
  - Mengonversi nilai dalam kolom `Serving`, `Grams`, dan `Calories` ke dalam format numerik yang benar.
  - Menghapus pemisah ribuan (jika ada) dari kolom `Calories` untuk memastikan angka dibaca dengan benar.
  - Menggunakan titik (`.`) sebagai pemisah desimal dan menghapus koma (`,`) jika digunakan sebagai pemisah ribuan.
  - Menambahkan fitur `serving_numeric`, `grams_in_serving`, dan `grams_numeric`.

---

## **2. Penambahan Fitur**
- **Fungsi `add_features`**:
  - Menambahkan fitur tambahan untuk model regresi linear:
    - `interaction_term`: Hasil perkalian antara `serving_numeric` dan `grams_numeric`.
    - `serving_squared`: Kuadrat dari `serving_numeric`.
    - `grams_squared`: Kuadrat dari `grams_numeric`.
    - `interaction_log`: Logaritma dari `grams_numeric + 1` (mencegah log(0)).
    - `sqrt_grams`: Akar kuadrat dari `grams_numeric`.
    - `interaction_squared`: Kuadrat dari `interaction_term`.
    - `calories_log`: Logaritma natural dari `calories` dengan fungsi `log1p` untuk stabilitas.

---

## **3. Implementasi Regresi Linear Manual**
- **Kelas `LinearRegressionManual`**:
  - Implementasi algoritma regresi linear manual.
  - **Fungsi `fit`**:
    - Menghitung bobot menggunakan metode matriks normal dengan regularisasi (Ridge Regression).
  - **Fungsi `predict`**:
    - Menghasilkan prediksi berdasarkan bobot yang telah dihitung.

---

## **4. Evaluasi Model**
- **Metrik Evaluasi**:
  - Fungsi-fungsi evaluasi model ditambahkan:
    - **`mean_absolute_error (MAE)`**
    - **`mean_squared_error (MSE)`**
    - **`r2_score`** untuk menghitung skor R².

---

## **5. Standardisasi Data**
- **MinMaxScaler**:
  - Data fitur distandarisasi agar model lebih stabil.

---

## **6. Optimasi Hyperparameter**
- **Grid Search untuk Lambda (Regularisasi Ridge)**:
  - Rentang nilai lambda diuji (`[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]`) untuk menemukan nilai lambda terbaik menggunakan validasi silang (cross-validation) dengan `KFold` (15 folds).

---

## **7. Prediksi Data Uji**
- Dataset uji diproses dengan langkah yang sama seperti data training.
- **Normalisasi**: Dataset uji distandarisasi dengan skala dari data training.
- **Transformasi Kembali**: Prediksi log (`calories_log`) dikembalikan ke skala asli dengan `expm1`.

---

## **8. Penyimpanan dan Tampilan Hasil**
- **Simpan ke CSV**:
  - Prediksi kalori pada data uji disimpan ke file CSV bernama `Food_Dataset_test_with_calories.csv`.
- **Tampilkan Hasil**:
  - Lima baris pertama dari dataset uji ditampilkan di terminal.

---

### **Poin-Poin Utama:**
1. **Stabilitas Model**:
   - Model memanfaatkan regularisasi Ridge untuk menghindari overfitting.
2. **Penanganan Data Negatif**:
   - Prediksi negatif diganti menjadi nol untuk menghindari kesalahan pada transformasi skala asli.
3. **Evaluasi Akurasi**:
   - Model mencapai skor R² sebesar 88% pada data pelatihan (dengan transformasi log).
